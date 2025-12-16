"""
RL Agent for Recipe Generation (Phase 1: Single Agent)
Uses Stable Baselines3 for training
HRM-Ready: Designed to work as low-level policy in future hierarchical setup
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from typing import Optional, Dict
import config


class RecipePolicyNetwork(BaseFeaturesExtractor):
    """
    Custom feature extractor for recipe generation
    
    Architecture designed to:
    - Process current nutrient state
    - Process target constraints
    - Output ingredient selection policy
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Neural network architecture
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.feature_net(observations)


class RecipeAgent:
    """
    RL Agent for Recipe Generation
    
    Phase 1: Standalone agent
    Phase 2: Can be used as low-level policy in HRM
    """
    
    def __init__(
        self,
        env,
        algorithm: str = "PPO",
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Initialize Recipe Agent
        
        Args:
            env: RecipeEnv instance (or vectorized env)
            algorithm: "PPO" or "DQN"
            policy_kwargs: Custom policy network config
            verbose: Logging verbosity
        """
        self.env = env
        self.algorithm = algorithm
        
        # Default policy kwargs
        if policy_kwargs is None:
            policy_kwargs = {
                'features_extractor_class': RecipePolicyNetwork,
                'features_extractor_kwargs': {'features_dim': 256},
                'net_arch': [256, 256]  # Actor-critic network sizes
            }
        
        # Initialize RL algorithm
        if algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config.RL_CONFIG['learning_rate'],
                gamma=config.RL_CONFIG['gamma'],
                batch_size=config.RL_CONFIG['batch_size'],
                n_steps=config.RL_CONFIG['n_steps'],
                n_epochs=config.RL_CONFIG['n_epochs'],
                ent_coef=config.RL_CONFIG.get('ent_coef', 0.01),  # Entropy for exploration
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                tensorboard_log=config.TENSORBOARD_LOG_DIR,
                device='cpu'  # Use CPU to avoid GPU warnings
            )
        elif algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=config.RL_CONFIG['learning_rate'],
                gamma=config.RL_CONFIG['gamma'],
                batch_size=config.RL_CONFIG['batch_size'],
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                tensorboard_log=config.TENSORBOARD_LOG_DIR
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train(self, total_timesteps: int, callback=None):
        """
        Train the agent
        
        Args:
            total_timesteps: Number of training steps
            callback: Optional training callback
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(self, observation, deterministic: bool = True):
        """
        Predict action for given observation
        
        Args:
            observation: Current state
            deterministic: Use deterministic policy
            
        Returns:
            action, state
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        if self.algorithm == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.algorithm == "DQN":
            self.model = DQN.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def generate_recipe(self, constraints: Optional[Dict] = None, render: bool = False, deterministic: bool = True):
        """
        Generate a complete recipe
        
        Args:
            constraints: Optional constraint override (for HRM)
            render: Whether to render each step
            deterministic: Use deterministic policy (False for stochastic/diversity)
            
        Returns:
            recipe_info: Dict with ingredients and nutrients
        """
        # Reset environment with optional constraints
        obs, info = self.env.reset(options={'constraints': constraints} if constraints else None)
        
        done = False
        steps = 0
        max_steps = config.MAX_INGREDIENTS_PER_RECIPE
        
        while not done and steps < max_steps:
            action, _ = self.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
            
            if render:
                self.env.render()
        
        return info


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress
    """
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Log episode metrics
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    self.episode_count += 1
                    
                    # Log to tensorboard
                    if self.episode_count % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        avg_length = np.mean(self.episode_lengths[-10:])
                        self.logger.record('train/episode_reward_mean', avg_reward)
                        self.logger.record('train/episode_length_mean', avg_length)
        
        return True


def create_vectorized_env(env_fn, n_envs: int = 4):
    """
    Create vectorized environment for parallel training
    
    Args:
        env_fn: Function that returns a RecipeEnv instance
        n_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    return DummyVecEnv([env_fn for _ in range(n_envs)])


if __name__ == "__main__":
    print("Recipe Agent Module")
    print("Use train/train_recipe.py for training")
