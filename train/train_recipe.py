"""
Training script for Recipe Generation Agent (Phase 1)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import pandas as pd
import numpy as np
import config
from env.recipe_env import RecipeEnv
from models.ingredient_policy import RecipeAgent, TrainingCallback, create_vectorized_env
from utils.data_preprocessing import load_processed_data


def setup_training_environment(user_profile: str = 'standard'):
    """
    Set up training environment with specified user profile
    
    Args:
        user_profile: User profile name from config.USER_PROFILES
        
    Returns:
        env: RecipeEnv instance
    """
    # Load processed ingredients
    print("Loading processed ingredient data...")
    try:
        ingredients_df = load_processed_data()
    except FileNotFoundError:
        print("Processed data not found. Run data preprocessing first:")
        print("  python utils/data_preprocessing.py")
        sys.exit(1)
    
    # Get constraints for user profile
    constraints = config.USER_PROFILES.get(user_profile, config.DEFAULT_CONSTRAINTS)
    print(f"Using user profile: {user_profile}")
    
    # Create environment
    env = RecipeEnv(
        ingredient_df=ingredients_df,
        constraints=constraints,
        recipe_history=[],
        hrm_mode=False  # Phase 1: standalone agent
    )
    
    return env


def train_agent(
    user_profile: str = 'standard',
    algorithm: str = 'PPO',
    total_timesteps: int = None,
    save_path: str = None
):
    """
    Train recipe generation agent
    
    Args:
        user_profile: User profile name
        algorithm: 'PPO' or 'DQN'
        total_timesteps: Training steps (if None, uses config default)
        save_path: Model save path (if None, uses default)
    """
    print("="*60)
    print("RECIPE GENERATION AGENT - PHASE 1 TRAINING")
    print("="*60)
    
    # Setup environment
    env = setup_training_environment(user_profile)
    
    # Vectorize for faster training (optional)
    # env = create_vectorized_env(lambda: setup_training_environment(user_profile), n_envs=4)
    
    # Initialize agent
    print(f"\nInitializing {algorithm} agent...")
    agent = RecipeAgent(env, algorithm=algorithm)
    
    # Training parameters
    if total_timesteps is None:
        total_timesteps = config.RL_CONFIG['total_timesteps']
    
    # Create callback
    callback = TrainingCallback(eval_freq=1000)
    
    # Train
    print(f"\nTraining for {total_timesteps} timesteps...")
    print("Monitor progress with: tensorboard --logdir logs/tensorboard/")
    print("-"*60)
    
    agent.train(total_timesteps=total_timesteps, callback=callback)
    
    # Save model
    if save_path is None:
        save_path = Path(config.MODEL_SAVE_DIR) / f"recipe_agent_{user_profile}_{algorithm.lower()}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    agent.save(str(save_path))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {save_path}")
    print(f"Training episodes: {callback.episode_count}")
    if callback.episode_rewards:
        print(f"Average reward (last 100): {np.mean(callback.episode_rewards[-100:]):.2f}")
    
    return agent


def test_agent(agent: RecipeAgent, n_recipes: int = 5):
    """
    Test trained agent by generating sample recipes
    
    Args:
        agent: Trained RecipeAgent
        n_recipes: Number of recipes to generate
    """
    print("\n" + "="*60)
    print("GENERATING TEST RECIPES")
    print("="*60)
    
    for i in range(n_recipes):
        print(f"\n--- Recipe {i+1} ---")
        recipe_info = agent.generate_recipe(render=True)
        
        compliance = recipe_info['compliance']
        all_met = recipe_info['all_constraints_met']
        
        print(f"\nConstraints satisfied: {'✓ All' if all_met else '✗ Some failed'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Recipe Generation Agent')
    parser.add_argument('--profile', type=str, default='standard',
                       choices=['standard', 'low_sodium', 'high_protein', 'low_carb'],
                       help='User profile for training')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'DQN'],
                       help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Training timesteps (default from config)')
    parser.add_argument('--test', action='store_true',
                       help='Generate test recipes after training')
    
    args = parser.parse_args()
    
    # Train agent
    agent = train_agent(
        user_profile=args.profile,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps
    )
    
    # Test if requested
    if args.test:
        test_agent(agent, n_recipes=5)
