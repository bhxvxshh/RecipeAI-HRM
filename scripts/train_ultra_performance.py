"""
Ultra-High Performance Training - Target: 85%+ ALL nutrients

Strategy:
1. 5-phase FAT-focused curriculum (800k timesteps)
2. Fat gets special treatment with wider initial ranges
3. Aggressive entropy annealing for exploitation
4. Enhanced reward system v3 with fat priority
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.recipe_env import RecipeEnv
from config import DEFAULT_CONSTRAINTS
import config_reward_v3 as reward_config


class FatFocusedCurriculumScheduler:
    """
    5-phase curriculum with SPECIAL FOCUS on fat constraint.
    Fat gets progressively tighter ranges while other nutrients follow standard curriculum.
    """
    
    def __init__(self, total_timesteps=800000):
        self.total_timesteps = total_timesteps
        self.phases = reward_config.FAT_CURRICULUM_PHASES
        
        self.cumulative_steps = [0]
        for phase in self.phases:
            self.cumulative_steps.append(self.cumulative_steps[-1] + phase['steps'])
    
    def get_current_phase(self, step):
        for i, threshold in enumerate(self.cumulative_steps[1:]):
            if step < threshold:
                return i, self.phases[i]
        return len(self.phases) - 1, self.phases[-1]
    
    def get_constraint_multipliers(self, step):
        """
        Returns multipliers for each nutrient.
        Fat gets special treatment, others follow standard curriculum.
        """
        phase_idx, phase = self.get_current_phase(step)
        
        fat_multiplier = phase['fat_multiplier']
        
        # Standard multipliers for other nutrients (less aggressive)
        if phase_idx == 0:  # Easy
            standard_multiplier = 1.5
        elif phase_idx == 1:  # Medium
            standard_multiplier = 1.35
        elif phase_idx == 2:  # Normal
            standard_multiplier = 1.2
        elif phase_idx == 3:  # Tight
            standard_multiplier = 1.1
        else:  # Target
            standard_multiplier = 1.0
        
        return {
            'fat': fat_multiplier,
            'calories': standard_multiplier,
            'protein': standard_multiplier,
            'carbs': standard_multiplier,
            'sodium': standard_multiplier,
        }
    
    def get_entropy_coef(self, step):
        """Aggressive entropy annealing for exploitation"""
        phase_idx, phase = self.get_current_phase(step)
        
        # Start high, end very low
        entropies = [0.15, 0.10, 0.05, 0.02, 0.005]
        
        if phase_idx < len(entropies):
            current_entropy = entropies[phase_idx]
            if phase_idx < len(entropies) - 1:
                next_entropy = entropies[phase_idx + 1]
                phase_start = self.cumulative_steps[phase_idx]
                phase_end = self.cumulative_steps[phase_idx + 1]
                phase_progress = (step - phase_start) / (phase_end - phase_start)
                return current_entropy + (next_entropy - current_entropy) * phase_progress
            return current_entropy
        return entropies[-1]
    
    def print_status(self, step):
        phase_idx, phase = self.get_current_phase(step)
        progress = (step / self.total_timesteps) * 100
        multipliers = self.get_constraint_multipliers(step)
        
        print(f"\n{'='*70}")
        print(f"FAT-FOCUSED CURRICULUM - Step {step:,} / {self.total_timesteps:,} ({progress:.1f}%)")
        print(f"{'='*70}")
        print(f"Phase {phase_idx + 1}/5: {phase['name']}")
        print(f"Fat Range Multiplier: {multipliers['fat']:.2f}x (PRIORITY)")
        print(f"Others Range: {multipliers['calories']:.2f}x")
        print(f"Entropy: {self.get_entropy_coef(step):.4f}")
        print(f"{'='*70}\n")


class FatCurriculumCallback(BaseCallback):
    """Callback to update parameters according to fat-focused curriculum"""
    
    def __init__(self, scheduler, update_frequency=10000, verbose=1):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.update_frequency = update_frequency
        self.last_update = 0
        self.phase_history = []
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_frequency:
            new_entropy = self.scheduler.get_entropy_coef(self.num_timesteps)
            self.model.ent_coef = new_entropy
            
            phase_idx, phase = self.scheduler.get_current_phase(self.num_timesteps)
            if not self.phase_history or self.phase_history[-1] != phase_idx:
                self.phase_history.append(phase_idx)
                if self.verbose > 0:
                    self.scheduler.print_status(self.num_timesteps)
            
            self.last_update = self.num_timesteps
        
        return True


class FatFocusedRecipeEnv(RecipeEnv):
    """Recipe environment with FAT-PRIORITY curriculum"""
    
    def __init__(self, scheduler, base_constraints=None, **kwargs):
        self.scheduler = scheduler
        self.base_constraints = base_constraints or DEFAULT_CONSTRAINTS
        self.current_step = 0
        super().__init__(**kwargs)
    
    def reset(self, **kwargs):
        """Reset with fat-focused constraint adjustment"""
        multipliers = self.scheduler.get_constraint_multipliers(self.current_step)
        
        adjusted_constraints = {}
        for nutrient, constraint in self.base_constraints.items():
            target = constraint['target']
            base_range = constraint['max'] - constraint['min']
            
            # Use nutrient-specific multiplier (fat gets special treatment)
            nutrient_multiplier = multipliers.get(nutrient, multipliers['calories'])
            adjusted_range = base_range * nutrient_multiplier
            
            adjusted_constraints[nutrient] = {
                'target': target,
                'min': target - adjusted_range / 2,
                'max': target + adjusted_range / 2,
            }
        
        self.constraints = adjusted_constraints
        return super().reset(**kwargs)
    
    def step(self, action):
        self.current_step += 1
        return super().step(action)


def train_ultra_performance(
    total_timesteps=800000,
    save_dir='models/saved',
    log_dir='logs/ultra_performance',
    eval_freq=20000,
    checkpoint_freq=100000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """Train model targeting 85%+ performance on ALL nutrients"""
    
    print(f"\n{'='*70}")
    print("ULTRA-HIGH PERFORMANCE TRAINING - 85%+ TARGET")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Strategy: FAT-FOCUSED 5-PHASE CURRICULUM")
    print(f"Reward System: v3 (Fat Priority)")
    print(f"{'='*70}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Print reward configuration
    reward_config.print_config_summary()
    
    # Initialize fat-focused scheduler
    scheduler = FatFocusedCurriculumScheduler(total_timesteps)
    scheduler.print_status(0)
    
    # Load ingredients
    import pandas as pd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    ingredients_path = os.path.join(project_dir, 'data', 'ingredients_enriched.csv')
    ingredients_df = pd.read_csv(ingredients_path)
    print(f"\n✓ Loaded {len(ingredients_df)} ingredients\n")
    
    # Create fat-focused environment
    def make_fat_focused_env():
        env = FatFocusedRecipeEnv(
            scheduler=scheduler,
            base_constraints=DEFAULT_CONSTRAINTS,
            ingredient_df=ingredients_df,
        )
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_fat_focused_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Enhanced policy architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Deeper policy network
            vf=[256, 256, 128],  # Deeper value network
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    # Initialize PPO with optimized settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,  # Larger batch
        n_epochs=15,     # More epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,   # Will be annealed by curriculum
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Setup callbacks
    fat_curriculum_callback = FatCurriculumCallback(scheduler, update_frequency=10000)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix='ultra_performance',
        save_vecnormalize=True,
    )
    
    # Evaluation environment
    eval_env = RecipeEnv(ingredient_df=ingredients_df)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=20,
    )
    
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([fat_curriculum_callback, checkpoint_callback, eval_callback])
    
    print(f"\nStarting ultra-performance training...")
    print(f"Expected duration: ~35-50 minutes on GPU\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(save_dir, 'ultra_performance_final.zip')
        model.save(final_path)
        env.save(os.path.join(save_dir, 'vec_normalize_ultra.pkl'))
        
        print(f"\n{'='*70}")
        print("ULTRA-PERFORMANCE TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Final model: {final_path}")
        print(f"Target: 85%+ satisfaction on ALL nutrients")
        print(f"{'='*70}\n")
        
        return model, env
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving checkpoint...")
        checkpoint_path = os.path.join(save_dir, 'ultra_performance_interrupted.zip')
        model.save(checkpoint_path)
        env.save(os.path.join(save_dir, 'vec_normalize_interrupted.pkl'))
        print(f"Checkpoint: {checkpoint_path}")
        return model, env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ultra-high performance model (85%+ target)')
    parser.add_argument('--timesteps', type=int, default=800000,
                        help='Total training timesteps (default: 800000)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--save_dir', type=str, default='models/saved')
    parser.add_argument('--log_dir', type=str, default='logs/ultra_performance')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    model, env = train_ultra_performance(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=device,
    )
    
    print("\n✓ Training complete!")
    print("Next: Test with 'python scripts/quick_test.py --curriculum models/saved/ultra_performance_final.zip'")
