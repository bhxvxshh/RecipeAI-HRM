"""
Curriculum Learning Training Script - Research Grade

Implements progressive constraint tightening and entropy annealing to improve
model accuracy and stability.

Training Strategy:
- Phase 1 (0-175k): Loose constraints (1.5x range), high entropy (0.15)
- Phase 2 (175k-350k): Medium constraints (1.25x range), medium entropy (0.08)
- Phase 3 (350k-525k): Tight constraints (1.1x range), low entropy (0.04)
- Phase 4 (525k-700k): Target constraints (1.0x range), minimal entropy (0.01)

Expected Improvements:
- Smoother learning curve
- Reduced reward variance: ±200 → ±50
- Higher final satisfaction: 57.2% → 65%+
- Better constraint-specific performance (Calories 40%+, Fat 45%+)
"""

import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.recipe_env import RecipeEnv
from config import DEFAULT_CONSTRAINTS, RL_CONFIG
import config_reward_v2 as reward_config


# ============================================================================
# CURRICULUM SCHEDULER
# ============================================================================

class CurriculumScheduler:
    """
    Progressive constraint tightening and entropy annealing.
    
    Phases:
    1. Exploration (0-175k): Learn basic constraint satisfaction with easy targets
    2. Refinement (175k-350k): Tighten constraints, reduce exploration
    3. Fine-tuning (350k-525k): Near-target constraints, focus on accuracy
    4. Mastery (525k-700k): Final constraints, pure exploitation
    """
    
    def __init__(self, total_timesteps=700000):
        self.total_timesteps = total_timesteps
        
        # Define training phases
        self.phases = [
            {
                'name': 'Exploration',
                'range_multiplier': 1.5,   # 50% wider constraint ranges
                'entropy': 0.15,            # High exploration
                'steps': 175000,
                'diversity_weight': 2.0,    # Emphasize diversity
            },
            {
                'name': 'Refinement',
                'range_multiplier': 1.25,  # 25% wider
                'entropy': 0.08,            # Medium exploration
                'steps': 175000,
                'diversity_weight': 1.5,
            },
            {
                'name': 'Fine-tuning',
                'range_multiplier': 1.1,   # 10% wider
                'entropy': 0.04,            # Low exploration
                'steps': 175000,
                'diversity_weight': 1.0,
            },
            {
                'name': 'Mastery',
                'range_multiplier': 1.0,   # Target constraints
                'entropy': 0.01,            # Minimal exploration
                'steps': total_timesteps - 525000,
                'diversity_weight': 0.8,   # Focus on accuracy
            },
        ]
        
        self.current_phase_idx = 0
        self.cumulative_steps = [0]
        for phase in self.phases:
            self.cumulative_steps.append(self.cumulative_steps[-1] + phase['steps'])
    
    def get_current_phase(self, step):
        """Get current training phase based on step count."""
        for i, threshold in enumerate(self.cumulative_steps[1:]):
            if step < threshold:
                return i, self.phases[i]
        return len(self.phases) - 1, self.phases[-1]
    
    def get_constraint_multiplier(self, step):
        """Get constraint range multiplier for current step."""
        _, phase = self.get_current_phase(step)
        return phase['range_multiplier']
    
    def get_entropy_coef(self, step):
        """Get entropy coefficient for current step (with smooth annealing)."""
        phase_idx, phase = self.get_current_phase(step)
        
        # Smooth interpolation within phase
        if phase_idx < len(self.phases) - 1:
            phase_start = self.cumulative_steps[phase_idx]
            phase_end = self.cumulative_steps[phase_idx + 1]
            phase_progress = (step - phase_start) / (phase_end - phase_start)
            
            current_entropy = phase['entropy']
            next_entropy = self.phases[phase_idx + 1]['entropy']
            
            # Linear interpolation
            entropy = current_entropy + (next_entropy - current_entropy) * phase_progress
            return entropy
        else:
            return phase['entropy']
    
    def get_diversity_weight(self, step):
        """Get diversity bonus weight for current step."""
        _, phase = self.get_current_phase(step)
        return phase['diversity_weight']
    
    def print_status(self, step):
        """Print current curriculum status."""
        phase_idx, phase = self.get_current_phase(step)
        progress = (step / self.total_timesteps) * 100
        
        print(f"\n{'='*70}")
        print(f"CURRICULUM STATUS - Step {step:,} / {self.total_timesteps:,} ({progress:.1f}%)")
        print(f"{'='*70}")
        print(f"Phase: {phase['name']} (Phase {phase_idx + 1}/4)")
        print(f"Constraint Range: {phase['range_multiplier']:.2f}x")
        print(f"Entropy Coef: {self.get_entropy_coef(step):.4f}")
        print(f"Diversity Weight: {self.get_diversity_weight(step):.2f}x")
        print(f"{'='*70}\n")


# ============================================================================
# CURRICULUM CALLBACK
# ============================================================================

class CurriculumCallback(BaseCallback):
    """
    Callback to update model parameters according to curriculum schedule.
    """
    
    def __init__(self, scheduler, update_frequency=10000, verbose=1):
        super().__init__(verbose)
        self.scheduler = scheduler
        self.update_frequency = update_frequency
        self.last_update = 0
        self.phase_history = []
    
    def _on_step(self) -> bool:
        # Update entropy coefficient every update_frequency steps
        if self.num_timesteps - self.last_update >= self.update_frequency:
            new_entropy = self.scheduler.get_entropy_coef(self.num_timesteps)
            self.model.ent_coef = new_entropy
            
            # Log phase transition
            phase_idx, phase = self.scheduler.get_current_phase(self.num_timesteps)
            if not self.phase_history or self.phase_history[-1] != phase_idx:
                self.phase_history.append(phase_idx)
                if self.verbose > 0:
                    self.scheduler.print_status(self.num_timesteps)
            
            self.last_update = self.num_timesteps
        
        return True


# ============================================================================
# CURRICULUM ENVIRONMENT WRAPPER
# ============================================================================

class CurriculumRecipeEnv(RecipeEnv):
    """
    Recipe environment with curriculum-adjusted constraints.
    """
    
    def __init__(self, scheduler, base_constraints=None, **kwargs):
        self.scheduler = scheduler
        self.base_constraints = base_constraints or DEFAULT_CONSTRAINTS
        self.current_step = 0
        
        super().__init__(**kwargs)
    
    def reset(self, **kwargs):
        """Reset with curriculum-adjusted constraints."""
        # Get constraint multiplier for current training step
        multiplier = self.scheduler.get_constraint_multiplier(self.current_step)
        
        # Adjust constraint ranges (widen them early in training)
        adjusted_constraints = {}
        for nutrient, constraint in self.base_constraints.items():
            target = constraint['target']
            base_range = constraint['max'] - constraint['min']
            adjusted_range = base_range * multiplier
            
            adjusted_constraints[nutrient] = {
                'target': target,
                'min': target - adjusted_range / 2,
                'max': target + adjusted_range / 2,
            }
        
        # Update environment constraints
        self.target_constraints = adjusted_constraints
        
        return super().reset(**kwargs)
    
    def step(self, action):
        """Step with curriculum tracking."""
        self.current_step += 1
        return super().step(action)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_with_curriculum(
    total_timesteps=700000,
    save_dir='models/saved',
    log_dir='logs/curriculum',
    eval_freq=20000,
    checkpoint_freq=50000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train model with curriculum learning.
    
    Args:
        total_timesteps: Total training steps (default 700k)
        save_dir: Directory to save models
        log_dir: Directory for TensorBoard logs
        eval_freq: Evaluation frequency
        checkpoint_freq: Checkpoint save frequency
        device: Training device (cuda/cpu)
    """
    
    print(f"\n{'='*70}")
    print("CURRICULUM LEARNING TRAINING - RESEARCH GRADE")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Save Directory: {save_dir}")
    print(f"Log Directory: {log_dir}")
    print(f"Using Reward Config v2: Improved weights and penalties")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize curriculum scheduler
    scheduler = CurriculumScheduler(total_timesteps)
    scheduler.print_status(0)
    
    # Load ingredient data
    print("\nLoading ingredient data...")
    import pandas as pd
    
    # Get script directory and construct absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    ingredients_path = os.path.join(project_dir, 'data', 'ingredients_enriched.csv')
    
    ingredients_df = pd.read_csv(ingredients_path)
    print(f"✓ Loaded {len(ingredients_df)} ingredients from {ingredients_path}")
    
    # Create curriculum environment
    def make_curriculum_env():
        env = CurriculumRecipeEnv(
            scheduler=scheduler,
            base_constraints=DEFAULT_CONSTRAINTS,
            ingredient_df=ingredients_df,
        )
        env = Monitor(env, log_dir)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_curriculum_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Enhanced policy architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256],  # Value network
        ),
        activation_fn=torch.nn.ReLU,
    )
    
    # Initialize PPO with curriculum-aware settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,           # Longer rollouts for better credit assignment
        batch_size=64,           # Larger batch for stability
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.15,          # Initial entropy (will be annealed by curriculum)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Setup callbacks
    curriculum_callback = CurriculumCallback(scheduler, update_frequency=10000)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix='curriculum_model',
        save_vecnormalize=True,
    )
    
    # Evaluation environment (with target constraints, not curriculum)
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
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([curriculum_callback, checkpoint_callback, eval_callback])
    
    # Train
    print(f"\nStarting curriculum training...")
    print(f"Expected duration: ~30-45 minutes on GPU\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = os.path.join(save_dir, 'curriculum_final_model.zip')
        model.save(final_model_path)
        env.save(os.path.join(save_dir, 'vec_normalize_curriculum.pkl'))
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Final model saved to: {final_model_path}")
        print(f"VecNormalize saved to: {os.path.join(save_dir, 'vec_normalize_curriculum.pkl')}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*70}\n")
        
        return model, env
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        checkpoint_path = os.path.join(save_dir, 'curriculum_interrupted.zip')
        model.save(checkpoint_path)
        env.save(os.path.join(save_dir, 'vec_normalize_interrupted.pkl'))
        print(f"Checkpoint saved to: {checkpoint_path}")
        return model, env


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RecipeAI with curriculum learning')
    parser.add_argument('--timesteps', type=int, default=700000,
                        help='Total training timesteps (default: 700000)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Training device (default: auto)')
    parser.add_argument('--save_dir', type=str, default='models/saved',
                        help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs/curriculum',
                        help='TensorBoard log directory')
    parser.add_argument('--eval_freq', type=int, default=20000,
                        help='Evaluation frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=50000,
                        help='Checkpoint save frequency')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Print reward configuration
    reward_config.print_config_summary()
    
    # Train model
    model, env = train_with_curriculum(
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        device=device,
    )
    
    print("\nTraining complete! Next steps:")
    print("1. Evaluate model: python scripts/comprehensive_model_analysis.py")
    print("2. Test live generation: python scripts/test_live_generation.py")
    print("3. Compare with baseline: python scripts/compare_models.py")
