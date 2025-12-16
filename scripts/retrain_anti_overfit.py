#!/usr/bin/env python3
"""
Retrain model with anti-overfitting improvements:
- 10x higher entropy (0.01 → 0.1) for exploration
- Constraint variation during training (\u00b115% noise)
- Curiosity bonus for novel ingredients
- Smaller batch size (64 → 32) for frequent updates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import config
from env.recipe_env import RecipeEnv

def main():
    print("="*80)
    print("RETRAINING WITH ANTI-OVERFITTING IMPROVEMENTS")
    print("="*80)
    print(f"✓ Entropy coefficient: 0.01 → 0.1 (10x exploration)")
    print(f"✓ Batch size: 64 → 32 (more frequent updates)")
    print(f"✓ Constraint variation: \u00b1{config.CONSTRAINT_NOISE_STD*100}% during training")
    print(f"✓ Curiosity bonus: {config.CURIOSITY_BONUS_WEIGHT} for novel ingredients")
    print("="*80)
    
    # Load ingredient data
    print("\n[1/5] Loading ingredient data...")
    ingredients_df = pd.read_csv(config.PROCESSED_INGREDIENT_FILE)
    print(f"✓ Loaded {len(ingredients_df)} ingredients")
    
    # Create environment with constraint variation enabled
    print("\n[2/5] Creating training environment...")
    def make_env():
        return RecipeEnv(
            ingredient_df=ingredients_df,
            constraints=config.DEFAULT_CONSTRAINTS,
            hrm_mode=False
        )
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Create evaluation environment (without constraint variation for consistent metrics)
    config_backup = config.VARY_CONSTRAINTS_TRAINING
    config.VARY_CONSTRAINTS_TRAINING = False
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    config.VARY_CONSTRAINTS_TRAINING = config_backup
    
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n} actions")
    
    # Create model with new hyperparameters
    print("\n[3/5] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.RL_CONFIG['learning_rate'],
        n_steps=config.RL_CONFIG['n_steps'],
        batch_size=config.RL_CONFIG['batch_size'],
        n_epochs=config.RL_CONFIG['n_epochs'],
        gamma=config.RL_CONFIG['gamma'],
        ent_coef=config.RL_CONFIG['ent_coef'],
        verbose=1,
        tensorboard_log=config.TENSORBOARD_LOG_DIR
    )
    print(f"✓ Model created with improved hyperparameters")
    
    # Setup callbacks
    print("\n[4/5] Setting up callbacks...")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=config.MODEL_SAVE_DIR,
        name_prefix="recipe_agent_anti_overfit"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_SAVE_DIR,
        log_path=config.EVAL_RESULTS_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=False,  # Stochastic evaluation to check diversity
        render=False
    )
    
    print("✓ Callbacks configured")
    
    # Train
    print("\n[5/5] Starting training...")
    print(f"Training for {config.RL_CONFIG['total_timesteps']} timesteps...")
    print("-"*80)
    
    model.learn(
        total_timesteps=config.RL_CONFIG['total_timesteps'],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model_path = os.path.join(config.MODEL_SAVE_DIR, "recipe_agent_anti_overfit_final.zip")
    model.save(model_path)
    env.save(os.path.join(config.MODEL_SAVE_DIR, "vec_normalize_anti_overfit.pkl"))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"✓ Model saved: {model_path}")
    print(f"✓ VecNormalize saved: vec_normalize_anti_overfit.pkl")
    print("\nTo evaluate: python eval/eval_phase1.py --model recipe_agent_anti_overfit_final.zip")
    print("="*80)

if __name__ == "__main__":
    main()
