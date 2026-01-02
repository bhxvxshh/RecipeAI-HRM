#!/usr/bin/env python3
"""
Train with IMPROVED reward system that fixes calorie and fat bias.
- Weighted penalties (calories 30%, fat 25%)
- Exponential penalties for large violations
- Progressive rewards for getting close to targets
- 500k timesteps on GPU
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import config
from env.recipe_env import RecipeEnv
from datetime import datetime

def main():
    print("="*80)
    print("IMPROVED REWARD TRAINING - FIXING CALORIE & FAT BIAS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🔧 NEW IMPROVEMENTS:")
    print(f"  ✓ Weighted penalties: Calories 30%, Fat 25%, Protein 20%")
    print(f"  ✓ Exponential scaling for large violations")
    print(f"  ✓ Progressive rewards: Within 10%=100pts, 20%=50pts, 30%=20pts")
    print(f"  ✓ Completion bonus: 200pts for perfect, 50pts for 4/5")
    print(f"  ✓ Milestone rewards: +10 at min ingredients, +5 at halfway")
    print("="*80)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("\n⚠ Using CPU")
        device = "cpu"
    
    # Load data
    print("\n[1/6] Loading ingredient data...")
    ingredients_df = pd.read_csv(config.PROCESSED_INGREDIENT_FILE)
    print(f"✓ Loaded {len(ingredients_df)} ingredients")
    
    # Create environments
    print("\n[2/6] Creating training environment...")
    
    def make_env():
        env = RecipeEnv(
            ingredient_df=ingredients_df,
            constraints=config.DEFAULT_CONSTRAINTS,
            hrm_mode=False
        )
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    config_backup = config.VARY_CONSTRAINTS_TRAINING
    config.VARY_CONSTRAINTS_TRAINING = False
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    config.VARY_CONSTRAINTS_TRAINING = config_backup
    
    print(f"✓ Environments created (device: {device})")
    
    # Create model
    print("\n[3/6] Creating PPO model with improved rewards...")
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.RL_CONFIG['learning_rate'],
        n_steps=config.RL_CONFIG['n_steps'],
        batch_size=config.RL_CONFIG['batch_size'],
        n_epochs=config.RL_CONFIG['n_epochs'],
        gamma=config.RL_CONFIG['gamma'],
        ent_coef=config.RL_CONFIG['ent_coef'],
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=config.TENSORBOARD_LOG_DIR,
        seed=42
    )
    
    print(f"✓ Model created on {model.device}")
    
    # Setup callbacks
    print("\n[4/6] Setting up callbacks...")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=config.MODEL_SAVE_DIR,
        name_prefix="recipe_improved_rewards",
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_SAVE_DIR,
        log_path=config.EVAL_RESULTS_DIR,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=False,
        render=False,
        verbose=1
    )
    
    print("✓ Callbacks configured")
    
    # Train
    print("\n[5/6] Starting improved reward training...")
    print("="*80)
    
    total_timesteps = 500000
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        print("\n" + "="*80)
        print("✓ Training completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise
    
    # Save final model
    print("\n[6/6] Saving final improved model...")
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, "best_model_improved")
    model.save(final_save_path)
    print(f"✓ Final model saved: {final_save_path}")
    
    env.save(os.path.join(config.MODEL_SAVE_DIR, "vec_normalize_improved.pkl"))
    print(f"✓ Normalization stats saved")
    
    # Quick evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (20 test episodes)")
    print("="*80)
    
    rewards = []
    for i in range(20):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward[0]
        
        rewards.append(episode_reward)
        if (i + 1) % 5 == 0:
            print(f"  Episodes {i-3}-{i+1}: avg reward = {np.mean(rewards[-5:]):.2f}")
    
    print("\n" + "="*80)
    print(f"Final Average Reward: {np.mean(rewards):.2f} (±{np.std(rewards):.2f})")
    print(f"Improvement over baseline: {((np.mean(rewards) - (-346)) / 346 * 100):.1f}%")
    print(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n✅ TRAINING COMPLETE WITH IMPROVED REWARDS!")
    print(f"\nModel saved to: {final_save_path}.zip")
    print("\nNext steps:")
    print("  1. Test generation: python scripts/test_live_generation.py")
    print("  2. Full evaluation: python scripts/comprehensive_model_analysis.py")
    print("  3. Compare with baseline and GPU models")

if __name__ == "__main__":
    main()
