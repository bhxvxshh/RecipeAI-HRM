#!/usr/bin/env python3
"""
Train model with GPU acceleration for improved performance.
Extended training: 500k timesteps (vs 100k baseline)
Optimized for CUDA with proper device configuration.
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

def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "="*80)
    print("GPU CONFIGURATION")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        print(f"✓ Current device: {torch.cuda.current_device()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ Total GPU memory: {total_memory:.2f} GB")
        
        device = "cuda"
    else:
        print("⚠ CUDA not available, using CPU")
        print("  To enable GPU:")
        print("  1. Install CUDA toolkit")
        print("  2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        device = "cpu"
    
    print("="*80)
    return device

def main():
    print("="*80)
    print("GPU-ACCELERATED MODEL TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTraining Configuration:")
    print(f"  • Timesteps: 500,000 (5x longer than baseline)")
    print(f"  • Entropy coefficient: {config.RL_CONFIG['ent_coef']} (high exploration)")
    print(f"  • Batch size: {config.RL_CONFIG['batch_size']} (optimized for GPU)")
    print(f"  • Learning rate: {config.RL_CONFIG['learning_rate']}")
    print(f"  • Constraint variation: ±{config.CONSTRAINT_NOISE_STD*100}%")
    print(f"  • Curiosity bonus: {config.CURIOSITY_BONUS_WEIGHT}")
    print("="*80)
    
    # Check GPU
    device = check_gpu()
    
    # Load ingredient data
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
        return Monitor(env)  # Monitor for better logging
    
    # Training environment with vectorization
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Evaluation environment (no constraint variation for consistent metrics)
    config_backup = config.VARY_CONSTRAINTS_TRAINING
    config.VARY_CONSTRAINTS_TRAINING = False
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    config.VARY_CONSTRAINTS_TRAINING = config_backup
    
    print(f"✓ Environments created")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n} actions")
    print(f"  - Device: {device}")
    
    # Create model with GPU support
    print("\n[3/6] Creating PPO model with GPU acceleration...")
    
    # Determine optimal policy kwargs for GPU
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Larger networks benefit from GPU
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
        device=device,  # Force device
        verbose=1,
        tensorboard_log=config.TENSORBOARD_LOG_DIR,
        seed=42  # For reproducibility
    )
    
    print(f"✓ Model created on device: {model.device}")
    print(f"  - Policy architecture: {policy_kwargs['net_arch']}")
    print(f"  - Total parameters: ~{sum(p.numel() for p in model.policy.parameters())/1e6:.2f}M")
    
    # Setup callbacks
    print("\n[4/6] Setting up callbacks...")
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.EVAL_RESULTS_DIR, exist_ok=True)
    
    # Checkpoint every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=config.MODEL_SAVE_DIR,
        name_prefix="recipe_agent_gpu",
        verbose=1
    )
    
    # Evaluate every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_SAVE_DIR,
        log_path=config.EVAL_RESULTS_DIR,
        eval_freq=10000,
        n_eval_episodes=20,  # More episodes for better stats
        deterministic=False,
        render=False,
        verbose=1
    )
    
    print("✓ Callbacks configured")
    print(f"  - Checkpoints: every 50,000 steps")
    print(f"  - Evaluation: every 10,000 steps (20 episodes)")
    print(f"  - Save path: {config.MODEL_SAVE_DIR}")
    
    # Train
    print("\n[5/6] Starting GPU-accelerated training...")
    print("="*80)
    
    total_timesteps = 500000  # 5x longer training
    
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
    print("\n[6/6] Saving final model...")
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, "best_model_gpu_500k")
    model.save(final_save_path)
    print(f"✓ Final model saved: {final_save_path}")
    
    # Also save VecNormalize stats
    env.save(os.path.join(config.MODEL_SAVE_DIR, "vec_normalize_gpu_500k.pkl"))
    print(f"✓ Normalization stats saved")
    
    # Quick evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (10 test episodes)")
    print("="*80)
    
    rewards = []
    for i in range(10):
        obs = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward[0]
        
        rewards.append(episode_reward)
        print(f"  Episode {i+1}: reward = {episode_reward:.2f}")
    
    print("\n" + "="*80)
    print(f"Average reward: {np.mean(rewards):.2f} (±{np.std(rewards):.2f})")
    print(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n✓ Training complete! Model saved to:")
    print(f"  {final_save_path}.zip")
    print("\nNext steps:")
    print("  1. Run evaluation: python scripts/comprehensive_model_analysis.py")
    print("  2. Update app.py to use new model")
    print("  3. Compare with baseline (100k model)")

if __name__ == "__main__":
    main()
