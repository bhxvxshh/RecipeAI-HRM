#!/usr/bin/env python3
"""
Demo script to test the Recipe Generation system
Run this after data preprocessing to verify everything works
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from utils.data_preprocessing import load_processed_data
from env.recipe_env import RecipeEnv
from models.ingredient_policy import RecipeAgent


def test_data_loading():
    """Test 1: Data loading"""
    print("="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        ingredients = load_processed_data()
        print(f"✓ Loaded {len(ingredients)} ingredients")
        print(f"✓ Columns: {list(ingredients.columns)}")
        print("\nSample ingredients:")
        print(ingredients.head())
        return True, ingredients
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False, None


def test_environment(ingredients):
    """Test 2: Environment creation and interaction"""
    print("\n" + "="*60)
    print("TEST 2: Environment Creation")
    print("="*60)
    
    try:
        env = RecipeEnv(
            ingredient_df=ingredients,
            constraints=config.DEFAULT_CONSTRAINTS
        )
        print("✓ Environment created")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Reward: {reward:.2f}")
        print(f"  Ingredient count: {info['ingredient_count']}")
        
        return True, env
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_random_policy(env, n_episodes=5):
    """Test 3: Random policy episodes"""
    print("\n" + "="*60)
    print("TEST 3: Random Policy Episodes")
    print("="*60)
    
    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < config.MAX_INGREDIENTS_PER_RECIPE:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            
            print(f"Episode {episode+1}: "
                  f"{steps} ingredients, "
                  f"reward: {episode_reward:.2f}, "
                  f"constraints met: {info['all_constraints_met']}")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation(env):
    """Test 4: Agent creation"""
    print("\n" + "="*60)
    print("TEST 4: Agent Creation")
    print("="*60)
    
    try:
        agent = RecipeAgent(env, algorithm='PPO', verbose=0)
        print("✓ PPO agent created")
        
        # Test prediction
        obs, _ = env.reset()
        action, _ = agent.predict(obs)
        print(f"✓ Prediction successful: action = {action}")
        
        return True, agent
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_short_training(agent, steps=1000):
    """Test 5: Short training run"""
    print("\n" + "="*60)
    print("TEST 5: Short Training Run")
    print("="*60)
    
    try:
        print(f"Training for {steps} steps...")
        agent.train(total_timesteps=steps)
        print("✓ Training completed")
        
        # Test recipe generation
        print("\nGenerating test recipe:")
        recipe_info = agent.generate_recipe(render=True)
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "RECIPE GENERATION SYSTEM - DEMO")
    print("="*70)
    
    results = []
    
    # Test 1: Data loading
    success, ingredients = test_data_loading()
    results.append(("Data Loading", success))
    if not success:
        print("\n⚠️  Cannot proceed without data. Run data preprocessing first:")
        print("   python utils/data_preprocessing.py")
        return
    
    # Test 2: Environment
    success, env = test_environment(ingredients)
    results.append(("Environment", success))
    if not success:
        return
    
    # Test 3: Random policy
    success = test_random_policy(env)
    results.append(("Random Policy", success))
    
    # Test 4: Agent creation
    success, agent = test_agent_creation(env)
    results.append(("Agent Creation", success))
    if not success:
        return
    
    # Test 5: Short training
    success = test_short_training(agent)
    results.append(("Short Training", success))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ All tests passed! System is ready for training.")
        print("="*70)
        print("\nNext steps:")
        print("1. Run full training:")
        print("   python train/train_recipe.py --profile standard --timesteps 100000")
        print("\n2. Monitor training:")
        print("   tensorboard --logdir logs/tensorboard/")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
