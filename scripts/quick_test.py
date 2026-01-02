"""
Live Recipe Generation Test - Quick comparison of models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from config import DEFAULT_CONSTRAINTS

def test_model(model_path, model_name, n_recipes=5):
    """Generate and analyze recipes"""
    
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}\n")
    
    # Load ingredients
    ingredients_df = pd.read_csv('/home/bhavesh/MajorB/RecipeAI/data/ingredients_enriched.csv')
    
    # Load model
    model = PPO.load(model_path)
    print(f"✓ Model loaded: {os.path.basename(model_path)}\n")
    
    # Create environment
    env = RecipeEnv(ingredient_df=ingredients_df, constraints=DEFAULT_CONSTRAINTS)
    
    satisfied_count = {n: 0 for n in ['calories', 'protein', 'fat', 'carbs', 'sodium']}
    all_rewards = []
    
    for i in range(n_recipes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        all_rewards.append(reward)
        
        # Get nutrients and constraints
        nutrients = env.current_nutrients
        constraints = env.constraints
        
        # Check satisfaction
        total_sat = 0
        print(f"Recipe {i+1}:")
        
        for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
            value = nutrients[nutrient]
            target = constraints[nutrient]['target']
            min_val = constraints[nutrient]['min']
            max_val = constraints[nutrient]['max']
            
            is_satisfied = min_val <= value <= max_val
            deviation = ((value - target) / target) * 100 if target > 0 else 0
            
            if is_satisfied:
                satisfied_count[nutrient] += 1
                total_sat += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {status} {nutrient:8s}: {value:7.1f} / {target:7.1f} ({deviation:+6.1f}%)")
        
        print(f"  Satisfied: {total_sat}/5, Reward: {reward:.1f}\n")
    
    # Summary
    print(f"\n{'-'*80}")
    print(f"SUMMARY - {model_name}")
    print(f"{'-'*80}")
    
    overall = sum(satisfied_count.values()) / (n_recipes * 5) * 100
    print(f"Overall Satisfaction: {overall:.1f}%")
    print(f"Average Reward: {np.mean(all_rewards):.1f}")
    print(f"\nBy Nutrient:")
    for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
        pct = (satisfied_count[nutrient] / n_recipes) * 100
        print(f"  {nutrient:8s}: {pct:5.1f}% ({satisfied_count[nutrient]}/{n_recipes})")
    print(f"{'-'*80}\n")
    
    return {
        'overall': overall,
        'avg_reward': np.mean(all_rewards),
        'nutrients': {n: (satisfied_count[n] / n_recipes) * 100 for n in satisfied_count}
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='models/saved/best_model.zip')
    parser.add_argument('--curriculum', type=str, default='models/saved/curriculum_final_model.zip')
    parser.add_argument('--n', type=int, default=5, help='Number of recipes per model')
    
    args = parser.parse_args()
    
    results = {}
    
    # Test baseline
    if os.path.exists(args.baseline):
        results['baseline'] = test_model(args.baseline, "BASELINE MODEL (100k steps)", args.n)
    else:
        print(f"⚠ Baseline model not found: {args.baseline}")
    
    # Test curriculum
    if os.path.exists(args.curriculum):
        results['curriculum'] = test_model(args.curriculum, "CURRICULUM MODEL (700k steps)", args.n)
    else:
        print(f"⚠ Curriculum model not found: {args.curriculum}")
    
    # Comparison
    if len(results) == 2:
        print(f"\n{'='*80}")
        print("COMPARISON: Curriculum vs Baseline")
        print(f"{'='*80}")
        
        diff_overall = results['curriculum']['overall'] - results['baseline']['overall']
        diff_reward = results['curriculum']['avg_reward'] - results['baseline']['avg_reward']
        
        print(f"Overall Satisfaction: {diff_overall:+.1f} percentage points")
        print(f"Average Reward: {diff_reward:+.1f} points")
        print(f"\nBy Nutrient:")
        
        for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
            diff = results['curriculum']['nutrients'][nutrient] - results['baseline']['nutrients'][nutrient]
            print(f"  {nutrient:8s}: {diff:+6.1f} pp")
        
        print(f"{'='*80}\n")
