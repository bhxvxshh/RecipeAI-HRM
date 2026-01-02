"""
Quick recipe generation test to verify model behavior.
Generate 5 recipes and check constraint satisfaction in detail.
"""

import sys
sys.path.append('/home/bhavesh/MajorB/RecipeAI')

from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from utils.data_preprocessing import load_processed_data
import pandas as pd
import numpy as np

def test_recipe_generation():
    print("="*80)
    print("LIVE RECIPE GENERATION TEST")
    print("="*80)
    
    # Load model and data
    print("\n📊 Loading model...")
    model = PPO.load('models/saved/best_model')
    ingredients_df = load_processed_data()
    env = RecipeEnv(ingredients_df)
    
    print(f"✓ Model loaded")
    print(f"✓ Ingredients: {len(ingredients_df)}")
    
    # Generate 5 recipes
    n_recipes = 5
    
    print(f"\n{'='*80}")
    print(f"GENERATING {n_recipes} RECIPES")
    print(f"{'='*80}")
    
    for recipe_num in range(1, n_recipes + 1):
        print(f"\n{'─'*80}")
        print(f"RECIPE #{recipe_num}")
        print(f"{'─'*80}")
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        ingredients = []
        
        # Generate recipe
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if action < env.n_ingredients:
                ing = env.ingredient_df.iloc[action]
                ingredients.append(ing)
        
        # Display recipe
        print(f"\n📝 INGREDIENTS ({len(ingredients)}):")
        for i, ing in enumerate(ingredients, 1):
            print(f"  {i}. {ing['food_name']}")
        
        # Nutrition summary
        nutrients = env.current_nutrients
        constraints = env.constraints
        
        print(f"\n📊 NUTRITIONAL BREAKDOWN:")
        print(f"  {'Nutrient':<12} {'Actual':<10} {'Target':<20} {'Status':<10}")
        print(f"  {'-'*60}")
        
        violations = []
        
        for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
            actual = nutrients[nutrient]
            min_val = constraints[nutrient]['min']
            max_val = constraints[nutrient]['max']
            target = constraints[nutrient]['target']
            
            if actual < min_val:
                status = f"❌ TOO LOW"
                violations.append(f"{nutrient} too low")
            elif actual > max_val:
                status = f"❌ TOO HIGH"
                violations.append(f"{nutrient} too high")
            else:
                status = "✅ OK"
            
            range_str = f"{min_val}-{max_val} (target: {target})"
            print(f"  {nutrient.capitalize():<12} {actual:<10.1f} {range_str:<20} {status}")
        
        print(f"\n🎯 REWARD: {episode_reward:.2f}")
        
        if violations:
            print(f"⚠️  VIOLATIONS ({len(violations)}):")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"✅ ALL CONSTRAINTS SATISFIED!")
        
        # Calculate how far off from target
        print(f"\n📏 DEVIATION FROM TARGET:")
        for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
            actual = nutrients[nutrient]
            target = constraints[nutrient]['target']
            deviation = ((actual - target) / target) * 100
            
            if abs(deviation) < 10:
                indicator = "✅"
            elif abs(deviation) < 30:
                indicator = "⚠️ "
            else:
                indicator = "❌"
            
            print(f"  {indicator} {nutrient.capitalize():<12} {deviation:+6.1f}% from target")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print("\nNOTE: This is the CURRENT model (100k timesteps)")
    print("GPU training is running in background for improved 500k model")
    print("\nExpected issues with current model:")
    print("  • Poor calorie control (23.5% satisfaction)")
    print("  • Weak fat control (36% satisfaction)")
    print("  • Good carb/sodium control (79-84% satisfaction)")

if __name__ == "__main__":
    test_recipe_generation()
