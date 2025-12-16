#!/usr/bin/env python3
"""
Compare old (overfitted) model vs new (anti-overfitting) model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from collections import Counter
import config
from env.recipe_env import RecipeEnv

def evaluate_model(model_path, n_episodes=20):
    """Evaluate a model and return metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"{'='*60}")
    
    # Load data
    ingredients_df = pd.read_csv(config.PROCESSED_INGREDIENT_FILE)
    
    # Load model
    model = PPO.load(model_path)
    
    # Create env (disable constraint variation for fair comparison)
    config.VARY_CONSTRAINTS_TRAINING = False
    env = RecipeEnv(ingredient_df=ingredients_df)
    
    # Metrics
    recipes = []
    all_ingredients = []
    constraint_compliant = 0
    total_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Stochastic sampling
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        # Get recipe
        recipe = env.current_recipe
        recipes.append(tuple(sorted(recipe)))
        all_ingredients.extend(recipe)
        total_rewards.append(episode_reward)
        
        # Check constraints
        nutrients = env.current_nutrients
        compliant = all([
            config.DEFAULT_CONSTRAINTS[n]['min'] <= nutrients[n] <= config.DEFAULT_CONSTRAINTS[n]['max']
            for n in ['calories', 'protein', 'sodium', 'carbs', 'fat']
        ])
        if compliant:
            constraint_compliant += 1
    
    # Calculate metrics
    unique_recipes = len(set(recipes))
    unique_ingredients = len(set(all_ingredients))
    diversity_pct = (unique_recipes / n_episodes) * 100
    compliance_pct = (constraint_compliant / n_episodes) * 100
    avg_reward = np.mean(total_rewards)
    
    # Most common ingredients
    ingredient_counts = Counter(all_ingredients)
    top_ingredients = ingredient_counts.most_common(10)
    
    print(f"\n📊 RESULTS:")
    print(f"  Constraint Compliance: {compliance_pct:.1f}% ({constraint_compliant}/{n_episodes})")
    print(f"  Recipe Diversity: {diversity_pct:.1f}% ({unique_recipes}/{n_episodes} unique)")
    print(f"  Unique Ingredients Used: {unique_ingredients}/{len(ingredients_df)}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    
    print(f"\n🔝 Top 10 Most Used Ingredients:")
    for idx, count in top_ingredients:
        ing_name = ingredients_df.iloc[idx]['food_name']
        print(f"    {ing_name}: {count} times ({count/n_episodes*100:.0f}%)")
    
    return {
        'compliance': compliance_pct,
        'diversity': diversity_pct,
        'unique_ingredients': unique_ingredients,
        'avg_reward': avg_reward,
        'unique_recipes': unique_recipes
    }

def main():
    print("="*60)
    print("MODEL COMPARISON: Overfitted vs Anti-Overfitting")
    print("="*60)
    
    old_model = "/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_standard_ppo.zip"
    new_model = "/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_anti_overfit_final.zip"
    
    # Check if models exist
    if not os.path.exists(new_model):
        print(f"\n⚠️  New model not found: {new_model}")
        print("Training may not be complete yet.")
        return
    
    # Evaluate both
    old_metrics = evaluate_model(old_model, n_episodes=20)
    new_metrics = evaluate_model(new_model, n_episodes=20)
    
    # Comparison
    print(f"\n{'='*60}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'Old Model':<15} {'New Model':<15} {'Change'}")
    print("-"*60)
    
    metrics = [
        ('Constraint Compliance', 'compliance', '%'),
        ('Recipe Diversity', 'diversity', '%'),
        ('Unique Ingredients', 'unique_ingredients', ''),
        ('Avg Reward', 'avg_reward', ''),
        ('Unique Recipes', 'unique_recipes', '/20')
    ]
    
    for label, key, unit in metrics:
        old_val = old_metrics[key]
        new_val = new_metrics[key]
        
        if unit == '%':
            change = f"{new_val - old_val:+.1f}%"
            old_str = f"{old_val:.1f}%"
            new_str = f"{new_val:.1f}%"
        else:
            change = f"{new_val - old_val:+.0f}"
            if unit:
                old_str = f"{old_val:.0f}{unit}"
                new_str = f"{new_val:.0f}{unit}"
            else:
                old_str = f"{old_val:.1f}"
                new_str = f"{new_val:.1f}"
        
        # Add emoji for improvement
        if new_val > old_val and 'Reward' not in label:
            emoji = "✅"
        elif new_val > old_val and 'Reward' in label:
            emoji = "✅"
        elif new_val < old_val:
            emoji = "⬇️"
        else:
            emoji = "➡️"
        
        print(f"{label:<25} {old_str:<15} {new_str:<15} {change} {emoji}")
    
    print(f"\n{'='*60}")
    print("VERDICT:")
    print(f"{'='*60}")
    
    if new_metrics['diversity'] > old_metrics['diversity'] * 2:
        print("✅ Mode collapse FIXED! Recipe diversity dramatically improved.")
    elif new_metrics['diversity'] > old_metrics['diversity']:
        print("✅ Recipe diversity improved!")
    else:
        print("⚠️  Diversity not significantly improved.")
    
    if new_metrics['compliance'] >= 90:
        print("✅ Constraint compliance excellent!")
    elif new_metrics['compliance'] >= 70:
        print("⚠️  Constraint compliance acceptable but could be better.")
    else:
        print("❌ Constraint compliance needs improvement.")
    
    if new_metrics['unique_ingredients'] > old_metrics['unique_ingredients'] * 1.5:
        print("✅ Ingredient exploration dramatically increased!")
    elif new_metrics['unique_ingredients'] > old_metrics['unique_ingredients']:
        print("✅ Ingredient exploration improved!")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
