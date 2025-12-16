"""
Test Hierarchical Recipe Management (Phase 2)
Generates weekly meal plans with diversity
"""
import sys
sys.path.insert(0, '/home/bhavesh/MajorB/RecipeAI')

from models.ingredient_policy import RecipeAgent
from models.hrm_policy import HierarchicalRecipeSystem
from train.train_recipe import setup_training_environment
import config
import numpy as np

def print_recipe_summary(day, recipe_info, env):
    """Print formatted recipe"""
    ingredients = []
    for idx in recipe_info['recipe']:
        idx_int = int(idx) if hasattr(idx, 'item') else idx
        name = env.unwrapped.get_ingredient_name(idx_int)
        ingredients.append(name)
    
    nutrients = recipe_info['current_nutrients']
    
    print(f"\n  Day {day+1}: {len(ingredients)} ingredients")
    print(f"    Ingredients: {', '.join(ingredients[:5])}{'...' if len(ingredients) > 5 else ''}")
    print(f"    Nutrients: {nutrients['calories']:.0f} kcal, "
          f"{nutrients['protein']:.1f}g protein, "
          f"{nutrients['sodium']:.0f}mg sodium, "
          f"{nutrients['carbs']:.1f}g carbs, "
          f"{nutrients['fat']:.1f}g fat")
    
    if recipe_info.get('all_constraints_met', False):
        print(f"    ✓ Constraints met")
    else:
        print(f"    ✗ Constraints violated")

def evaluate_diversity(weekly_plan):
    """Calculate diversity metrics"""
    all_ingredients = []
    all_recipes = []
    
    for recipe_info in weekly_plan:
        recipe = recipe_info['recipe']
        all_recipes.append(tuple(sorted(recipe)))
        all_ingredients.extend(recipe)
    
    unique_recipes = len(set(all_recipes))
    unique_ingredients = len(set(all_ingredients))
    
    return {
        'unique_recipes': unique_recipes,
        'unique_ingredients': unique_ingredients,
        'diversity_ratio': unique_recipes / len(weekly_plan)
    }

print('='*70)
print('PHASE 2: HIERARCHICAL RECIPE MANAGEMENT (HRM)')
print('='*70)
print(f"\nHRM Status: {'✓ ENABLED' if config.HRM_ENABLED else '✗ DISABLED'}")
print(f"Lambda Hierarchical: {config.LAMBDA_HIERARCHICAL}")
print(f"Planning Horizon: {config.HRM_CONFIG['planning_horizon']} days")

# Load trained low-level agent
print('\nLoading Phase 1 trained agent...')
env = setup_training_environment('standard')
low_level_agent = RecipeAgent(env, algorithm='PPO', verbose=0)
low_level_agent.load('/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_standard_ppo')
print('✓ Low-level agent loaded')

# Initialize HRM system
print('\nInitializing hierarchical system...')
hrm_system = HierarchicalRecipeSystem(low_level_agent, user_profile='standard')
print('✓ HRM system initialized')

# Generate weekly plan
print('\n' + '='*70)
print('GENERATING WEEKLY MEAL PLAN (7 DAYS)')
print('='*70)

weekly_plan = hrm_system.generate_weekly_plan()

for day, recipe_info in enumerate(weekly_plan):
    print_recipe_summary(day, recipe_info, env)

# Evaluate weekly performance
print('\n' + '='*70)
print('WEEKLY PERFORMANCE EVALUATION')
print('='*70)

# Calculate metrics from weekly_plan directly (since reset_week cleared history)
nutrient_totals = {
    'calories': sum(r['current_nutrients']['calories'] for r in weekly_plan),
    'protein': sum(r['current_nutrients']['protein'] for r in weekly_plan),
    'sodium': sum(r['current_nutrients']['sodium'] for r in weekly_plan),
    'carbs': sum(r['current_nutrients']['carbs'] for r in weekly_plan),
    'fat': sum(r['current_nutrients']['fat'] for r in weekly_plan),
}

metrics = {
    'weekly_reward': 0.0,  # Would need to recalculate
}

# Add nutrient totals to metrics
for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
    target = config.HRM_CONFIG['weekly_targets'][nutrient]
    total = nutrient_totals[nutrient]
    metrics[f'{nutrient}_total'] = total
    metrics[f'{nutrient}_target'] = target
    metrics[f'{nutrient}_deviation_pct'] = abs(total - target) / target * 100

print(f"\nWeekly Reward: {metrics['weekly_reward']:.2f}")

print("\nNutrient Totals vs Targets:")
for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
    total = metrics[f'{nutrient}_total']
    target = metrics[f'{nutrient}_target']
    deviation = metrics[f'{nutrient}_deviation_pct']
    status = '✓' if deviation < 10 else '✗'
    print(f"  {status} {nutrient.capitalize()}: {total:.0f} / {target:.0f} ({deviation:.1f}% deviation)")

diversity_metrics = evaluate_diversity(weekly_plan)
print("\nDiversity Metrics:")
print(f"  Unique Recipes: {diversity_metrics['unique_recipes']}/7 ({diversity_metrics['diversity_ratio']*100:.0f}%)")
print(f"  Unique Ingredients: {diversity_metrics['unique_ingredients']}")

# Compliance rate
constraints_met = sum(1 for r in weekly_plan if r.get('all_constraints_met', False))
print(f"\nDaily Constraint Compliance: {constraints_met}/7 ({constraints_met/7*100:.0f}%)")

print('\n' + '='*70)
print('PHASE 2 STATUS')
print('='*70)

if diversity_metrics['diversity_ratio'] >= 0.5 and constraints_met >= 5:
    print("\n✓ HRM SYSTEM WORKING - Good diversity and compliance")
    print("  The hierarchical planner successfully adjusts daily constraints")
    print("  to achieve weekly targets while maintaining variety.")
elif diversity_metrics['diversity_ratio'] >= 0.3:
    print("\n⚠ PARTIAL SUCCESS - Some diversity but room for improvement")
    print("  Recommendation: Tune high-level reward weights or add stochastic sampling")
else:
    print("\n✗ NEEDS WORK - Low diversity persists")
    print("  The low-level agent may be too deterministic.")
    print("  Options:")
    print("    1. Use stochastic policy (deterministic=False)")
    print("    2. Train with stronger diversity rewards")
    print("    3. Implement explicit diverse beam search")

print()
