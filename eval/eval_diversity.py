"""
Comprehensive evaluation with diversity analysis
"""
import sys
sys.path.insert(0, '/home/bhavesh/MajorB/RecipeAI')
from models.ingredient_policy import RecipeAgent
from train.train_recipe import setup_training_environment
import numpy as np

def evaluate_diversity(agent, env, n_recipes=100, deterministic=True):
    """
    Generate recipes and analyze diversity
    """
    recipes = []
    nutrients_list = []
    constraints_met = 0
    
    for i in range(n_recipes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 20:
            action, _ = agent.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        recipes.append(info['recipe'])
        nutrients_list.append(info['current_nutrients'])
        if info.get('all_constraints_met', False):
            constraints_met += 1
    
    # Calculate diversity metrics
    unique_recipes = len(set([tuple(sorted(r)) for r in recipes]))
    unique_ratio = unique_recipes / n_recipes
    
    # Average ingredient count
    avg_ingredients = np.mean([len(r) for r in recipes])
    
    # Ingredient frequency
    all_ingredients = []
    for recipe in recipes:
        all_ingredients.extend(recipe)
    unique_ingredients = len(set(all_ingredients))
    
    # Nutrient statistics
    cal_mean = np.mean([n['calories'] for n in nutrients_list])
    cal_std = np.std([n['calories'] for n in nutrients_list])
    
    return {
        'n_recipes': n_recipes,
        'deterministic': deterministic,
        'constraint_compliance': constraints_met / n_recipes,
        'unique_recipes': unique_recipes,
        'unique_ratio': unique_ratio,
        'avg_ingredients': avg_ingredients,
        'unique_ingredients_used': unique_ingredients,
        'total_ingredients_available': 324,
        'calories_mean': cal_mean,
        'calories_std': cal_std
    }

print('Loading trained agent...')
env = setup_training_environment('standard')
agent = RecipeAgent(env, algorithm='PPO', verbose=0)
agent.load('/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_standard_ppo')

print('\n' + '='*70)
print('RECIPE DIVERSITY EVALUATION')
print('='*70 + '\n')

# Test deterministic
print('Evaluating with DETERMINISTIC policy (100 recipes)...')
results_det = evaluate_diversity(agent, env, n_recipes=100, deterministic=True)

print('\nDETERMINISTIC RESULTS:')
print(f"  Constraint Compliance: {results_det['constraint_compliance']*100:.1f}%")
print(f"  Unique Recipes: {results_det['unique_recipes']}/{results_det['n_recipes']} ({results_det['unique_ratio']*100:.1f}%)")
print(f"  Avg Ingredients per Recipe: {results_det['avg_ingredients']:.1f}")
print(f"  Unique Ingredients Used: {results_det['unique_ingredients_used']}/{results_det['total_ingredients_available']}")
print(f"  Calories: {results_det['calories_mean']:.0f} ± {results_det['calories_std']:.0f}")

# Test stochastic
print('\n\nEvaluating with STOCHASTIC policy (100 recipes)...')
results_stoch = evaluate_diversity(agent, env, n_recipes=100, deterministic=False)

print('\nSTOCHASTIC RESULTS:')
print(f"  Constraint Compliance: {results_stoch['constraint_compliance']*100:.1f}%")
print(f"  Unique Recipes: {results_stoch['unique_recipes']}/{results_stoch['n_recipes']} ({results_stoch['unique_ratio']*100:.1f}%)")
print(f"  Avg Ingredients per Recipe: {results_stoch['avg_ingredients']:.1f}")
print(f"  Unique Ingredients Used: {results_stoch['unique_ingredients_used']}/{results_stoch['total_ingredients_available']}")
print(f"  Calories: {results_stoch['calories_mean']:.0f} ± {results_stoch['calories_std']:.0f}")

print('\n' + '='*70)
print('PHASE 1 ASSESSMENT')
print('='*70)

# Decision criteria
det_compliance = results_det['constraint_compliance']
stoch_compliance = results_stoch['constraint_compliance']
stoch_diversity = results_stoch['unique_ratio']

print(f"\nTarget: >70% compliance with good diversity")
print(f"\nDeterministic: {det_compliance*100:.1f}% compliance, {results_det['unique_ratio']*100:.1f}% unique")
print(f"Stochastic: {stoch_compliance*100:.1f}% compliance, {stoch_diversity*100:.1f}% unique")

if det_compliance >= 0.7 and results_det['unique_ratio'] >= 0.5:
    print("\n✓ PHASE 1 COMPLETE - Ready for Phase 2 (HRM)")
elif stoch_compliance >= 0.5 and stoch_diversity >= 0.3:
    print("\n⚠ PHASE 1 PARTIAL - Stochastic mode shows promise")
    print("  Recommendation: Tune diversity rewards or proceed to Phase 2")
else:
    print("\n✗ PHASE 1 NEEDS WORK")
    print("  Recommendation: Increase diversity bonus or adjust reward structure")

print('\n')
