"""
Hybrid HRM System - Balance compliance and diversity
Uses best-of-N sampling: generate N stochastic recipes, pick best one
"""
import sys
sys.path.insert(0, '/home/bhavesh/MajorB/RecipeAI')

from models.ingredient_policy import RecipeAgent
from models.hrm_policy import HierarchicalRecipeSystem
from train.train_recipe import setup_training_environment
import config
import numpy as np

class HybridHRM(HierarchicalRecipeSystem):
    """
    Enhanced HRM with best-of-N sampling for diversity
    """
    
    def generate_daily_recipe_diverse(self, n_candidates=10) -> dict:
        """
        Generate N candidate recipes, pick best one that satisfies constraints
        
        Args:
            n_candidates: Number of stochastic recipes to generate
            
        Returns:
            Best recipe that meets constraints (or best available)
        """
        daily_constraints = self.high_level_policy.get_daily_constraints(self.current_day)
        
        candidates = []
        
        # Generate N stochastic candidates
        for _ in range(n_candidates):
            recipe = self.low_level_agent.generate_recipe(
                constraints=daily_constraints,
                render=False,
                deterministic=False  # Stochastic for diversity
            )
            candidates.append(recipe)
        
        # Score candidates
        scored_candidates = []
        for recipe in candidates:
            score = self._score_recipe(recipe, daily_constraints)
            scored_candidates.append((score, recipe))
        
        # Sort by score (higher is better)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pick best recipe
        best_recipe = scored_candidates[0][1]
        
        # Update high-level policy
        self.high_level_policy.update_history(best_recipe)
        self.current_day = (self.current_day + 1) % self.high_level_policy.planning_horizon
        
        return best_recipe
    
    def _score_recipe(self, recipe_info, constraints) -> float:
        """Score recipe based on constraint satisfaction"""
        score = 0.0
        nutrients = recipe_info['current_nutrients']
        
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            value = nutrients[nutrient]
            min_val = constraints[nutrient]['min']
            max_val = constraints[nutrient]['max']
            target = constraints[nutrient]['target']
            
            if min_val <= value <= max_val:
                # Within range: bonus based on proximity to target
                deviation = abs(value - target) / (target + 1e-8)
                score += (1.0 - deviation) * 10
            else:
                # Out of range: penalty
                if value < min_val:
                    score -= (min_val - value) / min_val * 5
                else:
                    score -= (value - max_val) / max_val * 5
        
        return score
    
    def generate_weekly_plan_hybrid(self, n_candidates=10):
        """Generate weekly plan with best-of-N sampling"""
        weekly_plan = []
        self.high_level_policy.reset_week()
        
        for day in range(7):
            recipe = self.generate_daily_recipe_diverse(n_candidates=n_candidates)
            weekly_plan.append(recipe)
        
        return weekly_plan


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
print('HYBRID HRM: BEST-OF-N SAMPLING (Phase 2 Enhanced)')
print('='*70)

# Load trained low-level agent
print('\nLoading Phase 1 trained agent...')
env = setup_training_environment('standard')
low_level_agent = RecipeAgent(env, algorithm='PPO', verbose=0)
low_level_agent.load('/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_standard_ppo')
print('✓ Low-level agent loaded')

# Initialize Hybrid HRM
print('\nInitializing Hybrid HRM (best-of-10 sampling)...')
hrm_system = HybridHRM(low_level_agent, user_profile='standard')
print('✓ Hybrid HRM initialized')

# Generate weekly plan
print('\n' + '='*70)
print('GENERATING WEEKLY MEAL PLAN (7 DAYS)')
print('Strategy: Generate 10 stochastic recipes per day, pick best one')
print('='*70)

weekly_plan = hrm_system.generate_weekly_plan_hybrid(n_candidates=10)

for day, recipe_info in enumerate(weekly_plan):
    print_recipe_summary(day, recipe_info, env)

# Evaluate performance
print('\n' + '='*70)
print('WEEKLY PERFORMANCE EVALUATION')
print('='*70)

nutrient_totals = {
    'calories': sum(r['current_nutrients']['calories'] for r in weekly_plan),
    'protein': sum(r['current_nutrients']['protein'] for r in weekly_plan),
    'sodium': sum(r['current_nutrients']['sodium'] for r in weekly_plan),
    'carbs': sum(r['current_nutrients']['carbs'] for r in weekly_plan),
    'fat': sum(r['current_nutrients']['fat'] for r in weekly_plan),
}

print("\nNutrient Totals vs Targets:")
for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
    total = nutrient_totals[nutrient]
    target = config.HRM_CONFIG['weekly_targets'][nutrient]
    deviation = abs(total - target) / target * 100
    status = '✓' if deviation < 10 else '✗'
    print(f"  {status} {nutrient.capitalize()}: {total:.0f} / {target:.0f} ({deviation:.1f}% deviation)")

diversity_metrics = evaluate_diversity(weekly_plan)
print("\nDiversity Metrics:")
print(f"  Unique Recipes: {diversity_metrics['unique_recipes']}/7 ({diversity_metrics['diversity_ratio']*100:.0f}%)")
print(f"  Unique Ingredients: {diversity_metrics['unique_ingredients']}")

constraints_met = sum(1 for r in weekly_plan if r.get('all_constraints_met', False))
print(f"\nDaily Constraint Compliance: {constraints_met}/7 ({constraints_met/7*100:.0f}%)")

print('\n' + '='*70)
print('PHASE 2 FINAL ASSESSMENT')
print('='*70)

weekly_target_met = all(
    abs(nutrient_totals[n] - config.HRM_CONFIG['weekly_targets'][n]) / config.HRM_CONFIG['weekly_targets'][n] < 0.15
    for n in ['calories', 'protein', 'carbs', 'fat']
)

if diversity_metrics['diversity_ratio'] >= 0.7 and constraints_met >= 5 and weekly_target_met:
    print("\n✓✓ PHASE 2 SUCCESS - HRM System Working Excellently!")
    print("  ✓ High diversity (>70% unique recipes)")
    print("  ✓ Good daily compliance (≥5/7 days)")
    print("  ✓ Weekly targets achieved")
    print("\n  The hierarchical system successfully:")
    print("    - Adjusts daily constraints based on weekly progress")
    print("    - Maintains diversity through best-of-N sampling")
    print("    - Balances exploration and exploitation")
elif diversity_metrics['diversity_ratio'] >= 0.5 and constraints_met >= 3:
    print("\n✓ PHASE 2 FUNCTIONAL - System working reasonably well")
    print(f"  Diversity: {diversity_metrics['diversity_ratio']*100:.0f}%")
    print(f"  Compliance: {constraints_met/7*100:.0f}%")
    print("\n  Improvements possible:")
    print("    - Tune N candidates (try 15-20)")
    print("    - Adjust scoring function weights")
    print("    - Fine-tune high-level policy parameters")
else:
    print("\n⚠ PHASE 2 NEEDS TUNING")
    print(f"  Diversity: {diversity_metrics['diversity_ratio']*100:.0f}% (target >50%)")
    print(f"  Compliance: {constraints_met/7*100:.0f}% (target >40%)")

print()
