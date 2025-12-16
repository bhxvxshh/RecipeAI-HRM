"""
Quick evaluation script for trained agent
"""
import sys
sys.path.insert(0, '/home/bhavesh/MajorB/RecipeAI')
from models.ingredient_policy import RecipeAgent
from train.train_recipe import setup_training_environment

print('Loading trained agent...')
env = setup_training_environment('standard')
agent = RecipeAgent(env, algorithm='PPO', verbose=0)
agent.load('/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_standard_ppo')

print('\n' + '='*60)
print('GENERATING 10 SAMPLE RECIPES')
print('='*60 + '\n')

constraints_met = 0
for i in range(10):
    print(f'Recipe {i+1}:')
    recipe_info = agent.generate_recipe(render=False)
    
    # Get ingredient names
    ingredients = []
    for idx in recipe_info['recipe']:
        idx_int = int(idx) if hasattr(idx, 'item') else idx
        name = env.unwrapped.get_ingredient_name(idx_int)
        ingredients.append(name)
    
    ellipsis = '...' if len(ingredients) > 5 else ''
    ing_list = ', '.join(ingredients[:5])
    print(f'  Ingredients ({len(ingredients)}): {ing_list}{ellipsis}')
    
    nutrients = recipe_info['current_nutrients']
    print(f'  Nutrients: Cal={nutrients["calories"]:.0f}, Pro={nutrients["protein"]:.1f}g, Na={nutrients["sodium"]:.0f}mg, Carb={nutrients["carbs"]:.1f}g, Fat={nutrients["fat"]:.1f}g')
    
    if recipe_info['all_constraints_met']:
        print('  ✓ All constraints met')
        constraints_met += 1
    else:
        failed = [k for k, v in recipe_info['compliance'].items() if not v]
        print(f'  ✗ Failed: {", ".join(failed)}')
    print()

print('='*60)
print(f'Constraint compliance: {constraints_met}/10 ({constraints_met*10}%)')
print('='*60)
