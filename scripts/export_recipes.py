"""
Export generated recipes to human-readable formats (JSON, text).
Includes ingredient names, quantities, nutrition facts, and dietary information.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class RecipeExporter:
    def __init__(self, ingredients_file='data/ingredients_enriched.csv'):
        """Initialize with enriched ingredient data."""
        self.ingredients_df = pd.read_csv(ingredients_file)
        self.ingredients_dict = self.ingredients_df.set_index('food_id').to_dict('index')
    
    def get_ingredient_info(self, food_id):
        """Get full ingredient information by ID."""
        return self.ingredients_dict.get(food_id, None)
    
    def calculate_recipe_nutrition(self, ingredients_with_amounts):
        """
        Calculate total nutrition for recipe.
        ingredients_with_amounts: list of tuples (food_id, amount_in_grams)
        """
        nutrition = {
            'total_calories': 0,
            'total_protein': 0,
            'total_fat': 0,
            'total_carbs': 0,
            'total_sodium': 0,
            'total_weight': 0
        }
        
        for food_id, amount in ingredients_with_amounts:
            info = self.get_ingredient_info(food_id)
            if info:
                # Nutrition data is per 100g, scale to actual amount
                factor = amount / 100.0
                nutrition['total_calories'] += info['calories'] * factor
                nutrition['total_protein'] += info['protein'] * factor
                nutrition['total_fat'] += info['fat'] * factor
                nutrition['total_carbs'] += info['carbs'] * factor
                nutrition['total_sodium'] += info['sodium'] * factor
                nutrition['total_weight'] += amount
        
        return nutrition
    
    def analyze_recipe_properties(self, ingredients_list):
        """Analyze dietary properties and categories of recipe."""
        categories = set()
        allergens = set()
        dietary_tags = set(['vegan', 'vegetarian', 'gluten_free', 'dairy_free', 'nut_free', 'paleo', 'keto'])
        cost_tiers = []
        
        for food_id in ingredients_list:
            info = self.get_ingredient_info(food_id)
            if info:
                categories.add(info['category'])
                
                # Collect allergens
                ing_allergens = str(info['allergens']).split(',')
                for allergen in ing_allergens:
                    if allergen.strip() and allergen.strip() != 'none':
                        allergens.add(allergen.strip())
                
                # Intersect dietary tags (recipe is only compatible if ALL ingredients are)
                ing_tags = set(str(info['dietary_tags']).split(','))
                dietary_tags = dietary_tags.intersection(ing_tags)
                
                cost_tiers.append(info['cost_tier'])
        
        # Determine overall cost (highest tier wins)
        cost_priority = {'low': 1, 'medium': 2, 'high': 3}
        overall_cost = max(cost_tiers, key=lambda x: cost_priority.get(x, 2)) if cost_tiers else 'medium'
        
        return {
            'categories': list(categories),
            'allergens': list(allergens) if allergens else ['none'],
            'dietary_compatible': list(dietary_tags) if dietary_tags else ['none'],
            'cost_tier': overall_cost
        }
    
    def export_recipe_json(self, ingredients_list, output_file=None, 
                          recipe_name=None, serving_size=1, 
                          ingredient_amounts=None):
        """
        Export recipe to JSON format.
        
        Args:
            ingredients_list: list of food_ids
            output_file: path to save JSON (if None, returns dict)
            recipe_name: optional name for recipe
            serving_size: number of servings
            ingredient_amounts: optional dict {food_id: amount_in_grams}
                               if None, estimates equal portions totaling 500g
        """
        if not ingredient_amounts:
            # Estimate equal portions
            total_weight = 500  # grams
            equal_portion = total_weight / len(ingredients_list)
            ingredient_amounts = {fid: equal_portion for fid in ingredients_list}
        
        # Build ingredient list with details
        ingredients_with_amounts = [(fid, ingredient_amounts.get(fid, 100)) for fid in ingredients_list]
        
        ingredients_detailed = []
        for food_id, amount in ingredients_with_amounts:
            info = self.get_ingredient_info(food_id)
            if info:
                ingredients_detailed.append({
                    'food_id': int(food_id),
                    'name': str(info['food_name']),
                    'amount_g': round(float(amount), 1),
                    'category': str(info['category']),
                    'nutrition_per_100g': {
                        'calories': float(info['calories']),
                        'protein': float(info['protein']),
                        'fat': float(info['fat']),
                        'carbs': float(info['carbs']),
                        'sodium': float(info['sodium'])
                    }
                })
        
        # Calculate totals
        nutrition = self.calculate_recipe_nutrition(ingredients_with_amounts)
        properties = self.analyze_recipe_properties(ingredients_list)
        
        # Build recipe object
        recipe = {
            'recipe_name': recipe_name or f"AI Recipe {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'serving_size': serving_size,
            'ingredients': ingredients_detailed,
            'nutrition_totals': {
                'calories': round(nutrition['total_calories'], 1),
                'protein_g': round(nutrition['total_protein'], 1),
                'fat_g': round(nutrition['total_fat'], 1),
                'carbs_g': round(nutrition['total_carbs'], 1),
                'sodium_mg': round(nutrition['total_sodium'], 1),
                'total_weight_g': round(nutrition['total_weight'], 1)
            },
            'nutrition_per_serving': {
                'calories': round(nutrition['total_calories'] / serving_size, 1),
                'protein_g': round(nutrition['total_protein'] / serving_size, 1),
                'fat_g': round(nutrition['total_fat'] / serving_size, 1),
                'carbs_g': round(nutrition['total_carbs'] / serving_size, 1),
                'sodium_mg': round(nutrition['total_sodium'] / serving_size, 1)
            },
            'properties': properties
        }
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(recipe, f, indent=2)
            print(f"✓ Recipe exported to {output_file}")
        
        return recipe
    
    def export_recipe_text(self, ingredients_list, output_file=None, 
                          recipe_name=None, serving_size=1, 
                          ingredient_amounts=None):
        """Export recipe to human-readable text format."""
        recipe = self.export_recipe_json(ingredients_list, None, recipe_name, 
                                        serving_size, ingredient_amounts)
        
        # Build text output
        lines = []
        lines.append("=" * 70)
        lines.append(f"  {recipe['recipe_name']}")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Servings: {recipe['serving_size']}")
        lines.append("")
        
        # Ingredients section
        lines.append("INGREDIENTS:")
        lines.append("-" * 70)
        for ing in recipe['ingredients']:
            lines.append(f"  • {ing['amount_g']:.0f}g  {ing['name']}")
        lines.append("")
        
        # Nutrition section
        lines.append("NUTRITION (Total):")
        lines.append("-" * 70)
        nutr = recipe['nutrition_totals']
        lines.append(f"  Calories:  {nutr['calories']:.0f} kcal")
        lines.append(f"  Protein:   {nutr['protein_g']:.1f}g")
        lines.append(f"  Fat:       {nutr['fat_g']:.1f}g")
        lines.append(f"  Carbs:     {nutr['carbs_g']:.1f}g")
        lines.append(f"  Sodium:    {nutr['sodium_mg']:.0f}mg")
        lines.append(f"  Weight:    {nutr['total_weight_g']:.0f}g")
        lines.append("")
        
        # Per serving
        if serving_size > 1:
            lines.append("NUTRITION (Per Serving):")
            lines.append("-" * 70)
            nutr_serv = recipe['nutrition_per_serving']
            lines.append(f"  Calories:  {nutr_serv['calories']:.0f} kcal")
            lines.append(f"  Protein:   {nutr_serv['protein_g']:.1f}g")
            lines.append(f"  Fat:       {nutr_serv['fat_g']:.1f}g")
            lines.append(f"  Carbs:     {nutr_serv['carbs_g']:.1f}g")
            lines.append(f"  Sodium:    {nutr_serv['sodium_mg']:.0f}mg")
            lines.append("")
        
        # Properties
        props = recipe['properties']
        lines.append("RECIPE PROPERTIES:")
        lines.append("-" * 70)
        lines.append(f"  Categories: {', '.join(props['categories'])}")
        lines.append(f"  Allergens:  {', '.join(props['allergens'])}")
        lines.append(f"  Dietary:    {', '.join(props['dietary_compatible'])}")
        lines.append(f"  Cost:       {props['cost_tier']}")
        lines.append("=" * 70)
        
        text_output = '\n'.join(lines)
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(text_output)
            print(f"✓ Recipe exported to {output_file}")
        
        return text_output


# Example usage and testing
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/bhavesh/MajorB/RecipeAI')
    
    from stable_baselines3 import PPO
    from env.recipe_env import RecipeEnv
    from utils.data_preprocessing import load_processed_data
    
    print("Loading improved model...")
    model = PPO.load('models/saved/best_model')
    
    print("Loading ingredient data...")
    ingredients_df = load_processed_data()
    
    print("Creating environment...")
    env = RecipeEnv(ingredients_df)
    exporter = RecipeExporter()
    
    print("\nGenerating 3 sample recipes...\n")
    
    for i in range(3):
        obs, info = env.reset()
        done = False
        ingredients = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            if action < env.n_ingredients:
                ingredients.append(env.ingredient_df.iloc[action]['food_id'])
        
        if ingredients:
            print(f"\n{'='*70}")
            print(f"RECIPE #{i+1}")
            print('='*70)
            
            # Export as text
            text = exporter.export_recipe_text(
                ingredients,
                output_file=f'recipes/recipe_{i+1}.txt',
                recipe_name=f"Healthy AI Recipe #{i+1}",
                serving_size=2
            )
            print(text)
            
            # Also save JSON
            exporter.export_recipe_json(
                ingredients,
                output_file=f'recipes/recipe_{i+1}.json',
                recipe_name=f"Healthy AI Recipe #{i+1}",
                serving_size=2
            )
            print(f"\n✓ Saved JSON version to recipes/recipe_{i+1}.json")
    
    print(f"\n{'='*70}")
    print("✓ All recipes generated and exported!")
    print("="*70)
