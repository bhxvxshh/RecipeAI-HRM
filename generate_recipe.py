"""
Simple script to generate recipes using your trained model.
Just run: python generate_recipe.py
"""

from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from utils.data_preprocessing import load_processed_data
import pandas as pd

def generate_recipe(num_recipes=1):
    """Generate recipes using the trained model."""
    
    print("Loading model and data...")
    model = PPO.load('models/saved/best_model')
    ingredients_df = load_processed_data()
    ingredients_enriched = pd.read_csv('data/ingredients_enriched.csv')
    
    env = RecipeEnv(ingredients_df)
    
    for i in range(num_recipes):
        print(f"\n{'='*70}")
        print(f"RECIPE #{i+1}")
        print('='*70)
        
        obs, info = env.reset()
        done = False
        recipe_ingredients = []
        
        # Generate recipe by having model select ingredients
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            if action < env.n_ingredients:
                ing = env.ingredient_df.iloc[action]
                recipe_ingredients.append(ing)
        
        if recipe_ingredients:
            # Display recipe
            print("\nINGREDIENTS:")
            for ing in recipe_ingredients:
                print(f"  • {ing['food_name']}")
            
            # Calculate nutrition
            total_cal = sum(ing['calories'] for ing in recipe_ingredients)
            total_protein = sum(ing['protein'] for ing in recipe_ingredients)
            total_carbs = sum(ing['carbs'] for ing in recipe_ingredients)
            total_fat = sum(ing['fat'] for ing in recipe_ingredients)
            
            print(f"\nNUTRITION (per 100g average):")
            print(f"  Calories: {total_cal/len(recipe_ingredients):.0f} kcal")
            print(f"  Protein:  {total_protein/len(recipe_ingredients):.1f}g")
            print(f"  Carbs:    {total_carbs/len(recipe_ingredients):.1f}g")
            print(f"  Fat:      {total_fat/len(recipe_ingredients):.1f}g")
            
            # Show categories
            food_ids = [ing['food_id'] for ing in recipe_ingredients]
            categories = set()
            for fid in food_ids:
                enriched = ingredients_enriched[ingredients_enriched['food_id'] == fid]
                if not enriched.empty:
                    categories.add(enriched.iloc[0]['category'])
            
            print(f"\nCATEGORIES: {', '.join(categories)}")
            print('='*70)

if __name__ == '__main__':
    print("\n🍳 RecipeAI - Healthy Recipe Generator")
    print("Using your trained anti-overfitting model\n")
    
    num = input("How many recipes do you want? (default 3): ").strip()
    num = int(num) if num.isdigit() else 3
    
    generate_recipe(num)
    
    print("\n✅ Done! Want more features?")
    print("   - Use scripts/export_recipes.py for detailed nutrition")
    print("   - Edit constraints in config.py")
