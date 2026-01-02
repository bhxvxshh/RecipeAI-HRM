"""
Generate personalized recipes based on user health data.
Calculates daily caloric needs using Mifflin-St Jeor equation and creates
custom nutritional constraints for recipe generation.
"""

from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from utils.data_preprocessing import load_processed_data
import pandas as pd
import sys


class HealthBasedRecipeGenerator:
    def __init__(self):
        """Initialize the generator with model and data."""
        print("Loading AI model and ingredient database...")
        self.model = PPO.load('models/saved/best_model')
        self.ingredients_df = load_processed_data()
        self.ingredients_enriched = pd.read_csv('data/ingredients_enriched.csv')
    
    def calculate_calories(self, age, weight, height, gender, activity_level):
        """
        Calculate daily caloric needs using Mifflin-St Jeor equation.
        
        Args:
            age: years
            weight: kg
            height: cm
            gender: 'male' or 'female'
            activity_level: sedentary, light, moderate, active, very_active
        
        Returns:
            Daily caloric needs (kcal)
        """
        # Basal Metabolic Rate (BMR)
        if gender.lower() == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        # Activity multipliers
        activity_multipliers = {
            'sedentary': 1.2,      # Little to no exercise
            'light': 1.375,        # Light exercise 1-3 days/week
            'moderate': 1.55,      # Moderate exercise 3-5 days/week
            'active': 1.725,       # Heavy exercise 6-7 days/week
            'very_active': 1.9     # Very heavy exercise, physical job
        }
        
        multiplier = activity_multipliers.get(activity_level.lower(), 1.55)
        daily_calories = bmr * multiplier
        
        return daily_calories
    
    def adjust_for_goals(self, calories, goal):
        """
        Adjust caloric needs based on health goals.
        
        Args:
            calories: Base daily calories
            goal: 'lose_weight', 'maintain', 'gain_muscle'
        
        Returns:
            Adjusted calories and macro ratios
        """
        if goal == 'lose_weight':
            # 500 kcal deficit for ~0.5kg/week loss
            calories -= 500
            # Higher protein to preserve muscle
            protein_ratio = 0.30  # 30% protein
            carbs_ratio = 0.35    # 35% carbs
            fat_ratio = 0.35      # 35% fat
        
        elif goal == 'gain_muscle':
            # 300 kcal surplus for muscle gain
            calories += 300
            # High protein for muscle building
            protein_ratio = 0.30  # 30% protein
            carbs_ratio = 0.45    # 45% carbs
            fat_ratio = 0.25      # 25% fat
        
        else:  # maintain
            protein_ratio = 0.25  # 25% protein
            carbs_ratio = 0.45    # 45% carbs
            fat_ratio = 0.30      # 30% fat
        
        # Calculate macro targets (in grams)
        protein_g = (calories * protein_ratio) / 4  # 4 kcal per gram
        carbs_g = (calories * carbs_ratio) / 4
        fat_g = (calories * fat_ratio) / 9  # 9 kcal per gram
        
        return {
            'calories': calories,
            'protein': protein_g,
            'carbs': carbs_g,
            'fat': fat_g
        }
    
    def create_meal_constraints(self, daily_needs, meal_type='lunch'):
        """
        Create constraints for a specific meal.
        
        Args:
            daily_needs: Dict with daily calorie/macro needs
            meal_type: breakfast, lunch, dinner, snack
        
        Returns:
            Meal-specific constraints
        """
        # Meal proportion of daily needs
        meal_proportions = {
            'breakfast': 0.25,  # 25% of daily
            'lunch': 0.35,      # 35% of daily
            'dinner': 0.30,     # 30% of daily
            'snack': 0.10       # 10% of daily
        }
        
        proportion = meal_proportions.get(meal_type, 0.33)
        
        return {
            'max_calories': daily_needs['calories'] * proportion * 1.2,  # 20% tolerance
            'min_protein': daily_needs['protein'] * proportion * 0.8,
            'max_carbs': daily_needs['carbs'] * proportion * 1.3,
            'max_fat': daily_needs['fat'] * proportion * 1.3,
            'max_sodium': 800 * proportion  # Max ~800mg per meal
        }
    
    def filter_ingredients_by_diet(self, dietary_restrictions):
        """
        Filter ingredients based on dietary restrictions.
        
        Args:
            dietary_restrictions: list like ['vegan', 'gluten_free', 'nut_free']
        
        Returns:
            Filtered ingredient dataframe
        """
        if not dietary_restrictions:
            return self.ingredients_df
        
        # Get food_ids that match ALL restrictions
        valid_food_ids = set(self.ingredients_enriched['food_id'].unique())
        
        for restriction in dietary_restrictions:
            # Find ingredients that have this tag
            matching = self.ingredients_enriched[
                self.ingredients_enriched['dietary_tags'].str.contains(restriction, na=False)
            ]
            valid_food_ids &= set(matching['food_id'].unique())
        
        # Filter original dataframe
        filtered_df = self.ingredients_df[
            self.ingredients_df['food_id'].isin(valid_food_ids)
        ].reset_index(drop=True)
        
        return filtered_df
    
    def generate_personalized_recipe(self, health_data, meal_type='lunch', 
                                     dietary_restrictions=None):
        """
        Generate recipe based on user health data.
        
        Args:
            health_data: dict with age, weight, height, gender, activity_level, goal
            meal_type: breakfast, lunch, dinner, snack
            dietary_restrictions: list like ['vegan', 'gluten_free']
        
        Returns:
            Generated recipe with ingredients and nutrition
        """
        # Calculate personalized needs
        daily_cals = self.calculate_calories(
            health_data['age'],
            health_data['weight'],
            health_data['height'],
            health_data['gender'],
            health_data['activity_level']
        )
        
        daily_needs = self.adjust_for_goals(daily_cals, health_data['goal'])
        meal_constraints = self.create_meal_constraints(daily_needs, meal_type)
        
        print(f"\n📊 YOUR DAILY NEEDS (goal: {health_data['goal'].replace('_', ' ')}):")
        print(f"  Calories: {daily_needs['calories']:.0f} kcal")
        print(f"  Protein:  {daily_needs['protein']:.0f}g")
        print(f"  Carbs:    {daily_needs['carbs']:.0f}g")
        print(f"  Fat:      {daily_needs['fat']:.0f}g")
        
        print(f"\n🍽️  {meal_type.upper()} TARGET:")
        print(f"  Calories: {meal_constraints['max_calories']:.0f} kcal (max)")
        print(f"  Protein:  {meal_constraints['min_protein']:.0f}g (min)")
        
        # Filter ingredients if dietary restrictions
        ingredients_df = self.filter_ingredients_by_diet(dietary_restrictions or [])
        
        if len(ingredients_df) < 10:
            print(f"\n⚠️  Warning: Only {len(ingredients_df)} ingredients match your restrictions.")
            print("    Consider relaxing some restrictions for more variety.")
        else:
            print(f"\n✓ {len(ingredients_df)} ingredients available for your diet")
        
        # Create environment with custom constraints
        env = RecipeEnv(
            ingredients_df,
            constraints={
                'max_calories': meal_constraints['max_calories'],
                'min_protein': meal_constraints['min_protein'],
                'max_carbs': meal_constraints['max_carbs'],
                'max_fat': meal_constraints['max_fat'],
                'max_sodium': meal_constraints['max_sodium']
            }
        )
        
        # Generate recipe
        print(f"\n🤖 Generating personalized recipe...\n")
        
        obs, info = env.reset()
        done = False
        recipe_ingredients = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            if action < env.n_ingredients:
                ing = env.ingredient_df.iloc[action]
                recipe_ingredients.append(ing)
        
        return self.format_recipe(recipe_ingredients, meal_type, meal_constraints)
    
    def format_recipe(self, ingredients, meal_type, targets):
        """Format and display recipe with nutrition info."""
        if not ingredients:
            print("❌ No recipe generated. Try different constraints.")
            return None
        
        print(f"{'='*70}")
        print(f"  PERSONALIZED {meal_type.upper()} RECIPE")
        print(f"{'='*70}")
        
        print("\nINGREDIENTS:")
        for ing in ingredients:
            print(f"  • {ing['food_name']}")
        
        # Calculate nutrition (average per 100g)
        total_cal = sum(ing['calories'] for ing in ingredients)
        total_protein = sum(ing['protein'] for ing in ingredients)
        total_carbs = sum(ing['carbs'] for ing in ingredients)
        total_fat = sum(ing['fat'] for ing in ingredients)
        total_sodium = sum(ing['sodium'] for ing in ingredients)
        
        avg_cal = total_cal / len(ingredients)
        avg_protein = total_protein / len(ingredients)
        avg_carbs = total_carbs / len(ingredients)
        avg_fat = total_fat / len(ingredients)
        avg_sodium = total_sodium / len(ingredients)
        
        print(f"\nNUTRITION (per 100g average):")
        print(f"  Calories: {avg_cal:.0f} kcal")
        print(f"  Protein:  {avg_protein:.1f}g")
        print(f"  Carbs:    {avg_carbs:.1f}g")
        print(f"  Fat:      {avg_fat:.1f}g")
        print(f"  Sodium:   {avg_sodium:.0f}mg")
        
        # Show if targets are met
        print(f"\n✓ TARGETS:")
        cal_status = "✓" if avg_cal <= targets['max_calories']/100 else "⚠"
        protein_status = "✓" if avg_protein >= targets['min_protein']/100 else "⚠"
        print(f"  {cal_status} Calories within limit")
        print(f"  {protein_status} Protein goal met")
        
        # Show categories
        food_ids = [ing['food_id'] for ing in ingredients]
        categories = set()
        allergens = set()
        
        for fid in food_ids:
            enriched = self.ingredients_enriched[self.ingredients_enriched['food_id'] == fid]
            if not enriched.empty:
                categories.add(enriched.iloc[0]['category'])
                allerg = str(enriched.iloc[0]['allergens']).split(',')
                for a in allerg:
                    if a.strip() and a.strip() != 'none':
                        allergens.add(a.strip())
        
        print(f"\nCATEGORIES: {', '.join(categories)}")
        if allergens:
            print(f"⚠️  ALLERGENS: {', '.join(allergens)}")
        
        print(f"{'='*70}\n")
        
        return {
            'ingredients': ingredients,
            'nutrition': {
                'calories': avg_cal,
                'protein': avg_protein,
                'carbs': avg_carbs,
                'fat': avg_fat,
                'sodium': avg_sodium
            },
            'categories': list(categories),
            'allergens': list(allergens)
        }


def interactive_mode():
    """Interactive mode to collect user health data."""
    print("\n" + "="*70)
    print("  🍳 PERSONALIZED RECIPE GENERATOR")
    print("     Based on YOUR health data")
    print("="*70)
    
    print("\n📋 Please enter your health information:\n")
    
    # Collect data
    age = int(input("Age (years): "))
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    
    print("\nGender:")
    print("  1. Male")
    print("  2. Female")
    gender_choice = input("Choose (1/2): ")
    gender = 'male' if gender_choice == '1' else 'female'
    
    print("\nActivity Level:")
    print("  1. Sedentary (little/no exercise)")
    print("  2. Light (exercise 1-3 days/week)")
    print("  3. Moderate (exercise 3-5 days/week)")
    print("  4. Active (exercise 6-7 days/week)")
    print("  5. Very Active (intense exercise daily)")
    activity_choice = input("Choose (1-5): ")
    activity_map = {
        '1': 'sedentary',
        '2': 'light',
        '3': 'moderate',
        '4': 'active',
        '5': 'very_active'
    }
    activity = activity_map.get(activity_choice, 'moderate')
    
    print("\nHealth Goal:")
    print("  1. Lose weight")
    print("  2. Maintain weight")
    print("  3. Gain muscle")
    goal_choice = input("Choose (1-3): ")
    goal_map = {
        '1': 'lose_weight',
        '2': 'maintain',
        '3': 'gain_muscle'
    }
    goal = goal_map.get(goal_choice, 'maintain')
    
    print("\nMeal Type:")
    print("  1. Breakfast")
    print("  2. Lunch")
    print("  3. Dinner")
    print("  4. Snack")
    meal_choice = input("Choose (1-4): ")
    meal_map = {'1': 'breakfast', '2': 'lunch', '3': 'dinner', '4': 'snack'}
    meal_type = meal_map.get(meal_choice, 'lunch')
    
    print("\nDietary Restrictions (optional):")
    print("  Type restrictions separated by commas (e.g., vegan,gluten_free)")
    print("  Available: vegan, vegetarian, gluten_free, dairy_free, nut_free, keto, paleo")
    print("  Or press Enter to skip")
    restrictions_input = input("Restrictions: ").strip()
    dietary_restrictions = [r.strip() for r in restrictions_input.split(',')] if restrictions_input else None
    
    health_data = {
        'age': age,
        'weight': weight,
        'height': height,
        'gender': gender,
        'activity_level': activity,
        'goal': goal
    }
    
    # Generate recipe
    generator = HealthBasedRecipeGenerator()
    recipe = generator.generate_personalized_recipe(
        health_data,
        meal_type,
        dietary_restrictions
    )
    
    # Ask if user wants another
    print("\n" + "="*70)
    another = input("Generate another recipe? (y/n): ")
    if another.lower() == 'y':
        recipe = generator.generate_personalized_recipe(
            health_data,
            meal_type,
            dietary_restrictions
        )


if __name__ == '__main__':
    interactive_mode()
