"""
Health-Based Recipe Modification System
User enters: health data + recipe request ("dosa", "sandwich")
System outputs: That recipe, modified for their health goals
"""

from scripts.recipe_database import RecipeDatabase
from personalized_recipe import HealthBasedRecipeGenerator
import pandas as pd


class RecipeModificationSystem:
    def __init__(self):
        """Initialize with recipe database and health calculator."""
        print("Initializing Recipe Modification System...")
        self.recipe_db = RecipeDatabase()
        self.health_gen = HealthBasedRecipeGenerator()
        print("✓ System ready!\n")
    
    def modify_recipe_for_health(self, recipe, health_data, meal_type='lunch'):
        """
        Modify a recipe to fit user's health needs.
        
        Strategy:
        1. Calculate user's nutritional needs
        2. Compare recipe nutrition to targets
        3. Suggest modifications:
           - Portion adjustments
           - Ingredient substitutions
           - Cooking method changes
        """
        # Calculate health needs
        daily_cals = self.health_gen.calculate_calories(
            health_data['age'],
            health_data['weight'],
            health_data['height'],
            health_data['gender'],
            health_data['activity_level']
        )
        
        daily_needs = self.health_gen.adjust_for_goals(daily_cals, health_data['goal'])
        meal_constraints = self.health_gen.create_meal_constraints(daily_needs, meal_type)
        
        # Get recipe details
        full_recipe = self.recipe_db.get_recipe_details(recipe)
        
        # Analyze recipe vs targets
        original_cal = recipe.get('calories', 0)
        original_protein = recipe.get('protein', 0)
        target_cal = meal_constraints['max_calories']
        target_protein = meal_constraints['min_protein']
        
        modifications = []
        
        # Calorie adjustment
        if original_cal > target_cal * 1.2:
            reduction = ((original_cal - target_cal) / original_cal) * 100
            modifications.append({
                'type': 'portion',
                'change': f'Reduce portion size by {reduction:.0f}%',
                'reason': f'Original has {original_cal:.0f} cal, target is {target_cal:.0f} cal'
            })
        
        # Protein adjustment
        if original_protein < target_protein * 0.8:
            modifications.append({
                'type': 'protein',
                'change': 'Add protein-rich ingredients (chicken, tofu, legumes)',
                'reason': f'Original has {original_protein:.0f}g protein, need {target_protein:.0f}g'
            })
        
        # Goal-specific modifications
        if health_data['goal'] == 'lose_weight':
            modifications.append({
                'type': 'oil',
                'change': 'Reduce oil/butter by 50%, use cooking spray',
                'reason': 'Weight loss goal - reduce fat calories'
            })
            modifications.append({
                'type': 'vegetables',
                'change': 'Add more vegetables for volume and fiber',
                'reason': 'Increase satiety while lowering calorie density'
            })
        
        elif health_data['goal'] == 'gain_muscle':
            modifications.append({
                'type': 'protein',
                'change': 'Double protein sources (extra chicken, eggs, legumes)',
                'reason': 'Muscle building requires higher protein intake'
            })
        
        # Create modified recipe
        modified_recipe = {
            'original_name': full_recipe['name'],
            'modified_name': f"Healthy {full_recipe['name']} (for {health_data['goal'].replace('_', ' ')})",
            'source': full_recipe['source'],
            'original_nutrition': {
                'calories': original_cal,
                'protein': original_protein
            },
            'target_nutrition': {
                'calories': target_cal,
                'protein': target_protein
            },
            'modifications': modifications,
            'ingredients': full_recipe.get('ingredients', []),
            'instructions': full_recipe.get('instructions', [])
        }
        
        return modified_recipe
    
    def generate_from_request(self, health_data, recipe_request, meal_type='lunch'):
        """
        Main function: User requests recipe + health data → Modified recipe.
        
        Args:
            health_data: dict with age, weight, height, gender, activity_level, goal
            recipe_request: str like "dosa", "sandwich", "pasta"
            meal_type: breakfast, lunch, dinner, snack
        
        Returns:
            Modified recipe with health-based adjustments
        """
        print(f"🔍 Searching for '{recipe_request}' recipes...")
        
        # Search for matching recipes
        results = self.recipe_db.search_recipes(recipe_request, max_results=5)
        
        if not results:
            return {
                'error': f"Sorry, couldn't find any recipes matching '{recipe_request}'",
                'suggestion': 'Try: sandwich, pasta, pizza, salad, chicken, soup'
            }
        
        # Pick best match
        best_match = results[0]
        print(f"✓ Found: {best_match['name']} ({best_match['source']})")
        print(f"   Original: {best_match.get('calories', 'N/A')} cal, {best_match.get('protein', 'N/A')}g protein\n")
        
        # Modify for health
        print(f"🔧 Modifying recipe for your health goals...")
        modified = self.modify_recipe_for_health(best_match, health_data, meal_type)
        
        return modified
    
    def format_output(self, modified_recipe):
        """Format modified recipe for display."""
        if 'error' in modified_recipe:
            return f"❌ {modified_recipe['error']}\n💡 {modified_recipe['suggestion']}"
        
        output = f"""
{'='*70}
{modified_recipe['modified_name']}
{'='*70}
Original: {modified_recipe['original_name']}
Source: {modified_recipe['source']}

📊 NUTRITION COMPARISON:
------------------------
Original: {modified_recipe['original_nutrition']['calories']:.0f} cal, {modified_recipe['original_nutrition']['protein']:.0f}g protein
Your Target: {modified_recipe['target_nutrition']['calories']:.0f} cal, {modified_recipe['target_nutrition']['protein']:.0f}g protein

🔧 RECOMMENDED MODIFICATIONS:
------------------------------
"""
        for i, mod in enumerate(modified_recipe['modifications'], 1):
            output += f"\n{i}. {mod['change']}"
            output += f"\n   → {mod['reason']}\n"
        
        if modified_recipe['ingredients']:
            output += f"\n📝 INGREDIENTS:\n"
            for ing in modified_recipe['ingredients'][:10]:  # Show first 10
                output += f"  • {ing}\n"
            if len(modified_recipe['ingredients']) > 10:
                output += f"  ... and {len(modified_recipe['ingredients'])-10} more\n"
        
        output += f"\n{'='*70}\n"
        
        return output


# Test example
if __name__ == '__main__':
    system = RecipeModificationSystem()
    
    # Example: User wants dosa, trying to lose weight
    health_data = {
        'age': 30,
        'weight': 80,  # kg
        'height': 175,  # cm
        'gender': 'male',
        'activity_level': 'moderate',
        'goal': 'lose_weight'
    }
    
    test_requests = ['dosa', 'sandwich', 'pasta']
    
    for request in test_requests:
        print(f"\n{'#'*70}")
        print(f"# USER REQUEST: '{request}' for lunch (weight loss goal)")
        print(f"{'#'*70}\n")
        
        modified = system.generate_from_request(health_data, request, 'lunch')
        print(system.format_output(modified))
