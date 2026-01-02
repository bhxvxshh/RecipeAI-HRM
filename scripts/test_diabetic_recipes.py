"""
Test recipe modification system with real diabetic patient profiles.
Shows how recipes are adjusted for diabetic nutritional needs.
"""

import sys
sys.path.append('/home/bhavesh/MajorB/RecipeAI')

from recipe_modifier import RecipeModificationSystem
import pandas as pd
import ast


def load_diabetic_profiles():
    """Load the 66 diabetes patient profiles."""
    df = pd.read_csv('/home/bhavesh/MajorB/RecipeAI/data/health_profiles_diabetes.csv')
    
    # Convert string representation of list to actual list
    df['dietary_restrictions'] = df['dietary_restrictions'].apply(ast.literal_eval)
    
    return df


def convert_profile_to_health_data(profile):
    """Convert diabetes profile to health_data format."""
    return {
        'age': int(profile['estimated_age']),
        'weight': float(profile['estimated_weight_kg']),
        'height': float(profile['estimated_height_cm']),
        'gender': profile['gender'] if profile['gender'] != 'unknown' else 'male',
        'activity_level': profile['activity_level'],
        'goal': profile['health_goal'],
        
        # Diabetes-specific data
        'has_diabetes': True,
        'avg_glucose': profile['avg_glucose_mg_dl'],
        'glucose_control': profile['glucose_control'],
        'dietary_restrictions': profile['dietary_restrictions'],
        'target_carbs_percent': int(profile['target_carbs_percent']),
        'target_fiber_g': int(profile['target_fiber_g']),
        'target_sugar_g': int(profile['target_sugar_g'])
    }


def add_diabetic_modifications(modifications, health_data, recipe):
    """Add diabetes-specific recipe modifications."""
    
    glucose_control = health_data['glucose_control']
    avg_glucose = health_data['avg_glucose']
    
    # Carbohydrate modifications based on glucose control
    if glucose_control == 'poor':
        modifications.append({
            'type': 'carbs_strict',
            'change': 'Reduce refined carbs by 60%, use low-GI alternatives',
            'reason': f'Poor glucose control (avg {avg_glucose:.0f} mg/dL, target <140)',
            'substitutions': {
                'white rice': 'brown rice or cauliflower rice',
                'white bread': 'whole grain bread',
                'potato': 'sweet potato',
                'regular pasta': 'whole wheat pasta or zoodles'
            }
        })
        modifications.append({
            'type': 'fiber_boost',
            'change': 'Add extra fiber: beans, lentils, vegetables',
            'reason': 'High fiber slows glucose absorption'
        })
    
    elif glucose_control == 'fair':
        modifications.append({
            'type': 'carbs_moderate',
            'change': 'Replace 40% of refined carbs with complex carbs',
            'reason': f'Fair glucose control (avg {avg_glucose:.0f} mg/dL)',
            'substitutions': {
                'white rice': 'mix of white and brown rice',
                'regular flour': 'whole wheat flour'
            }
        })
    
    # Sugar elimination
    modifications.append({
        'type': 'sugar_elimination',
        'change': 'Remove all added sugars, use stevia/erythritol if needed',
        'reason': 'Diabetic diet requires <25g sugar/day',
        'substitutions': {
            'sugar': 'stevia or erythritol',
            'honey': 'sugar-free sweetener',
            'maple syrup': 'sugar-free syrup',
            'dried fruits': 'fresh berries (small portion)'
        }
    })
    
    # Portion timing for blood sugar management
    modifications.append({
        'type': 'portion_timing',
        'change': 'Divide into smaller portions, eat slowly over 20+ minutes',
        'reason': 'Prevents blood sugar spikes'
    })
    
    # Protein pairing
    modifications.append({
        'type': 'protein_pairing',
        'change': 'Ensure protein in every meal (lean chicken, fish, tofu, legumes)',
        'reason': 'Protein slows carb digestion and stabilizes blood sugar'
    })
    
    # Fat quality
    modifications.append({
        'type': 'healthy_fats',
        'change': 'Use olive oil, avocado, nuts instead of butter/ghee',
        'reason': 'Improve insulin sensitivity with healthy fats'
    })
    
    return modifications


def test_recipes_for_diabetic_patients():
    """Test recipe modification for diabetes patients."""
    
    print("="*80)
    print("DIABETIC RECIPE MODIFICATION TEST")
    print("="*80)
    
    # Load profiles
    profiles_df = load_diabetic_profiles()
    print(f"\n✓ Loaded {len(profiles_df)} diabetic patient profiles")
    
    # Initialize system
    modifier = RecipeModificationSystem()
    
    # Test with 3 different patients representing different glucose control levels
    test_patients = [
        profiles_df[profiles_df['glucose_control'] == 'good'].iloc[0],  # Good control
        profiles_df[profiles_df['glucose_control'] == 'fair'].iloc[0],  # Fair control
        profiles_df[profiles_df['glucose_control'] == 'poor'].iloc[0],  # Poor control
    ]
    
    # Test recipes
    test_recipe_queries = [
        ('dosa', 'breakfast'),
        ('sandwich', 'lunch'),
        ('pasta', 'dinner')
    ]
    
    for patient_idx, patient in enumerate(test_patients, 1):
        health_data = convert_profile_to_health_data(patient)
        
        print("\n" + "="*80)
        print(f"PATIENT #{patient_idx}: {patient['patient_id']}")
        print("="*80)
        print(f"Glucose Control: {patient['glucose_control'].upper()}")
        print(f"Average Glucose: {patient['avg_glucose_mg_dl']:.0f} mg/dL (target: 70-140)")
        print(f"Average Insulin: {patient['avg_insulin_units']:.1f} units/dose")
        print(f"Activity Level: {patient['activity_level']}")
        print(f"Days Tracked: {patient['days_tracked']}")
        
        # Test one recipe per patient
        recipe_query, meal_type = test_recipe_queries[patient_idx - 1]
        
        print(f"\n{'─'*80}")
        print(f"REQUESTED RECIPE: {recipe_query.upper()} ({meal_type})")
        print('─'*80)
        
        # Search for recipe
        recipes = modifier.recipe_db.search_recipes(recipe_query, limit=1)
        
        if recipes:
            recipe = recipes[0]
            print(f"\n✓ Found: {recipe['name']}")
            print(f"  Source: {recipe['source']}")
            
            # Get base modifications
            result = modifier.modify_recipe_for_health(recipe, health_data, meal_type)
            modifications = result['modifications']
            
            # Add diabetes-specific modifications
            modifications = add_diabetic_modifications(modifications, health_data, recipe)
            
            # Display modifications
            print(f"\n📋 DIABETIC-FRIENDLY MODIFICATIONS:")
            print(f"   (Total: {len(modifications)} changes)")
            
            for i, mod in enumerate(modifications, 1):
                print(f"\n   {i}. [{mod['type'].upper()}]")
                print(f"      Change: {mod['change']}")
                print(f"      Reason: {mod['reason']}")
                
                if 'substitutions' in mod:
                    print(f"      Substitutions:")
                    for original, replacement in mod['substitutions'].items():
                        print(f"        • {original} → {replacement}")
            
            # Nutritional targets
            print(f"\n📊 NUTRITIONAL TARGETS FOR THIS MEAL:")
            print(f"   Calories: ~{result['target_nutrition']['calories']:.0f} cal")
            print(f"   Protein: ≥{result['target_nutrition']['protein']:.0f}g")
            print(f"   Carbs: ≤{health_data['target_carbs_percent']}% of calories (complex carbs)")
            print(f"   Fiber: ≥{health_data['target_fiber_g']//3}g per meal")
            print(f"   Sugar: <{health_data['target_sugar_g']//3}g per meal")
            
        else:
            print(f"\n✗ No recipes found for '{recipe_query}'")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: DIABETIC PATIENT PROFILES")
    print("="*80)
    print(f"Total Patients: {len(profiles_df)}")
    print(f"\nGlucose Control Distribution:")
    for control, count in profiles_df['glucose_control'].value_counts().items():
        pct = (count / len(profiles_df)) * 100
        print(f"  {control.capitalize():8s}: {count:2d} patients ({pct:4.1f}%)")
    
    print(f"\nActivity Level Distribution:")
    for activity, count in profiles_df['activity_level'].value_counts().items():
        pct = (count / len(profiles_df)) * 100
        print(f"  {activity.capitalize():12s}: {count:2d} patients ({pct:4.1f}%)")
    
    print(f"\nAverage Metrics:")
    print(f"  Glucose: {profiles_df['avg_glucose_mg_dl'].mean():.1f} mg/dL (target: 70-140)")
    print(f"  Insulin: {profiles_df['avg_insulin_units'].mean():.1f} units/dose")
    print(f"  Tracking: {profiles_df['days_tracked'].mean():.1f} days/patient")
    
    print("\n✓ Integration complete! System can now modify recipes for diabetic patients.")


if __name__ == "__main__":
    test_recipes_for_diabetic_patients()
