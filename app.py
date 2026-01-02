"""
Web interface for personalized recipe generation.
Users input health data through a clean web form.
"""

import gradio as gr
from personalized_recipe import HealthBasedRecipeGenerator
import warnings
warnings.filterwarnings('ignore')

# Initialize generator (load model once)
print("Loading AI model...")
generator = HealthBasedRecipeGenerator()
print("✓ Ready!")


def generate_recipe_web(age, weight, height, gender, activity, goal, meal_type, dietary):
    """Generate recipe from web form inputs."""
    
    # Validate inputs
    if not all([age, weight, height]):
        return "❌ Please fill in all required fields (age, weight, height)"
    
    # Parse dietary restrictions
    dietary_restrictions = None
    if dietary:
        dietary_restrictions = [d.strip() for d in dietary.split(',') if d.strip()]
    
    # Build health data
    health_data = {
        'age': int(age),
        'weight': float(weight),
        'height': float(height),
        'gender': gender.lower(),
        'activity_level': activity.lower().replace(' ', '_'),
        'goal': goal.lower().replace(' ', '_')
    }
    
    # Generate recipe
    try:
        # Calculate needs
        daily_cals = generator.calculate_calories(
            health_data['age'],
            health_data['weight'],
            health_data['height'],
            health_data['gender'],
            health_data['activity_level']
        )
        
        daily_needs = generator.adjust_for_goals(daily_cals, health_data['goal'])
        meal_constraints = generator.create_meal_constraints(daily_needs, meal_type.lower())
        
        # Summary
        output = f"""
## 📊 YOUR DAILY NUTRITIONAL NEEDS
**Goal:** {health_data['goal'].replace('_', ' ').title()}

- **Calories:** {daily_needs['calories']:.0f} kcal/day
- **Protein:** {daily_needs['protein']:.0f}g/day
- **Carbs:** {daily_needs['carbs']:.0f}g/day
- **Fat:** {daily_needs['fat']:.0f}g/day

---

## 🍽️ {meal_type.upper()} TARGET
- **Calories:** {meal_constraints['max_calories']:.0f} kcal (max)
- **Protein:** {meal_constraints['min_protein']:.0f}g (min)

---

"""
        
        # Filter ingredients
        ingredients_df = generator.filter_ingredients_by_diet(dietary_restrictions)
        
        if len(ingredients_df) < 10:
            output += f"⚠️ Warning: Only {len(ingredients_df)} ingredients match your restrictions.\n\n"
        
        # Create environment
        from env.recipe_env import RecipeEnv
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
        
        # Generate
        obs, info = env.reset()
        done = False
        recipe_ingredients = []
        
        while not done:
            action, _ = generator.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            
            if action < env.n_ingredients:
                ing = env.ingredient_df.iloc[action]
                recipe_ingredients.append(ing)
        
        if not recipe_ingredients:
            return output + "❌ No recipe generated. Try different constraints."
        
        # Format recipe
        output += "## 🤖 AI-GENERATED RECIPE\n\n### INGREDIENTS:\n"
        for ing in recipe_ingredients:
            output += f"- {ing['food_name']}\n"
        
        # Calculate nutrition
        total_cal = sum(ing['calories'] for ing in recipe_ingredients)
        total_protein = sum(ing['protein'] for ing in recipe_ingredients)
        total_carbs = sum(ing['carbs'] for ing in recipe_ingredients)
        total_fat = sum(ing['fat'] for ing in recipe_ingredients)
        total_sodium = sum(ing['sodium'] for ing in recipe_ingredients)
        
        n = len(recipe_ingredients)
        
        output += f"""
---

### NUTRITION (per 100g average):
- **Calories:** {total_cal/n:.0f} kcal
- **Protein:** {total_protein/n:.1f}g
- **Carbs:** {total_carbs/n:.1f}g
- **Fat:** {total_fat/n:.1f}g
- **Sodium:** {total_sodium/n:.0f}mg

---

### TARGET ANALYSIS:
"""
        
        if total_cal/n <= meal_constraints['max_calories']/100:
            output += "✓ Calories within limit\n"
        else:
            output += "⚠️ Calories slightly over target\n"
        
        if total_protein/n >= meal_constraints['min_protein']/100:
            output += "✓ Protein goal met\n"
        else:
            output += "⚠️ Could use more protein\n"
        
        # Categories & allergens
        food_ids = [ing['food_id'] for ing in recipe_ingredients]
        categories = set()
        allergens = set()
        
        for fid in food_ids:
            enriched = generator.ingredients_enriched[
                generator.ingredients_enriched['food_id'] == fid
            ]
            if not enriched.empty:
                categories.add(enriched.iloc[0]['category'])
                allerg = str(enriched.iloc[0]['allergens']).split(',')
                for a in allerg:
                    if a.strip() and a.strip() != 'none':
                        allergens.add(a.strip())
        
        output += f"\n**Categories:** {', '.join(categories)}\n"
        if allergens:
            output += f"**⚠️ Allergens:** {', '.join(allergens)}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your inputs and try again."


# Create Gradio interface
with gr.Blocks(title="Personalized Recipe AI", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🍳 Personalized Recipe Generator AI
    
    Get healthy recipes tailored to **your body and goals**!  
    Based on scientifically-calculated nutritional needs.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Your Health Profile")
            
            age = gr.Number(label="Age (years)", value=25, minimum=10, maximum=100)
            weight = gr.Number(label="Weight (kg)", value=70, minimum=30, maximum=200)
            height = gr.Number(label="Height (cm)", value=170, minimum=100, maximum=250)
            
            gender = gr.Radio(
                label="Gender",
                choices=["Male", "Female"],
                value="Male"
            )
            
            activity = gr.Dropdown(
                label="Activity Level",
                choices=[
                    "Sedentary",
                    "Light", 
                    "Moderate",
                    "Active",
                    "Very Active"
                ],
                value="Moderate"
            )
            
            goal = gr.Dropdown(
                label="Health Goal",
                choices=[
                    "Lose Weight",
                    "Maintain",
                    "Gain Muscle"
                ],
                value="Maintain"
            )
            
            meal_type = gr.Dropdown(
                label="Meal Type",
                choices=["Breakfast", "Lunch", "Dinner", "Snack"],
                value="Lunch"
            )
            
            dietary = gr.Textbox(
                label="Dietary Restrictions (optional)",
                placeholder="e.g., vegan,gluten_free,nut_free",
                info="Separate multiple with commas. Options: vegan, vegetarian, gluten_free, dairy_free, nut_free, keto, paleo"
            )
            
            generate_btn = gr.Button("🤖 Generate Recipe", variant="primary", size="lg")
        
        with gr.Column():
            output = gr.Markdown(label="Your Personalized Recipe")
    
    generate_btn.click(
        fn=generate_recipe_web,
        inputs=[age, weight, height, gender, activity, goal, meal_type, dietary],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    
    ### How it works:
    1. **Calculates your daily needs** using Mifflin-St Jeor equation
    2. **Adjusts for your goal** (weight loss/gain, muscle building)
    3. **AI generates recipe** using reinforcement learning model
    4. **Balances nutrition** while respecting your dietary restrictions
    
    *Made with RecipeAI - Trained on 324 ingredients from USDA database*
    """)

# Launch
if __name__ == "__main__":
    app.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",
        server_port=7860
    )
