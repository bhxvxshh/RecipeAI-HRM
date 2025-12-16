# RecipeAI Project Structure

## Directory Contents

### `/data`
- Place your USDA FoodData Central files here:
  - `food.csv` - Food descriptions and metadata
  - `food_nutrient.csv` - Nutrient values for each food
- After preprocessing: `ingredients_processed.csv` will be generated

### `/env`
- `recipe_env.py` - Gymnasium environment for recipe generation
  - State: Current recipe nutrients + constraints
  - Action: Select ingredient or DONE
  - Reward: Constraint satisfaction + diversity

### `/models`
- `ingredient_policy.py` - RL agent for ingredient selection (Phase 1)
- `hrm_policy.py` - High-level weekly planner policy (Phase 2)

### `/train`
- `train_recipe.py` - Training script for Phase 1 baseline
- `train_hrm.py` - Training script for Phase 2 HRM system

### `/eval`
- `nutrition_metrics.py` - Evaluation metrics and analysis

### `/utils`
- `data_preprocessing.py` - USDA data preprocessing pipeline

### `/logs`
- Created automatically during training
- Contains TensorBoard logs for monitoring

### `/models/saved`
- Saved trained models

## Model Checkpoints

Trained models will be saved as:
- `models/saved/recipe_agent_<profile>_<algorithm>`

Example:
- `models/saved/recipe_agent_standard_ppo`
- `models/saved/recipe_agent_low_sodium_ppo`
