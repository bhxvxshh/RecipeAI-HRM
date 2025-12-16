"""
Quick Start Guide for Recipe Generation RL System
"""

# Phase 1: Baseline Single-Agent Training

## Step 1: Prepare Data
print("="*60)
print("STEP 1: DATA PREPARATION")
print("="*60)
print("""
1. Download USDA FoodData Central data:
   - food.csv
   - food_nutrient.csv

2. Place them in: RecipeAI/data/

3. Run preprocessing:
   cd RecipeAI
   python utils/data_preprocessing.py
""")

## Step 2: Train Baseline Agent
print("\n" + "="*60)
print("STEP 2: TRAIN BASELINE AGENT (PHASE 1)")
print("="*60)
print("""
Train with default profile:
   python train/train_recipe.py --profile standard --algorithm PPO --timesteps 100000 --test

Train with different profiles:
   python train/train_recipe.py --profile low_sodium --algorithm PPO --timesteps 100000
   python train/train_recipe.py --profile high_protein --algorithm PPO --timesteps 100000
   python train/train_recipe.py --profile low_carb --algorithm PPO --timesteps 100000

Monitor training:
   tensorboard --logdir logs/tensorboard/
""")

## Step 3: Evaluate Baseline
print("\n" + "="*60)
print("STEP 3: EVALUATE BASELINE")
print("="*60)
print("""
From Python:
   from models.ingredient_policy import RecipeAgent
   from train.train_recipe import setup_training_environment
   from eval.nutrition_metrics import evaluate_trained_agent
   
   env = setup_training_environment('standard')
   agent = RecipeAgent(env, algorithm='PPO')
   agent.load('models/saved/recipe_agent_standard_ppo')
   
   metrics = evaluate_trained_agent(agent, n_recipes=100)
""")

# Phase 2: Hierarchical RL Training

print("\n" + "="*60)
print("STEP 4: TRAIN HRM SYSTEM (PHASE 2)")
print("="*60)
print("""
After Phase 1 is complete:

1. Enable HRM in config.py:
   HRM_ENABLED = True
   LAMBDA_HIERARCHICAL = 0.5

2. Train hierarchical system:
   python train/train_hrm.py --pretrained models/saved/recipe_agent_standard_ppo --profile standard --weeks 10 --eval

This will:
   - Load pretrained low-level agent
   - Initialize high-level weekly planner
   - Train hierarchical system with weekly planning
""")

# Configuration Options

print("\n" + "="*60)
print("CONFIGURATION OPTIONS")
print("="*60)
print("""
Edit config.py to customize:

User Profiles:
   - standard: Balanced nutrition
   - low_sodium: Restricted sodium intake
   - high_protein: Increased protein targets
   - low_carb: Reduced carbohydrates

RL Parameters:
   - algorithm: 'PPO' or 'DQN'
   - learning_rate: 3e-4 (default)
   - total_timesteps: 100000 (default)

Reward Weights:
   - constraint_satisfaction: 10.0
   - nutrient_balance: 1.0
   - diversity_bonus: 2.0
   - constraint_violation_penalty: -10.0

HRM Parameters (Phase 2):
   - planning_horizon: 7 days
   - weekly_target_bonus: 5.0
   - long_term_health_stability: 3.0
""")

# Expected Results

print("\n" + "="*60)
print("EXPECTED RESULTS")
print("="*60)
print("""
Phase 1 (Baseline):
   ✓ 80%+ constraint compliance rate
   ✓ 70%+ nutrient balance score
   ✓ 5-8 ingredients per recipe
   ✓ Recipe diversity improves over time

Phase 2 (HRM):
   ✓ 90%+ weekly target achievement
   ✓ Improved long-term nutrient stability
   ✓ Better cross-day planning
   ✓ <10% deviation from weekly goals
""")

# Troubleshooting

print("\n" + "="*60)
print("TROUBLESHOOTING")
print("="*60)
print("""
Issue: Agent always violates constraints
   → Increase constraint_violation_penalty
   → Decrease max_ingredients_per_recipe
   → Check data preprocessing (normalize nutrients)

Issue: Low recipe diversity
   → Increase diversity_bonus weight
   → Increase recipe_history_length
   → Check ingredient pool size

Issue: Training not converging
   → Try different algorithm (PPO vs DQN)
   → Adjust learning_rate (try 1e-4 to 1e-3)
   → Increase total_timesteps

Issue: HRM not improving over baseline
   → Increase LAMBDA_HIERARCHICAL (0.5 → 1.0)
   → Increase weekly_target_bonus
   → Check weekly_targets in config
""")

print("\n" + "="*60)
print("PROJECT TIMELINE")
print("="*60)
print("""
Day 1: Data preprocessing + environment setup
Day 2-3: Train baseline agent (Phase 1)
Day 4: Evaluate baseline, tune hyperparameters
Day 5-6: Implement and train HRM (Phase 2)
Day 7: Final evaluation, documentation, results
""")

print("\n" + "="*60)
print("Ready to start! 🚀")
print("="*60)
