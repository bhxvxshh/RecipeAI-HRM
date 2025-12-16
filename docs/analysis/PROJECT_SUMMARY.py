"""
Project Summary: Recipe Generation with Hierarchical RL
========================================================

WHAT WE BUILT
-------------
✓ Complete HRM-ready recipe generation system
✓ Phase 1: Single-agent baseline (working now)
✓ Phase 2: Hierarchical architecture (ready to activate)
✓ Real USDA nutritional data pipeline
✓ Multiple user health profiles
✓ Training and evaluation scripts
✓ Comprehensive documentation

PROJECT STRUCTURE
-----------------
RecipeAI/
├── config.py                    # Central configuration (HRM-ready)
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── IMPLEMENTATION_GUIDE.md      # Detailed guide (read this!)
├── QUICKSTART.py               # Quick reference
├── demo.py                      # System verification test
│
├── data/                        # USDA datasets
│   └── README.md               # Data instructions
│
├── utils/
│   └── data_preprocessing.py   # USDA → ingredients pipeline
│
├── env/
│   └── recipe_env.py           # RL environment (Gymnasium)
│                               # - State: nutrients + constraints
│                               # - Action: select ingredient
│                               # - Reward: compliance + diversity
│
├── models/
│   ├── ingredient_policy.py    # Low-level RL agent (PPO/DQN)
│   └── hrm_policy.py           # High-level weekly planner
│
├── train/
│   ├── train_recipe.py         # Phase 1 training script
│   └── train_hrm.py            # Phase 2 training script
│
└── eval/
    └── nutrition_metrics.py    # Evaluation metrics

KEY FEATURES
------------
1. HRM-Ready Design
   - Same code for Phase 1 and Phase 2
   - Switch via config flags
   - No redesign needed

2. Modular Rewards
   - Low-level: immediate constraint satisfaction
   - High-level: weekly planning (Phase 2)
   - Lambda parameter controls hierarchy

3. User Profiles
   - Standard: balanced nutrition
   - Low sodium: <400mg
   - High protein: 40-70g
   - Low carb: <40g

4. Real Data
   - USDA FoodData Central
   - 500 diverse ingredients
   - Accurate per-100g nutrients

HOW TO USE
----------
Step 1: Get USDA data
   Download food.csv and food_nutrient.csv
   Place in data/

Step 2: Preprocess
   python utils/data_preprocessing.py

Step 3: Verify
   python demo.py

Step 4: Train Phase 1
   python train/train_recipe.py --profile standard --timesteps 100000

Step 5: Evaluate
   python -c "
   from models.ingredient_policy import RecipeAgent
   from train.train_recipe import setup_training_environment
   from eval.nutrition_metrics import evaluate_trained_agent
   
   env = setup_training_environment('standard')
   agent = RecipeAgent(env, algorithm='PPO')
   agent.load('models/saved/recipe_agent_standard_ppo')
   evaluate_trained_agent(agent, n_recipes=100)
   "

Step 6: Train Phase 2 (after Phase 1 works)
   Edit config.py:
      HRM_ENABLED = True
      LAMBDA_HIERARCHICAL = 0.5
   
   python train/train_hrm.py --pretrained models/saved/recipe_agent_standard_ppo --weeks 10

ARCHITECTURE COMPARISON
-----------------------

Phase 1 (Baseline):
   User Profile
       ↓
   RL Agent (PPO/DQN)
       ↓
   Ingredient Selection
       ↓
   Recipe
   
   Reward = R_low (immediate)

Phase 2 (HRM):
   User Profile
       ↓
   High-Level Policy (Weekly Planner)
       ↓ (daily constraints)
   Low-Level Policy (Ingredient Selection)
       ↓
   Recipe
   
   Reward = R_low + λ × R_high

STATE-ACTION-REWARD
-------------------
State (11-dim):
   [current nutrients (5), target nutrients (5), ingredient count (1)]

Action:
   - Select ingredient [0...N-1]
   - Or DONE [N]

Reward (Phase 1):
   + constraint_satisfaction (10.0)
   + nutrient_balance (1.0 per nutrient)
   + diversity_bonus (2.0)
   - constraint_violation (-10.0)
   - ingredient_repeat (-5.0)

Reward (Phase 2):
   Phase1_Reward + λ × (weekly_target + stability)

EXPECTED RESULTS
----------------
Phase 1:
   ✓ 80%+ constraint compliance
   ✓ 70%+ nutrient balance
   ✓ 5-8 ingredients/recipe
   ✓ Good diversity

Phase 2:
   ✓ 90%+ weekly target achievement
   ✓ <10% weekly deviation
   ✓ Better long-term stability
   ✓ Cross-day planning

CONFIGURATION
-------------
Edit config.py to tune:

# User constraints
USER_PROFILES = {'standard', 'low_sodium', 'high_protein', 'low_carb'}

# RL parameters
RL_CONFIG = {
    'algorithm': 'PPO',
    'learning_rate': 3e-4,
    'total_timesteps': 100000,
}

# Reward weights
REWARD_WEIGHTS = {
    'constraint_satisfaction': 10.0,
    'diversity_bonus': 2.0,
    ...
}

# HRM (Phase 2 only)
HRM_ENABLED = False  # Set True for Phase 2
LAMBDA_HIERARCHICAL = 0.0  # Set 0.5 for Phase 2

TIMELINE
--------
Day 1: Data preprocessing
Day 2-3: Train Phase 1 baseline
Day 4: Evaluate, tune hyperparameters
Day 5-6: Train Phase 2 HRM
Day 7: Final evaluation, documentation

TROUBLESHOOTING
---------------
Problem: No data files
   → Download USDA FoodData Central
   → Place food.csv and food_nutrient.csv in data/

Problem: Constraints violated
   → Increase constraint_violation_penalty
   → Check ingredient pool quality

Problem: Low diversity
   → Increase diversity_bonus
   → Increase recipe_history_length

Problem: Not converging
   → Try PPO (more stable than DQN)
   → Increase timesteps
   → Reduce learning rate

WHY THIS DESIGN WORKS
---------------------
✓ HRM-ready from start → no rewrite needed
✓ Real USDA data → realistic results
✓ Gymnasium interface → works with any RL lib
✓ Modular rewards → easy to debug
✓ User profiles → multiple health scenarios

IMPORTANT FILES TO READ
-----------------------
1. IMPLEMENTATION_GUIDE.md - Complete guide (READ THIS FIRST!)
2. config.py - All tunable parameters
3. demo.py - Working test script
4. QUICKSTART.py - Usage examples

NEXT STEPS
----------
1. Get USDA data files
2. Run data preprocessing
3. Run demo.py to verify setup
4. Train Phase 1 baseline
5. Evaluate results
6. If Phase 1 works well, proceed to Phase 2

Remember: Make Phase 1 solid before adding Phase 2!

Good luck! 🚀
"""

print(__doc__)
