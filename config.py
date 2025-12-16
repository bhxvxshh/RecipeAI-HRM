"""
Configuration file for Recipe Generation RL System
Designed to be HRM-ready with hierarchical constraint support
"""

# ====================
# Dataset Configuration
# ====================
DATA_DIR = "data/FoodData_Central_foundation_food_csv_2024-10-31/"
RAW_FOOD_FILE = "food.csv"
RAW_NUTRIENT_FILE = "food_nutrient.csv"
PROCESSED_INGREDIENT_FILE = "data/ingredients_processed.csv"

# Nutrients to track (aligned with USDA nutrient IDs)
# Foundation Foods uses Atwater Specific Factors for energy
NUTRIENT_IDS = {
    'calories': 2048,      # Energy (Atwater Specific Factors) (kcal)
    'protein': 1003,       # Protein (g)
    'sodium': 1093,        # Sodium (mg)
    'carbs': 1005,         # Carbohydrates (g)
    'fat': 1004,           # Total fat (g)
}

# ====================
# Recipe Generation Configuration
# ====================
MAX_INGREDIENTS_PER_RECIPE = 10
MIN_INGREDIENTS_PER_RECIPE = 3
INGREDIENT_POOL_SIZE = 500  # Top N most common ingredients
INGREDIENT_SERVING_SIZE_G = 50  # Grams per ingredient (scaled from 100g USDA data)

# ====================
# Health Constraints (Daily targets)
# ====================
# These can be overridden by user profiles or HRM high-level policy
DEFAULT_CONSTRAINTS = {
    'calories': {'min': 400, 'max': 800, 'target': 600},
    'protein': {'min': 15, 'max': 50, 'target': 30},
    'sodium': {'min': 0, 'max': 800, 'target': 500},
    'carbs': {'min': 30, 'max': 100, 'target': 60},
    'fat': {'min': 10, 'max': 30, 'target': 20},
}

# User profile templates
USER_PROFILES = {
    'standard': DEFAULT_CONSTRAINTS,
    'low_sodium': {
        'calories': {'min': 400, 'max': 800, 'target': 600},
        'protein': {'min': 20, 'max': 50, 'target': 35},
        'sodium': {'min': 0, 'max': 400, 'target': 300},
        'carbs': {'min': 30, 'max': 100, 'target': 60},
        'fat': {'min': 10, 'max': 30, 'target': 20},
    },
    'high_protein': {
        'calories': {'min': 500, 'max': 900, 'target': 700},
        'protein': {'min': 40, 'max': 70, 'target': 50},
        'sodium': {'min': 0, 'max': 800, 'target': 500},
        'carbs': {'min': 20, 'max': 80, 'target': 50},
        'fat': {'min': 15, 'max': 35, 'target': 25},
    },
    'low_carb': {
        'calories': {'min': 400, 'max': 800, 'target': 600},
        'protein': {'min': 25, 'max': 50, 'target': 35},
        'sodium': {'min': 0, 'max': 800, 'target': 500},
        'carbs': {'min': 10, 'max': 40, 'target': 25},
        'fat': {'min': 20, 'max': 40, 'target': 30},
    }
}

# ====================
# RL Configuration (Phase 1: Single Agent)
# ====================
RL_CONFIG = {
    'algorithm': 'PPO',  # or 'DQN'
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'batch_size': 32,  # Reduced for more frequent updates (anti-overfitting)
    'n_steps': 2048,
    'n_epochs': 10,
    'total_timesteps': 100000,
    'ent_coef': 0.1,  # Increased 10x for more exploration (anti-mode-collapse)
}

# ====================
# Anti-Overfitting Configuration
# ====================
VARY_CONSTRAINTS_TRAINING = True  # Randomize target constraints each episode
CONSTRAINT_NOISE_STD = 0.15  # ±15% variation in targets during training
CURIOSITY_BONUS_WEIGHT = 1.0  # Bonus for using less-visited ingredients

# ====================
# Reward Configuration (HRM-Ready)
# ====================
REWARD_WEIGHTS = {
    # Low-level rewards (immediate)
    'constraint_satisfaction': 20.0,  # Increased reward for meeting all constraints
    'nutrient_balance': 2.0,  # Increased for better balance
    'diversity_bonus': 5.0,  # Increased to encourage variety
    'ingredient_repeat_penalty': -20.0,  # Much stronger penalty for repetition
    'constraint_violation_penalty': -15.0,  # Stronger penalty
    
    # High-level rewards (HRM Phase 2)
    'weekly_target_bonus': 10.0,  # Weekly aggregate constraint bonus
    'long_term_health_stability': 5.0,  # Stability across week
}

# Lambda for hierarchical reward shaping
# Phase 1 (baseline): λ = 0
# Phase 2 (HRM): λ > 0
LAMBDA_HIERARCHICAL = 0.5  # PHASE 2 ACTIVATED

# ====================
# HRM Configuration (Phase 2 - ACTIVE)
# ====================
HRM_ENABLED = True  # PHASE 2 ACTIVATED

HRM_CONFIG = {
    'planning_horizon': 7,  # Weekly planning
    'high_level_update_freq': 7,  # Update meta-policy every N recipes
    'weekly_targets': {
        'calories': 4200,  # 600 * 7
        'protein': 210,
        'sodium': 3500,
        'carbs': 420,
        'fat': 140,
    }
}

# ====================
# Environment Configuration
# ====================
ENV_CONFIG = {
    'normalize_observations': True,
    'normalize_rewards': False,
    'recipe_history_length': 10,  # For diversity tracking
    'done_action': True,  # Allow agent to terminate recipe early
}

# ====================
# Evaluation Metrics
# ====================
EVAL_METRICS = [
    'constraint_compliance_rate',
    'nutrient_balance_score',
    'recipe_diversity',
    'average_ingredients_per_recipe',
    'reward_per_episode',
]

# ====================
# Paths
# ====================
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models/saved/")
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, "logs/tensorboard/")
EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "eval/results/")
