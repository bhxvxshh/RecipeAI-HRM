"""
Reward Configuration v3 - Ultra-High Performance (Target: 85%+ all nutrients)

FAT-FOCUSED optimization to address critical bottleneck (10% → 85%+ target).
"""

import numpy as np

# ============================================================================
# NUTRIENT IMPORTANCE WEIGHTS - FAT PRIORITY
# ============================================================================

NUTRIENT_IMPORTANCE = {
    'fat': 0.35,       # INCREASED from 0.25
    'calories': 0.18,  # REDUCED from 0.25  
    'protein': 0.18,   # REDUCED from 0.20
    'carbs': 0.19,     # Similar to v2
    'sodium': 0.10,    # Same as v2
}

# Verify weights sum to 1.0
assert abs(sum(NUTRIENT_IMPORTANCE.values()) - 1.0) < 1e-6


# ============================================================================
# FAT-SPECIFIC PARAMETERS
# ============================================================================

FAT_STRICT_MODE = True
FAT_EXPONENTIAL_FACTOR = 2.0  # vs 1.5 for others
FAT_BONUS_MULTIPLIER = 2.0
FAT_SATISFACTION_BONUS = 100

FAT_PROGRESSIVE_THRESHOLDS = {
    'excellent': 0.10,  # vs 0.15 for others
    'good': 0.20,       # vs 0.25
    'acceptable': 0.30, # vs 0.35
}


# ============================================================================
# CURRICULUM PHASES - FAT FOCUSED
# ============================================================================

FAT_CURRICULUM_PHASES = [
    {'name': 'Easy Fat', 'fat_multiplier': 2.5, 'steps': 150000},
    {'name': 'Medium Fat', 'fat_multiplier': 2.0, 'steps': 150000},
    {'name': 'Normal Fat', 'fat_multiplier': 1.5, 'steps': 150000},
    {'name': 'Tight Fat', 'fat_multiplier': 1.2, 'steps': 150000},
    {'name': 'Target Fat', 'fat_multiplier': 1.0, 'steps': 200000},
]


# ============================================================================
# BONUSES & PENALTIES
# ============================================================================

COMPLETION_BONUS = 200
PARTIAL_BONUS = 60
DIVERSITY_BONUS_WEIGHT = 12.0
NOVELTY_BONUS_WEIGHT = 6.0
CATEGORY_BALANCE_BONUS = 25.0

REPETITION_PENALTY = -25.0
MIN_INGREDIENT_PENALTY = -120.0
MAX_INGREDIENT_PENALTY = -60.0
EMPTY_RECIPE_PENALTY = -600.0


# ============================================================================
# VIOLATION PENALTY CALCULATION
# ============================================================================

def calculate_violation_penalty(value, min_val, max_val, target, nutrient_weight, nutrient_name=''):
    """Calculate penalty/reward with fat-specific handling"""
    
    is_fat = nutrient_name.lower() == 'fat'
    
    # Within range - progressive reward
    if min_val <= value <= max_val:
        deviation_pct = abs(value - target) / (target + 1e-8)
        
        if is_fat:
            if deviation_pct < FAT_PROGRESSIVE_THRESHOLDS['excellent']:
                return 100 * nutrient_weight * FAT_BONUS_MULTIPLIER
            elif deviation_pct < FAT_PROGRESSIVE_THRESHOLDS['good']:
                return 50 * nutrient_weight * FAT_BONUS_MULTIPLIER
            elif deviation_pct < FAT_PROGRESSIVE_THRESHOLDS['acceptable']:
                return 20 * nutrient_weight * FAT_BONUS_MULTIPLIER
            else:
                return 5 * nutrient_weight
        else:
            if deviation_pct < 0.15:
                return 100 * nutrient_weight
            elif deviation_pct < 0.25:
                return 50 * nutrient_weight
            elif deviation_pct < 0.35:
                return 20 * nutrient_weight
            else:
                return 5 * nutrient_weight
    
    # Out of range - exponential penalty
    else:
        if value < min_val:
            violation_pct = (min_val - value) / (min_val + 1e-8)
        else:
            violation_pct = (value - max_val) / (max_val + 1e-8)
        
        base_penalty = -50 * nutrient_weight
        
        if is_fat and FAT_STRICT_MODE:
            exponential_factor = (1 + violation_pct) ** FAT_EXPONENTIAL_FACTOR
        else:
            exponential_factor = (1 + violation_pct) ** 1.5
        
        penalty = base_penalty * exponential_factor
        penalty = max(penalty, -200 * nutrient_weight)
        
        return penalty


def calculate_weighted_constraint_reward(current_nutrients, target_constraints):
    """Calculate total reward with fat priority"""
    total_reward = 0.0
    satisfied_count = 0
    details = {}
    
    for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
        value = current_nutrients.get(nutrient, 0)
        min_val = target_constraints[nutrient]['min']
        max_val = target_constraints[nutrient]['max']
        target = target_constraints[nutrient]['target']
        weight = NUTRIENT_IMPORTANCE[nutrient]
        
        reward = calculate_violation_penalty(value, min_val, max_val, target, weight, nutrient)
        total_reward += reward
        
        if min_val <= value <= max_val:
            satisfied_count += 1
        
        details[nutrient] = {
            'value': value,
            'target': target,
            'min': min_val,
            'max': max_val,
            'satisfied': min_val <= value <= max_val,
            'reward': reward,
            'weight': weight,
        }
    
    return total_reward, satisfied_count, details


def calculate_completion_bonus(satisfied_count, total_constraints=5, fat_satisfied=False):
    """Calculate completion bonus with fat bonus"""
    bonus = 0
    
    if satisfied_count == total_constraints:
        bonus += COMPLETION_BONUS
    elif satisfied_count == total_constraints - 1:
        bonus += PARTIAL_BONUS
    elif satisfied_count == total_constraints - 2:
        bonus += 30
    
    if fat_satisfied:
        bonus += FAT_SATISFACTION_BONUS
    
    return bonus


def calculate_milestone_reward(ingredient_count, min_ingredients, max_ingredients):
    """Calculate milestone rewards"""
    reward = 0.0
    
    if ingredient_count == min_ingredients:
        reward += 15.0
    
    halfway = (min_ingredients + max_ingredients) / 2
    if abs(ingredient_count - halfway) < 0.5:
        reward += 8.0
    
    if ingredient_count <= max_ingredients:
        reward += 7.0
    
    return reward


def print_config_summary():
    """Print configuration summary"""
    print("=" * 70)
    print("REWARD CONFIGURATION v3 - FAT-FOCUSED (85%+ TARGET)")
    print("=" * 70)
    print("\nNutrient Weights:")
    for nutrient, weight in sorted(NUTRIENT_IMPORTANCE.items(), key=lambda x: x[1], reverse=True):
        print(f"  {nutrient:8s}: {weight:.2f} ({weight*100:.0f}%)")
    
    print(f"\nFat-Specific:")
    print(f"  Exponential: {FAT_EXPONENTIAL_FACTOR}")
    print(f"  Bonus Multiplier: {FAT_BONUS_MULTIPLIER}x")
    print(f"  Satisfaction Bonus: {FAT_SATISFACTION_BONUS} pts")
    print(f"  Thresholds: 10%/20%/30%")
    
    print(f"\nCurriculum: {len(FAT_CURRICULUM_PHASES)} phases, 800k steps")
    print("=" * 70)
