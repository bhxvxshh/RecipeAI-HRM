"""
Improved Reward Configuration v2 - Research-Grade
Based on analysis of calorie bias (23.5% satisfaction) and fat imbalance (36%)

Key Changes from v1:
1. Reduced exponential factor: 2.0 → 1.5 (less aggressive penalties)
2. Loosened progressive thresholds: 10%/20%/30% → 15%/25%/35%
3. Rebalanced weights: Calories 0.25 (from 0.30), Carbs 0.20 (from 0.15)
4. Reduced completion bonus: 150 (from 200) to prevent overfitting
5. Added gradient clipping for stability

Expected Improvements:
- Calorie satisfaction: 23.5% → 40%+
- Fat satisfaction: 36% → 45%+
- Overall satisfaction: 57.2% → 65%+
- Reward variance: ±200 → ±50
"""

import numpy as np

# ============================================================================
# NUTRIENT IMPORTANCE WEIGHTS (sum to 1.0)
# ============================================================================

NUTRIENT_IMPORTANCE = {
    'calories': 0.25,  # Reduced from 0.30 (was too aggressive)
    'protein': 0.20,   # Keep (performing adequately at 63.5%)
    'fat': 0.25,       # Keep (critical issue, needs focus)
    'carbs': 0.20,     # Increased from 0.15 (rebalance weight distribution)
    'sodium': 0.10,    # Keep (performing well at 84%)
}

# Validate weights sum to 1.0
assert abs(sum(NUTRIENT_IMPORTANCE.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ============================================================================
# CONSTRAINT VIOLATION PENALTIES
# ============================================================================

def calculate_violation_penalty(value, min_val, max_val, target, nutrient_weight):
    """
    Calculate penalty/reward with REDUCED exponential scaling for violations.
    
    Changes from v1:
    - Exponential factor: 2.0 → 1.5 (less harsh)
    - Progressive thresholds: 10/20/30 → 15/25/35 (more lenient)
    - Gradient clipping to prevent extreme penalties
    
    Args:
        value: Current nutrient value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        target: Target value (center of range)
        nutrient_weight: Importance weight from NUTRIENT_IMPORTANCE (0-1)
    
    Returns:
        Penalty value (negative for violation, positive for satisfaction)
    """
    # Within acceptable range - reward proximity to target
    if min_val <= value <= max_val:
        deviation_pct = abs(value - target) / (target + 1e-8)
        
        # Progressive rewards (LOOSENED THRESHOLDS)
        if deviation_pct < 0.15:  # Within 15% of target (was 10%)
            return 100 * nutrient_weight
        elif deviation_pct < 0.25:  # Within 25% (was 20%)
            return 50 * nutrient_weight
        elif deviation_pct < 0.35:  # Within 35% (was 30%)
            return 20 * nutrient_weight
        else:  # Further but still in acceptable range
            return 5 * nutrient_weight
    
    # Out of range - exponential penalty (REDUCED AGGRESSION)
    else:
        if value < min_val:
            violation_pct = (min_val - value) / (min_val + 1e-8)
        else:
            violation_pct = (value - max_val) / (max_val + 1e-8)
        
        # Exponential penalty with REDUCED factor
        base_penalty = -50 * nutrient_weight
        exponential_factor = (1 + violation_pct) ** 1.5  # Was 2.0
        
        penalty = base_penalty * exponential_factor
        
        # Gradient clipping to prevent extreme penalties
        penalty = max(penalty, -200 * nutrient_weight)  # Cap max penalty
        
        return penalty


def calculate_weighted_constraint_reward(current_nutrients, target_constraints):
    """
    Calculate total reward using weighted penalties for all nutrients.
    
    Args:
        current_nutrients: Dict with current nutrient values
        target_constraints: Dict with min/max/target for each nutrient
    
    Returns:
        total_reward: Sum of all nutrient penalties/rewards
        satisfied_count: Number of constraints satisfied
        details: Dict with per-nutrient breakdown
    """
    total_reward = 0.0
    satisfied_count = 0
    details = {}
    
    for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
        value = current_nutrients.get(nutrient, 0)
        min_val = target_constraints[nutrient]['min']
        max_val = target_constraints[nutrient]['max']
        target = target_constraints[nutrient]['target']
        weight = NUTRIENT_IMPORTANCE[nutrient]
        
        # Calculate reward for this nutrient
        reward = calculate_violation_penalty(value, min_val, max_val, target, weight)
        total_reward += reward
        
        # Track satisfaction
        if min_val <= value <= max_val:
            satisfied_count += 1
        
        # Store details for analysis
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


# ============================================================================
# COMPLETION BONUSES
# ============================================================================

COMPLETION_BONUS = 150  # Reduced from 200 (prevent overfitting to 5/5)
PARTIAL_BONUS = 50      # Keep (reward 4/5 satisfaction)

def calculate_completion_bonus(satisfied_count, total_constraints=5):
    """
    Calculate bonus for satisfying multiple constraints.
    
    Changes from v1:
    - Reduced perfect bonus: 200 → 150
    - Added graduated bonuses for 3/5
    
    Args:
        satisfied_count: Number of constraints satisfied
        total_constraints: Total number of constraints (default 5)
    
    Returns:
        bonus: Completion bonus value
    """
    if satisfied_count == total_constraints:
        return COMPLETION_BONUS  # All constraints met
    elif satisfied_count == total_constraints - 1:
        return PARTIAL_BONUS  # Almost perfect (4/5)
    elif satisfied_count == total_constraints - 2:
        return 25  # Good attempt (3/5) - NEW
    else:
        return 0  # Too few satisfied


# ============================================================================
# DIVERSITY AND BALANCE REWARDS
# ============================================================================

DIVERSITY_BONUS_WEIGHT = 10.0  # Keep from v1
NOVELTY_BONUS_WEIGHT = 5.0     # Keep from v1

# Category balance rewards (encourage varied food groups)
CATEGORY_BALANCE_BONUS = 20.0

def calculate_category_balance_reward(ingredient_categories):
    """
    Reward recipes that include diverse food categories.
    
    Args:
        ingredient_categories: List of categories for current ingredients
    
    Returns:
        bonus: Category diversity reward
    """
    unique_categories = len(set(ingredient_categories))
    
    # Reward having 4+ different categories (vegetables, proteins, grains, dairy)
    if unique_categories >= 4:
        return CATEGORY_BALANCE_BONUS
    elif unique_categories >= 3:
        return CATEGORY_BALANCE_BONUS * 0.5
    else:
        return 0


# ============================================================================
# PENALTIES
# ============================================================================

REPETITION_PENALTY = -20.0      # Keep (per repeated ingredient)
MIN_INGREDIENT_PENALTY = -100.0  # Keep (too few ingredients)
MAX_INGREDIENT_PENALTY = -50.0   # NEW (too many ingredients - impractical)

# Empty recipe penalty (stronger discouragement)
EMPTY_RECIPE_PENALTY = -500.0  # NEW


# ============================================================================
# PROGRESSIVE MILESTONE REWARDS
# ============================================================================

def calculate_milestone_reward(ingredient_count, min_ingredients, max_ingredients):
    """
    Reward reaching recipe construction milestones.
    
    Args:
        ingredient_count: Current number of ingredients
        min_ingredients: Minimum required ingredients
        max_ingredients: Maximum allowed ingredients
    
    Returns:
        reward: Milestone reward
    """
    reward = 0.0
    
    # Reached minimum viable recipe
    if ingredient_count == min_ingredients:
        reward += 10.0
    
    # Halfway to max (good progress)
    halfway = (min_ingredients + max_ingredients) / 2
    if abs(ingredient_count - halfway) < 0.5:
        reward += 5.0
    
    # Per-ingredient progress reward (small incremental bonus)
    if ingredient_count <= max_ingredients:
        reward += 5.0  # Increased from 2.0 (encourage building)
    
    return reward


# ============================================================================
# CONSTRAINT SATISFACTION THRESHOLDS (for curriculum learning)
# ============================================================================

SATISFACTION_THRESHOLDS = {
    'strict': 0.10,   # Within 10% of target
    'normal': 0.15,   # Within 15% (default)
    'lenient': 0.25,  # Within 25% (early training)
}


# ============================================================================
# REWARD SCALING AND NORMALIZATION
# ============================================================================

def normalize_reward(raw_reward, min_reward=-500, max_reward=500):
    """
    Normalize reward to [-1, 1] range for stable learning.
    
    Args:
        raw_reward: Raw calculated reward
        min_reward: Expected minimum reward (for scaling)
        max_reward: Expected maximum reward (for scaling)
    
    Returns:
        normalized_reward: Scaled to [-1, 1]
    """
    # Clip to expected range
    clipped = np.clip(raw_reward, min_reward, max_reward)
    
    # Scale to [-1, 1]
    normalized = 2 * (clipped - min_reward) / (max_reward - min_reward) - 1
    
    return normalized


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary():
    """Print reward configuration summary for debugging."""
    print("=" * 70)
    print("REWARD CONFIGURATION v2 - RESEARCH GRADE")
    print("=" * 70)
    print("\nNutrient Importance Weights:")
    for nutrient, weight in NUTRIENT_IMPORTANCE.items():
        print(f"  {nutrient:10s}: {weight:.2f} ({weight*100:.0f}%)")
    
    print(f"\nCompletion Bonuses:")
    print(f"  Perfect (5/5): {COMPLETION_BONUS}")
    print(f"  Almost (4/5):  {PARTIAL_BONUS}")
    print(f"  Good (3/5):    25")
    
    print(f"\nProgressive Reward Thresholds:")
    print(f"  Excellent (<15%): 100 pts × weight")
    print(f"  Good (<25%):      50 pts × weight")
    print(f"  Acceptable (<35%): 20 pts × weight")
    
    print(f"\nPenalties:")
    print(f"  Repetition:      {REPETITION_PENALTY}")
    print(f"  Too few ingr.:   {MIN_INGREDIENT_PENALTY}")
    print(f"  Too many ingr.:  {MAX_INGREDIENT_PENALTY}")
    print(f"  Empty recipe:    {EMPTY_RECIPE_PENALTY}")
    
    print(f"\nExponential Factor: 1.5 (reduced from 2.0)")
    print(f"Max Penalty Cap: -200 × weight (gradient clipping)")
    print("=" * 70)


if __name__ == "__main__":
    # Print configuration
    print_config_summary()
    
    # Test reward calculation
    print("\nTesting Reward Calculation:")
    print("-" * 70)
    
    test_nutrients = {
        'calories': 2000,
        'protein': 150,
        'fat': 60,
        'carbs': 250,
        'sodium': 2000,
    }
    
    test_constraints = {
        'calories': {'min': 1800, 'max': 2200, 'target': 2000},
        'protein': {'min': 140, 'max': 180, 'target': 160},
        'fat': {'min': 50, 'max': 70, 'target': 60},
        'carbs': {'min': 200, 'max': 300, 'target': 250},
        'sodium': {'min': 1500, 'max': 2500, 'target': 2000},
    }
    
    total_reward, satisfied, details = calculate_weighted_constraint_reward(
        test_nutrients, test_constraints
    )
    
    print(f"\nTest Recipe (Perfect Match):")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Satisfied: {satisfied}/5")
    print(f"\nPer-Nutrient Breakdown:")
    for nutrient, info in details.items():
        status = "✓" if info['satisfied'] else "✗"
        print(f"  {status} {nutrient:10s}: {info['value']:7.1f} / {info['target']:7.1f} "
              f"(reward: {info['reward']:6.1f})")
    
    # Test with violations
    print("\n" + "-" * 70)
    print("\nTest Recipe (With Violations):")
    
    violation_nutrients = {
        'calories': 1500,  # Too low (-16.7%)
        'protein': 200,    # Too high (+11%)
        'fat': 40,         # Too low (-33%)
        'carbs': 250,      # Perfect
        'sodium': 3000,    # Too high (+20%)
    }
    
    total_reward2, satisfied2, details2 = calculate_weighted_constraint_reward(
        violation_nutrients, test_constraints
    )
    
    print(f"Total Reward: {total_reward2:.2f}")
    print(f"Satisfied: {satisfied2}/5")
    print(f"\nPer-Nutrient Breakdown:")
    for nutrient, info in details2.items():
        status = "✓" if info['satisfied'] else "✗"
        deviation = ((info['value'] - info['target']) / info['target']) * 100
        print(f"  {status} {nutrient:10s}: {info['value']:7.1f} / {info['target']:7.1f} "
              f"({deviation:+6.1f}%, reward: {info['reward']:6.1f})")
