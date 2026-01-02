"""
Improved reward configuration to fix calorie and fat bias.
Changes:
1. Weighted penalties for critical nutrients (calories, fat)
2. Progressive rewards for getting closer to targets
3. Balanced constraint satisfaction
4. Exponential penalties for large violations
"""

# Nutrient importance weights (sum to 1.0)
NUTRIENT_IMPORTANCE = {
    'calories': 0.30,  # Most critical - 30% weight (was equal before)
    'protein': 0.20,   # Important for health
    'fat': 0.25,       # Critical and problematic - 25% weight
    'carbs': 0.15,     # Moderate importance
    'sodium': 0.10,    # Least critical (easier to control)
}

# Constraint violation penalties (exponential scaling)
def calculate_violation_penalty(value, min_val, max_val, target, nutrient_weight):
    """
    Calculate penalty with exponential scaling for large violations.
    
    Args:
        value: Current nutrient value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        target: Target value
        nutrient_weight: Importance weight (0-1)
    
    Returns:
        Penalty value (negative)
    """
    if min_val <= value <= max_val:
        # Within range - reward proximity to target
        deviation_pct = abs(value - target) / (target + 1e-8)
        
        if deviation_pct < 0.10:  # Within 10% of target
            return 100 * nutrient_weight
        elif deviation_pct < 0.20:  # Within 20%
            return 50 * nutrient_weight
        elif deviation_pct < 0.30:  # Within 30%
            return 20 * nutrient_weight
        else:  # Further but still in range
            return 5 * nutrient_weight
    else:
        # Out of range - exponential penalty
        if value < min_val:
            violation_pct = (min_val - value) / (min_val + 1e-8)
        else:
            violation_pct = (value - max_val) / (max_val + 1e-8)
        
        # Exponential penalty: -50 * weight * (1 + violation%)^2
        base_penalty = -50 * nutrient_weight
        exponential_factor = (1 + violation_pct) ** 2
        
        return base_penalty * exponential_factor


# Bonus rewards
COMPLETION_BONUS = 200  # All constraints satisfied
PARTIAL_BONUS = 50      # Most constraints satisfied (4/5)

# Diversity rewards
DIVERSITY_BONUS_WEIGHT = 10.0
NOVELTY_BONUS_WEIGHT = 5.0

# Penalties
REPETITION_PENALTY = -20.0  # Per repeated ingredient
MIN_INGREDIENT_PENALTY = -100.0  # Too few ingredients
