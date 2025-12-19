import numpy as np
from collections import Counter

# ---------------------------
# Nutrient Deviation Score
# ---------------------------
def nutrient_deviation(nutrients, targets):
    total_dev = 0.0
    for k in nutrients:
        total_dev += abs(nutrients[k] - targets[k]["target"]) / targets[k]["target"]
    return total_dev / len(nutrients)


# ---------------------------
# Constraint Compliance
# ---------------------------
def constraints_met(nutrients, constraints):
    for k in nutrients:
        if not (constraints[k]["min"] <= nutrients[k] <= constraints[k]["max"]):
            return False
    return True


# ---------------------------
# Diversity Score
# ---------------------------
def diversity_score(recipe):
    if len(recipe) == 0:
        return 0.0
    return len(set(recipe)) / len(recipe)


# ---------------------------
# Repetition Penalty
# ---------------------------
def repetition_penalty(recipe):
    counts = Counter(recipe)
    return sum(v - 1 for v in counts.values() if v > 1)


# ---------------------------
# Weekly Stability (HRM)
# ---------------------------
def weekly_stability(weekly_nutrients, target):
    return np.mean([abs(day - target) for day in weekly_nutrients])
