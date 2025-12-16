"""
High-Level Policy for Hierarchical RL (Phase 2)
Weekly nutrient planning meta-policy

This will be activated when config.HRM_ENABLED = True
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import config


class WeeklyPlannerPolicy:
    """
    High-Level HRM Policy
    
    Responsibilities:
    - Set weekly nutrient targets
    - Adjust daily constraints based on history
    - Provide guidance to low-level ingredient policy
    
    State: Weekly average nutrients + user health goals
    Action: Daily constraint adjustments
    Reward: Long-term health stability + weekly target achievement
    """
    
    def __init__(self, user_profile: str = 'standard'):
        """
        Initialize weekly planner
        
        Args:
            user_profile: User health profile
        """
        self.user_profile = user_profile
        self.base_constraints = config.USER_PROFILES[user_profile]
        self.weekly_targets = config.HRM_CONFIG['weekly_targets']
        self.planning_horizon = config.HRM_CONFIG['planning_horizon']
        
        # History tracking
        self.daily_recipes = []
        self.weekly_nutrients = {
            'calories': [],
            'protein': [],
            'sodium': [],
            'carbs': [],
            'fat': []
        }
    
    def get_daily_constraints(self, current_day: int) -> Dict:
        """
        Generate daily constraints based on weekly progress
        
        Args:
            current_day: Current day in week (0-6)
            
        Returns:
            Adjusted constraint dict for low-level policy
        """
        if len(self.daily_recipes) == 0:
            # First day: use base constraints
            return self.base_constraints.copy()
        
        # Calculate remaining budget
        days_remaining = self.planning_horizon - len(self.daily_recipes)
        
        if days_remaining <= 0:
            return self.base_constraints.copy()
        
        adjusted_constraints = {}
        
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            # Calculate consumed so far
            consumed = sum(self.weekly_nutrients[nutrient])
            
            # Weekly target
            weekly_target = self.weekly_targets[nutrient]
            
            # Remaining budget
            remaining = weekly_target - consumed
            daily_target = remaining / days_remaining
            
            # Adjust constraints around new target
            base = self.base_constraints[nutrient]
            adjusted_constraints[nutrient] = {
                'min': max(base['min'], daily_target * 0.7),
                'max': min(base['max'], daily_target * 1.3),
                'target': daily_target
            }
        
        return adjusted_constraints
    
    def update_history(self, recipe_info: Dict):
        """
        Update weekly history with new recipe
        
        Args:
            recipe_info: Recipe info from environment
        """
        self.daily_recipes.append(recipe_info)
        
        # Update nutrient totals
        nutrients = recipe_info['current_nutrients']
        for nutrient in self.weekly_nutrients.keys():
            self.weekly_nutrients[nutrient].append(nutrients[nutrient])
        
        # Reset weekly history if week complete
        if len(self.daily_recipes) >= self.planning_horizon:
            self.reset_week()
    
    def reset_week(self):
        """Reset weekly tracking"""
        self.daily_recipes = []
        for nutrient in self.weekly_nutrients.keys():
            self.weekly_nutrients[nutrient] = []
    
    def get_weekly_reward(self) -> float:
        """
        Calculate high-level reward based on weekly performance
        
        Returns:
            Weekly reward component
        """
        if len(self.daily_recipes) < self.planning_horizon:
            return 0.0  # Week not complete
        
        reward = 0.0
        
        # Check weekly targets
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            total = sum(self.weekly_nutrients[nutrient])
            target = self.weekly_targets[nutrient]
            
            deviation = abs(total - target) / (target + 1e-8)
            
            if deviation < 0.1:  # Within 10%
                reward += config.REWARD_WEIGHTS['weekly_target_bonus']
            else:
                reward -= deviation * 10  # Penalty for deviation
        
        # Health stability bonus (low variance)
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            values = self.weekly_nutrients[nutrient]
            if len(values) > 0:
                variance = np.var(values)
                normalized_variance = variance / (np.mean(values) + 1e-8)
                
                if normalized_variance < 0.2:  # Stable
                    reward += config.REWARD_WEIGHTS['long_term_health_stability']
        
        return reward
    
    def get_state(self) -> np.ndarray:
        """
        Get high-level state representation
        
        Returns:
            State vector for meta-policy
        """
        state = []
        
        # Weekly progress
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            consumed = sum(self.weekly_nutrients[nutrient])
            target = self.weekly_targets[nutrient]
            state.append(consumed / (target + 1e-8))
        
        # Days completed
        state.append(len(self.daily_recipes) / self.planning_horizon)
        
        # Nutrient variance (stability)
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            if len(self.weekly_nutrients[nutrient]) > 1:
                variance = np.var(self.weekly_nutrients[nutrient])
                state.append(variance)
            else:
                state.append(0.0)
        
        return np.array(state, dtype=np.float32)


class HierarchicalRecipeSystem:
    """
    Complete HRM System (Phase 2)
    
    Combines:
    - High-level: WeeklyPlannerPolicy
    - Low-level: RecipeAgent (ingredient selection)
    """
    
    def __init__(self, low_level_agent, user_profile: str = 'standard'):
        """
        Initialize hierarchical system
        
        Args:
            low_level_agent: Trained RecipeAgent
            user_profile: User health profile
        """
        self.low_level_agent = low_level_agent
        self.high_level_policy = WeeklyPlannerPolicy(user_profile)
        self.current_day = 0
    
    def generate_daily_recipe(self, deterministic: bool = False) -> Dict:
        """
        Generate recipe for current day using hierarchical planning
        
        Args:
            deterministic: Use deterministic policy (default False for diversity)
        
        Returns:
            Recipe info dict
        """
        # Get constraints from high-level policy
        daily_constraints = self.high_level_policy.get_daily_constraints(self.current_day)
        
        # Generate recipe with low-level policy (stochastic for diversity)
        recipe_info = self.low_level_agent.generate_recipe(
            constraints=daily_constraints,
            render=False,
            deterministic=deterministic
        )
        
        # Update high-level policy
        self.high_level_policy.update_history(recipe_info)
        
        self.current_day = (self.current_day + 1) % self.high_level_policy.planning_horizon
        
        return recipe_info
    
    def generate_weekly_plan(self) -> List[Dict]:
        """
        Generate complete weekly meal plan
        
        Returns:
            List of 7 daily recipes
        """
        weekly_plan = []
        
        self.high_level_policy.reset_week()
        
        for day in range(7):
            recipe = self.generate_daily_recipe()
            weekly_plan.append(recipe)
        
        return weekly_plan
    
    def evaluate_weekly_performance(self) -> Dict:
        """
        Evaluate weekly performance
        
        Returns:
            Performance metrics
        """
        metrics = {
            'weekly_reward': self.high_level_policy.get_weekly_reward(),
        }
        
        # Nutrient totals vs targets
        for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
            total = sum(self.high_level_policy.weekly_nutrients[nutrient])
            target = self.high_level_policy.weekly_targets[nutrient]
            metrics[f'{nutrient}_total'] = total
            metrics[f'{nutrient}_target'] = target
            metrics[f'{nutrient}_deviation_pct'] = abs(total - target) / target * 100
        
        return metrics


if __name__ == "__main__":
    print("HRM Weekly Planner Module (Phase 2)")
    print("This will be activated when config.HRM_ENABLED = True")
