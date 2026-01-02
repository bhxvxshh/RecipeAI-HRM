"""
Recipe Environment for RL Agent
HRM-Ready Design: Supports both standalone and hierarchical constraint passing
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
from typing import Dict, List, Optional, Tuple
import config


class RecipeEnv(gym.Env):
    """
    Reinforcement Learning Environment for Recipe Generation
    
    State: Current recipe nutrient totals + ingredient count + constraints
    Action: Select next ingredient from pool (or DONE action)
    Reward: Constraint satisfaction + diversity + nutrient balance
    
    HRM-Ready Features:
    - Accepts external constraints (from high-level policy)
    - Tracks recipe history for diversity
    - Returns hierarchical reward components
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        ingredient_df: pd.DataFrame,
        constraints: Optional[Dict] = None,
        recipe_history: Optional[List] = None,
        hrm_mode: bool = False,
        config_overrides: Optional[Dict] = None
    ):
        """
        Initialize Recipe Environment
        
        Args:
            ingredient_df: DataFrame with columns [food_id, food_name, calories, protein, sodium, carbs, fat]
            constraints: Nutrient constraints dict (if None, uses DEFAULT_CONSTRAINTS)
            recipe_history: List of past recipes for diversity tracking
            hrm_mode: If True, returns hierarchical reward components
            config_overrides: Override default config values
        """
        super().__init__()
        
        self.ingredient_df = ingredient_df.reset_index(drop=True)
        self.n_ingredients = len(ingredient_df)
        self.nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
        
        # Constraints (can be overridden by HRM high-level policy)
        self.constraints = constraints if constraints is not None else config.DEFAULT_CONSTRAINTS
        self.hrm_mode = hrm_mode
        
        # Recipe history for diversity tracking
        self.recipe_history = recipe_history if recipe_history is not None else []
        
        # Config
        self.max_ingredients = config.MAX_INGREDIENTS_PER_RECIPE
        self.min_ingredients = config.MIN_INGREDIENTS_PER_RECIPE
        self.done_action_enabled = config.ENV_CONFIG['done_action']
        
        # Action space: [0 to n_ingredients-1] for ingredients + [n_ingredients] for DONE
        if self.done_action_enabled:
            self.action_space = spaces.Discrete(self.n_ingredients + 1)
        else:
            self.action_space = spaces.Discrete(self.n_ingredients)
        
        # Observation space (normalized)
        # [current_calories, protein, sodium, carbs, fat, ingredient_count, 
        #  target_calories, target_protein, target_sodium, target_carbs, target_fat]
        obs_dim = len(self.nutrient_cols) * 2 + 1  # current + targets + count
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,  # Normalized values
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Current recipe state
        self.current_recipe = []
        self.current_nutrients = {nutrient: 0.0 for nutrient in self.nutrient_cols}
        self.ingredient_count = 0
        
        # Anti-overfitting: Track ingredient usage for curiosity bonus
        self.ingredient_visit_counts = np.zeros(self.n_ingredients)
        
        # Normalization constants (for observation)
        self.norm_factors = {
            'calories': 1000.0,
            'protein': 100.0,
            'sodium': 1000.0,
            'carbs': 100.0,
            'fat': 100.0,
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Optional dict with 'constraints' key for HRM
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Allow HRM to pass new constraints
        if options and 'constraints' in options:
            self.constraints = options['constraints']
        elif config.VARY_CONSTRAINTS_TRAINING:
            # Anti-overfitting: Vary constraints during training
            noise = np.random.normal(1.0, config.CONSTRAINT_NOISE_STD, size=5)
            base_constraints = config.DEFAULT_CONSTRAINTS
            self.constraints = {
                nutrient: {
                    'min': base_constraints[nutrient]['min'] * max(0.5, noise[i]),
                    'max': base_constraints[nutrient]['max'] * min(1.5, noise[i]),
                    'target': base_constraints[nutrient]['target'] * noise[i]
                }
                for i, nutrient in enumerate(self.nutrient_cols)
            }
        
        self.current_recipe = []
        self.current_nutrients = {nutrient: 0.0 for nutrient in self.nutrient_cols}
        self.ingredient_count = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Ingredient index (or DONE action)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        terminated = False
        truncated = False
        
        # Check if DONE action
        if self.done_action_enabled and action == self.n_ingredients:
            # Agent wants to finish recipe
            if self.ingredient_count >= self.min_ingredients:
                terminated = True
            else:
                # Terminate with heavy penalty for too few ingredients
                # Penalty scales with how many ingredients short
                penalty_per_missing = 50.0
                missing_count = self.min_ingredients - self.ingredient_count
                reward = -penalty_per_missing * missing_count - 100.0
                obs = self._get_observation()
                info = self._get_info()
                return obs, reward, True, False, info  # Terminate episode
        
        else:
            # Add ingredient to recipe
            if action >= self.n_ingredients:
                # Invalid action
                reward = -10.0
                obs = self._get_observation()
                info = self._get_info()
                return obs, reward, False, False, info
            
            ingredient = self.ingredient_df.iloc[action]
            # Convert action to int for consistent storage
            action_int = int(action) if hasattr(action, 'item') else action
            self.current_recipe.append(action_int)
            
            # Update nutrient totals (scaled to serving size)
            serving_factor = config.INGREDIENT_SERVING_SIZE_G / 100.0
            for nutrient in self.nutrient_cols:
                self.current_nutrients[nutrient] += ingredient[nutrient] * serving_factor
            
            self.ingredient_count += 1
            
            # Check if max ingredients reached
            if self.ingredient_count >= self.max_ingredients:
                terminated = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Add progress reward for each ingredient added (not for DONE action)
        if action != self.n_ingredients:
            # Base progress reward
            reward += 5.0
            
            # Anti-overfitting: Curiosity bonus for exploring novel ingredients
            visit_count = self.ingredient_visit_counts[action]
            
            try:
                from config_reward import NOVELTY_BONUS_WEIGHT
                curiosity_reward = NOVELTY_BONUS_WEIGHT / (1 + visit_count)
            except ImportError:
                curiosity_reward = config.CURIOSITY_BONUS_WEIGHT / (1 + visit_count)
            
            reward += curiosity_reward
            self.ingredient_visit_counts[action] += 1
            
            # Intermediate milestone rewards (encourage building toward targets)
            if self.ingredient_count == self.min_ingredients:
                reward += 10.0  # Reached minimum
            elif self.ingredient_count == self.max_ingredients // 2:
                reward += 5.0  # Halfway point
        
        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Add recipe to history if terminated
        if terminated:
            self.recipe_history.append({
                'ingredients': self.current_recipe.copy(),
                'nutrients': self.current_nutrients.copy()
            })
            # Keep only recent history
            if len(self.recipe_history) > config.ENV_CONFIG['recipe_history_length']:
                self.recipe_history.pop(0)
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (normalized)
        
        Returns:
            Normalized state vector
        """
        obs = []
        
        # Current nutrient values (normalized)
        for nutrient in self.nutrient_cols:
            normalized = self.current_nutrients[nutrient] / self.norm_factors[nutrient]
            obs.append(normalized)
        
        # Target nutrient values (normalized)
        for nutrient in self.nutrient_cols:
            target = self.constraints[nutrient]['target']
            normalized = target / self.norm_factors[nutrient]
            obs.append(normalized)
        
        # Ingredient count (normalized)
        obs.append(self.ingredient_count / self.max_ingredients)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward (HRM-ready with component breakdown)
        Improved: Weighted penalties for calories/fat, progressive rewards
        
        Returns:
            Total reward (or dict if hrm_mode=True)
        """
        reward_components = {}
        
        # Import improved reward config
        try:
            from config_reward import (
                NUTRIENT_IMPORTANCE,
                calculate_violation_penalty,
                COMPLETION_BONUS,
                PARTIAL_BONUS
            )
            use_improved_rewards = True
        except ImportError:
            use_improved_rewards = False
        
        # 1. Constraint satisfaction reward (IMPROVED)
        constraint_score = 0
        satisfied_count = 0
        
        for nutrient in self.nutrient_cols:
            value = self.current_nutrients[nutrient]
            min_val = self.constraints[nutrient]['min']
            max_val = self.constraints[nutrient]['max']
            target = self.constraints[nutrient]['target']
            
            if use_improved_rewards:
                # Use weighted, exponential penalty system
                weight = NUTRIENT_IMPORTANCE.get(nutrient, 0.2)
                nutrient_reward = calculate_violation_penalty(value, min_val, max_val, target, weight)
                constraint_score += nutrient_reward
                
                if min_val <= value <= max_val:
                    satisfied_count += 1
            else:
                # Original reward logic (fallback)
                if min_val <= value <= max_val:
                    deviation = abs(value - target) / (target + 1e-8)
                    constraint_score += (1.0 - deviation) * config.REWARD_WEIGHTS['nutrient_balance']
                    satisfied_count += 1
                else:
                    constraint_score += config.REWARD_WEIGHTS['constraint_violation_penalty']
        
        # Completion bonuses
        if use_improved_rewards:
            if satisfied_count == 5 and self.ingredient_count >= self.min_ingredients:
                constraint_score += COMPLETION_BONUS  # Perfect recipe
            elif satisfied_count >= 4:
                constraint_score += PARTIAL_BONUS  # Almost perfect
        else:
            if satisfied_count == 5 and self.ingredient_count >= self.min_ingredients:
                constraint_score += config.REWARD_WEIGHTS['constraint_satisfaction']
        
        reward_components['constraint_reward'] = constraint_score
        
        # 2. Diversity bonus
        diversity_score = self._calculate_diversity_bonus()
        reward_components['diversity_reward'] = diversity_score
        
        # 3. Ingredient repetition penalty (within same recipe)
        repetition_penalty = 0
        # Convert to hashable types for set comparison
        recipe_ints = [int(x) if hasattr(x, 'item') else x for x in self.current_recipe]
        unique_count = len(set(recipe_ints))
        total_count = len(recipe_ints)
        
        if total_count > unique_count:
            # Penalize based on number of repetitions
            num_repeats = total_count - unique_count
            repetition_penalty = config.REWARD_WEIGHTS['ingredient_repeat_penalty'] * num_repeats
        reward_components['repetition_penalty'] = repetition_penalty
        
        # 4. Low-level total
        low_level_reward = constraint_score + diversity_score + repetition_penalty
        reward_components['low_level_total'] = low_level_reward
        
        # 5. High-level reward (for future HRM integration)
        high_level_reward = 0.0
        if config.HRM_ENABLED:
            high_level_reward = self._calculate_high_level_reward()
        reward_components['high_level_total'] = high_level_reward
        
        # 6. Combined reward with hierarchical weighting
        total_reward = (
            low_level_reward + 
            config.LAMBDA_HIERARCHICAL * high_level_reward
        )
        
        if self.hrm_mode:
            return total_reward, reward_components
        else:
            return total_reward
    
    def _calculate_diversity_bonus(self) -> float:
        """
        Calculate diversity bonus based on recipe history
        
        Returns:
            Diversity bonus score
        """
        if len(self.recipe_history) == 0:
            return 0.0
        
        # Check ingredient overlap with recent recipes
        # Convert to list of ints to handle numpy arrays
        current_set = set([int(x) if hasattr(x, 'item') else x for x in self.current_recipe])
        overlap_scores = []
        
        for past_recipe in self.recipe_history[-5:]:  # Last 5 recipes
            past_set = set([int(x) if hasattr(x, 'item') else x for x in past_recipe['ingredients']])
            overlap = len(current_set & past_set) / (len(current_set) + 1e-8)
            overlap_scores.append(overlap)
        
        avg_overlap = np.mean(overlap_scores)
        diversity_bonus = (1.0 - avg_overlap) * config.REWARD_WEIGHTS['diversity_bonus']
        
        return diversity_bonus
    
    def _calculate_high_level_reward(self) -> float:
        """
        Calculate high-level reward for HRM (weekly nutrient tracking)
        
        This will be properly implemented when HRM is enabled
        
        Returns:
            High-level reward component
        """
        # Placeholder for Phase 2 HRM integration
        # Will calculate deviation from weekly nutrient targets
        return 0.0
    
    def _get_info(self) -> Dict:
        """
        Get additional info about current state
        
        Returns:
            Info dict
        """
        info = {
            'ingredient_count': self.ingredient_count,
            'current_nutrients': self.current_nutrients.copy(),
            'recipe': self.current_recipe.copy(),
            'constraints': self.constraints,
        }
        
        # Add constraint compliance check
        compliance = {}
        for nutrient in self.nutrient_cols:
            value = self.current_nutrients[nutrient]
            min_val = self.constraints[nutrient]['min']
            max_val = self.constraints[nutrient]['max']
            compliance[nutrient] = min_val <= value <= max_val
        
        info['compliance'] = compliance
        info['all_constraints_met'] = all(compliance.values())
        
        return info
    
    def render(self):
        """Render current recipe state"""
        print("\n=== Current Recipe ===")
        print(f"Ingredients: {self.ingredient_count}")
        for idx in self.current_recipe:
            ingredient = self.ingredient_df.iloc[idx]
            print(f"  - {ingredient['food_name']}")
        
        print("\n=== Nutrient Totals ===")
        for nutrient in self.nutrient_cols:
            value = self.current_nutrients[nutrient]
            target = self.constraints[nutrient]['target']
            min_val = self.constraints[nutrient]['min']
            max_val = self.constraints[nutrient]['max']
            status = "✓" if min_val <= value <= max_val else "✗"
            print(f"  {nutrient}: {value:.1f} / {target:.1f} [{min_val}-{max_val}] {status}")
    
    def set_constraints(self, new_constraints: Dict):
        """
        Update constraints (for HRM high-level policy)
        
        Args:
            new_constraints: New constraint dict
        """
        self.constraints = new_constraints
    
    def get_ingredient_name(self, action: int) -> str:
        """Get ingredient name from action index"""
        if action < self.n_ingredients:
            return self.ingredient_df.iloc[action]['food_name']
        return "DONE"
