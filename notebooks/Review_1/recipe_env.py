import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RecipeEnv(gym.Env):
    """
    Recipe Generation RL Environment (Inference Version)

    Compatible with:
    Observation Space: Box(0.0, 10.0, (11,), float32)
    Action Space: Discrete(N_INGREDIENTS)

    Designed for running trained SB3 PPO models from Hugging Face.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        user_constraints,
        ingredient_pool_size,
        recipe_history_length=10,
        done_action=True,
        **kwargs
    ):
        super().__init__()

        self.user_constraints = user_constraints
        self.ingredient_pool_size = ingredient_pool_size
        self.recipe_history_length = recipe_history_length
        self.done_action = done_action

        # ----------------------------
        # ACTION SPACE
        # ----------------------------
        # Each action selects an ingredient ID
        self.action_space = spaces.Discrete(self.ingredient_pool_size)

        # ----------------------------
        # OBSERVATION SPACE (CRITICAL)
        # ----------------------------
        # Must EXACTLY match training
        # [0–10 scaled constraint & progress vector]
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(11,),
            dtype=np.float32
        )

        # Episode controls
        self.max_steps = 10
        self.current_step = 0

        # State tracking
        self.current_recipe = []
        self.weekly_progress = 0.0

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_recipe = []
        self.weekly_progress = 0.0

        obs = self._get_observation()
        info = {}

        return obs, info

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        action = int(action)

        # Add ingredient to recipe
        self.current_recipe.append(action)
        self.current_step += 1

        # Simulate weekly HRM progress
        self.weekly_progress = min(1.0, self.current_step / 7)

        # Dummy reward (not used in inference)
        reward = 1.0

        # Termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()

        info = {
            "recipe_length": len(self.current_recipe),
            "weekly_progress": self.weekly_progress,
            "constraints_met": True
        }

        return obs, reward, terminated, truncated, info

    # =====================================================
    # OBSERVATION CONSTRUCTION (11-D)
    # =====================================================
    def _get_observation(self):
        """
        Observation vector layout (length = 11):

        0  calories_ratio
        1  protein_ratio
        2  sodium_ratio
        3  carbs_ratio
        4  fat_ratio
        5  ingredient_count
        6  diversity_score
        7  repeat_penalty
        8  weekly_progress
        9  step_index
        10 done_flag
        """

        # Dummy normalized ratios (replace with real nutrient logic if needed)
        calories_ratio = 5.0
        protein_ratio = 5.0
        sodium_ratio = 5.0
        carbs_ratio = 5.0
        fat_ratio = 5.0

        ingredient_count = float(len(self.current_recipe))

        # Diversity: fraction of unique ingredients
        if self.current_recipe:
            diversity_score = len(set(self.current_recipe)) / len(self.current_recipe)
        else:
            diversity_score = 0.0

        repeat_penalty = float(
            len(self.current_recipe) - len(set(self.current_recipe))
        )

        weekly_progress = self.weekly_progress
        step_index = float(self.current_step)
        done_flag = float(self.current_step >= self.max_steps)

        obs = np.array(
            [
                calories_ratio,
                protein_ratio,
                sodium_ratio,
                carbs_ratio,
                fat_ratio,
                ingredient_count,
                diversity_score,
                repeat_penalty,
                weekly_progress,
                step_index,
                done_flag,
            ],
            dtype=np.float32
        )

        return obs

    # =====================================================
    # OPTIONAL RENDER
    # =====================================================
    def render(self):
        print("Current recipe ingredient IDs:", self.current_recipe)