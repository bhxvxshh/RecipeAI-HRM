# Adaptive Health-Aware Recipe Generation using Hierarchical RL

## Project Overview
This project implements a reinforcement learning-based recipe generation system that dynamically composes ingredient combinations while balancing taste preference, nutritional adequacy, and personalized health constraints using real-world nutritional data.

## Architecture
- **Phase 1 (Current)**: Single-agent RL recipe generator with HRM-ready design
- **Phase 2 (Planned)**: Hierarchical RL with weekly nutrient planning meta-policy

## Project Structure
```
RecipeAI/
├── data/              # USDA ingredient datasets
├── env/               # RL environments (recipe_env, weekly_env for HRM)
├── models/            # RL policies (ingredient_policy, hrm_policy)
├── train/             # Training scripts
├── eval/              # Evaluation and metrics
├── utils/             # Helper functions
└── notebooks/         # Jupyter notebooks for experiments
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
- USDA FoodData Central: food.csv, food_nutrient.csv
- Processed ingredient-level nutrients: calories, protein, sodium, carbs, fat

## Phase: 1 - Simplified Single-Agent
- One RL policy for ingredient selection
- Direct constraint satisfaction in reward
- Recipe diversity tracking
- HRM-ready state/action/reward design

## Phase: 2 - Full HRM 
- High-level: Weekly nutrient budget planner
- Low-level: Ingredient selection policy
- Hierarchical reward shaping
