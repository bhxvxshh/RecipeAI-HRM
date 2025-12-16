# RecipeAI Project Structure

```
RecipeAI/
│
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
├── config.py                      # Central configuration (HRM settings, constraints)
│
├── data/                          # Data files
│   ├── FoodData_Central.../       # USDA dataset (raw)
│   └── ingredients_processed.csv  # Processed ingredient database (324 items)
│
├── env/                           # RL Environment
│   └── recipe_env.py              # RecipeEnv (Gymnasium environment)
│
├── models/                        # RL Models & Policies
│   ├── ingredient_policy.py       # Phase 1: Low-level PPO agent
│   ├── hrm_policy.py              # Phase 2: High-level weekly planner
│   └── saved/                     # Trained model checkpoints
│       └── recipe_agent_standard_ppo.zip  # Phase 1 trained agent (5.2MB)
│
├── train/                         # Training Scripts
│   └── train_recipe.py            # Main training script
│
├── eval/                          # Evaluation Scripts
│   ├── nutrition_metrics.py       # Evaluation metrics
│   ├── quick_eval.py              # Quick 10-recipe evaluation
│   ├── eval_diversity.py          # Comprehensive diversity analysis
│   └── results/                   # Evaluation results
│
├── scripts/                       # Test & Demo Scripts
│   ├── test_hrm.py                # Test Phase 2 HRM system
│   └── test_hrm_hybrid.py         # Test hybrid best-of-N sampling
│
├── utils/                         # Utilities
│   └── data_preprocessing.py      # USDA data processing
│
├── docs/                          # Documentation
│   ├── analysis/                  # Analysis & reports
│   │   ├── ANALYSIS_CORRECTION.py # ChatGPT cross-check analysis
│   │   ├── PROJECT_SUMMARY.py     # Project summary
│   │   └── IMPLEMENTATION_GUIDE.md
│   └── demos/                     # Demo scripts
│       ├── QUICKSTART.py
│       └── demo.py
│
├── logs/                          # Training logs
│   ├── tensorboard/               # TensorBoard logs
│   └── training.log               # Training output
│
└── venv/                          # Virtual environment (Python 3.12)
```

## Key Files

### Core System
- **`config.py`**: Central configuration
  - Phase 1/2 settings
  - HRM enabled: `HRM_ENABLED = True`
  - Hierarchical weight: `LAMBDA_HIERARCHICAL = 0.5`
  - Nutrient constraints and reward weights

- **`env/recipe_env.py`**: Recipe generation environment
  - State: 11-dim [current_nutrients, target_nutrients, ingredient_count]
  - Action: 325 discrete (324 ingredients + DONE)
  - Reward: Constraint satisfaction + diversity + repetition penalties

- **`models/ingredient_policy.py`**: Phase 1 low-level agent
  - PPO with custom policy network
  - 100% constraint compliance (deterministic)
  - Trained on fixed constraints

- **`models/hrm_policy.py`**: Phase 2 hierarchical system
  - `WeeklyPlannerPolicy`: Adjusts daily constraints
  - `HierarchicalRecipeSystem`: Combines high/low-level policies
  - Balances weekly nutrition targets

### Training & Evaluation
- **`train/train_recipe.py`**: Main training script
  - Usage: `python train/train_recipe.py --profile standard --algorithm PPO --timesteps 150000`
  - Saves to: `models/saved/recipe_agent_standard_ppo.zip`

- **`eval/quick_eval.py`**: Fast evaluation (10 recipes)
- **`eval/eval_diversity.py`**: Comprehensive analysis (100 recipes, deterministic vs stochastic)
- **`scripts/test_hrm_hybrid.py`**: Phase 2 weekly planning with best-of-N sampling

### Data
- **Raw**: `data/FoodData_Central_foundation_food_csv_2024-10-31/`
  - `food.csv`: 68,875 foods
  - `food_nutrient.csv`: Nutrient values
- **Processed**: `data/ingredients_processed.csv`
  - 324 ingredients after filtering
  - 5 nutrients: calories, protein, sodium, carbs, fat

## Results Summary

### Phase 1 (Baseline)
- **Deterministic**: 100% compliance, 1% diversity
- **Stochastic**: 14% compliance, 96% diversity
- **Training**: 150k timesteps, -164 avg reward

### Phase 2 (HRM)
- **Weekly targets**: 4/5 within 10% (excellent)
- **Recipe diversity**: 100% unique (7/7 days)
- **Unique ingredients**: 55/324 used
- **Daily compliance**: 14% (expected due to train/test distribution mismatch)

## Running the System

### Quick Test
```bash
cd /home/bhavesh/MajorB/RecipeAI
source venv/bin/activate
python eval/quick_eval.py
```

### Full Evaluation
```bash
python eval/eval_diversity.py
```

### Test HRM System
```bash
python scripts/test_hrm_hybrid.py
```

### Retrain (if needed)
```bash
python train/train_recipe.py --profile standard --algorithm PPO --timesteps 150000
```

## Important Notes

1. **Don't modify** `models/saved/` - contains trained model
2. **Don't modify** `data/ingredients_processed.csv` - preprocessed data
3. **All scripts expect** to run from `/home/bhavesh/MajorB/RecipeAI/`
4. **HRM is enabled** in `config.py` - set `HRM_ENABLED = False` for Phase 1 only

## Dependencies
See `requirements.txt`:
- gymnasium
- stable-baselines3
- numpy
- pandas
- torch (CPU mode)
