# Implementation Guide: Adaptive Health-Aware Recipe Generation

## Project Overview

This project implements a **Hierarchical Reinforcement Learning (HRM)** system for recipe generation that:
- ✓ Generates recipes ingredient-by-ingredient (not recommendation)
- ✓ Ensures nutritional and health constraint compliance
- ✓ Learns from feedback over time
- ✓ Uses real USDA nutritional data
- ✓ Is designed HRM-ready from the start

## Two-Phase Approach

### Phase 1: Single-Agent Baseline (Current)
**Goal**: Build a working RL agent that generates recipes satisfying constraints

**What you get**:
- One policy: ingredient selection
- Direct constraint satisfaction
- Recipe diversity tracking
- Immediate rewards only (λ = 0)

**Timeline**: Days 1-4

### Phase 2: Full HRM (Future)
**Goal**: Add hierarchical planning with weekly nutrient targets

**What you add**:
- High-level policy: weekly nutrient planner
- Low-level policy: reuse Phase 1 agent
- Hierarchical rewards (λ > 0)
- Long-term health stability

**Timeline**: Days 5-7

---

## Architecture (HRM-Ready Design)

```
Phase 1 (Baseline):
    User Profile → RL Agent → Ingredient Selection → Recipe
                      ↑              ↓
                      └─── Reward ───┘

Phase 2 (HRM):
    User Profile
         ↓
    High-Level Policy (Weekly Planner)
         ↓ (constraints)
    Low-Level Policy (Ingredient Selection)
         ↓
    Recipe
```

**Key Design Decision**: Both phases use the same code structure. Phase 2 simply activates HRM mode by:
1. Setting `config.HRM_ENABLED = True`
2. Setting `config.LAMBDA_HIERARCHICAL > 0`
3. Using `HierarchicalRecipeSystem` wrapper

---

## File Structure

```
RecipeAI/
├── config.py                    # All configuration (HRM-ready)
├── README.md                    # Project overview
├── requirements.txt             # Dependencies
├── demo.py                      # Quick test script
├── QUICKSTART.py               # Usage guide
│
├── data/                        # USDA datasets
│   ├── food.csv                # (you provide)
│   ├── food_nutrient.csv       # (you provide)
│   └── ingredients_processed.csv  # (generated)
│
├── utils/
│   └── data_preprocessing.py   # USDA data pipeline
│
├── env/
│   └── recipe_env.py           # RL environment (HRM-ready)
│
├── models/
│   ├── ingredient_policy.py    # Low-level agent (Phase 1)
│   └── hrm_policy.py           # High-level planner (Phase 2)
│
├── train/
│   ├── train_recipe.py         # Phase 1 training
│   └── train_hrm.py            # Phase 2 training
│
└── eval/
    └── nutrition_metrics.py    # Evaluation metrics
```

---

## Quick Start

### Step 0: Installation
```bash
cd /home/bhavesh/MajorB/RecipeAI
pip install -r requirements.txt
```

### Step 1: Get USDA Data
Download from: https://fdc.nal.usda.gov/download-datasets.html

Required files:
- `food.csv`
- `food_nutrient.csv`

Place in: `RecipeAI/data/`

### Step 2: Preprocess Data
```bash
python utils/data_preprocessing.py
```

This creates: `data/ingredients_processed.csv` with 500 ingredients

### Step 3: Verify Setup
```bash
python demo.py
```

Runs 5 tests to verify everything works.

### Step 4: Train Phase 1 (Baseline)
```bash
# Standard profile
python train/train_recipe.py --profile standard --algorithm PPO --timesteps 100000 --test

# Or other profiles
python train/train_recipe.py --profile low_sodium --algorithm PPO --timesteps 50000
python train/train_recipe.py --profile high_protein --algorithm PPO --timesteps 50000
```

### Step 5: Monitor Training
```bash
tensorboard --logdir logs/tensorboard/
```

Open: http://localhost:6006

### Step 6: Evaluate Baseline
```python
from models.ingredient_policy import RecipeAgent
from train.train_recipe import setup_training_environment
from eval.nutrition_metrics import evaluate_trained_agent

env = setup_training_environment('standard')
agent = RecipeAgent(env, algorithm='PPO')
agent.load('models/saved/recipe_agent_standard_ppo')

metrics = evaluate_trained_agent(agent, n_recipes=100)
```

### Step 7: Train Phase 2 (HRM) - After Phase 1 works
```bash
# Edit config.py first:
# HRM_ENABLED = True
# LAMBDA_HIERARCHICAL = 0.5

python train/train_hrm.py \
    --pretrained models/saved/recipe_agent_standard_ppo \
    --profile standard \
    --weeks 10 \
    --eval
```

---

## Configuration

Edit `config.py` to customize:

### User Profiles
```python
USER_PROFILES = {
    'standard': {...},      # Balanced
    'low_sodium': {...},    # <400mg sodium
    'high_protein': {...},  # 40-70g protein
    'low_carb': {...}       # <40g carbs
}
```

### RL Parameters
```python
RL_CONFIG = {
    'algorithm': 'PPO',     # or 'DQN'
    'learning_rate': 3e-4,
    'total_timesteps': 100000,
}
```

### Reward Weights
```python
REWARD_WEIGHTS = {
    'constraint_satisfaction': 10.0,
    'nutrient_balance': 1.0,
    'diversity_bonus': 2.0,
    'constraint_violation_penalty': -10.0,
    
    # Phase 2 only:
    'weekly_target_bonus': 0.0,  # Set to 5.0 for HRM
    'long_term_health_stability': 0.0,  # Set to 3.0 for HRM
}

LAMBDA_HIERARCHICAL = 0.0  # Phase 1: 0.0, Phase 2: 0.5
```

### HRM Parameters (Phase 2)
```python
HRM_ENABLED = False  # Set True for Phase 2

HRM_CONFIG = {
    'planning_horizon': 7,  # Weekly planning
    'weekly_targets': {
        'calories': 4200,   # 7 days × 600
        'protein': 210,
        'sodium': 3500,
        ...
    }
}
```

---

## How It Works

### State Representation
```
Low-Level State (11-dim):
[calories_current, protein_current, sodium_current, carbs_current, fat_current,
 calories_target, protein_target, sodium_target, carbs_target, fat_target,
 ingredient_count_normalized]
```

### Action Space
```
Actions:
- [0 to N-1]: Select ingredient from pool
- [N]: DONE (finish recipe)
```

### Reward Function
```python
# Phase 1 (λ = 0):
reward = R_low
    where R_low = constraint_satisfaction + nutrient_balance + diversity - penalties

# Phase 2 (λ > 0):
reward = R_low + λ × R_high
    where R_high = weekly_target_achievement + health_stability
```

### Recipe Generation Process
```
1. Reset environment → empty recipe
2. Agent selects ingredient → add to recipe
3. Update nutrient totals
4. Calculate reward
5. Repeat until:
   - Agent says DONE
   - Max ingredients reached
   - Constraints violated too much
```

---

## Expected Results

### Phase 1 Baseline (Single Agent)
| Metric | Target | Typical |
|--------|--------|---------|
| Constraint Compliance | >80% | 75-85% |
| Nutrient Balance Score | >70% | 65-80% |
| Ingredients per Recipe | 5-8 | 6-7 |
| Recipe Diversity | Increasing | Good after 50k steps |

### Phase 2 HRM (Hierarchical)
| Metric | Target | Improvement over Phase 1 |
|--------|--------|--------------------------|
| Weekly Target Achievement | >90% | +10-15% |
| Weekly Deviation | <10% | -5-10% |
| Long-term Stability | High | +20-30% |
| Cross-day Planning | Yes | New capability |

---

## Debugging Tips

### Problem: Agent always violates constraints
**Solutions**:
- Increase `constraint_violation_penalty` (-10 → -20)
- Decrease `max_ingredients_per_recipe` (10 → 7)
- Check ingredient pool (some ingredients might dominate)
- Normalize observations properly

### Problem: Low recipe diversity
**Solutions**:
- Increase `diversity_bonus` (2.0 → 5.0)
- Increase `recipe_history_length` (10 → 20)
- Add ingredient repetition penalty within recipe

### Problem: Training not converging
**Solutions**:
- Try PPO instead of DQN (more stable)
- Reduce learning rate (3e-4 → 1e-4)
- Increase training steps (100k → 200k)
- Check reward scaling (normalize?)

### Problem: HRM not better than baseline
**Solutions**:
- Increase `LAMBDA_HIERARCHICAL` (0.5 → 1.0)
- Increase `weekly_target_bonus` (5.0 → 10.0)
- Ensure Phase 1 agent is well-trained first
- Check weekly target reasonableness

---

## Key Design Choices (Why This Works)

### 1. HRM-Ready from Start
**Decision**: Single codebase for both phases
**Benefit**: No redesign needed; just flip config flags

### 2. Real USDA Data
**Decision**: Use actual nutrient values per 100g
**Benefit**: Realistic, generalizable results

### 3. Gymnasium Environment
**Decision**: Standard RL interface
**Benefit**: Works with any RL library (SB3, RLlib, etc.)

### 4. Modular Reward
**Decision**: Separate low/high level reward components
**Benefit**: Easy to debug and tune independently

### 5. User Profile Templates
**Decision**: Predefined constraint sets
**Benefit**: Quick testing of different health scenarios

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Data preprocessing | `ingredients_processed.csv` |
| 2-3 | Train Phase 1 baseline | Trained agent model |
| 4 | Evaluate baseline | Metrics report |
| 5 | Implement HRM system | HRM wrapper ready |
| 6 | Train Phase 2 HRM | Hierarchical agent |
| 7 | Final evaluation | Results comparison |

---

## Next Steps

1. **Immediate**: Get USDA data and run preprocessing
2. **Day 1-4**: Focus on Phase 1 until you have 80%+ compliance
3. **Day 5+**: Add HRM only after Phase 1 works well

**Remember**: Phase 2 is an enhancement, not a replacement. Make Phase 1 solid first!

---

## Questions?

Check:
- `QUICKSTART.py` - Usage examples
- `demo.py` - Working code to test setup
- `config.py` - All tunable parameters
- Code comments - Detailed explanations

Good luck! 🚀
