# Anti-Overfitting Improvements

## Problem Diagnosis

The original model exhibited **mode collapse** - a form of overfitting where the agent:
- Converged to a single recipe ("Flour, coconut")
- 100% constraint compliance but only 1% recipe diversity
- Failed to generalize to varied constraints (HRM scenario)

Root cause: **Low exploration** (entropy_coef=0.01) + early discovery of satisfactory solution

## Solutions Implemented

### 1. Increased Exploration (Entropy Coefficient)
```python
# Before
RL_CONFIG['ent_coef'] = 0.01

# After  
RL_CONFIG['ent_coef'] = 0.1  # 10x increase
```

**Impact:** Forces agent to try more diverse actions instead of exploiting known solution

### 2. Constraint Variation During Training
```python
# New configuration
VARY_CONSTRAINTS_TRAINING = True
CONSTRAINT_NOISE_STD = 0.15  # ±15% variation

# During env.reset()
noise = np.random.normal(1.0, 0.15, size=5)
target = base_target * noise[i]  # Different targets each episode
```

**Impact:** Agent sees diverse constraint scenarios during training, preventing overfitting to fixed targets

### 3. Curiosity-Driven Exploration
```python
# Track ingredient usage
self.ingredient_visit_counts = np.zeros(n_ingredients)

# Reward novel ingredients
visit_count = self.ingredient_visit_counts[action]
curiosity_reward = 1.0 / (1 + visit_count)
reward += curiosity_reward
```

**Impact:** Intrinsic motivation to explore less-visited ingredients

### 4. More Frequent Updates
```python
# Before
RL_CONFIG['batch_size'] = 64

# After
RL_CONFIG['batch_size'] = 32  # Smaller batches = more updates
```

**Impact:** Agent updates policy more frequently, reducing tendency to get stuck in local optima

## Expected Improvements

| Metric | Old Model | Expected New Model |
|--------|-----------|-------------------|
| Constraint Compliance | 100% | 90-100% (slight trade-off) |
| Recipe Diversity | 1% | 50-90% |
| Unique Ingredients | ~10 | 50-150 |
| Generalization | Poor | Good |

## Key Insights

1. **Trade-off:** Slightly lower compliance acceptable for much better diversity
2. **Efficiency over accuracy:** Model doesn't need 100% accuracy, needs good coverage
3. **Real-world readiness:** Diverse models better handle user preferences and ingredient availability
4. **Research contribution:** Demonstrates importance of training distribution matching deployment scenarios

## Files Modified

1. `config.py` - Updated hyperparameters
2. `env/recipe_env.py` - Added constraint variation and curiosity bonus
3. `scripts/retrain_anti_overfit.py` - Training script
4. `scripts/compare_models.py` - Evaluation comparison

## Usage

```bash
# Train new model (already running)
python scripts/retrain_anti_overfit.py

# Compare old vs new
python scripts/compare_models.py

# Evaluate new model
python eval/eval_phase1.py --model recipe_agent_anti_overfit_final.zip
```

## Training Progress

Training for 100k timesteps with:
- ✓ 10x entropy for exploration
- ✓ ±15% constraint variation
- ✓ Curiosity bonus for novel ingredients  
- ✓ Smaller batches for frequent updates

Current status: ~30% complete (30k/100k timesteps)

## References

- Entropy regularization: [OpenAI Spinning Up](https://spinningup.openai.com/)
- Curiosity-driven learning: "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)
- Distribution shift: Standard ML generalization theory
