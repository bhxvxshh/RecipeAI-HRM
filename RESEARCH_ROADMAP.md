# Research Publication Roadmap - RecipeAI-HRM

## Current Status Assessment

### Performance Metrics (Baseline)
- **Overall Constraint Satisfaction**: 57.2%
- **By Nutrient**:
  - Sodium: 84% ✓ (acceptable)
  - Carbs: 79% ✓ (acceptable)
  - Protein: 63.5% (borderline)
  - Fat: 36% ✗ (critical issue)
  - Calories: 23.5% ✗ (critical issue)

### Critical Issues
1. **Calorie Bias**: -46% average deviation, model underestimates consistently
2. **Fat Imbalance**: No middle ground (±30-63% deviations)
3. **Reward Variance**: -450 to -14 range (unstable learning)
4. **Evaluation Limitations**: No negative class samples, 100% precision is misleading

---

## Phase 1: Model Accuracy Improvements (Weeks 1-2)

### 1.1 Reward System Optimization
**Priority: CRITICAL**

**Current Issues:**
- Exponential penalties may be too aggressive (^2 factor)
- Weights unvalidated (Calories 30%, Fat 25%)
- Progressive thresholds too tight (10%, 20%, 30%)
- Completion bonus may cause overfitting (200 pts)

**Proposed Changes:**
```python
# config_reward_v2.py

NUTRIENT_IMPORTANCE = {
    'calories': 0.25,  # Reduce from 0.30 (too aggressive)
    'protein': 0.20,   # Keep
    'fat': 0.25,       # Keep (still critical)
    'carbs': 0.20,     # Increase from 0.15 (rebalance)
    'sodium': 0.10,    # Keep
}

# Reduce exponential factor
exponential_factor = (1 + violation_pct) ** 1.5  # Was 2.0

# Looser progressive rewards
if deviation_pct < 0.15:  # Was 0.10
    return 100 * nutrient_weight
elif deviation_pct < 0.25:  # Was 0.20
    return 50 * nutrient_weight
elif deviation_pct < 0.35:  # Was 0.30
    return 20 * nutrient_weight

# Reduce completion bonus
COMPLETION_BONUS = 150  # Was 200
```

**Expected Improvement**: Calories 35%+, Fat 45%+, Overall 65%+

### 1.2 Curriculum Learning
**Priority: HIGH**

**Implementation:**
```python
# scripts/train_curriculum.py

class CurriculumScheduler:
    """Progressive constraint tightening during training"""
    
    def __init__(self, total_timesteps):
        self.phases = [
            # Phase 1: Loose constraints (0-200k steps)
            {'range_multiplier': 2.0, 'steps': 200000},
            # Phase 2: Medium constraints (200k-400k)
            {'range_multiplier': 1.5, 'steps': 200000},
            # Phase 3: Tight constraints (400k-600k)
            {'range_multiplier': 1.2, 'steps': 200000},
            # Phase 4: Target constraints (600k+)
            {'range_multiplier': 1.0, 'steps': total_timesteps - 600000},
        ]
    
    def get_constraint_multiplier(self, current_step):
        """Returns constraint range multiplier for current training step"""
        cumulative = 0
        for phase in self.phases:
            cumulative += phase['steps']
            if current_step < cumulative:
                return phase['range_multiplier']
        return 1.0  # Final phase

# Usage in training:
# min_calories = base_min * scheduler.get_constraint_multiplier(step)
# max_calories = base_max * scheduler.get_constraint_multiplier(step)
```

**Expected Improvement**: Smoother learning, reduced reward variance ±50 (from ±200+)

### 1.3 Temperature Annealing for Exploration
**Priority: MEDIUM**

```python
# Entropy coefficient annealing
initial_entropy = 0.1
final_entropy = 0.01
annealing_steps = 300000

current_entropy = initial_entropy + (final_entropy - initial_entropy) * (step / annealing_steps)
model.ent_coef = max(current_entropy, final_entropy)
```

**Expected Improvement**: Better exploration early, exploitation late

---

## Phase 2: Evaluation Framework (Weeks 2-3)

### 2.1 Standardized Test Sets

**Create 3 Test Sets:**

1. **Easy Set** (n=100):
   - Moderate calorie targets (1800-2200 kcal)
   - Balanced macros (30/40/30 P/C/F)
   - Wide constraint ranges (±20%)

2. **Medium Set** (n=100):
   - Diverse targets (1500-2500 kcal)
   - Varied macros (25-35/30-50/20-30)
   - Normal ranges (±15%)

3. **Hard Set** (n=100):
   - Extreme targets (1200 or 3000+ kcal)
   - Restrictive (low-carb, low-fat)
   - Tight ranges (±10%)

**Metrics:**
- Constraint satisfaction rate per difficulty
- Average deviation per nutrient
- Recipe quality scores (diversity, practicality)
- Generation success rate
- Time per recipe

### 2.2 Statistical Significance Testing

```python
# scripts/statistical_analysis.py

from scipy import stats
import numpy as np

def compare_models(model_a_results, model_b_results, metric='satisfaction_rate'):
    """
    Compare two models with statistical significance testing
    
    Returns:
        - Mean difference
        - p-value (paired t-test)
        - Effect size (Cohen's d)
        - 95% confidence interval
    """
    a_scores = [r[metric] for r in model_a_results]
    b_scores = [r[metric] for r in model_b_results]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a_scores, b_scores)
    
    # Effect size
    mean_diff = np.mean(a_scores) - np.mean(b_scores)
    pooled_std = np.sqrt((np.std(a_scores)**2 + np.std(b_scores)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    # Confidence interval
    ci = stats.t.interval(0.95, len(a_scores)-1, 
                          loc=mean_diff, 
                          scale=stats.sem(np.array(a_scores) - np.array(b_scores)))
    
    return {
        'mean_difference': mean_diff,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        '95_ci': ci,
        'interpretation': interpret_effect_size(cohens_d)
    }

def interpret_effect_size(d):
    """Cohen's d interpretation"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

### 2.3 Ablation Studies

**Test Each Component Independently:**

| Model Variant | Components Removed | Expected Satisfaction |
|---------------|-------------------|---------------------|
| Full Model | None | 65% (target) |
| No Weighted Penalties | Use equal weights | ~55% |
| No Exponential Scaling | Linear penalties | ~50% |
| No Progressive Rewards | Binary only | ~52% |
| No Completion Bonus | Remove 150pt bonus | ~60% |
| No Curriculum | Fixed constraints | ~58% |
| Baseline (Original) | All improvements | 57.2% (current) |

**Statistical Requirement**: Each variant p < 0.05 vs baseline

---

## Phase 3: Advanced Techniques (Weeks 3-4)

### 3.1 Multi-Stage Training

```python
# Stage 1: Exploration (200k steps)
# - High entropy (0.15)
# - Loose constraints (2x range)
# - Diversity bonus 2x

# Stage 2: Refinement (300k steps)  
# - Medium entropy (0.05)
# - Normal constraints
# - Balanced rewards

# Stage 3: Fine-tuning (200k steps)
# - Low entropy (0.01)
# - Tight constraints (0.9x range)
# - Accuracy focus
```

### 3.2 Constraint-Specific Networks

**Architecture Enhancement:**
```python
class MultiHeadConstraintCritic(nn.Module):
    """Separate value heads for each nutrient constraint"""
    
    def __init__(self, features_dim, n_constraints=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Separate heads for each constraint
        self.calorie_head = nn.Linear(256, 1)
        self.protein_head = nn.Linear(256, 1)
        self.fat_head = nn.Linear(256, 1)
        self.carbs_head = nn.Linear(256, 1)
        self.sodium_head = nn.Linear(256, 1)
    
    def forward(self, x):
        shared_features = self.shared(x)
        return {
            'calories': self.calorie_head(shared_features),
            'protein': self.protein_head(shared_features),
            'fat': self.fat_head(shared_features),
            'carbs': self.carbs_head(shared_features),
            'sodium': self.sodium_head(shared_features),
        }
```

**Expected Improvement**: Better constraint-specific learning, +5-8% overall

### 3.3 Data Augmentation

**Synthetic Constraint Generation:**
```python
def augment_constraints(base_constraints, n_augmented=10):
    """
    Generate varied constraint sets from base constraints
    - Noise injection (±5% per nutrient)
    - Ratio shifts (exchange protein/carbs)
    - Temporal variations (meal timing)
    """
    augmented = []
    for _ in range(n_augmented):
        new_constraints = base_constraints.copy()
        
        # Add Gaussian noise
        for nutrient in ['calories', 'protein', 'fat', 'carbs']:
            noise = np.random.normal(0, 0.05)  # 5% std
            new_constraints[nutrient] *= (1 + noise)
        
        # Occasionally swap macros (protein ↔ carbs)
        if np.random.rand() < 0.3:
            new_constraints['protein'], new_constraints['carbs'] = \
                new_constraints['carbs'], new_constraints['protein']
        
        augmented.append(new_constraints)
    
    return augmented
```

---

## Phase 4: Human Evaluation (Week 4)

### 4.1 Expert Rating Protocol

**Recipe Quality Dimensions** (1-5 Likert scale):
1. **Nutritional Accuracy**: "Does this recipe meet the stated constraints?"
2. **Practicality**: "Is this a realistic, cookable recipe?"
3. **Taste Potential**: "Would this combination taste good?"
4. **Ingredient Harmony**: "Do these ingredients work together?"
5. **Dietary Appropriateness**: "Is this suitable for stated health goals?"

**Raters**: 3 nutritionists + 2 chefs (inter-rater reliability: Cronbach's α > 0.7)

**Sample Size**: 50 recipes per model × 3 difficulty levels = 150 total ratings

### 4.2 User Study Design

**A/B Testing:**
- Group A: Baseline model recipes (n=30 users)
- Group B: Improved model recipes (n=30 users)
- Group C: Human-created recipes (gold standard, n=30)

**Metrics:**
- Satisfaction score (1-10)
- Intent to cook (Yes/No)
- Perceived healthiness (1-10)
- Taste expectation (1-10)

**Analysis**: ANOVA + post-hoc Tukey HSD

---

## Phase 5: Baseline Comparisons (Week 5)

### 5.1 Comparison Models

| Model | Type | Description |
|-------|------|-------------|
| **Random** | Baseline | Random ingredient selection |
| **Greedy** | Heuristic | Select by calorie density until target |
| **Rule-Based** | Expert System | Predefined recipe templates |
| **Recipe Retrieval** | IR | Nearest neighbor from database |
| **GPT-4** | LLM | Prompted for recipe generation |
| **RecipeAI (Ours)** | RL | PPO with improved rewards |

### 5.2 Benchmark Results Table

```markdown
| Model | Constraint Satisfaction | Diversity | Time (s) | Human Rating |
|-------|----------------------|-----------|----------|--------------|
| Random | 12.5% ± 3.2 | 0.95 | 0.01 | 2.1 ± 0.5 |
| Greedy | 45.2% ± 5.1 | 0.32 | 0.05 | 3.4 ± 0.6 |
| Rule-Based | 58.3% ± 4.8 | 0.51 | 0.10 | 4.1 ± 0.5 |
| Recipe Retrieval | 62.1% ± 6.2 | 0.88 | 2.50 | 4.5 ± 0.4 |
| GPT-4 | 55.8% ± 7.5 | 0.91 | 3.20 | 4.7 ± 0.5 |
| **RecipeAI (Ours)** | **67.5% ± 4.1** | **0.86** | **0.11** | **4.6 ± 0.4** |
```

**Note**: Target values after all improvements

---

## Phase 6: Publication Preparation (Week 6)

### 6.1 Paper Structure

**Title**: "Hierarchical Reinforcement Learning for Constraint-Satisfying Recipe Generation with Real-World Nutritional Data"

**Sections**:

1. **Abstract** (250 words)
   - Problem: Personalized nutrition automation
   - Method: PPO with curriculum learning + weighted rewards
   - Results: 67.5% satisfaction (baseline 57.2%, p<0.001)
   - Impact: First RL system integrating real diabetic patient data

2. **Introduction**
   - Motivation: Obesity epidemic, diabetic care costs
   - Challenge: Multi-constraint optimization + taste
   - Contribution: Novel reward shaping + curriculum

3. **Related Work**
   - Recipe generation (neural, template-based)
   - Constrained RL (safe RL, Lagrangian methods)
   - Nutritional AI (diet planning, meal recommendation)

4. **Method**
   - Environment design (RecipeEnv)
   - Reward engineering (weighted + exponential + progressive)
   - Curriculum learning schedule
   - Training procedure (PPO, 700k steps, GPU)

5. **Experiments**
   - Datasets (USDA, UCI diabetes, 301k recipes)
   - Evaluation protocol (3 test sets, statistical tests)
   - Ablation studies (7 variants)
   - Human evaluation (5 raters, 150 recipes)
   - Baseline comparisons (6 models)

6. **Results**
   - Constraint satisfaction improvements
   - Nutrient-wise breakdown (calorie bias fixed)
   - Statistical significance (all p<0.01)
   - Human ratings comparable to GPT-4
   - Ablation confirms all components essential

7. **Discussion**
   - Calorie bias analysis (why underestimation occurred)
   - Generalization to other dietary needs
   - Limitations (ingredient set fixed, no cooking methods)
   - Future: Hierarchical weekly planning (HRM Phase 2)

8. **Conclusion**
   - First medical-data-driven RL recipe system
   - Practical accuracy (67.5% satisfaction)
   - Open-source contribution

### 6.2 Key Figures

**Figure 1**: System architecture diagram
**Figure 2**: Constraint satisfaction by nutrient (before/after)
**Figure 3**: Learning curves (3 stages curriculum)
**Figure 4**: Ablation study results (bar chart)
**Figure 5**: Human evaluation comparison (box plots)
**Figure 6**: Qualitative examples (3 recipes with analysis)

### 6.3 Supplementary Materials

- Full hyperparameters table
- Extended ablation results
- Recipe examples (50 generated, annotated)
- Code repository (GitHub)
- Datasets and preprocessing scripts

---

## Target Venues

### Top-Tier Conferences
1. **NeurIPS** (Neural Information Processing Systems)
   - Workshop: ML for Health (ML4H)
   - Deadline: May/September

2. **AAAI** (Association for Advancement of AI)
   - Main track or Special Track on AI for Social Impact
   - Deadline: August

3. **IJCAI** (International Joint Conference on AI)
   - AI and Healthcare track
   - Deadline: January

### Domain Conferences
4. **AAMAS** (Autonomous Agents and Multi-Agent Systems)
   - RL applications track
   - Deadline: February

5. **ACM RecSys** (Recommender Systems)
   - Health and wellness track
   - Deadline: May

### Journals (if expanding)
6. **JMLR** (Journal of Machine Learning Research)
7. **Artificial Intelligence in Medicine**
8. **IEEE Transactions on Neural Networks and Learning Systems**

---

## Success Criteria for Publication

### Minimum Requirements
- ✅ Overall satisfaction: **≥65%** (currently 57.2%)
- ✅ Calorie satisfaction: **≥40%** (currently 23.5%)
- ✅ Fat satisfaction: **≥45%** (currently 36%)
- ✅ Statistical significance: **p < 0.01** vs baseline
- ✅ Effect size: **Cohen's d > 0.5** (medium-large)
- ✅ Ablation studies: All components significant
- ✅ Human ratings: **≥4.0/5.0** (comparable to GPT-4)
- ✅ Baselines beaten: 5/6 models outperformed

### Competitive Targets
- 🎯 Overall satisfaction: **70%+** (strong paper)
- 🎯 Calorie satisfaction: **50%+**
- 🎯 Human ratings: **≥4.5/5.0** (excellent)
- 🎯 Generation speed: **<0.2s** (real-time)
- 🎯 Diversity: **>0.80** ingredient coverage

---

## Timeline Summary

| Week | Phase | Deliverables | Estimated Hours |
|------|-------|--------------|-----------------|
| 1-2 | Model Improvements | Re-tuned rewards, curriculum training | 40h |
| 2-3 | Evaluation Framework | Test sets, statistical tests, ablations | 30h |
| 3-4 | Advanced Techniques | Multi-head network, augmentation | 35h |
| 4 | Human Evaluation | Expert ratings, user study | 25h |
| 5 | Baselines | Implement 6 comparison models | 30h |
| 6 | Publication | Write paper, prepare figures | 40h |
| **Total** | | | **~200 hours** |

---

## Immediate Next Steps (Priority Order)

### This Week:
1. ✅ **Implement config_reward_v2.py** (2h)
   - Adjust exponential factor: 2.0 → 1.5
   - Loosen progressive thresholds: 10%/20%/30% → 15%/25%/35%
   - Rebalance weights: Calories 0.25, Carbs 0.20

2. ✅ **Create train_curriculum.py** (3h)
   - 4-phase constraint tightening
   - Entropy annealing: 0.1 → 0.01
   - 700k total timesteps

3. ✅ **Train improved model** (12h compute)
   - Expected reward: -20 to +80 (vs current -1685)
   - Expected satisfaction: 65%+ overall

4. ✅ **Evaluate on standardized test set** (2h)
   - Generate 100 recipes (easy/medium/hard)
   - Measure per-nutrient satisfaction
   - Compare to baseline with t-test

### Next Week:
5. Implement ablation studies
6. Create baseline comparison models
7. Begin human evaluation protocol

---

## Questions for Discussion

1. **Target Venue**: Which conference/journal should we aim for?
   - NeurIPS ML4H (high impact, competitive)
   - AAAI (broader audience)
   - Domain journal (more space for details)

2. **Human Evaluation**: Can you recruit nutritionists/chefs?
   - Alternative: Crowdsourcing (MTurk, Prolific)

3. **Compute Budget**: GPU time for full experiments?
   - Need: ~50-70 GPU hours total
   - Can use free Colab/Kaggle or institutional cluster

4. **Timeline**: Aiming for which deadline?
   - Next NeurIPS: May 2026 (4 months)
   - AAAI 2027: August 2026 (7 months)

Should I proceed with implementing config_reward_v2.py and train_curriculum.py?
