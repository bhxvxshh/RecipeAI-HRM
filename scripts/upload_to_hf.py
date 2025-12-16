"""
Upload RecipeAI model to Hugging Face Hub
"""
import os
from pathlib import Path

def upload_to_huggingface():
    """
    Upload the trained model to Hugging Face Hub
    """
    print("="*70)
    print("HUGGING FACE MODEL UPLOAD")
    print("="*70)
    
    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import HfApi, create_repo, upload_file
        print("✓ huggingface_hub library found")
    except ImportError:
        print("\n❌ huggingface_hub not installed")
        print("\nInstall with:")
        print("  pip install huggingface_hub")
        return
    
    # Check if model exists
    model_path = "/home/bhavesh/MajorB/RecipeAI/models/saved/recipe_agent_anti_overfit_final.zip"
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        return
    
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✓ Model found (anti-overfitting version): {model_size:.2f} MB")
    
    # Get user credentials
    print("\n" + "="*70)
    print("AUTHENTICATION")
    print("="*70)
    print("\nYou need a Hugging Face account token.")
    print("Get it from: https://huggingface.co/settings/tokens")
    print("\nOptions:")
    print("  1. Login via CLI: huggingface-cli login")
    print("  2. Set environment variable: export HF_TOKEN=your_token")
    print("  3. Enter token when prompted below")
    
    token = input("\nEnter your HF token (or press Enter if already logged in): ").strip()
    
    # Initialize API
    api = HfApi(token=token if token else None)
    
    # Get repository name
    print("\n" + "="*70)
    print("REPOSITORY SETUP")
    print("="*70)
    username = input("Enter your HF username: ").strip()
    
    default_repo = f"{username}/recipe-ai-hrm"
    repo_name = input(f"Enter repository name (default: {default_repo}): ").strip()
    if not repo_name:
        repo_name = default_repo
    
    # Create repository
    print(f"\n📦 Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            token=token if token else None,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✓ Repository created/verified: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return
    
    # Create model card
    model_card = f"""---
language: en
tags:
- reinforcement-learning
- recipe-generation
- nutrition
- hierarchical-rl
- stable-baselines3
- ppo
- anti-overfitting
license: mit
---

# RecipeAI: Anti-Overfitting Recipe Generation with RL

## Model Description

This is an **improved** PPO agent for nutritionally-balanced recipe generation, trained with anti-overfitting techniques. Part of a hierarchical reinforcement learning system (HRM).

**v2.0 - Anti-Overfitting Version**:
- Algorithm: Proximal Policy Optimization (PPO)
- Training: 100,000 timesteps
- Framework: Stable Baselines3
- Environment: Custom Gymnasium env (RecipeEnv)
- **Key Improvement**: 10x higher entropy coefficient, constraint variation, curiosity bonus

## What's New in v2.0

### Anti-Overfitting Improvements
1. **10x More Exploration** (entropy 0.01 → 0.1)
2. **Constraint Variation** (±15% noise during training)
3. **Curiosity Bonus** (rewards exploring new ingredients)
4. **Faster Updates** (batch size 64 → 32)

### Results vs v1.0
- ✅ **Mode Collapse Fixed**: Old model used coconut flour in 60% of recipes → New model max 25%
- ✅ **Better Exploration**: 122 → 130 unique ingredients (+6.6%)
- ✅ **Improved Generalization**: 10% → 30% constraint compliance (3x improvement!)
- ✅ **Healthier Recipes**: Switched from high-fat ingredients to vegetables, fruits, low-fat dairy

## Performance

### Model Comparison (Stochastic Sampling)

| Metric | v1.0 (Overfitted) | v2.0 (Anti-Overfit) | Change |
|--------|-------------------|---------------------|---------|
| Constraint Compliance | 10% | 30% | +200% ✅ |
| Recipe Diversity | 100% | 100% | Maintained ✅ |
| Unique Ingredients | 122/324 | 130/324 | +8 ✅ |
| Mode Collapse | Yes (60%) | Fixed (25% max) | ✅ |

### Ingredient Distribution
**v1.0**: Dominated by "Flour, coconut" (60% of recipes)
**v2.0**: Balanced distribution with healthy ingredients:
- Vegetables: lettuce, tomatoes, onions, peppers, cabbage (15-20% each)
- Fruits: nectarines, bananas, pineapple (15% each)
- Dairy: low-fat milk 1% (25%)

### Phase 2 (HRM Integration)
- **Weekly Target Achievement**: 4/5 nutrients within 10%
- **Recipe Diversity**: 100% unique recipes
- **Unique Ingredients**: 55/324 used

## Training Details

- **State Space**: 11-dimensional
  - Current nutrients (5): calories, protein, sodium, carbs, fat
  - Target nutrients (5) - **varied during training**
  - Ingredient count (1)

- **Action Space**: 325 discrete actions
  - 324 ingredients (with curiosity bonus)
  - 1 DONE action

- **Constraints** (per meal, with ±15% variation):
  - Calories: 400-800 kcal
  - Protein: 15-50g
  - Sodium: 0-800mg
  - Carbs: 30-100g
  - Fat: 10-30g

- **Dataset**: USDA FoodData Central (324 processed ingredients)

- **Hyperparameters** (Anti-Overfitting):
  - Entropy coefficient: 0.1 (10x higher)
  - Batch size: 32 (2x more updates)
  - Constraint noise: 15% std
  - Curiosity bonus: 1.0

## Usage

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Load model
model = PPO.load("recipe_agent_anti_overfit_final")

# Generate recipe (requires RecipeEnv - see repository)
# For best results, use stochastic sampling or best-of-N
# obs, info = env.reset()
# action, _states = model.predict(obs, deterministic=False)  # Stochastic!
```

## Repository

Full code: [RecipeAI GitHub Repository]

## Citation

```bibtex
@software{{recipeai2024,
  title={{RecipeAI: Hierarchical Reinforcement Learning for Recipe Generation}},
  author={{Your Name}},
  year={{2024}},
  url={{https://huggingface.co/{repo_name}}}
}}
```

## License

MIT License
"""
    
    # Save model card
    card_path = "/tmp/README.md"
    with open(card_path, "w") as f:
        f.write(model_card)
    
    # Upload files
    print("\n" + "="*70)
    print("UPLOADING FILES")
    print("="*70)
    
    try:
        # Upload model
        print(f"\n📤 Uploading model ({model_size:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="recipe_agent_standard_ppo.zip",
            repo_id=repo_name,
            token=token if token else None,
        )
        print("✓ Model uploaded")
        
        # Upload model card
        print("\n📤 Uploading README...")
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token if token else None,
        )
        print("✓ README uploaded")
        
        # Upload config
        print("\n📤 Uploading config...")
        api.upload_file(
            path_or_fileobj="/home/bhavesh/MajorB/RecipeAI/config.py",
            path_in_repo="config.py",
            repo_id=repo_name,
            token=token if token else None,
        )
        print("✓ Config uploaded")
        
        print("\n" + "="*70)
        print("✓✓ UPLOAD COMPLETE")
        print("="*70)
        print(f"\n🎉 Model available at: https://huggingface.co/{repo_name}")
        print(f"\n📥 Your friend can download with:")
        print(f"   from huggingface_hub import hf_hub_download")
        print(f"   model_path = hf_hub_download(repo_id='{repo_name}', filename='recipe_agent_standard_ppo.zip')")
        print("\n   Or directly load:")
        print(f"   from stable_baselines3 import PPO")
        print(f"   model = PPO.load(hf_hub_download('{repo_name}', 'recipe_agent_standard_ppo.zip'))")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify token has write access")
        print("  2. Check internet connection")
        print("  3. Try: huggingface-cli login")


if __name__ == "__main__":
    upload_to_huggingface()
