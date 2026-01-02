"""
Training script for Hierarchical RL System (Phase 2)
This will be used after Phase 1 baseline is complete
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path
import config
from models.ingredient_policy import RecipeAgent
from models.hrm_policy import HierarchicalRecipeSystem, WeeklyPlannerPolicy
from eval.nutrition_metrics import RecipeEvaluator
from train.train_recipe import setup_training_environment


def load_pretrained_agent(model_path: str, user_profile: str = 'standard'):
    """
    Load pretrained low-level agent from Phase 1
    
    Args:
        model_path: Path to saved model
        user_profile: User profile
        
    Returns:
        Loaded RecipeAgent
    """
    print(f"Loading pretrained agent from {model_path}...")
    
    env = setup_training_environment(user_profile)
    agent = RecipeAgent(env, algorithm='PPO')
    agent.load(model_path)
    
    print("✓ Pretrained agent loaded")
    return agent


def train_hrm_system(
    pretrained_agent_path: str,
    user_profile: str = 'standard',
    n_weeks: int = 10
):
    """
    Train hierarchical RL system
    
    Phase 2 Strategy:
    1. Use pretrained low-level agent (frozen or fine-tuned)
    2. Train high-level policy for weekly planning
    3. Jointly optimize hierarchical rewards
    
    Args:
        pretrained_agent_path: Path to Phase 1 trained agent
        user_profile: User profile
        n_weeks: Number of training weeks
    """
    print("="*60)
    print("HIERARCHICAL RL TRAINING - PHASE 2")
    print("="*60)
    
    # Enable HRM in config
    config.HRM_ENABLED = True
    config.LAMBDA_HIERARCHICAL = 0.5  # Balance low/high level rewards
    config.REWARD_WEIGHTS['weekly_target_bonus'] = 5.0
    config.REWARD_WEIGHTS['long_term_health_stability'] = 3.0
    
    # Load pretrained agent
    low_level_agent = load_pretrained_agent(pretrained_agent_path, user_profile)
    
    # Create hierarchical system
    hrm_system = HierarchicalRecipeSystem(low_level_agent, user_profile)
    
    print(f"\nTraining HRM system for {n_weeks} weeks...")
    print("-"*60)
    
    # Training loop
    weekly_metrics = []
    
    for week in range(n_weeks):
        print(f"\n--- Week {week+1}/{n_weeks} ---")
        
        # Generate weekly plan
        weekly_plan = hrm_system.generate_weekly_plan()
        
        # Evaluate performance
        performance = hrm_system.evaluate_weekly_performance()
        weekly_metrics.append(performance)
        
        print(f"Weekly reward: {performance['weekly_reward']:.2f}")
        print(f"Calories: {performance['calories_total']:.0f} / {performance['calories_target']:.0f} "
              f"(dev: {performance['calories_deviation_pct']:.1f}%)")
        print(f"Protein: {performance['protein_total']:.0f} / {performance['protein_target']:.0f} "
              f"(dev: {performance['protein_deviation_pct']:.1f}%)")
        print(f"Sodium: {performance['sodium_total']:.0f} / {performance['sodium_target']:.0f} "
              f"(dev: {performance['sodium_deviation_pct']:.1f}%)")
        
        # TODO: Implement high-level policy gradient update here
        # For Phase 2, you would:
        # 1. Calculate policy gradient for weekly planner
        # 2. Update high-level policy parameters
        # 3. Optionally fine-tune low-level agent
    
    print("\n" + "="*60)
    print("HRM TRAINING COMPLETE")
    print("="*60)
    
    # Aggregate metrics
    print("\n--- Overall Performance ---")
    avg_weekly_reward = np.mean([m['weekly_reward'] for m in weekly_metrics])
    print(f"Average weekly reward: {avg_weekly_reward:.2f}")
    
    for nutrient in ['calories', 'protein', 'sodium']:
        avg_dev = np.mean([m[f'{nutrient}_deviation_pct'] for m in weekly_metrics])
        print(f"Average {nutrient} deviation: {avg_dev:.1f}%")
    
    return hrm_system, weekly_metrics


def evaluate_hrm_system(hrm_system: HierarchicalRecipeSystem, n_weeks: int = 5):
    """
    Evaluate trained HRM system
    
    Args:
        hrm_system: Trained HierarchicalRecipeSystem
        n_weeks: Number of evaluation weeks
    """
    print("\n" + "="*60)
    print("HRM SYSTEM EVALUATION")
    print("="*60)
    
    all_metrics = []
    
    for week in range(n_weeks):
        print(f"\nEvaluation Week {week+1}/{n_weeks}")
        
        weekly_plan = hrm_system.generate_weekly_plan()
        performance = hrm_system.evaluate_weekly_performance()
        all_metrics.append(performance)
        
        # Print daily recipes
        for day, recipe in enumerate(weekly_plan, 1):
            print(f"  Day {day}: {recipe['ingredient_count']} ingredients, "
                  f"Calories: {recipe['current_nutrients']['calories']:.0f}")
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for nutrient in ['calories', 'protein', 'sodium', 'carbs', 'fat']:
        deviations = [m[f'{nutrient}_deviation_pct'] for m in all_metrics]
        print(f"{nutrient.capitalize()} avg deviation: {np.mean(deviations):.1f}% ± {np.std(deviations):.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hierarchical RL System (Phase 2)')
    parser.add_argument('--pretrained', type=str, required=True,
                       help='Path to pretrained Phase 1 agent')
    parser.add_argument('--profile', type=str, default='standard',
                       choices=['standard', 'low_sodium', 'high_protein', 'low_carb'],
                       help='User profile')
    parser.add_argument('--weeks', type=int, default=10,
                       help='Training weeks')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Train HRM system
    hrm_system, metrics = train_hrm_system(
        pretrained_agent_path=args.pretrained,
        user_profile=args.profile,
        n_weeks=args.weeks
    )
    
    # Evaluate if requested
    if args.eval:
        evaluate_hrm_system(hrm_system, n_weeks=5)
