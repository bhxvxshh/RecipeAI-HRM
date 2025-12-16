"""
Evaluation metrics for Recipe Generation System
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import config


class RecipeEvaluator:
    """
    Evaluates recipe generation performance
    """
    
    def __init__(self, constraints: Dict):
        """
        Initialize evaluator
        
        Args:
            constraints: Nutrient constraints dict
        """
        self.constraints = constraints
        self.nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
    
    def evaluate_recipe(self, recipe_info: Dict) -> Dict:
        """
        Evaluate a single recipe
        
        Args:
            recipe_info: Recipe info from environment
            
        Returns:
            Evaluation metrics dict
        """
        metrics = {}
        
        # 1. Constraint compliance
        nutrients = recipe_info['current_nutrients']
        compliance = recipe_info['compliance']
        
        metrics['all_constraints_met'] = all(compliance.values())
        metrics['constraints_met_count'] = sum(compliance.values())
        metrics['constraint_compliance_rate'] = sum(compliance.values()) / len(compliance)
        
        # 2. Nutrient balance score (how close to targets)
        balance_scores = []
        for nutrient in self.nutrient_cols:
            value = nutrients[nutrient]
            target = self.constraints[nutrient]['target']
            deviation = abs(value - target) / (target + 1e-8)
            balance_score = max(0, 1 - deviation)
            balance_scores.append(balance_score)
        
        metrics['nutrient_balance_score'] = np.mean(balance_scores)
        
        # 3. Ingredient count
        metrics['ingredient_count'] = recipe_info['ingredient_count']
        
        # 4. Nutrient details
        for nutrient in self.nutrient_cols:
            metrics[f'{nutrient}_value'] = nutrients[nutrient]
            metrics[f'{nutrient}_target'] = self.constraints[nutrient]['target']
            metrics[f'{nutrient}_deviation_pct'] = (
                abs(nutrients[nutrient] - self.constraints[nutrient]['target']) / 
                (self.constraints[nutrient]['target'] + 1e-8) * 100
            )
        
        return metrics
    
    def evaluate_batch(self, recipes: List[Dict]) -> Dict:
        """
        Evaluate a batch of recipes
        
        Args:
            recipes: List of recipe_info dicts
            
        Returns:
            Aggregated metrics
        """
        individual_metrics = [self.evaluate_recipe(recipe) for recipe in recipes]
        
        aggregated = {}
        
        # Average metrics
        for key in individual_metrics[0].keys():
            if isinstance(individual_metrics[0][key], (int, float, bool)):
                values = [m[key] for m in individual_metrics]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        # Recipe diversity (unique ingredient usage)
        all_ingredients = []
        for recipe in recipes:
            all_ingredients.extend(recipe['recipe'])
        
        unique_ingredients = len(set(all_ingredients))
        total_ingredient_uses = len(all_ingredients)
        
        aggregated['unique_ingredients_used'] = unique_ingredients
        aggregated['total_ingredient_uses'] = total_ingredient_uses
        aggregated['recipe_diversity_score'] = unique_ingredients / (total_ingredient_uses + 1e-8)
        
        return aggregated
    
    def print_evaluation(self, metrics: Dict):
        """
        Pretty print evaluation metrics
        
        Args:
            metrics: Metrics dict
        """
        print("\n" + "="*60)
        print("RECIPE EVALUATION RESULTS")
        print("="*60)
        
        print("\n--- Constraint Compliance ---")
        print(f"Constraint Compliance Rate: {metrics.get('constraint_compliance_rate_mean', 0)*100:.1f}%")
        print(f"All Constraints Met Rate: {metrics.get('all_constraints_met_mean', 0)*100:.1f}%")
        
        print("\n--- Nutrient Balance ---")
        print(f"Overall Balance Score: {metrics.get('nutrient_balance_score_mean', 0)*100:.1f}%")
        
        nutrient_cols = ['calories', 'protein', 'sodium', 'carbs', 'fat']
        for nutrient in nutrient_cols:
            if f'{nutrient}_deviation_pct_mean' in metrics:
                print(f"  {nutrient.capitalize()} deviation: {metrics[f'{nutrient}_deviation_pct_mean']:.1f}%")
        
        print("\n--- Recipe Characteristics ---")
        print(f"Avg ingredients per recipe: {metrics.get('ingredient_count_mean', 0):.1f}")
        print(f"Recipe diversity score: {metrics.get('recipe_diversity_score', 0)*100:.1f}%")
        print(f"Unique ingredients used: {metrics.get('unique_ingredients_used', 0)}")
        
        print("\n" + "="*60)


def evaluate_trained_agent(agent, n_recipes: int = 100, constraints: Dict = None) -> Dict:
    """
    Evaluate a trained agent by generating recipes
    
    Args:
        agent: Trained RecipeAgent
        n_recipes: Number of recipes to generate
        constraints: Optional constraint override
        
    Returns:
        Evaluation metrics dict
    """
    print(f"Evaluating agent on {n_recipes} recipes...")
    
    recipes = []
    for i in range(n_recipes):
        recipe_info = agent.generate_recipe(constraints=constraints, render=False)
        recipes.append(recipe_info)
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{n_recipes} recipes")
    
    # Get constraints from first recipe if not provided
    if constraints is None:
        constraints = recipes[0]['constraints']
    
    # Evaluate
    evaluator = RecipeEvaluator(constraints)
    metrics = evaluator.evaluate_batch(recipes)
    evaluator.print_evaluation(metrics)
    
    return metrics


def save_evaluation_results(metrics: Dict, save_path: str):
    """
    Save evaluation results to file
    
    Args:
        metrics: Evaluation metrics dict
        save_path: Path to save results
    """
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
    print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    print("Recipe Evaluation Module")
    print("Use this module to evaluate trained agents")
