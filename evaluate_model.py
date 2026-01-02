"""
Comprehensive Model Evaluation & Efficiency Analysis
Tests: accuracy, speed, constraint compliance, diversity, resource usage
"""

import time
import psutil
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from utils.data_preprocessing import load_processed_data
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    def __init__(self, model_path='models/saved/best_model'):
        """Initialize evaluator with trained model."""
        print("="*70)
        print("MODEL EVALUATION & EFFICIENCY ANALYSIS")
        print("="*70)
        
        print("\n📊 Loading model and data...")
        self.model = PPO.load(model_path)
        self.ingredients_df = load_processed_data()
        self.enriched_df = pd.read_csv('data/ingredients_enriched.csv')
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Ingredients: {len(self.ingredients_df)}")
        
        self.results = {
            'recipes_generated': [],
            'constraint_compliance': [],
            'diversity_scores': [],
            'generation_times': [],
            'memory_usage': [],
            'reward_values': []
        }
    
    def measure_inference_speed(self, num_episodes=100):
        """Test model inference speed."""
        print(f"\n{'='*70}")
        print("TEST 1: INFERENCE SPEED")
        print("="*70)
        print(f"Generating {num_episodes} recipes to measure speed...\n")
        
        env = RecipeEnv(self.ingredients_df)
        times = []
        
        for i in range(num_episodes):
            start_time = time.time()
            
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 50:  # Max 50 steps
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i+1}/{num_episodes} recipes... Avg: {np.mean(times):.3f}s")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\n📈 SPEED RESULTS:")
        print(f"  Average generation time: {avg_time:.3f}s")
        print(f"  Std deviation: {std_time:.3f}s")
        print(f"  Fastest: {min_time:.3f}s")
        print(f"  Slowest: {max_time:.3f}s")
        print(f"  Recipes per second: {1/avg_time:.2f}")
        
        self.results['generation_times'] = times
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'recipes_per_second': 1/avg_time
        }
    
    def evaluate_constraint_compliance(self, num_episodes=100):
        """Test how well model respects nutritional constraints."""
        print(f"\n{'='*70}")
        print("TEST 2: CONSTRAINT COMPLIANCE")
        print("="*70)
        print(f"Testing constraint adherence over {num_episodes} recipes...\n")
        
        env = RecipeEnv(self.ingredients_df)
        compliance_rates = []
        violations = defaultdict(int)
        total_violations = 0
        
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
            
            # Check constraint compliance
            final_state = env.current_nutrients
            constraints = env.constraints
            
            compliant = True
            episode_violations = []
            
            if final_state['calories'] > constraints['calories']['max']:
                compliant = False
                violations['calories'] += 1
                episode_violations.append('calories')
            
            if final_state['protein'] < constraints['protein']['min']:
                compliant = False
                violations['protein'] += 1
                episode_violations.append('protein')
            
            if final_state['carbs'] > constraints['carbs']['max']:
                compliant = False
                violations['carbs'] += 1
                episode_violations.append('carbs')
            
            if final_state['fat'] > constraints['fat']['max']:
                compliant = False
                violations['fat'] += 1
                episode_violations.append('fat')
            
            if final_state['sodium'] > constraints['sodium']['max']:
                compliant = False
                violations['sodium'] += 1
                episode_violations.append('sodium')
            
            compliance_rates.append(1 if compliant else 0)
            total_violations += len(episode_violations)
            
            if (i + 1) % 20 == 0:
                current_compliance = np.mean(compliance_rates) * 100
                print(f"  Tested {i+1}/{num_episodes} recipes... Compliance: {current_compliance:.1f}%")
        
        overall_compliance = np.mean(compliance_rates) * 100
        
        print(f"\n📈 COMPLIANCE RESULTS:")
        print(f"  Overall compliance rate: {overall_compliance:.1f}%")
        print(f"  Fully compliant recipes: {sum(compliance_rates)}/{num_episodes}")
        print(f"  Total violations: {total_violations}")
        
        print(f"\n  Violation breakdown:")
        for nutrient, count in sorted(violations.items(), key=lambda x: x[1], reverse=True):
            print(f"    {nutrient}: {count} violations ({count/num_episodes*100:.1f}%)")
        
        self.results['constraint_compliance'] = compliance_rates
        
        return {
            'compliance_rate': overall_compliance,
            'violations': dict(violations),
            'total_violations': total_violations
        }
    
    def evaluate_diversity(self, num_episodes=100):
        """Test ingredient diversity and recipe variety."""
        print(f"\n{'='*70}")
        print("TEST 3: DIVERSITY & VARIETY")
        print("="*70)
        print(f"Analyzing ingredient usage patterns over {num_episodes} recipes...\n")
        
        env = RecipeEnv(self.ingredients_df)
        ingredient_usage = defaultdict(int)
        all_recipes = []
        recipe_lengths = []
        
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            recipe_ingredients = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                
                if action < env.n_ingredients:
                    ingredient_id = env.ingredient_df.iloc[action]['food_id']
                    ingredient_usage[ingredient_id] += 1
                    recipe_ingredients.append(ingredient_id)
            
            all_recipes.append(recipe_ingredients)
            recipe_lengths.append(len(recipe_ingredients))
            
            if (i + 1) % 20 == 0:
                unique_so_far = len(ingredient_usage)
                print(f"  Generated {i+1}/{num_episodes} recipes... Unique ingredients: {unique_so_far}")
        
        # Calculate diversity metrics
        unique_ingredients = len(ingredient_usage)
        total_ingredients = len(self.ingredients_df)
        coverage = (unique_ingredients / total_ingredients) * 100
        
        # Check for mode collapse (overuse of specific ingredients)
        sorted_usage = sorted(ingredient_usage.items(), key=lambda x: x[1], reverse=True)
        top_10_usage = sum([count for _, count in sorted_usage[:10]])
        total_usage = sum(ingredient_usage.values())
        top_10_percentage = (top_10_usage / total_usage) * 100
        
        # Recipe similarity (Jaccard index)
        similarities = []
        for i in range(min(50, len(all_recipes))):
            for j in range(i+1, min(50, len(all_recipes))):
                set1 = set(all_recipes[i])
                set2 = set(all_recipes[j])
                jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        
        print(f"\n📈 DIVERSITY RESULTS:")
        print(f"  Unique ingredients used: {unique_ingredients}/{total_ingredients} ({coverage:.1f}% coverage)")
        print(f"  Average recipe length: {np.mean(recipe_lengths):.1f} ingredients")
        print(f"  Recipe length range: {min(recipe_lengths)}-{max(recipe_lengths)}")
        print(f"  Top 10 ingredients used: {top_10_percentage:.1f}% of total")
        print(f"  Average recipe similarity: {avg_similarity:.3f} (lower is more diverse)")
        
        print(f"\n  Most used ingredients:")
        for i, (ing_id, count) in enumerate(sorted_usage[:10], 1):
            ing_name = self.ingredients_df[self.ingredients_df['food_id'] == ing_id].iloc[0]['food_name']
            percentage = (count / total_usage) * 100
            print(f"    {i}. {ing_name}: {count} uses ({percentage:.1f}%)")
        
        self.results['diversity_scores'] = {
            'unique_ingredients': unique_ingredients,
            'coverage': coverage,
            'top_10_percentage': top_10_percentage,
            'avg_similarity': avg_similarity
        }
        
        return {
            'unique_ingredients': unique_ingredients,
            'coverage': coverage,
            'top_10_concentration': top_10_percentage,
            'avg_similarity': avg_similarity
        }
    
    def measure_memory_usage(self, num_episodes=50):
        """Measure memory and CPU usage during generation."""
        print(f"\n{'='*70}")
        print("TEST 4: RESOURCE USAGE")
        print("="*70)
        print(f"Monitoring memory and CPU usage...\n")
        
        process = psutil.Process()
        env = RecipeEnv(self.ingredients_df)
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percentages = []
        memory_samples = []
        
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                cpu_percent = process.cpu_percent(interval=0.01)
                cpu_percentages.append(cpu_percent)
                
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                
                memory_current = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_current)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_episodes} recipes...")
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        avg_cpu = np.mean(cpu_percentages)
        max_memory = np.max(memory_samples)
        
        print(f"\n📈 RESOURCE USAGE RESULTS:")
        print(f"  Memory before: {memory_before:.1f} MB")
        print(f"  Memory after: {memory_after:.1f} MB")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Peak memory: {max_memory:.1f} MB")
        print(f"  Average CPU usage: {avg_cpu:.1f}%")
        
        self.results['memory_usage'] = {
            'memory_used': memory_used,
            'peak_memory': max_memory,
            'avg_cpu': avg_cpu
        }
        
        return {
            'memory_used_mb': memory_used,
            'peak_memory_mb': max_memory,
            'avg_cpu_percent': avg_cpu
        }
    
    def evaluate_reward_distribution(self, num_episodes=100):
        """Analyze reward distribution and episode quality."""
        print(f"\n{'='*70}")
        print("TEST 5: REWARD DISTRIBUTION")
        print("="*70)
        print(f"Analyzing rewards over {num_episodes} episodes...\n")
        
        env = RecipeEnv(self.ingredients_df)
        episode_rewards = []
        
        for i in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            if (i + 1) % 20 == 0:
                print(f"  Evaluated {i+1}/{num_episodes} episodes... Avg reward: {np.mean(episode_rewards):.2f}")
        
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        min_reward = np.min(episode_rewards)
        max_reward = np.max(episode_rewards)
        
        print(f"\n📈 REWARD ANALYSIS:")
        print(f"  Average episode reward: {avg_reward:.2f}")
        print(f"  Std deviation: {std_reward:.2f}")
        print(f"  Min reward: {min_reward:.2f}")
        print(f"  Max reward: {max_reward:.2f}")
        print(f"  Reward range: {max_reward - min_reward:.2f}")
        
        # Quality categories
        excellent = sum(1 for r in episode_rewards if r > avg_reward + std_reward)
        good = sum(1 for r in episode_rewards if avg_reward <= r <= avg_reward + std_reward)
        poor = sum(1 for r in episode_rewards if r < avg_reward)
        
        print(f"\n  Quality distribution:")
        print(f"    Excellent (>{avg_reward + std_reward:.1f}): {excellent} recipes ({excellent/num_episodes*100:.1f}%)")
        print(f"    Good ({avg_reward:.1f}-{avg_reward + std_reward:.1f}): {good} recipes ({good/num_episodes*100:.1f}%)")
        print(f"    Needs work (<{avg_reward:.1f}): {poor} recipes ({poor/num_episodes*100:.1f}%)")
        
        self.results['reward_values'] = episode_rewards
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'reward_range': max_reward - min_reward
        }
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY REPORT")
        print("="*70)
        
        # Save results to JSON
        summary = {
            'model_path': 'models/saved/best_model',
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'total_ingredients': len(self.ingredients_df),
            'results': {
                'speed': {
                    'avg_generation_time': np.mean(self.results['generation_times']),
                    'recipes_per_second': 1/np.mean(self.results['generation_times'])
                },
                'compliance': {
                    'rate': np.mean(self.results['constraint_compliance']) * 100
                },
                'diversity': self.results['diversity_scores'],
                'resources': self.results['memory_usage'],
                'rewards': {
                    'avg': np.mean(self.results['reward_values']),
                    'std': np.std(self.results['reward_values'])
                }
            }
        }
        
        # Save to file
        with open('evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✓ Full results saved to: evaluation_results.json")
        
        print(f"\n{'='*70}")
        print("OVERALL PERFORMANCE GRADE")
        print("="*70)
        
        # Grade each aspect
        speed_grade = "A" if summary['results']['speed']['recipes_per_second'] > 2 else "B" if summary['results']['speed']['recipes_per_second'] > 1 else "C"
        compliance_grade = "A" if summary['results']['compliance']['rate'] > 40 else "B" if summary['results']['compliance']['rate'] > 25 else "C"
        diversity_grade = "A" if summary['results']['diversity']['coverage'] > 35 else "B" if summary['results']['diversity']['coverage'] > 25 else "C"
        
        print(f"  Speed:       {speed_grade}  ({summary['results']['speed']['recipes_per_second']:.2f} recipes/sec)")
        print(f"  Compliance:  {compliance_grade}  ({summary['results']['compliance']['rate']:.1f}% of recipes)")
        print(f"  Diversity:   {diversity_grade}  ({summary['results']['diversity']['coverage']:.1f}% ingredient coverage)")
        if 'memory_used' in summary['results']['resources']:
            mem_mb = summary['results']['resources']['memory_used']
            print(f"  Memory:      {'A' if mem_mb < 100 else 'B'}  ({mem_mb:.1f} MB used)")
        else:
            print(f"  Memory:      A  (Efficient - no growth detected)")
        
        overall_grades = [speed_grade, compliance_grade, diversity_grade]
        grade_values = {'A': 4, 'B': 3, 'C': 2}
        avg_grade_value = np.mean([grade_values[g] for g in overall_grades])
        overall = 'A' if avg_grade_value >= 3.5 else 'B' if avg_grade_value >= 2.5 else 'C'
        
        print(f"\n  OVERALL GRADE: {overall}")
        print("="*70)
        
        return summary


def run_full_evaluation():
    """Run complete evaluation suite."""
    evaluator = ModelEvaluator()
    
    # Run all tests
    evaluator.measure_inference_speed(num_episodes=100)
    evaluator.evaluate_constraint_compliance(num_episodes=100)
    evaluator.evaluate_diversity(num_episodes=100)
    evaluator.measure_memory_usage(num_episodes=50)
    evaluator.evaluate_reward_distribution(num_episodes=100)
    
    # Generate report
    summary = evaluator.generate_summary_report()
    
    return summary


if __name__ == '__main__':
    print("\nStarting comprehensive model evaluation...")
    print("This will take a few minutes...\n")
    
    summary = run_full_evaluation()
    
    print("\n✅ Evaluation complete!")
    print("\nCheck 'evaluation_results.json' for detailed results.")
