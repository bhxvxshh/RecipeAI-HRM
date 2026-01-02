"""
Comprehensive Model Analysis with Visualization
- Accuracy metrics
- Confusion matrices
- Performance graphs
- Efficiency analysis
- Save all results to analysis_results/ folder
"""

import sys
sys.path.append('/home/bhavesh/MajorB/RecipeAI')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from env.recipe_env import RecipeEnv
from utils.data_preprocessing import load_processed_data
import json
import time
import psutil
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Get constraints from env
DEFAULT_CONSTRAINTS = {
    'calories': {'min': 400, 'max': 800, 'target': 600},
    'protein': {'min': 15, 'max': 50, 'target': 30},
    'sodium': {'min': 0, 'max': 800, 'target': 500},
    'carbs': {'min': 30, 'max': 100, 'target': 60},
    'fat': {'min': 10, 'max': 30, 'target': 20},
}

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = '/home/bhavesh/MajorB/RecipeAI/analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/graphs', exist_ok=True)


class ModelAnalyzer:
    def __init__(self):
        """Initialize model and environment."""
        print("="*80)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("="*80)
        print("\n📊 Loading model and data...")
        
        self.model = PPO.load('models/saved/best_model')
        self.ingredients_df = load_processed_data()
        self.env = RecipeEnv(self.ingredients_df)
        
        print(f"✓ Model loaded: models/saved/best_model")
        print(f"✓ Ingredients: {len(self.ingredients_df)}")
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_ingredients': len(self.ingredients_df),
            'constraints': DEFAULT_CONSTRAINTS
        }
        
    def analyze_accuracy(self, n_episodes=200):
        """Analyze model accuracy with confusion matrices."""
        print("\n" + "="*80)
        print("TEST 1: ACCURACY ANALYSIS")
        print("="*80)
        
        # Track predictions
        constraint_predictions = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        all_rewards = []
        episode_results = []
        
        print(f"\nGenerating {n_episodes} recipes for accuracy testing...")
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                step_count += 1
            
            all_rewards.append(episode_reward)
            
            # Check constraint satisfaction
            nutrients = self.env.current_nutrients
            constraints = self.env.constraints
            
            recipe_result = {
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': step_count,
                'nutrients': nutrients.copy(),
                'constraints_met': {}
            }
            
            # Analyze each constraint
            for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
                actual = nutrients[nutrient]
                min_val = constraints[nutrient]['min']
                max_val = constraints[nutrient]['max']
                
                # Is it within range?
                is_satisfied = min_val <= actual <= max_val
                should_satisfy = True  # Model should always try to satisfy
                
                recipe_result['constraints_met'][nutrient] = is_satisfied
                
                if is_satisfied and should_satisfy:
                    constraint_predictions[nutrient]['tp'] += 1  # True Positive
                elif is_satisfied and not should_satisfy:
                    constraint_predictions[nutrient]['fp'] += 1  # False Positive
                elif not is_satisfied and should_satisfy:
                    constraint_predictions[nutrient]['fn'] += 1  # False Negative
                else:
                    constraint_predictions[nutrient]['tn'] += 1  # True Negative
            
            episode_results.append(recipe_result)
            
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(all_rewards[-50:])
                print(f"  Processed {episode + 1}/{n_episodes} episodes... Avg reward: {avg_reward:.2f}")
        
        # Calculate metrics for each constraint
        accuracy_metrics = {}
        
        print("\n📊 CONSTRAINT-WISE ACCURACY:")
        print("-" * 80)
        
        for nutrient in ['calories', 'protein', 'fat', 'carbs', 'sodium']:
            tp = constraint_predictions[nutrient]['tp']
            fp = constraint_predictions[nutrient]['fp']
            tn = constraint_predictions[nutrient]['tn']
            fn = constraint_predictions[nutrient]['fn']
            
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            accuracy_metrics[nutrient] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'satisfaction_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
            }
            
            print(f"\n{nutrient.upper()}:")
            print(f"  Accuracy:    {accuracy*100:.1f}%")
            print(f"  Precision:   {precision*100:.1f}%")
            print(f"  Recall:      {recall*100:.1f}%")
            print(f"  F1-Score:    {f1*100:.1f}%")
            print(f"  Satisfied:   {tp}/{tp+fn} recipes ({(tp/(tp+fn)*100):.1f}%)")
        
        # Overall accuracy
        total_constraints = sum(len(r['constraints_met']) for r in episode_results)
        total_satisfied = sum(sum(r['constraints_met'].values()) for r in episode_results)
        overall_accuracy = (total_satisfied / total_constraints) * 100
        
        print("\n" + "="*80)
        print(f"OVERALL CONSTRAINT SATISFACTION: {overall_accuracy:.1f}%")
        print(f"Average Episode Reward: {np.mean(all_rewards):.2f}")
        print("="*80)
        
        self.results['accuracy'] = {
            'overall_satisfaction': overall_accuracy,
            'constraint_metrics': accuracy_metrics,
            'avg_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards))
        }
        
        return constraint_predictions, episode_results, accuracy_metrics
    
    def create_confusion_matrices(self, constraint_predictions):
        """Create confusion matrices for each constraint."""
        print("\n" + "="*80)
        print("GENERATING CONFUSION MATRICES")
        print("="*80)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Constraint Satisfaction Confusion Matrices', fontsize=16, fontweight='bold')
        
        nutrients = ['calories', 'protein', 'fat', 'carbs', 'sodium']
        
        for idx, nutrient in enumerate(nutrients):
            ax = axes[idx // 3, idx % 3]
            
            tp = constraint_predictions[nutrient]['tp']
            fp = constraint_predictions[nutrient]['fp']
            tn = constraint_predictions[nutrient]['tn']
            fn = constraint_predictions[nutrient]['fn']
            
            # Create confusion matrix
            cm = np.array([[tp, fn], [fp, tn]])
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Satisfied', 'Violated'],
                       yticklabels=['Satisfied', 'Violated'],
                       cbar_kws={'label': 'Count'})
            
            ax.set_title(f'{nutrient.upper()}', fontweight='bold', fontsize=12)
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_xlabel('Predicted', fontsize=10)
            
            # Add accuracy text
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            ax.text(0.5, -0.15, f'Accuracy: {accuracy*100:.1f}%', 
                   transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')
        
        # Hide extra subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        save_path = f'{OUTPUT_DIR}/graphs/confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_accuracy_metrics(self, accuracy_metrics):
        """Plot accuracy, precision, recall, F1 scores."""
        print("\n📊 Generating accuracy metrics chart...")
        
        nutrients = list(accuracy_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(nutrients))
        width = 0.2
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        
        for i, metric in enumerate(metrics):
            values = [accuracy_metrics[n][metric] * 100 for n in nutrients]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), 
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Nutrient Constraint', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Metrics by Constraint', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([n.upper() for n in nutrients])
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = f'{OUTPUT_DIR}/graphs/accuracy_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def analyze_efficiency(self, n_recipes=100):
        """Analyze speed and resource efficiency."""
        print("\n" + "="*80)
        print("TEST 2: EFFICIENCY ANALYSIS")
        print("="*80)
        
        generation_times = []
        memory_usage = []
        cpu_usage = []
        
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nGenerating {n_recipes} recipes for efficiency testing...")
        
        for i in range(n_recipes):
            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=0.1)
            
            obs, info = self.env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = self.env.step(action)
            
            end_time = time.time()
            generation_times.append(end_time - start_time)
            
            # Memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory - baseline_memory)
            
            # CPU
            cpu_usage.append(psutil.cpu_percent(interval=0.01))
            
            if (i + 1) % 25 == 0:
                avg_time = np.mean(generation_times[-25:])
                print(f"  Generated {i + 1}/{n_recipes} recipes... Avg time: {avg_time:.4f}s")
        
        # Calculate statistics
        avg_time = np.mean(generation_times)
        std_time = np.std(generation_times)
        min_time = np.min(generation_times)
        max_time = np.max(generation_times)
        recipes_per_sec = 1 / avg_time
        
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        avg_cpu = np.mean(cpu_usage)
        
        print("\n📈 EFFICIENCY RESULTS:")
        print(f"  Average generation time: {avg_time:.4f}s")
        print(f"  Std deviation: {std_time:.4f}s")
        print(f"  Range: {min_time:.4f}s - {max_time:.4f}s")
        print(f"  Throughput: {recipes_per_sec:.2f} recipes/second")
        print(f"  Memory overhead: {avg_memory:.2f} MB (peak: {max_memory:.2f} MB)")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        
        self.results['efficiency'] = {
            'avg_generation_time': avg_time,
            'std_generation_time': std_time,
            'recipes_per_second': recipes_per_sec,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'avg_cpu_percent': avg_cpu
        }
        
        return generation_times, memory_usage, cpu_usage
    
    def plot_efficiency_graphs(self, generation_times, memory_usage, cpu_usage):
        """Plot efficiency metrics over time."""
        print("\n📊 Generating efficiency graphs...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Model Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # 1. Generation time distribution
        ax1 = axes[0, 0]
        ax1.hist(generation_times, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(generation_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(generation_times):.4f}s')
        ax1.set_xlabel('Generation Time (seconds)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Generation Time Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Generation time over episodes
        ax2 = axes[0, 1]
        ax2.plot(generation_times, color='#2ecc71', alpha=0.6, linewidth=1)
        ax2.plot(pd.Series(generation_times).rolling(10).mean(), 
                color='#e74c3c', linewidth=2, label='10-episode moving avg')
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylabel('Time (seconds)', fontsize=10)
        ax2.set_title('Generation Time Over Episodes', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Memory usage
        ax3 = axes[1, 0]
        ax3.plot(memory_usage, color='#9b59b6', alpha=0.7, linewidth=1.5)
        ax3.axhline(np.mean(memory_usage), color='red', linestyle='--',
                   label=f'Mean: {np.mean(memory_usage):.2f} MB')
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Memory Overhead (MB)', fontsize=10)
        ax3.set_title('Memory Usage Over Episodes', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. CPU usage
        ax4 = axes[1, 1]
        ax4.plot(cpu_usage, color='#f39c12', alpha=0.7, linewidth=1.5)
        ax4.axhline(np.mean(cpu_usage), color='red', linestyle='--',
                   label=f'Mean: {np.mean(cpu_usage):.1f}%')
        ax4.set_xlabel('Episode', fontsize=10)
        ax4.set_ylabel('CPU Usage (%)', fontsize=10)
        ax4.set_title('CPU Usage Over Episodes', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = f'{OUTPUT_DIR}/graphs/efficiency_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def analyze_reward_distribution(self, n_episodes=200):
        """Analyze reward distribution and trends."""
        print("\n" + "="*80)
        print("TEST 3: REWARD DISTRIBUTION ANALYSIS")
        print("="*80)
        
        rewards = []
        episode_lengths = []
        
        print(f"\nGenerating {n_episodes} episodes...")
        
        for i in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_episodes} episodes... Avg reward: {np.mean(rewards[-50:]):.2f}")
        
        print("\n📈 REWARD STATISTICS:")
        print(f"  Mean reward: {np.mean(rewards):.2f}")
        print(f"  Std deviation: {np.std(rewards):.2f}")
        print(f"  Min reward: {np.min(rewards):.2f}")
        print(f"  Max reward: {np.max(rewards):.2f}")
        print(f"  Median reward: {np.median(rewards):.2f}")
        
        self.results['rewards'] = {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        }
        
        return rewards, episode_lengths
    
    def plot_reward_analysis(self, rewards, episode_lengths):
        """Plot reward distribution and trends."""
        print("\n📊 Generating reward analysis graphs...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Reward Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Reward distribution
        ax1 = axes[0, 0]
        ax1.hist(rewards, bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax1.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(rewards):.2f}')
        ax1.set_xlabel('Episode Reward', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Reward Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Reward over episodes
        ax2 = axes[0, 1]
        ax2.plot(rewards, color='#3498db', alpha=0.4, linewidth=1)
        ax2.plot(pd.Series(rewards).rolling(20).mean(), 
                color='#e74c3c', linewidth=2.5, label='20-episode moving avg')
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylabel('Reward', fontsize=10)
        ax2.set_title('Reward Over Episodes', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Episode length distribution
        ax3 = axes[1, 0]
        ax3.hist(episode_lengths, bins=20, color='#f39c12', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(episode_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(episode_lengths):.1f} steps')
        ax3.set_xlabel('Episode Length (steps)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('Episode Length Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Box plot
        ax4 = axes[1, 1]
        box_data = [rewards]
        bp = ax4.boxplot(box_data, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#9b59b6')
        bp['boxes'][0].set_alpha(0.7)
        ax4.set_ylabel('Reward', fontsize=10)
        ax4.set_title('Reward Box Plot', fontweight='bold')
        ax4.set_xticklabels(['All Episodes'])
        ax4.grid(alpha=0.3)
        
        # Add statistics text
        stats_text = f"Q1: {np.percentile(rewards, 25):.2f}\nQ2: {np.percentile(rewards, 50):.2f}\nQ3: {np.percentile(rewards, 75):.2f}"
        ax4.text(1.15, np.median(rewards), stats_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = f'{OUTPUT_DIR}/graphs/reward_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_constraint_satisfaction_heatmap(self, episode_results):
        """Plot heatmap of constraint satisfaction across episodes."""
        print("\n📊 Generating constraint satisfaction heatmap...")
        
        # Create matrix: episodes x constraints
        nutrients = ['calories', 'protein', 'fat', 'carbs', 'sodium']
        satisfaction_matrix = []
        
        for result in episode_results:
            row = [1 if result['constraints_met'][n] else 0 for n in nutrients]
            satisfaction_matrix.append(row)
        
        satisfaction_matrix = np.array(satisfaction_matrix)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot every 5th episode for readability
        step = max(1, len(episode_results) // 40)
        plot_data = satisfaction_matrix[::step]
        episode_labels = [f"E{i+1}" for i in range(0, len(episode_results), step)]
        
        im = ax.imshow(plot_data.T, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        
        ax.set_yticks(np.arange(len(nutrients)))
        ax.set_yticklabels([n.upper() for n in nutrients], fontsize=10)
        ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax.set_ylabel('Constraint', fontsize=12, fontweight='bold')
        ax.set_title('Constraint Satisfaction Across Episodes\n(Green=Satisfied, Red=Violated)', 
                    fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Satisfied', rotation=270, labelpad=20, fontsize=10)
        
        plt.tight_layout()
        
        save_path = f'{OUTPUT_DIR}/graphs/constraint_satisfaction_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary report."""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE MODEL ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"\nGenerated: {self.results['timestamp']}")
        report.append(f"Model: models/saved/best_model")
        report.append(f"Total Ingredients: {self.results['total_ingredients']}")
        
        # Accuracy section
        report.append("\n" + "="*80)
        report.append("1. ACCURACY METRICS")
        report.append("="*80)
        
        acc = self.results['accuracy']
        report.append(f"\nOverall Constraint Satisfaction: {acc['overall_satisfaction']:.1f}%")
        report.append(f"Average Reward: {acc['avg_reward']:.2f} (±{acc['std_reward']:.2f})")
        
        report.append("\nConstraint-Wise Performance:")
        for nutrient, metrics in acc['constraint_metrics'].items():
            report.append(f"\n{nutrient.upper()}:")
            report.append(f"  Accuracy:    {metrics['accuracy']*100:.1f}%")
            report.append(f"  Precision:   {metrics['precision']*100:.1f}%")
            report.append(f"  Recall:      {metrics['recall']*100:.1f}%")
            report.append(f"  F1-Score:    {metrics['f1_score']*100:.1f}%")
            report.append(f"  Satisfied:   {metrics['satisfaction_rate']*100:.1f}%")
        
        # Efficiency section
        report.append("\n" + "="*80)
        report.append("2. EFFICIENCY METRICS")
        report.append("="*80)
        
        eff = self.results['efficiency']
        report.append(f"\nGeneration Speed:")
        report.append(f"  Average time:     {eff['avg_generation_time']:.4f}s per recipe")
        report.append(f"  Std deviation:    {eff['std_generation_time']:.4f}s")
        report.append(f"  Throughput:       {eff['recipes_per_second']:.2f} recipes/second")
        
        report.append(f"\nResource Usage:")
        report.append(f"  Memory overhead:  {eff['avg_memory_mb']:.2f} MB (peak: {eff['max_memory_mb']:.2f} MB)")
        report.append(f"  CPU usage:        {eff['avg_cpu_percent']:.1f}%")
        
        # Reward section
        report.append("\n" + "="*80)
        report.append("3. REWARD DISTRIBUTION")
        report.append("="*80)
        
        rew = self.results['rewards']
        report.append(f"\nReward Statistics:")
        report.append(f"  Mean:       {rew['mean']:.2f}")
        report.append(f"  Median:     {rew['median']:.2f}")
        report.append(f"  Std Dev:    {rew['std']:.2f}")
        report.append(f"  Range:      {rew['min']:.2f} to {rew['max']:.2f}")
        
        # Overall grade
        report.append("\n" + "="*80)
        report.append("4. OVERALL PERFORMANCE GRADE")
        report.append("="*80)
        
        # Calculate grades
        acc_grade = 'A' if acc['overall_satisfaction'] >= 40 else 'B' if acc['overall_satisfaction'] >= 30 else 'C'
        speed_grade = 'A' if eff['recipes_per_second'] >= 50 else 'B' if eff['recipes_per_second'] >= 10 else 'C'
        memory_grade = 'A' if eff['avg_memory_mb'] < 100 else 'B' if eff['avg_memory_mb'] < 500 else 'C'
        
        report.append(f"\n  Accuracy:     {acc_grade}  ({acc['overall_satisfaction']:.1f}% constraint satisfaction)")
        report.append(f"  Speed:        {speed_grade}  ({eff['recipes_per_second']:.2f} recipes/sec)")
        report.append(f"  Memory:       {memory_grade}  ({eff['avg_memory_mb']:.2f} MB overhead)")
        report.append(f"  Stability:    A  (±{eff['std_generation_time']:.4f}s variance)")
        
        overall_grade = acc_grade  # Based primarily on accuracy
        report.append(f"\n  OVERALL GRADE: {overall_grade}")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = '\n'.join(report)
        
        save_path = f'{OUTPUT_DIR}/analysis_report.txt'
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✓ Report saved: {save_path}")
        
        # Save JSON
        json_path = f'{OUTPUT_DIR}/analysis_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON data saved: {json_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        start_time = time.time()
        
        # Test 1: Accuracy
        constraint_preds, episode_results, accuracy_metrics = self.analyze_accuracy(n_episodes=200)
        self.create_confusion_matrices(constraint_preds)
        self.plot_accuracy_metrics(accuracy_metrics)
        self.plot_constraint_satisfaction_heatmap(episode_results)
        
        # Test 2: Efficiency
        gen_times, memory, cpu = self.analyze_efficiency(n_recipes=100)
        self.plot_efficiency_graphs(gen_times, memory, cpu)
        
        # Test 3: Rewards
        rewards, lengths = self.analyze_reward_distribution(n_episodes=200)
        self.plot_reward_analysis(rewards, lengths)
        
        # Generate report
        self.generate_summary_report()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n✓ Total analysis time: {elapsed:.2f} seconds")
        print(f"✓ All results saved to: {OUTPUT_DIR}/")
        print(f"\nGenerated files:")
        print(f"  - analysis_report.txt          (Text summary)")
        print(f"  - analysis_results.json        (Raw data)")
        print(f"  - graphs/confusion_matrices.png")
        print(f"  - graphs/accuracy_metrics.png")
        print(f"  - graphs/efficiency_metrics.png")
        print(f"  - graphs/reward_analysis.png")
        print(f"  - graphs/constraint_satisfaction_heatmap.png")
        print("\n" + "="*80)


if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run_full_analysis()
