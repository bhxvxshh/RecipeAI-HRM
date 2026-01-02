"""
Statistical Analysis and Model Comparison Framework - Research Grade

Implements rigorous statistical testing for research publication:
- Paired t-tests for model comparisons
- Effect size calculations (Cohen's d)
- Confidence intervals
- Multiple comparison corrections (Bonferroni)
- Power analysis

Usage:
    python scripts/statistical_analysis.py --models baseline gpu curriculum
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind, f_oneway, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def paired_ttest_comparison(model_a_results, model_b_results, metric='satisfaction_rate'):
    """
    Compare two models using paired t-test (same test cases for both models).
    
    Args:
        model_a_results: List of result dicts from model A
        model_b_results: List of result dicts from model B (same order as A)
        metric: Metric to compare (e.g., 'satisfaction_rate', 'avg_reward')
    
    Returns:
        dict with statistical test results
    """
    # Extract metric values
    a_scores = np.array([r[metric] for r in model_a_results])
    b_scores = np.array([r[metric] for r in model_b_results])
    
    # Paired t-test
    t_stat, p_value = ttest_rel(a_scores, b_scores)
    
    # Effect size (Cohen's d for paired samples)
    differences = a_scores - b_scores
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Confidence interval for mean difference
    ci = stats.t.interval(
        0.95, 
        len(differences) - 1,
        loc=mean_diff,
        scale=stats.sem(differences)
    )
    
    # Power analysis (post-hoc)
    from statsmodels.stats.power import TTestPower
    power_analysis = TTestPower()
    power = power_analysis.solve_power(
        effect_size=abs(cohens_d),
        nobs=len(a_scores),
        alpha=0.05,
        alternative='two-sided'
    )
    
    return {
        'model_a_mean': np.mean(a_scores),
        'model_a_std': np.std(a_scores, ddof=1),
        'model_b_mean': np.mean(b_scores),
        'model_b_std': np.std(b_scores, ddof=1),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_interpretation': interpret_effect_size(cohens_d),
        'significant': p_value < 0.05,
        '95_ci_lower': ci[0],
        '95_ci_upper': ci[1],
        'statistical_power': power,
        'n_samples': len(a_scores),
    }


def independent_ttest_comparison(model_a_results, model_b_results, metric='satisfaction_rate'):
    """
    Compare two models using independent t-test (different test cases).
    """
    a_scores = np.array([r[metric] for r in model_a_results])
    b_scores = np.array([r[metric] for r in model_b_results])
    
    # Independent t-test
    t_stat, p_value = ttest_ind(a_scores, b_scores)
    
    # Effect size (Cohen's d for independent samples)
    mean_diff = np.mean(a_scores) - np.mean(b_scores)
    pooled_std = np.sqrt((np.var(a_scores, ddof=1) + np.var(b_scores, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'model_a_mean': np.mean(a_scores),
        'model_a_std': np.std(a_scores, ddof=1),
        'model_b_mean': np.mean(b_scores),
        'model_b_std': np.std(b_scores, ddof=1),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_interpretation': interpret_effect_size(cohens_d),
        'significant': p_value < 0.05,
        'n_samples_a': len(a_scores),
        'n_samples_b': len(b_scores),
    }


def mann_whitney_u_test(model_a_results, model_b_results, metric='satisfaction_rate'):
    """
    Non-parametric alternative to t-test (for non-normal distributions).
    """
    a_scores = np.array([r[metric] for r in model_a_results])
    b_scores = np.array([r[metric] for r in model_b_results])
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(a_scores, b_scores, alternative='two-sided')
    
    # Effect size (rank-biserial correlation)
    n_a, n_b = len(a_scores), len(b_scores)
    r = 1 - (2 * u_stat) / (n_a * n_b)  # Rank-biserial correlation
    
    return {
        'model_a_median': np.median(a_scores),
        'model_b_median': np.median(b_scores),
        'u_statistic': u_stat,
        'p_value': p_value,
        'rank_biserial_r': r,
        'significant': p_value < 0.05,
    }


def anova_multiple_models(model_results_dict, metric='satisfaction_rate'):
    """
    Compare 3+ models using one-way ANOVA.
    
    Args:
        model_results_dict: Dict of {model_name: [results]} for each model
        metric: Metric to compare
    
    Returns:
        ANOVA results and post-hoc pairwise comparisons
    """
    # Extract scores for each model
    groups = []
    model_names = []
    for name, results in model_results_dict.items():
        scores = np.array([r[metric] for r in results])
        groups.append(scores)
        model_names.append(name)
    
    # One-way ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    # Post-hoc pairwise comparisons (with Bonferroni correction)
    n_comparisons = len(model_names) * (len(model_names) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    
    pairwise_comparisons = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            comparison = paired_ttest_comparison(
                model_results_dict[model_names[i]],
                model_results_dict[model_names[j]],
                metric
            )
            comparison['model_a_name'] = model_names[i]
            comparison['model_b_name'] = model_names[j]
            comparison['bonferroni_alpha'] = bonferroni_alpha
            comparison['bonferroni_significant'] = comparison['p_value'] < bonferroni_alpha
            pairwise_comparisons.append(comparison)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_groups': len(model_names),
        'n_comparisons': n_comparisons,
        'bonferroni_alpha': bonferroni_alpha,
        'pairwise_comparisons': pairwise_comparisons,
    }


def interpret_effect_size(cohens_d):
    """
    Interpret Cohen's d effect size following standard conventions.
    
    Cohen's guidelines:
    - |d| < 0.2: Negligible
    - 0.2 ≤ |d| < 0.5: Small
    - 0.5 ≤ |d| < 0.8: Medium
    - |d| ≥ 0.8: Large
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================================
# RESULTS FORMATTING
# ============================================================================

def format_statistical_result(result):
    """
    Format statistical test results for publication.
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("STATISTICAL COMPARISON RESULTS")
    lines.append(f"{'='*70}\n")
    
    if 'model_a_name' in result:
        lines.append(f"Model A: {result['model_a_name']}")
        lines.append(f"Model B: {result['model_b_name']}\n")
    
    lines.append(f"Model A: M = {result['model_a_mean']:.4f}, SD = {result['model_a_std']:.4f}")
    lines.append(f"Model B: M = {result['model_b_mean']:.4f}, SD = {result['model_b_std']:.4f}")
    lines.append(f"Mean Difference: {result['mean_difference']:.4f}")
    lines.append(f"95% CI: [{result['95_ci_lower']:.4f}, {result['95_ci_upper']:.4f}]")
    lines.append(f"\nt({result['n_samples']-1}) = {result['t_statistic']:.4f}, p = {result['p_value']:.6f}")
    
    if result['significant']:
        lines.append("Result: SIGNIFICANT (p < 0.05) ***")
    else:
        lines.append("Result: NOT SIGNIFICANT (p ≥ 0.05)")
    
    lines.append(f"\nEffect Size: Cohen's d = {result['cohens_d']:.4f} ({result['effect_interpretation']})")
    lines.append(f"Statistical Power: {result['statistical_power']:.4f}")
    
    lines.append(f"\n{'='*70}\n")
    
    return "\n".join(lines)


def create_comparison_table(comparisons_dict):
    """
    Create publication-ready comparison table.
    
    Args:
        comparisons_dict: Dict of {comparison_name: statistical_result}
    
    Returns:
        pandas DataFrame formatted for publication
    """
    rows = []
    for name, result in comparisons_dict.items():
        row = {
            'Comparison': name,
            'Model A Mean': f"{result['model_a_mean']:.3f}",
            'Model B Mean': f"{result['model_b_mean']:.3f}",
            'Difference': f"{result['mean_difference']:.3f}",
            '95% CI': f"[{result['95_ci_lower']:.3f}, {result['95_ci_upper']:.3f}]",
            't': f"{result['t_statistic']:.3f}",
            'p': format_p_value(result['p_value']),
            "Cohen's d": f"{result['cohens_d']:.3f}",
            'Effect': result['effect_interpretation'].capitalize(),
            'Sig.': '***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns',
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def format_p_value(p):
    """Format p-value for publication."""
    if p < 0.001:
        return "< .001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(model_results_dict, metric='satisfaction_rate', 
                          output_path='analysis_results/model_comparison.png'):
    """
    Create publication-quality comparison visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    model_names = list(model_results_dict.keys())
    data = []
    for name in model_names:
        scores = [r[metric] for r in model_results_dict[name]]
        data.append(scores)
    
    # Box plot
    bp = axes[0].boxplot(data, labels=model_names, patch_artist=True, 
                         notch=True, showmeans=True)
    axes[0].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    axes[0].set_title('Model Comparison (Box Plot)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Color boxes
    colors = sns.color_palette('Set2', len(model_names))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Bar plot with error bars
    means = [np.mean(scores) for scores in data]
    stds = [np.std(scores, ddof=1) for scores in data]
    
    x_pos = np.arange(len(model_names))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7,
                error_kw={'linewidth': 2})
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    axes[1].set_title('Model Comparison (Mean ± SD)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add significance stars
    # (This would require pairwise comparison results - placeholder for now)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    return fig


def plot_effect_sizes(comparisons_dict, output_path='analysis_results/effect_sizes.png'):
    """
    Visualize effect sizes for multiple comparisons.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(comparisons_dict.keys())
    effect_sizes = [result['cohens_d'] for result in comparisons_dict.values()]
    ci_lowers = [result['95_ci_lower'] for result in comparisons_dict.values()]
    ci_uppers = [result['95_ci_upper'] for result in comparisons_dict.values()]
    
    y_pos = np.arange(len(names))
    
    # Plot effect sizes with error bars
    colors = ['green' if abs(d) >= 0.8 else 'orange' if abs(d) >= 0.5 else 'red' 
              for d in effect_sizes]
    
    ax.barh(y_pos, effect_sizes, color=colors, alpha=0.6)
    ax.errorbar(effect_sizes, y_pos, xerr=[np.array(effect_sizes) - np.array(ci_lowers),
                                            np.array(ci_uppers) - np.array(effect_sizes)],
                fmt='none', ecolor='black', capsize=5, linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Effect Sizes with 95% Confidence Intervals", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Small')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Medium')
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Large')
    ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Effect size plot saved to: {output_path}")
    
    return fig


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def run_statistical_analysis(model_paths, metric='satisfaction_rate', output_dir='analysis_results'):
    """
    Run complete statistical analysis comparing multiple models.
    
    Args:
        model_paths: Dict of {model_name: path_to_results_json}
        metric: Metric to analyze
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS PIPELINE - RESEARCH GRADE")
    print(f"{'='*70}\n")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results for each model
    model_results = {}
    for name, path in model_paths.items():
        with open(path, 'r') as f:
            results = json.load(f)
        model_results[name] = results
        print(f"Loaded {len(results)} results for {name}")
    
    print(f"\nAnalyzing metric: {metric}")
    print(f"{'='*70}\n")
    
    # Pairwise comparisons
    comparisons = {}
    model_names = list(model_results.keys())
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            comparison_name = f"{name_a} vs {name_b}"
            
            print(f"Comparing: {comparison_name}")
            result = paired_ttest_comparison(
                model_results[name_a],
                model_results[name_b],
                metric
            )
            result['model_a_name'] = name_a
            result['model_b_name'] = name_b
            comparisons[comparison_name] = result
            
            print(format_statistical_result(result))
    
    # ANOVA (if 3+ models)
    if len(model_names) >= 3:
        print(f"\nRunning ANOVA for {len(model_names)} models...")
        anova_result = anova_multiple_models(model_results, metric)
        print(f"F({anova_result['n_groups']-1}, {sum([len(r) for r in model_results.values()])-anova_result['n_groups']}) = "
              f"{anova_result['f_statistic']:.4f}, p = {anova_result['p_value']:.6f}")
        
        if anova_result['significant']:
            print("Result: SIGNIFICANT (at least one model differs) ***")
        else:
            print("Result: NOT SIGNIFICANT")
    
    # Create comparison table
    comparison_table = create_comparison_table(comparisons)
    table_path = Path(output_dir) / 'statistical_comparisons.csv'
    comparison_table.to_csv(table_path, index=False)
    print(f"\nComparison table saved to: {table_path}")
    
    # Generate visualizations
    plot_model_comparison(model_results, metric, Path(output_dir) / 'model_comparison.png')
    plot_effect_sizes(comparisons, Path(output_dir) / 'effect_sizes.png')
    
    # Save full results
    results_path = Path(output_dir) / 'statistical_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'metric': metric,
            'pairwise_comparisons': comparisons,
            'anova': anova_result if len(model_names) >= 3 else None,
        }, f, indent=2, default=str)
    print(f"Full results saved to: {results_path}")
    
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    return comparisons, comparison_table


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Statistical analysis for model comparison')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model names (e.g., baseline gpu curriculum)')
    parser.add_argument('--result_paths', nargs='+', required=True,
                        help='Paths to results JSON files for each model')
    parser.add_argument('--metric', default='satisfaction_rate',
                        help='Metric to analyze (default: satisfaction_rate)')
    parser.add_argument('--output_dir', default='analysis_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Build model_paths dict
    model_paths = dict(zip(args.models, args.result_paths))
    
    # Run analysis
    comparisons, table = run_statistical_analysis(
        model_paths,
        metric=args.metric,
        output_dir=args.output_dir
    )
    
    print("\nExample usage for further analysis:")
    print("  python scripts/statistical_analysis.py \\")
    print("    --models baseline gpu curriculum \\")
    print("    --result_paths baseline_results.json gpu_results.json curriculum_results.json \\")
    print("    --metric satisfaction_rate")
