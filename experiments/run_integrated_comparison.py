"""
Integrated GBWM Method Comparison with Paper Figure Generation

This script performs a rigorous comparison of all GBWM methods following the
methodology from Section III.D of the GBWM RL paper (Das et al.):

Methods compared:
1. Dynamic Programming (DP) - Optimal theoretical solution
2. Pure RL (PPO-based GBWM)
3. Sentiment-Aware RL (VIX-augmented GBWM)
4. Random Strategy
5. Buy and Hold Strategy
6. Greedy Goal Strategy
7. Conservative Strategy
8. Aggressive Strategy

Hybrid combinations:
- DP Investment + RL Goal-taking
- RL Investment + DP Goal-taking
- DP Investment + Sentiment Goal-taking
- Sentiment Investment + DP Goal-taking
- RL Investment + Sentiment Goal-taking
- Sentiment Investment + RL Goal-taking

Generates paper figures:
- Figure 1: Efficiency vs Number of Goals (all strategies)
- Figure 2: Hybrid Strategy Analysis

Usage:
    # Quick test (100 simulations)
    python experiments/run_integrated_comparison.py --num_simulations 100 --num_iterations 1

    # Full comparison (100,000 simulations)
    python experiments/run_integrated_comparison.py --num_simulations 100000 --num_iterations 10

    # With config preset
    python experiments/run_integrated_comparison.py --config_preset aggressive --num_simulations 1000

    # Skip training (use existing models)
    python experiments/run_integrated_comparison.py --skip_training --num_simulations 10000
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.gbwm_env import GBWMEnvironment, make_gbwm_env
from src.environment.gbwm_env_sentiment import make_sentiment_gbwm_env
from src.environment.gbwm_env_monthly import GBWMEnvMonthly, make_gbwm_env_monthly
from src.models.ppo_agent import PPOAgent
from src.models.sentiment_ppo_agent import SentimentAwarePPOAgent
from src.models.vix_market_model import VIXMarketModel, VIXModelParams
from src.algorithms.dynamic_programming import (
    GBWMDynamicProgramming, DPConfig,
    MultiGoalGBWMDP, MultiGoalDPConfig, solve_multi_goal_dp
)
from src.data.sentiment_provider import SentimentProvider
from config.training_config import TrainingConfig
from config.sentiment_config import get_sentiment_config

# Pre-training modules (efficient frontier and beta/delta learning)
from src.data.efficient_frontier import (
    EfficientFrontierCalculator,
    compute_efficient_frontier,
    get_portfolio_weights,
)
from src.data.beta_delta_learner import (
    BetaDeltaLearner,
    learn_beta_delta,
    get_default_beta_delta,
)

# Sentiment trainer (uses pre-trained β/δ and efficient frontier)
from src.training.sentiment_trainer import (
    SentimentGBWMTrainer,
    SentimentTrainingConfig,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation results"""
    method_name: str
    num_goals: int
    mean_reward: float
    std_reward: float
    efficiency: float
    num_simulations: int
    goal_success_rate: float
    mean_final_wealth: float
    std_final_wealth: float = 0.0


@dataclass
class SharedRandomState:
    """Container for shared random state across all methods"""
    random_seeds: np.ndarray
    market_samples: np.ndarray  # Pre-generated GBM shocks for fair comparison

    def __post_init__(self):
        if len(self.random_seeds) != self.market_samples.shape[0]:
            raise ValueError("Mismatch between seeds and market samples")


@dataclass
class YearlyMetrics:
    """Metrics for a single year within a backtest window"""
    calendar_year: int
    vix_level: float
    vix_regime: str  # 'tranquil', 'turmoil', 'crisis'
    portfolio_choice: int
    wealth_start: float
    wealth_end: float
    wealth_change_pct: float
    goal_available: bool
    goal_taken: bool
    goal_reward: float
    cumulative_reward: float


class CrisisYearAnalyzer:
    """
    Analyze strategy performance during crisis vs normal years.

    Generates visualizations comparing DP, RL, Sentiment RL during:
    - 2008 Financial Crisis
    - 2020 COVID-19 Crisis
    - Other high-VIX periods
    """

    # Define crisis years based on historical VIX spikes
    CRISIS_YEARS = {
        1974: "Oil Crisis",
        1987: "Black Monday",
        1998: "LTCM Crisis",
        2000: "Dot-com Crash",
        2001: "9/11",
        2002: "Market Bottom",
        2008: "Financial Crisis",
        2009: "Crisis Continuation",
        2011: "European Debt",
        2020: "COVID-19"
    }

    # VIX thresholds for regime classification
    VIX_TRANQUIL_MAX = 18
    VIX_TURMOIL_MAX = 30

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.yearly_data: Dict[str, List[Dict]] = {}  # method -> list of yearly records

    def classify_vix_regime(self, vix_level: float) -> str:
        """Classify VIX level into regime"""
        if vix_level < self.VIX_TRANQUIL_MAX:
            return 'tranquil'
        elif vix_level < self.VIX_TURMOIL_MAX:
            return 'turmoil'
        else:
            return 'crisis'

    def add_yearly_metrics(self, method_name: str, metrics: List[YearlyMetrics]):
        """Add yearly metrics for a method"""
        if method_name not in self.yearly_data:
            self.yearly_data[method_name] = []

        for m in metrics:
            self.yearly_data[method_name].append({
                'calendar_year': m.calendar_year,
                'vix_level': m.vix_level,
                'vix_regime': m.vix_regime,
                'portfolio_choice': m.portfolio_choice,
                'wealth_start': m.wealth_start,
                'wealth_end': m.wealth_end,
                'wealth_change_pct': m.wealth_change_pct,
                'goal_available': m.goal_available,
                'goal_taken': m.goal_taken,
                'goal_reward': m.goal_reward,
                'cumulative_reward': m.cumulative_reward,
                'is_crisis': m.calendar_year in self.CRISIS_YEARS
            })

    def get_dataframe(self) -> 'pd.DataFrame':
        """Convert all yearly data to pandas DataFrame"""
        import pandas as pd

        records = []
        for method_name, method_records in self.yearly_data.items():
            for record in method_records:
                record['method'] = method_name
                records.append(record)

        return pd.DataFrame(records)

    def generate_all_plots(self):
        """Generate all crisis analysis visualizations"""
        import pandas as pd
        import matplotlib.pyplot as plt

        df = self.get_dataframe()
        if df.empty:
            logger.warning("No data for crisis analysis plots")
            return

        # Create output directory for crisis analysis
        crisis_dir = self.output_dir / "crisis_analysis"
        crisis_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating crisis analysis plots in {crisis_dir}")

        # Generate each plot
        self._plot_portfolio_vs_vix(df, crisis_dir)
        self._plot_wealth_change_by_regime(df, crisis_dir)
        self._plot_crisis_year_comparison(df, crisis_dir)
        self._plot_sentiment_advantage(df, crisis_dir)
        self._plot_goal_timing_analysis(df, crisis_dir)
        self._plot_performance_heatmap(df, crisis_dir)

        # Save summary statistics
        self._save_crisis_statistics(df, crisis_dir)

        logger.info(f"Crisis analysis complete. Plots saved to {crisis_dir}")

    def _plot_portfolio_vs_vix(self, df: 'pd.DataFrame', output_dir: Path):
        """Plot portfolio choice adaptation vs VIX levels"""
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Get unique years and sort
        years = sorted(df['calendar_year'].unique())

        # Top plot: VIX level by year
        ax1 = axes[0]
        yearly_vix = df.groupby('calendar_year')['vix_level'].mean()
        ax1.fill_between(yearly_vix.index, 0, yearly_vix.values, alpha=0.3, color='gray', label='VIX Level')
        ax1.plot(yearly_vix.index, yearly_vix.values, color='black', linewidth=1)
        ax1.axhline(self.VIX_TRANQUIL_MAX, color='green', linestyle='--', alpha=0.7, label=f'Tranquil (<{self.VIX_TRANQUIL_MAX})')
        ax1.axhline(self.VIX_TURMOIL_MAX, color='orange', linestyle='--', alpha=0.7, label=f'Turmoil (<{self.VIX_TURMOIL_MAX})')
        ax1.axhline(50, color='red', linestyle='--', alpha=0.7, label='Crisis (>50)')
        ax1.set_ylabel('VIX Level', fontsize=12)
        ax1.set_title('VIX Level and Portfolio Adaptation Over Time', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.set_ylim(0, 80)

        # Highlight crisis years
        for year in self.CRISIS_YEARS.keys():
            if year in yearly_vix.index:
                ax1.axvspan(year - 0.4, year + 0.4, alpha=0.2, color='red')

        # Bottom plot: Portfolio choice by method
        ax2 = axes[1]
        methods_to_plot = ['RL', 'Sentiment RL']
        colors = {'RL': 'blue', 'Sentiment RL': 'green', 'DP': 'red'}

        for method in methods_to_plot:
            if method in df['method'].unique():
                method_df = df[df['method'] == method]
                yearly_portfolio = method_df.groupby('calendar_year')['portfolio_choice'].mean()
                ax2.plot(yearly_portfolio.index, yearly_portfolio.values,
                        label=method, color=colors.get(method, 'gray'), linewidth=2)

        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Avg Portfolio Choice\n(0=Conservative, 14=Aggressive)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.set_ylim(-0.5, 14.5)
        ax2.axhline(7, color='gray', linestyle=':', alpha=0.5, label='Moderate')

        # Highlight crisis years on bottom plot too
        for year in self.CRISIS_YEARS.keys():
            if year in years:
                ax2.axvspan(year - 0.4, year + 0.4, alpha=0.2, color='red')

        plt.tight_layout()
        plt.savefig(output_dir / 'portfolio_vs_vix.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: portfolio_vs_vix.png")

    def _plot_wealth_change_by_regime(self, df: 'pd.DataFrame', output_dir: Path):
        """Plot wealth change comparison by VIX regime"""
        import matplotlib.pyplot as plt
        import pandas as pd

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        regimes = ['tranquil', 'turmoil', 'crisis']
        regime_titles = ['Tranquil (VIX < 18)', 'Turmoil (18 ≤ VIX < 30)', 'Crisis (VIX ≥ 30)']

        methods = ['DP', 'RL', 'Sentiment RL', 'Buy & Hold', 'Conservative']
        method_colors = {
            'DP': '#2ecc71',
            'RL': '#3498db',
            'Sentiment RL': '#e74c3c',
            'Buy & Hold': '#9b59b6',
            'Conservative': '#1abc9c'
        }

        for idx, (regime, title) in enumerate(zip(regimes, regime_titles)):
            ax = axes[idx]
            regime_df = df[df['vix_regime'] == regime]

            wealth_changes = []
            method_labels = []
            colors = []

            for method in methods:
                if method in regime_df['method'].unique():
                    method_df = regime_df[regime_df['method'] == method]
                    avg_change = method_df['wealth_change_pct'].mean()
                    wealth_changes.append(avg_change)
                    method_labels.append(method)
                    colors.append(method_colors.get(method, 'gray'))

            if wealth_changes:
                bars = ax.barh(method_labels, wealth_changes, color=colors, alpha=0.8)
                ax.axvline(0, color='black', linewidth=0.5)
                ax.set_xlabel('Avg Wealth Change (%)')
                ax.set_title(title)

                # Add value labels
                for bar, val in zip(bars, wealth_changes):
                    ax.text(val + 0.5 if val >= 0 else val - 0.5, bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

        plt.suptitle('Wealth Change by VIX Regime', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'wealth_change_by_regime.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: wealth_change_by_regime.png")

    def _plot_crisis_year_comparison(self, df: 'pd.DataFrame', output_dir: Path):
        """Deep dive into specific crisis years (2008, 2020)"""
        import matplotlib.pyplot as plt

        crisis_focus = [(2008, 'Financial Crisis'), (2020, 'COVID-19')]
        available_crises = [(y, n) for y, n in crisis_focus if y in df['calendar_year'].unique()]

        if not available_crises:
            logger.warning("No crisis years in data for deep dive plot")
            return

        fig, axes = plt.subplots(1, len(available_crises), figsize=(7 * len(available_crises), 8))
        if len(available_crises) == 1:
            axes = [axes]

        methods = ['DP', 'RL', 'Sentiment RL', 'Buy & Hold']

        for idx, (year, crisis_name) in enumerate(available_crises):
            ax = axes[idx]
            year_df = df[df['calendar_year'] == year]

            wealth_changes = []
            method_labels = []
            colors = []

            for method in methods:
                if method in year_df['method'].unique():
                    method_df = year_df[year_df['method'] == method]
                    avg_change = method_df['wealth_change_pct'].mean()
                    wealth_changes.append(avg_change)
                    method_labels.append(method)

                    # Color based on performance
                    if avg_change > -5:
                        colors.append('#2ecc71')  # Green - good
                    elif avg_change > -15:
                        colors.append('#f39c12')  # Orange - moderate
                    else:
                        colors.append('#e74c3c')  # Red - bad

            bars = ax.barh(method_labels, wealth_changes, color=colors, alpha=0.8, edgecolor='black')
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Wealth Change (%)', fontsize=12)
            ax.set_title(f'{year} {crisis_name}', fontsize=14, fontweight='bold')

            # Get VIX for this year
            vix_level = year_df['vix_level'].mean()
            ax.text(0.02, 0.98, f'Avg VIX: {vix_level:.1f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Add value labels
            for bar, val in zip(bars, wealth_changes):
                ax.text(val - 1 if val < 0 else val + 1, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', ha='right' if val < 0 else 'left',
                       fontsize=11, fontweight='bold')

        plt.suptitle('Crisis Year Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'crisis_year_deep_dive.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: crisis_year_deep_dive.png")

    def _plot_sentiment_advantage(self, df: 'pd.DataFrame', output_dir: Path):
        """Show Sentiment RL advantage over pure RL by year"""
        import matplotlib.pyplot as plt
        import pandas as pd

        if 'RL' not in df['method'].unique() or 'Sentiment RL' not in df['method'].unique():
            logger.warning("Need both RL and Sentiment RL for advantage plot")
            return

        # Calculate yearly averages for each method
        rl_df = df[df['method'] == 'RL'].groupby('calendar_year').agg({
            'wealth_change_pct': 'mean',
            'portfolio_choice': 'mean',
            'vix_level': 'mean'
        }).rename(columns={'wealth_change_pct': 'rl_wealth', 'portfolio_choice': 'rl_portfolio'})

        sent_df = df[df['method'] == 'Sentiment RL'].groupby('calendar_year').agg({
            'wealth_change_pct': 'mean',
            'portfolio_choice': 'mean'
        }).rename(columns={'wealth_change_pct': 'sent_wealth', 'portfolio_choice': 'sent_portfolio'})

        # Merge
        combined = rl_df.join(sent_df, how='inner')
        combined['advantage'] = combined['sent_wealth'] - combined['rl_wealth']
        combined['portfolio_diff'] = combined['sent_portfolio'] - combined['rl_portfolio']

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        years = combined.index

        # Plot 1: Wealth advantage
        ax1 = axes[0]
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in combined['advantage']]
        ax1.bar(years, combined['advantage'], color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylabel('Wealth Change Advantage (%)\n(Sentiment RL - RL)', fontsize=11)
        ax1.set_title('Sentiment RL Advantage Over Pure RL', fontsize=14, fontweight='bold')

        # Annotate crisis years
        for year, name in self.CRISIS_YEARS.items():
            if year in years:
                ax1.annotate(name, (year, combined.loc[year, 'advantage']),
                           textcoords="offset points", xytext=(0, 10),
                           ha='center', fontsize=8, rotation=45)

        # Plot 2: Portfolio difference (Sentiment - RL)
        ax2 = axes[1]
        colors2 = ['#3498db' if x < 0 else '#e67e22' for x in combined['portfolio_diff']]
        ax2.bar(years, combined['portfolio_diff'], color=colors2, alpha=0.8, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('Portfolio Choice Diff\n(Sentiment RL - RL)\nNegative = More Conservative', fontsize=11)

        # Plot 3: VIX level for context
        ax3 = axes[2]
        ax3.fill_between(years, 0, combined['vix_level'], alpha=0.4, color='gray')
        ax3.plot(years, combined['vix_level'], color='black', linewidth=1)
        ax3.axhline(self.VIX_TURMOIL_MAX, color='orange', linestyle='--', alpha=0.7)
        ax3.set_ylabel('VIX Level', fontsize=11)
        ax3.set_xlabel('Year', fontsize=12)

        # Highlight crisis years
        for year in self.CRISIS_YEARS.keys():
            if year in years:
                for ax in axes:
                    ax.axvspan(year - 0.4, year + 0.4, alpha=0.15, color='red')

        plt.tight_layout()
        plt.savefig(output_dir / 'sentiment_advantage.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: sentiment_advantage.png")

    def _plot_goal_timing_analysis(self, df: 'pd.DataFrame', output_dir: Path):
        """Analyze goal-taking decisions during crisis vs normal periods"""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Filter to only rows where goals were available
        goal_df = df[df['goal_available'] == True].copy()

        if goal_df.empty:
            logger.warning("No goal data available for timing analysis")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        methods = ['DP', 'RL', 'Sentiment RL']

        # Plot 1: Goal take rate by regime
        ax1 = axes[0]
        regimes = ['tranquil', 'turmoil', 'crisis']
        x = np.arange(len(regimes))
        width = 0.25

        for i, method in enumerate(methods):
            if method in goal_df['method'].unique():
                method_df = goal_df[goal_df['method'] == method]
                take_rates = []
                for regime in regimes:
                    regime_df = method_df[method_df['vix_regime'] == regime]
                    if len(regime_df) > 0:
                        take_rates.append(regime_df['goal_taken'].mean() * 100)
                    else:
                        take_rates.append(0)
                ax1.bar(x + i * width, take_rates, width, label=method, alpha=0.8)

        ax1.set_ylabel('Goal Take Rate (%)')
        ax1.set_title('Goal-Taking Rate by VIX Regime')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['Tranquil\n(VIX<18)', 'Turmoil\n(18≤VIX<30)', 'Crisis\n(VIX≥30)'])
        ax1.legend()
        ax1.set_ylim(0, 100)

        # Plot 2: Goal take rate - crisis vs non-crisis years
        ax2 = axes[1]
        crisis_years_set = set(self.CRISIS_YEARS.keys())
        goal_df['is_crisis_year'] = goal_df['calendar_year'].isin(crisis_years_set)

        categories = ['Non-Crisis Years', 'Crisis Years']
        x = np.arange(len(categories))

        for i, method in enumerate(methods):
            if method in goal_df['method'].unique():
                method_df = goal_df[goal_df['method'] == method]
                take_rates = [
                    method_df[~method_df['is_crisis_year']]['goal_taken'].mean() * 100,
                    method_df[method_df['is_crisis_year']]['goal_taken'].mean() * 100
                ]
                ax2.bar(x + i * width, take_rates, width, label=method, alpha=0.8)

        ax2.set_ylabel('Goal Take Rate (%)')
        ax2.set_title('Goal-Taking Rate: Crisis vs Non-Crisis Years')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.set_ylim(0, 100)

        plt.suptitle('Goal Decision Timing Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'goal_timing_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: goal_timing_analysis.png")

    def _plot_performance_heatmap(self, df: 'pd.DataFrame', output_dir: Path):
        """Create heatmap of performance by year and method"""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Pivot to create year x method matrix
        pivot = df.pivot_table(
            values='wealth_change_pct',
            index='calendar_year',
            columns='method',
            aggfunc='mean'
        )

        if pivot.empty:
            logger.warning("No data for heatmap")
            return

        # Select methods to show
        methods_order = ['DP', 'RL', 'Sentiment RL', 'Buy & Hold', 'Conservative', 'Greedy Goal']
        available_methods = [m for m in methods_order if m in pivot.columns]
        pivot = pivot[available_methods]

        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.3)))

        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)

        # Set ticks
        ax.set_xticks(np.arange(len(available_methods)))
        ax.set_xticklabels(available_methods, rotation=45, ha='right')

        # Show every 2nd year for readability
        years = pivot.index.tolist()
        ax.set_yticks(np.arange(0, len(years), 2))
        ax.set_yticklabels([years[i] for i in range(0, len(years), 2)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Wealth Change (%)', fontsize=11)

        # Highlight crisis years
        for i, year in enumerate(years):
            if year in self.CRISIS_YEARS:
                ax.axhline(i - 0.5, color='red', linewidth=2)
                ax.axhline(i + 0.5, color='red', linewidth=2)
                ax.text(-0.5, i, f'← {self.CRISIS_YEARS[year]}', va='center', ha='right',
                       fontsize=8, color='red', fontweight='bold')

        ax.set_title('Yearly Wealth Change by Strategy\n(Crisis years highlighted in red)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Year')

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved: performance_heatmap.png")

    def _save_crisis_statistics(self, df: 'pd.DataFrame', output_dir: Path):
        """Save detailed crisis statistics to JSON"""
        import json

        stats = {
            'crisis_years': list(self.CRISIS_YEARS.keys()),
            'methods_analyzed': list(df['method'].unique()),
            'year_range': [int(df['calendar_year'].min()), int(df['calendar_year'].max())],
            'regime_summary': {},
            'crisis_year_performance': {},
            'sentiment_advantage': {}
        }

        # Regime summary
        for regime in ['tranquil', 'turmoil', 'crisis']:
            regime_df = df[df['vix_regime'] == regime]
            stats['regime_summary'][regime] = {
                'num_observations': len(regime_df),
                'by_method': {}
            }
            for method in df['method'].unique():
                method_df = regime_df[regime_df['method'] == method]
                if len(method_df) > 0:
                    stats['regime_summary'][regime]['by_method'][method] = {
                        'avg_wealth_change': float(method_df['wealth_change_pct'].mean()),
                        'avg_portfolio': float(method_df['portfolio_choice'].mean()),
                        'goal_take_rate': float(method_df[method_df['goal_available']]['goal_taken'].mean())
                                         if len(method_df[method_df['goal_available']]) > 0 else None
                    }

        # Crisis year performance
        for year in self.CRISIS_YEARS.keys():
            if year in df['calendar_year'].unique():
                year_df = df[df['calendar_year'] == year]
                stats['crisis_year_performance'][str(year)] = {
                    'crisis_name': self.CRISIS_YEARS[year],
                    'avg_vix': float(year_df['vix_level'].mean()),
                    'by_method': {}
                }
                for method in df['method'].unique():
                    method_df = year_df[year_df['method'] == method]
                    if len(method_df) > 0:
                        stats['crisis_year_performance'][str(year)]['by_method'][method] = {
                            'wealth_change_pct': float(method_df['wealth_change_pct'].mean()),
                            'portfolio_choice': float(method_df['portfolio_choice'].mean())
                        }

        # Sentiment advantage
        if 'RL' in df['method'].unique() and 'Sentiment RL' in df['method'].unique():
            rl_avg = df[df['method'] == 'RL'].groupby('vix_regime')['wealth_change_pct'].mean()
            sent_avg = df[df['method'] == 'Sentiment RL'].groupby('vix_regime')['wealth_change_pct'].mean()

            for regime in ['tranquil', 'turmoil', 'crisis']:
                if regime in rl_avg.index and regime in sent_avg.index:
                    stats['sentiment_advantage'][regime] = {
                        'rl_avg': float(rl_avg[regime]),
                        'sentiment_avg': float(sent_avg[regime]),
                        'advantage': float(sent_avg[regime] - rl_avg[regime])
                    }

        # Save
        with open(output_dir / 'crisis_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info("  Saved: crisis_statistics.json")


@dataclass
class ComparisonConfig:
    """Configuration for comparison experiment"""
    num_simulations: int = 100000
    goal_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    base_seed: int = 42
    time_horizon: int = 16  # Years (DP still uses yearly for tractability)
    num_portfolios: int = 15

    # Training config
    num_iterations: int = 10
    batch_size: int = 4800
    learning_rate: float = 0.01

    # Architecture config (from presets)
    policy_type: str = "standard"
    value_type: str = "standard"
    encoder_type: str = "feature"
    hidden_dim: int = 64
    use_batch_norm: bool = False
    dropout_rate: float = 0.0

    # DP config (grid_density: higher = more accurate, 0.5=fast, 1.5=default, 3.0=paper)
    dp_grid_density: float = 1.5

    # Data mode
    data_mode: str = "simulation"

    # Flags
    skip_training: bool = False
    skip_dp: bool = False
    skip_sentiment: bool = False
    quick_mode: bool = False

    # Rolling Efficient Frontier settings
    use_rolling_ef: bool = False  # Use rolling EF for historical backtesting
    ef_lookback_years: int = 20   # Years of data for EF calculation
    ef_min_years: int = 10        # Minimum years required for EF calculation

    # Crisis analysis settings
    crisis_analysis: bool = False  # Enable year-by-year crisis analysis

    # Monthly time step settings (NEW: correct VIX causality)
    # When True: Both Pure RL and Sentiment RL use monthly time steps (192 steps)
    # This provides more granular VIX learning opportunities
    use_monthly_steps: bool = True
    months_per_year: int = 12

    # VIX model parameters for correct VIX → Returns causality
    # VIX PREDICTS returns via β(VIX) and δ(VIX)
    # μ_adj = μ + β, σ_adj = σ - δ
    vix_beta_sensitivity: float = 0.03   # How much VIX affects μ
    vix_delta_sensitivity: float = 0.05  # How much VIX affects σ
    vix_kappa: float = 3.0               # VIX mean reversion speed
    vix_theta: float = 20.0              # VIX long-term mean

    # Sentiment adjustment settings (Option A vs Option B)
    # False (default) = Option A: Sentiment as pure information advantage
    #   - ALL methods use base mu/sigma for wealth evolution
    #   - Sentiment only affects agent's DECISIONS (portfolio choice, goal timing)
    #   - Fair comparison: same market dynamics for DP, RL, Sentiment RL
    # True = Option B: VIX-adjusted market dynamics with correct causality
    #   - Sentiment adjusts mu/sigma via β(VIX) and δ(VIX)
    #   - μ_adj = μ + β, σ_adj = σ - δ (professor's formula)
    #   - Pure RL uses base μ, σ (doesn't see VIX)
    #   - Sentiment RL uses μ_adj, σ_adj (knows VIX at beginning of period)
    use_sentiment_adjusted_returns: bool = False

    # Pre-training cache settings for efficient frontier and β/δ
    # Date ranges for computing efficient frontier from yfinance
    ef_start_date: str = "2003-01-01"  # Start date for efficient frontier data
    ef_end_date: Optional[str] = None  # End date (None = today)
    # Date ranges for learning β/δ from VIX/returns relationship
    beta_delta_start_date: str = "1990-01-02"  # Start date for β/δ learning
    beta_delta_end_date: Optional[str] = None  # End date (None = today)
    # Cache behavior
    use_pretrain_cache: bool = True  # Use cached efficient frontier and β/δ
    force_pretrain_recompute: bool = False  # Force recomputation even if cache exists

    # Use SentimentGBWMTrainer for sentiment training
    use_sentiment_trainer: bool = True  # Use new trainer vs legacy approach

    device: str = "cpu"


# =============================================================================
# BENCHMARK STRATEGIES
# =============================================================================

class BenchmarkStrategy(ABC):
    """Abstract base class for benchmark strategies"""

    def __init__(self, name: str, num_portfolios: int = 15):
        self.name = name
        self.num_portfolios = num_portfolios

    @abstractmethod
    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        """Get action (goal_action, portfolio_action)"""
        pass

    def reset(self):
        pass


class RandomStrategy(BenchmarkStrategy):
    """Random investment and goal decisions"""

    def __init__(self, num_portfolios: int = 15):
        super().__init__("Random", num_portfolios)

    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        return rng.randint(0, 2), rng.randint(0, self.num_portfolios)


class BuyAndHoldStrategy(BenchmarkStrategy):
    """Fixed portfolio with random goal decisions"""

    def __init__(self, num_portfolios: int = 15, hold_portfolio: int = 7):
        super().__init__("Buy & Hold", num_portfolios)
        self.hold_portfolio = hold_portfolio

    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        return rng.randint(0, 2), self.hold_portfolio


class GreedyGoalStrategy(BenchmarkStrategy):
    """Take goals when affordable, random portfolio"""

    def __init__(self, num_portfolios: int = 15, wealth_threshold: float = 1.2):
        super().__init__("Greedy Goal", num_portfolios)
        self.wealth_threshold = wealth_threshold

    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        goal_action = 0
        if info.get('goal_available', False):
            current_wealth = info.get('current_wealth', 0)
            goal_cost = info.get('goal_cost', float('inf'))
            if current_wealth >= self.wealth_threshold * goal_cost:
                goal_action = 1
        return goal_action, rng.randint(0, self.num_portfolios)


class ConservativeStrategy(BenchmarkStrategy):
    """Low-risk portfolio, cautious goal-taking"""

    def __init__(self, num_portfolios: int = 15, conservative_portfolio: int = 2, wealth_buffer: float = 1.5):
        super().__init__("Conservative", num_portfolios)
        self.conservative_portfolio = conservative_portfolio
        self.wealth_buffer = wealth_buffer

    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        goal_action = 0
        if info.get('goal_available', False):
            current_wealth = info.get('current_wealth', 0)
            goal_cost = info.get('goal_cost', float('inf'))
            if current_wealth >= self.wealth_buffer * goal_cost:
                goal_action = 1
        return goal_action, self.conservative_portfolio


class AggressiveStrategy(BenchmarkStrategy):
    """High-risk portfolio, aggressive goal-taking"""

    def __init__(self, num_portfolios: int = 15, aggressive_portfolio: int = 12):
        super().__init__("Aggressive", num_portfolios)
        self.aggressive_portfolio = aggressive_portfolio

    def get_action(self, state: np.ndarray, info: Dict, rng: np.random.RandomState) -> Tuple[int, int]:
        goal_action = 0
        if info.get('goal_available', False):
            current_wealth = info.get('current_wealth', 0)
            goal_cost = info.get('goal_cost', float('inf'))
            if current_wealth >= goal_cost:
                goal_action = 1
        return goal_action, self.aggressive_portfolio


# =============================================================================
# VIX REGIME-SWITCHING SIMULATOR
# =============================================================================

@dataclass
class VIXRegimeParams:
    """Parameters for a single VIX regime"""
    mean: float              # Long-term mean VIX level for this regime
    std: float               # Base volatility of VIX in this regime
    mean_reversion: float    # Speed of mean reversion (kappa)
    jump_intensity: float    # Probability of jump per time step
    jump_mean: float         # Mean jump size
    jump_std: float          # Jump size standard deviation


class VIXRegimeSimulator:
    """
    Regime-switching VIX simulator with mean reversion and jumps.

    Based on academic research:
    - Baba & Sakurai (2011): Three distinct VIX regimes (tranquil, turmoil, crisis)
    - Papanicolaou & Sircar (2014): Regime-switching Heston model for VIX
    - Jump clustering during financial crises (2008, COVID)

    References:
    - https://onlinelibrary.wiley.com/doi/10.1002/fut.70041 (Markov-Switching GARCH)
    - https://economics.princeton.edu/published-papers/a-regime-switching-heston-model-for-vix-and-sp-500-implied-volatilities/
    - https://link.springer.com/article/10.1007/s11156-009-0153-8 (Jump-diffusion VIX)
    """

    # Regime definitions based on empirical research
    # Parameters calibrated for ANNUAL time steps (dt=1.0 year):
    # - Historical VIX average: ~18-20
    # - Calm periods: VIX 12-18
    # - Stress periods: VIX 20-35
    # - Crisis periods: VIX 35-80
    # Note: Lower noise/jump params since each step = 1 year
    REGIMES = {
        'tranquil': VIXRegimeParams(
            mean=15.0,           # Low VIX environment (calm markets)
            std=1.5,             # Annual VIX volatility within regime
            mean_reversion=0.8,  # Slow mean reversion (VIX stays stable in calm)
            jump_intensity=0.05, # Low jump probability (5% per year)
            jump_mean=2.0,       # Small jumps
            jump_std=1.0
        ),
        'turmoil': VIXRegimeParams(
            mean=28.0,           # Elevated VIX (market stress)
            std=3.0,             # Moderate VIX volatility
            mean_reversion=0.5,  # Slower mean reversion
            jump_intensity=0.12, # More frequent jumps
            jump_mean=4.0,       # Medium jumps
            jump_std=2.0
        ),
        'crisis': VIXRegimeParams(
            mean=55.0,           # Crisis VIX (2008: peaked 80, COVID: peaked 82)
            std=6.0,             # High VIX volatility
            mean_reversion=0.3,  # Very slow mean reversion (fear persists)
            jump_intensity=0.25, # Frequent jumps during crises
            jump_mean=8.0,       # Large jumps
            jump_std=4.0
        )
    }

    # Transition matrix (annual probabilities)
    # Based on Baba & Sakurai (2011) empirical findings
    # Rows: from regime, Columns: to regime [tranquil, turmoil, crisis]
    BASE_TRANSITION_MATRIX = np.array([
        #  Tranquil  Turmoil  Crisis
        [    0.85,    0.12,    0.03],  # From Tranquil: mostly stays calm
        [    0.20,    0.70,    0.10],  # From Turmoil: can calm down or escalate
        [    0.05,    0.25,    0.70],  # From Crisis: persistent, slow recovery
    ])

    REGIME_NAMES = ['tranquil', 'turmoil', 'crisis']
    REGIME_TO_IDX = {'tranquil': 0, 'turmoil': 1, 'crisis': 2}

    # VIX-SPX correlation for Cholesky coupling
    # Empirically ~-0.70 (VIX rises when market falls)
    RHO = -0.70

    def __init__(self, base_seed: int = 42, dt: float = 1.0, use_cholesky_coupling: bool = True):
        """
        Initialize VIX regime simulator.

        Args:
            base_seed: Base random seed for reproducibility
            dt: Time step in years (default 1.0 for annual)
            use_cholesky_coupling: If True, couple VIX diffusion to market shocks
                                   via Cholesky decomposition. This ensures the
                                   model learns the true VIX-return relationship.
        """
        self.base_seed = base_seed
        self.dt = dt
        self.use_cholesky_coupling = use_cholesky_coupling
        self._rng = None
        self._current_regime = 'tranquil'
        self._current_vix = 18.0
        self._jump_cluster_count = 0  # For Hawkes-like clustering

    def reset(self, sim_idx: int, initial_vix: float = None) -> None:
        """
        Reset simulator for a new trajectory.

        Args:
            sim_idx: Simulation index for reproducible randomness
            initial_vix: Starting VIX level (default: regime-appropriate)
        """
        self._rng = np.random.RandomState(self.base_seed + sim_idx)

        # Determine initial regime based on initial VIX
        if initial_vix is not None:
            self._current_vix = initial_vix
            self._current_regime = self._classify_regime(initial_vix)
        else:
            # Start in tranquil regime with typical VIX
            self._current_regime = 'tranquil'
            self._current_vix = self.REGIMES['tranquil'].mean + self._rng.normal(0, 2)
            self._current_vix = np.clip(self._current_vix, 10, 80)

        self._jump_cluster_count = 0

    def _classify_regime(self, vix_level: float) -> str:
        """Classify VIX level into regime"""
        if vix_level < 18:
            return 'tranquil'
        elif vix_level < 30:
            return 'turmoil'
        else:
            return 'crisis'

    def _transition_regime(self, market_shock: float) -> str:
        """
        Transition to next regime based on market conditions.

        Market shocks influence regime transitions:
        - Severe negative shocks increase probability of crisis
        - Positive shocks increase probability of returning to tranquil

        Args:
            market_shock: Z-score of market return shock

        Returns:
            New regime name
        """
        current_idx = self.REGIME_TO_IDX[self._current_regime]
        probs = self.BASE_TRANSITION_MATRIX[current_idx].copy()

        # Market shock adjustments (asymmetric: crashes have bigger impact)
        if market_shock < -2.5:
            # Severe crash (>2.5 std): high probability of crisis
            probs[2] += 0.40  # Much more likely to enter/stay in crisis
            probs[0] -= 0.25  # Less likely to be tranquil
            probs[1] -= 0.15
        elif market_shock < -1.5:
            # Significant drop (1.5-2.5 std): elevated stress
            probs[2] += 0.25
            probs[1] += 0.10
            probs[0] -= 0.35
        elif market_shock < -0.5:
            # Moderate drop (0.5-1.5 std): mild stress increase
            probs[1] += 0.15
            probs[0] -= 0.10
            probs[2] += 0.05
        elif market_shock > 1.5:
            # Strong rally: calming effect
            probs[0] += 0.20
            probs[2] -= 0.15
            probs[1] -= 0.05
        elif market_shock > 0.5:
            # Moderate rally: slight calming
            probs[0] += 0.10
            probs[2] -= 0.05
            probs[1] -= 0.05

        # Jump clustering effect: recent jumps increase future jump probability
        # This creates the "jumps beget jumps" pattern seen in crises
        if self._jump_cluster_count > 0:
            probs[2] += 0.05 * min(self._jump_cluster_count, 3)
            probs[0] -= 0.05 * min(self._jump_cluster_count, 3)

        # Ensure probabilities are valid
        probs = np.clip(probs, 0.01, 0.99)  # Keep small probability for all transitions
        probs = probs / probs.sum()  # Normalize

        # Sample new regime
        new_regime = self._rng.choice(self.REGIME_NAMES, p=probs)
        return new_regime

    def simulate_step(self, market_shock: float) -> Tuple[float, float, float]:
        """
        Simulate one time step of VIX evolution.

        Uses mean-reverting jump-diffusion within current regime:
        dV = κ(θ - V)dt + σ dW + J dN

        Args:
            market_shock: Z-score of market return shock (for regime transitions)

        Returns:
            Tuple of (vix_level, vix_sentiment, vix_momentum)
            - vix_level: Raw VIX value [10, 80]
            - vix_sentiment: Normalized sentiment [-1, 1] where +1 = fear (high VIX)
            - vix_momentum: Rate of VIX change indicator [-1, 1]
        """
        # Store previous VIX for momentum calculation
        prev_vix = self._current_vix

        # 1. Regime transition (influenced by market)
        self._current_regime = self._transition_regime(market_shock)
        params = self.REGIMES[self._current_regime]

        # 2. Mean-reverting diffusion (Ornstein-Uhlenbeck style)
        # dV = κ(θ - V)dt + σ dW
        kappa = params.mean_reversion
        theta = params.mean
        sigma = params.std

        # Drift: mean reversion (pulls VIX toward regime mean)
        drift = kappa * (theta - self._current_vix) * self.dt

        # Diffusion: either coupled to market shock (Cholesky) or independent
        if self.use_cholesky_coupling:
            # Cholesky decomposition: Z_vix = rho * Z_market + sqrt(1-rho^2) * Z_indep
            # This creates the negative correlation between VIX and market returns:
            # - When market_shock < 0 (bad market day), z_vix tends to be positive (VIX rises)
            # - When market_shock > 0 (good market day), z_vix tends to be negative (VIX falls)
            z_indep = self._rng.normal(0, 1)
            z_vix = self.RHO * market_shock + np.sqrt(1 - self.RHO**2) * z_indep
            diffusion = sigma * z_vix * np.sqrt(self.dt)
        else:
            # Independent diffusion (original behavior)
            diffusion = sigma * self._rng.normal(0, np.sqrt(self.dt))

        # 3. Jump component (Poisson with clustering)
        # Adjust jump intensity based on clustering and market shock
        effective_intensity = params.jump_intensity

        # Hawkes-like clustering: recent jumps increase future jump probability
        if self._jump_cluster_count > 0:
            effective_intensity *= (1.0 + 0.4 * self._jump_cluster_count)

        # Negative market shocks increase jump probability
        if market_shock < -1:
            effective_intensity *= (1.0 + abs(market_shock) * 0.25)

        # Sample jump occurrence
        jump = 0.0
        if self._rng.random() < effective_intensity * self.dt:
            # Jump occurred - sample from exponential (mostly upward jumps in VIX)
            jump_size = self._rng.exponential(params.jump_mean)

            # Add some downward jumps in recovery periods
            if self._current_regime == 'tranquil' and market_shock > 0:
                # 30% chance of downward jump in calm markets with positive returns
                if self._rng.random() < 0.3:
                    jump_size = -abs(self._rng.normal(0, params.jump_std * 0.3))

            jump = jump_size
            self._jump_cluster_count = min(self._jump_cluster_count + 1, 5)
        else:
            # No jump - decay clustering effect
            self._jump_cluster_count = max(0, self._jump_cluster_count - 0.3)

        # 4. Update VIX
        self._current_vix = self._current_vix + drift + diffusion + jump

        # 5. Apply bounds (historical VIX range)
        self._current_vix = np.clip(self._current_vix, 10, 80)

        # 6. Calculate outputs
        vix_level = self._current_vix

        # Sentiment: normalized to [-1, 1] with +1 = fear (high VIX)
        # Based on long-term mean of ~20 and typical range of 10-80
        vix_sentiment = np.clip((vix_level - 20) / 30, -1, 1)

        # Momentum: rate of change, normalized
        # Positive = VIX rising (increasing fear), negative = VIX falling
        # Use gentler scaling to avoid hitting ±1 constantly
        raw_momentum = (vix_level - prev_vix) / max(prev_vix, 12)
        vix_momentum = np.clip(raw_momentum * 3, -1, 1)

        return vix_level, vix_sentiment, vix_momentum

    def simulate_trajectory(self, market_shocks: np.ndarray, sim_idx: int,
                           initial_vix: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate full VIX trajectory.

        Args:
            market_shocks: Array of market return Z-scores
            sim_idx: Simulation index for reproducibility
            initial_vix: Starting VIX level

        Returns:
            Tuple of arrays (vix_levels, vix_sentiments, vix_momentums)
        """
        n_steps = len(market_shocks)
        vix_levels = np.zeros(n_steps)
        vix_sentiments = np.zeros(n_steps)
        vix_momentums = np.zeros(n_steps)

        self.reset(sim_idx, initial_vix)

        for t in range(n_steps):
            vix_levels[t], vix_sentiments[t], vix_momentums[t] = self.simulate_step(market_shocks[t])

        return vix_levels, vix_sentiments, vix_momentums

    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulator state for debugging"""
        return {
            'regime': self._current_regime,
            'vix': self._current_vix,
            'jump_cluster_count': self._jump_cluster_count
        }

    def get_current_vix(self) -> float:
        """Get current VIX level"""
        return self._current_vix


# =============================================================================
# COUPLED VIX SIMULATOR (for Training with VIX-Return Correlation)
# =============================================================================

class CoupledVIXSimulator:
    """
    VIX simulator with correlated market returns for training.

    Key insight: Same market shock Z drives both VIX changes and wealth returns.
    This creates the coupling that allows the model to learn the true relationship.

    VIX dynamics:
        dV = kappa * (theta - V) * dt + sigma_v * sqrt(V) * (rho * Z + sqrt(1-rho^2) * Z_indep) + J * dN

    Where:
        - Z is the market shock (shared with wealth evolution)
        - rho ~ -0.7 (VIX-SPX correlation, negative: market down -> VIX up)
        - Z_indep is independent noise for VIX-specific factors
        - J * dN is a jump process for crisis events

    Wealth dynamics (GBM with sentiment-adjusted mu/sigma):
        W(t+1) = W(t) * exp((mu_adj - 0.5*sigma_adj^2) + sigma_adj * Z)

    Reference: Whaley (2009), "Understanding the VIX"
    """

    # Calibration parameters (from empirical research)
    KAPPA = 3.0        # Mean reversion speed (annual)
    THETA = 20.0       # Long-term VIX mean
    SIGMA_V = 0.8      # VIX volatility of volatility
    RHO = -0.70        # VIX-SPX correlation (strongly negative)
    JUMP_INTENSITY = 0.10  # 10% annual jump probability
    JUMP_MEAN = 15.0   # Mean jump size
    JUMP_STD = 8.0     # Jump size volatility

    def __init__(self, base_seed: int = 42):
        """
        Initialize coupled VIX simulator.

        Args:
            base_seed: Base random seed for reproducibility
        """
        self.base_seed = base_seed
        self._rng = None
        self._current_vix = 20.0
        self._prev_vix = 20.0

    def reset(self, sim_idx: int, initial_vix: float = 20.0):
        """
        Reset simulator for new trajectory.

        Args:
            sim_idx: Simulation index for reproducible randomness
            initial_vix: Starting VIX level
        """
        self._rng = np.random.RandomState(self.base_seed + sim_idx)
        self._current_vix = initial_vix
        self._prev_vix = initial_vix

    def simulate_step(self, market_shock_z: float) -> Tuple[float, float, float]:
        """
        Simulate one VIX step with coupling to market shock.

        This is the KEY method that creates the VIX-return coupling:
        - Uses the SAME Z shock that drives wealth evolution
        - VIX responds negatively to positive market shocks (rho ~ -0.7)

        Args:
            market_shock_z: The same Z used for wealth evolution (N(0,1))

        Returns:
            (vix_level, vix_sentiment, vix_momentum)
            - vix_level: Raw VIX value
            - vix_sentiment: Normalized to [-1, 1], high VIX = negative sentiment
            - vix_momentum: Rate of change normalized to [-1, 1]
        """
        dt = 1.0  # Annual timestep
        V = self._current_vix
        self._prev_vix = V

        # Generate VIX-correlated shock using Cholesky decomposition
        # Z_vix = rho * Z_market + sqrt(1 - rho^2) * Z_independent
        z_indep = self._rng.normal(0, 1)
        z_vix = self.RHO * market_shock_z + np.sqrt(1 - self.RHO**2) * z_indep

        # Mean reversion drift: pulls VIX toward long-term mean
        drift = self.KAPPA * (self.THETA - V) * dt

        # Diffusion with square-root process (CIR-like, ensures positivity)
        diffusion = self.SIGMA_V * np.sqrt(max(V, 1.0)) * z_vix * np.sqrt(dt)

        # Jump component (Poisson process for crisis events)
        jump = 0.0
        if self._rng.random() < self.JUMP_INTENSITY * dt:
            # Jump occurred - mostly upward for VIX
            jump = self._rng.normal(self.JUMP_MEAN, self.JUMP_STD)
            jump = max(jump, -V + 10)  # Don't let VIX go below 10

        # Update VIX
        new_vix = V + drift + diffusion + jump
        new_vix = np.clip(new_vix, 10.0, 80.0)  # Historical bounds

        # Calculate sentiment feature
        # High VIX = fear = negative sentiment for investing
        vix_sentiment = -np.clip((new_vix - self.THETA) / 30.0, -1.0, 1.0)

        # Calculate momentum feature (rate of change)
        vix_change = (new_vix - V) / max(V, 12.0)
        vix_momentum = np.clip(vix_change * 3.0, -1.0, 1.0)

        self._current_vix = new_vix
        return new_vix, vix_sentiment, vix_momentum

    def get_current_vix(self) -> float:
        """Get current VIX level"""
        return self._current_vix


# =============================================================================
# ROLLING EFFICIENT FRONTIER (for Dynamic Weight Calculation)
# =============================================================================

class RollingEfficientFrontier:
    """
    Calculate efficient frontier weights using rolling historical windows.

    Instead of using FIXED weights, this calculates optimal portfolio weights
    using only PAST data at each point in time, avoiding look-ahead bias.

    For a backtest at year T:
        - Use data from [T - lookback, T - 1] to calculate covariance/returns
        - Solve mean-variance optimization for each risk level
        - Apply resulting weights to year T returns

    This is more realistic than fixed weights because:
        1. Correlations and volatilities change over time
        2. An investor would rebalance based on recent market conditions
        3. No look-ahead bias (only uses past data)
    """

    # Asset indices
    BONDS = 0
    US_STOCKS = 1
    INTL_STOCKS = 2

    def __init__(self, lookback_years: int = 20, num_portfolios: int = 15,
                 min_lookback: int = 10, risk_free_rate: float = 0.02):
        """
        Initialize rolling efficient frontier calculator.

        Args:
            lookback_years: Years of historical data to use for optimization
            num_portfolios: Number of portfolios on efficient frontier
            min_lookback: Minimum years required (for early windows)
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.lookback_years = lookback_years
        self.num_portfolios = num_portfolios
        self.min_lookback = min_lookback
        self.risk_free_rate = risk_free_rate

        # Cache for computed weights
        self._weight_cache: Dict[int, np.ndarray] = {}

        # Asset return data (will be set by load_data)
        self.bond_returns: np.ndarray = None
        self.stock_returns: np.ndarray = None
        self.intl_returns: np.ndarray = None
        self.years: np.ndarray = None

    def load_data(self, historical_loader) -> None:
        """
        Load asset return data from historical loader.

        Args:
            historical_loader: HistoricalDataLoader instance with computed returns
        """
        # Get individual asset returns from the asset_returns dictionary
        if hasattr(historical_loader, 'asset_returns') and historical_loader.asset_returns is not None:
            asset_returns = historical_loader.asset_returns
            self.bond_returns = np.array(asset_returns['bonds'])
            self.stock_returns = np.array(asset_returns['us_stocks'])
            self.intl_returns = np.array(asset_returns['intl_stocks'])
        else:
            raise ValueError("HistoricalDataLoader must have asset_returns dict")

        # Get years (derive from data length starting from 1970)
        n_years = len(self.bond_returns)
        self.years = np.arange(1970, 1970 + n_years)

        # Store as combined array for convenience
        self.asset_returns = np.column_stack([
            self.bond_returns, self.stock_returns, self.intl_returns
        ])

        logger.info(f"RollingEF: Loaded {len(self.years)} years of data "
                   f"({self.years[0]}-{self.years[-1]})")

    def get_weights_for_year(self, target_year: int) -> np.ndarray:
        """
        Get optimal portfolio weights for a specific year using PAST data only.

        Args:
            target_year: Year for which to calculate weights (e.g., 1985)

        Returns:
            Array of shape (num_portfolios, 3) with weights for each portfolio
            Columns: [bonds, us_stocks, intl_stocks]
        """
        # Check cache
        if target_year in self._weight_cache:
            return self._weight_cache[target_year]

        # Find year index
        if target_year not in self.years:
            # Use fallback weights if year not in data
            logger.warning(f"Year {target_year} not in data, using fallback weights")
            return self._get_fallback_weights()

        year_idx = np.where(self.years == target_year)[0][0]

        # Determine lookback window (only PAST data)
        lookback_end = year_idx  # Exclusive - don't include target year
        lookback_start = max(0, lookback_end - self.lookback_years)

        # Check if we have enough data
        available_years = lookback_end - lookback_start
        if available_years < self.min_lookback:
            logger.debug(f"Year {target_year}: Only {available_years} years available, "
                        f"using fallback weights")
            return self._get_fallback_weights()

        # Get historical returns for lookback period
        bond_hist = self.bond_returns[lookback_start:lookback_end]
        stock_hist = self.stock_returns[lookback_start:lookback_end]
        intl_hist = self.intl_returns[lookback_start:lookback_end]

        # Stack into matrix (n_years, 3 assets)
        returns_matrix = np.column_stack([bond_hist, stock_hist, intl_hist])

        # Calculate mean returns and covariance
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)

        # Solve efficient frontier
        weights = self._solve_efficient_frontier(mean_returns, cov_matrix)

        # Cache result
        self._weight_cache[target_year] = weights

        return weights

    def _solve_efficient_frontier(self, mean_returns: np.ndarray,
                                   cov_matrix: np.ndarray) -> np.ndarray:
        """
        Solve mean-variance optimization to get efficient frontier portfolios.

        Uses analytical solution for the efficient frontier:
        For target return μ_p, minimize portfolio variance σ²_p
        subject to: w'μ = μ_p, w'1 = 1, w >= 0

        Args:
            mean_returns: Expected returns for each asset (3,)
            cov_matrix: Covariance matrix (3, 3)

        Returns:
            Array of shape (num_portfolios, 3) with optimal weights
        """
        n_assets = len(mean_returns)
        weights = np.zeros((self.num_portfolios, n_assets))

        # Get return range for efficient frontier
        min_return = np.min(mean_returns)
        max_return = np.max(mean_returns)

        # Target returns along efficient frontier
        target_returns = np.linspace(min_return, max_return, self.num_portfolios)

        for i, target_return in enumerate(target_returns):
            w = self._optimize_portfolio(mean_returns, cov_matrix, target_return)
            weights[i] = w

        return weights

    def _optimize_portfolio(self, mean_returns: np.ndarray,
                            cov_matrix: np.ndarray,
                            target_return: float) -> np.ndarray:
        """
        Find minimum variance portfolio for a target return.

        Uses quadratic programming with constraints:
        - minimize: w' Σ w (portfolio variance)
        - subject to: w' μ = target_return
        - subject to: sum(w) = 1
        - subject to: w >= 0 (long-only)

        Args:
            mean_returns: Expected returns (3,)
            cov_matrix: Covariance matrix (3, 3)
            target_return: Target portfolio return

        Returns:
            Optimal weights (3,)
        """
        n_assets = len(mean_returns)

        try:
            # Try scipy optimization
            from scipy.optimize import minimize

            def portfolio_variance(w):
                return w @ cov_matrix @ w

            def portfolio_return(w):
                return w @ mean_returns

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
                {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return}
            ]

            # Bounds (long-only: 0 <= w <= 1)
            bounds = [(0, 1) for _ in range(n_assets)]

            # Initial guess (equal weight)
            w0 = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-10}
            )

            if result.success:
                # Normalize to ensure sum = 1
                w = result.x
                w = np.maximum(w, 0)  # Ensure non-negative
                w = w / np.sum(w)
                return w

        except Exception as e:
            logger.debug(f"Optimization failed: {e}, using heuristic")

        # Fallback: heuristic approach
        return self._heuristic_weights(mean_returns, target_return)

    def _heuristic_weights(self, mean_returns: np.ndarray,
                           target_return: float) -> np.ndarray:
        """
        Simple heuristic for portfolio weights when optimization fails.

        Linearly interpolates between min and max return assets.
        """
        n_assets = len(mean_returns)
        min_ret = np.min(mean_returns)
        max_ret = np.max(mean_returns)

        if max_ret == min_ret:
            return np.ones(n_assets) / n_assets

        # Interpolation factor
        t = (target_return - min_ret) / (max_ret - min_ret)
        t = np.clip(t, 0, 1)

        # Assign weights based on return ranking
        sorted_idx = np.argsort(mean_returns)

        weights = np.zeros(n_assets)
        weights[sorted_idx[0]] = 1 - t  # Lowest return asset
        weights[sorted_idx[-1]] = t     # Highest return asset

        return weights

    def _get_fallback_weights(self) -> np.ndarray:
        """
        Get fallback weights when insufficient historical data.

        Uses the fixed weights from the paper.
        """
        weights = np.zeros((self.num_portfolios, 3))

        for i in range(self.num_portfolios):
            # Linear interpolation from conservative to aggressive
            t = i / (self.num_portfolios - 1)

            # Conservative: 70% bonds, 25% stocks, 5% intl
            # Aggressive: 10% bonds, 55% stocks, 35% intl
            bonds = 0.70 * (1 - t) + 0.10 * t
            stocks = 0.25 * (1 - t) + 0.55 * t
            intl = 0.05 * (1 - t) + 0.35 * t

            weights[i] = [bonds, stocks, intl]

        return weights

    def get_portfolio_return(self, year: int, portfolio_idx: int) -> float:
        """
        Get portfolio return for a specific year and portfolio using rolling weights.

        Args:
            year: Target year
            portfolio_idx: Portfolio index (0-14)

        Returns:
            Portfolio return for that year
        """
        # Get weights calculated from PAST data
        weights = self.get_weights_for_year(year)
        w = weights[portfolio_idx]

        # Find year index
        year_idx = np.where(self.years == year)[0][0]

        # Apply weights to CURRENT year returns
        portfolio_return = (
            w[0] * self.bond_returns[year_idx] +
            w[1] * self.stock_returns[year_idx] +
            w[2] * self.intl_returns[year_idx]
        )

        return portfolio_return

    def compute_all_returns(self) -> Dict[int, np.ndarray]:
        """
        Compute portfolio returns for all years using rolling EF.

        Returns:
            Dictionary {year: array(num_portfolios)} with portfolio returns
        """
        n_years = len(self.years)
        returns_dict: Dict[int, np.ndarray] = {}

        # Cache for storing the computed returns
        self._computed_returns = {}

        for year_idx, year in enumerate(self.years):
            weights = self.get_weights_for_year(int(year))

            returns = np.zeros(self.num_portfolios)
            for p in range(self.num_portfolios):
                returns[p] = (
                    weights[p, 0] * self.bond_returns[year_idx] +
                    weights[p, 1] * self.stock_returns[year_idx] +
                    weights[p, 2] * self.intl_returns[year_idx]
                )

            returns_dict[int(year)] = returns
            self._computed_returns[int(year)] = returns

        return returns_dict

    def get_weight_statistics(self, year: int) -> Dict[str, Any]:
        """Get statistics about weights for debugging."""
        weights = self.get_weights_for_year(year)

        return {
            'year': year,
            'conservative_weights': weights[0].tolist(),
            'moderate_weights': weights[7].tolist(),
            'aggressive_weights': weights[14].tolist(),
            'avg_bond_weight': np.mean(weights[:, 0]),
            'avg_stock_weight': np.mean(weights[:, 1]),
            'avg_intl_weight': np.mean(weights[:, 2])
        }


# =============================================================================
# HISTORICAL BACKTESTER (for Evaluation on Real Data)
# =============================================================================

class HistoricalBacktester:
    """
    Deterministic backtester using real historical data.

    Uses 39 overlapping 16-year windows from 1970-2023:
        - Window 0: 1970-1985
        - Window 1: 1971-1986
        - ...
        - Window 38: 2008-2023

    For each window:
        - Uses REAL portfolio returns (no sentiment adjustment)
        - Uses REAL VIX scores (from 1990+, or estimated for earlier)
        - NO randomness - pure historical replay

    This is "Historical Backtesting" not "Monte Carlo" since there's no randomness.

    Efficient Frontier Modes:
        - Fixed: Uses static weights computed from full historical period (faster)
        - Rolling: Recomputes weights using only past data at each point (more realistic)
    """

    def __init__(self, historical_data_path: str = "data/raw/market_data/",
                 use_rolling_ef: bool = False,
                 ef_lookback_years: int = 20,
                 ef_min_years: int = 10):
        """
        Initialize historical backtester.

        Args:
            historical_data_path: Path to historical market data
            use_rolling_ef: If True, use rolling EF calculation (more realistic)
                           If False, use fixed weights (faster, default)
            ef_lookback_years: Years of data to use for rolling EF (default 20)
            ef_min_years: Minimum years required for EF calculation (default 10)
        """
        self.data_path = historical_data_path
        self.use_rolling_ef = use_rolling_ef
        self.ef_lookback_years = ef_lookback_years
        self.ef_min_years = ef_min_years

        self.returns_data = None  # Shape: (54 years, 15 portfolios) - with fixed weights
        self.years = None         # [1970, 1971, ..., 2023]
        self.vix_data = None      # {year: {'vix_level', 'sentiment', 'momentum'}}
        self._loaded = False

        # Rolling EF calculator (initialized on demand)
        self._rolling_ef = None
        self._rolling_ef_returns = None  # Cached rolling EF returns

    def _ensure_loaded(self):
        """Lazy load data on first use"""
        if self._loaded:
            return

        try:
            loader = self._load_data()
            self._loaded = True

            # Initialize rolling EF if enabled
            if self.use_rolling_ef and loader is not None:
                self._init_rolling_ef(loader)

        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            self._create_fallback_data()
            self._loaded = True

    def _init_rolling_ef(self, historical_loader):
        """Initialize rolling efficient frontier calculator"""
        logger.info("Initializing Rolling Efficient Frontier calculator...")

        self._rolling_ef = RollingEfficientFrontier(
            lookback_years=self.ef_lookback_years,
            min_lookback=self.ef_min_years
        )

        # Load data from the historical loader
        self._rolling_ef.load_data(historical_loader)

        # Pre-compute rolling EF returns for all years
        self._rolling_ef_returns = self._rolling_ef.compute_all_returns()

        logger.info(f"Rolling EF initialized: {len(self._rolling_ef_returns)} years of returns computed")

    def _load_data(self):
        """Load and prepare historical data

        Returns:
            HistoricalDataLoader instance (for rolling EF initialization)
        """
        from src.data.historical_data_loader import HistoricalDataLoader

        # HistoricalDataLoader loads data in __init__
        loader = HistoricalDataLoader(data_path=self.data_path)

        # Get portfolio returns from loader (already computed in __init__)
        if hasattr(loader, 'portfolio_returns') and loader.portfolio_returns is not None:
            # portfolio_returns is a list of arrays, one per portfolio
            # Convert to (n_years, n_portfolios) array
            n_portfolios = len(loader.portfolio_returns)
            n_years = len(loader.portfolio_returns[0]) if n_portfolios > 0 else 54

            self.returns_data = np.zeros((n_years, n_portfolios))
            for p in range(n_portfolios):
                self.returns_data[:, p] = loader.portfolio_returns[p]
        else:
            # Fallback: compute from raw data
            self.returns_data = self._compute_portfolio_returns(loader)

        self.years = list(range(1970, 1970 + self.returns_data.shape[0]))  # Dynamic year range

        # Load VIX with 90-day momentum
        self._load_annual_vix()

        logger.info(f"Loaded historical data: {len(self.years)} years, "
                   f"{self.returns_data.shape[1]} portfolios")

        return loader  # Return loader for rolling EF initialization

    def _compute_portfolio_returns(self, loader) -> np.ndarray:
        """Compute annual portfolio returns from raw data"""
        # Portfolio weights on efficient frontier (15 portfolios)
        # From bonds (conservative) to stocks (aggressive)
        n_years = 54
        n_portfolios = 15

        returns = np.zeros((n_years, n_portfolios))

        # Get asset returns
        if hasattr(loader, 'annual_returns'):
            annual_data = loader.annual_returns
        else:
            # Fallback: use simulated returns
            np.random.seed(42)
            returns = np.random.normal(0.07, 0.15, (n_years, n_portfolios))
            for p in range(n_portfolios):
                # Increasing volatility and return for more aggressive portfolios
                mu = 0.05 + 0.03 * (p / 14)
                sigma = 0.04 + 0.16 * (p / 14)
                returns[:, p] = np.random.normal(mu, sigma, n_years)
            return returns

        # Compute portfolio returns using efficient frontier weights
        for p in range(n_portfolios):
            weight_bonds = 1.0 - p / 14.0
            weight_stocks = p / 14.0

            for y in range(n_years):
                year = 1970 + y
                if year in annual_data:
                    bond_return = annual_data[year].get('bonds', 0.05)
                    stock_return = annual_data[year].get('stocks', 0.08)
                    returns[y, p] = weight_bonds * bond_return + weight_stocks * stock_return
                else:
                    returns[y, p] = 0.05 + 0.03 * (p / 14)  # Fallback

        return returns

    def _load_annual_vix(self):
        """Load annual VIX with 3-4 month (90-day) momentum"""
        import pickle

        vix_path = Path("data/sentiment/vix_data.pkl")
        self.vix_data = {}

        if vix_path.exists():
            try:
                with open(vix_path, 'rb') as f:
                    vix_df = pickle.load(f)

                # Compute annual VIX features with 90-day momentum
                self.vix_data = self._compute_annual_vix_features(vix_df)
                logger.info(f"Loaded VIX data for {len(self.vix_data)} years")

            except Exception as e:
                logger.warning(f"Failed to load VIX data: {e}")

        # Estimate VIX for years without data (pre-1990)
        for year in self.years:
            if year not in self.vix_data:
                self.vix_data[year] = self._estimate_vix_for_year(year)

    def _compute_annual_vix_features(self, vix_df) -> Dict[int, Dict]:
        """
        Compute annual VIX features with 90-day (3-4 month) momentum.

        Returns:
            {year: {'vix_level': float, 'sentiment': float, 'momentum': float}}
        """
        annual_vix = {}

        # Handle both DataFrame with 'vix_close' column and Series
        if hasattr(vix_df, 'columns') and 'vix_close' in vix_df.columns:
            vix_series = vix_df['vix_close']
        elif hasattr(vix_df, 'columns') and 'Close' in vix_df.columns:
            vix_series = vix_df['Close']
        else:
            vix_series = vix_df  # Assume it's already a Series

        for year in range(1990, 2024):
            try:
                year_mask = vix_series.index.year == year
                if year_mask.sum() == 0:
                    continue

                year_data = vix_series[year_mask]

                # Year-end VIX
                vix_level = float(year_data.iloc[-1])

                # 90-day (3-4 month) momentum
                if len(year_data) >= 90:
                    vix_90d_ago = float(year_data.iloc[-90])
                    momentum_raw = (vix_level - vix_90d_ago) / vix_90d_ago
                elif len(year_data) >= 20:
                    # Fall back to 20-day if not enough data
                    vix_20d_ago = float(year_data.iloc[-min(20, len(year_data))])
                    momentum_raw = (vix_level - vix_20d_ago) / vix_20d_ago
                else:
                    momentum_raw = 0.0

                # Normalize features
                sentiment = -np.clip((vix_level - 20.0) / 30.0, -1.0, 1.0)
                momentum = np.clip(momentum_raw / 0.5, -1.0, 1.0)

                annual_vix[year] = {
                    'vix_level': vix_level,
                    'sentiment': sentiment,
                    'momentum': momentum
                }

            except Exception as e:
                logger.warning(f"Error processing VIX for year {year}: {e}")
                continue

        return annual_vix

    def _estimate_vix_for_year(self, year: int) -> Dict:
        """
        Estimate VIX for years without data (pre-1990).

        Uses historical market volatility and regime as proxy.
        """
        # Historical VIX estimates based on market conditions
        # Pre-1990: estimate from historical volatility patterns
        vix_estimates = {
            # 1970s: High inflation, oil crisis -> elevated VIX
            1970: 22, 1971: 18, 1972: 16, 1973: 28, 1974: 35,
            1975: 25, 1976: 18, 1977: 17, 1978: 20, 1979: 22,
            # 1980s: Volcker era, Black Monday
            1980: 24, 1981: 22, 1982: 28, 1983: 16, 1984: 17,
            1985: 15, 1986: 18, 1987: 35, 1988: 20, 1989: 17,
        }

        vix_level = vix_estimates.get(year, 20.0)

        return {
            'vix_level': vix_level,
            'sentiment': -np.clip((vix_level - 20.0) / 30.0, -1.0, 1.0),
            'momentum': 0.0  # Unknown for estimated data
        }

    def _create_fallback_data(self):
        """Create fallback data if loading fails"""
        logger.warning("Using fallback simulated data for historical backtesting")

        self.years = list(range(1970, 2024))
        n_years = len(self.years)
        n_portfolios = 15

        # Generate realistic historical-like returns
        np.random.seed(42)
        self.returns_data = np.zeros((n_years, n_portfolios))

        for p in range(n_portfolios):
            mu = 0.05 + 0.035 * (p / 14)  # 5% to 8.5%
            sigma = 0.04 + 0.16 * (p / 14)  # 4% to 20%
            self.returns_data[:, p] = np.random.normal(mu, sigma, n_years)

        # Create VIX data
        self.vix_data = {}
        for year in self.years:
            self.vix_data[year] = self._estimate_vix_for_year(year)

    def get_num_windows(self, horizon: int = 16) -> int:
        """
        Get number of available historical windows.

        Args:
            horizon: Time horizon (default 16 years)

        Returns:
            Number of windows (54 - 16 + 1 = 39 for default)
        """
        self._ensure_loaded()
        return len(self.years) - horizon + 1

    def get_window(self, window_idx: int, horizon: int = 16) -> Dict:
        """
        Get returns and VIX for a specific historical window.

        Args:
            window_idx: 0 to 38 for 16-year windows
            horizon: Time horizon (16 years)

        Returns:
            {
                'returns': np.ndarray (horizon, 15),  # Portfolio returns by year
                'vix_features': List[Dict],  # VIX features per year
                'start_year': int,
                'end_year': int,
                'ef_method': str  # 'fixed' or 'rolling'
            }
        """
        self._ensure_loaded()

        if window_idx < 0 or window_idx >= self.get_num_windows(horizon):
            raise ValueError(f"Invalid window_idx {window_idx}. Valid range: 0-{self.get_num_windows(horizon)-1}")

        start_idx = window_idx
        end_idx = start_idx + horizon

        # Get returns based on EF method
        if self.use_rolling_ef and self._rolling_ef_returns is not None:
            # Use rolling EF returns (computed with weights from past data only)
            returns = np.zeros((horizon, 15))
            for i in range(horizon):
                year = self.years[start_idx + i]
                if year in self._rolling_ef_returns:
                    returns[i] = self._rolling_ef_returns[year]
                else:
                    # Fallback to fixed weights for early years
                    returns[i] = self.returns_data[start_idx + i]
            ef_method = 'rolling'
        else:
            # Use fixed weight returns (original behavior)
            returns = self.returns_data[start_idx:end_idx].copy()  # (horizon, 15)
            ef_method = 'fixed'

        # Get VIX features for each year
        vix_features = []
        for i in range(horizon):
            year = self.years[start_idx + i]
            vix_features.append(self.vix_data.get(year, self._estimate_vix_for_year(year)))

        return {
            'returns': returns,
            'vix_features': vix_features,
            'start_year': self.years[start_idx],
            'end_year': self.years[end_idx - 1],
            'ef_method': ef_method
        }

    def backtest_all_windows(self, agent, initial_wealth: float, num_goals: int,
                             is_sentiment_agent: bool = False) -> List[Dict]:
        """
        Backtest agent on all historical windows.

        Args:
            agent: Trained agent (RL, Sentiment RL, or DP policy)
            initial_wealth: Starting wealth
            num_goals: Number of goals
            is_sentiment_agent: Whether agent uses VIX features

        Returns:
            List of results for each window
        """
        self._ensure_loaded()

        results = []
        n_windows = self.get_num_windows()

        # Get goal schedule
        goal_years = self._get_goal_years(num_goals)

        for window_idx in range(n_windows):
            window = self.get_window(window_idx)
            result = self._backtest_single_window(
                agent, window, initial_wealth, goal_years, is_sentiment_agent
            )
            result['window_idx'] = window_idx
            result['start_year'] = window['start_year']
            result['end_year'] = window['end_year']
            results.append(result)

        return results

    def _get_goal_years(self, num_goals: int) -> List[int]:
        """Get goal years based on number of goals"""
        if num_goals == 1:
            return [16]
        elif num_goals == 2:
            return [8, 16]
        elif num_goals == 4:
            return [4, 8, 12, 16]
        elif num_goals == 8:
            return [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            return list(range(1, 17))

    def _backtest_single_window(self, agent, window: Dict, initial_wealth: float,
                                goal_years: List[int], is_sentiment_agent: bool) -> Dict:
        """Backtest on a single historical window"""
        wealth = initial_wealth
        total_reward = 0.0
        goals_taken = 0

        returns = window['returns']
        vix_features = window['vix_features']
        horizon = len(returns)

        for t in range(horizon):
            # Build state
            normalized_time = t / horizon
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)

            if is_sentiment_agent:
                vix = vix_features[t]
                state = np.array([
                    normalized_time,
                    normalized_wealth,
                    vix['sentiment'],
                    vix['momentum']
                ], dtype=np.float32)
            else:
                state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

            # Get action from agent
            with torch.no_grad():
                action = agent.predict(state, deterministic=True)

            goal_action = int(action[0])
            portfolio_action = int(action[1])

            # Check for goal at this time
            if (t + 1) in goal_years:
                goal_cost = get_goal_cost(t + 1)
                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += 10 + (t + 1)
                    wealth -= goal_cost
                    goals_taken += 1

            # Wealth evolution using REAL historical returns (NO sentiment adjustment)
            if t < horizon - 1:
                real_return = returns[t, portfolio_action]
                wealth = wealth * (1 + real_return)
                wealth = max(wealth, 0.0)  # No negative wealth

        return {
            'total_reward': total_reward,
            'final_wealth': wealth,
            'goals_taken': goals_taken
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_initial_wealth(num_goals: int) -> float:
    """Paper formula: W0 = 12 * (NG)^0.85 * 10000"""
    return 12 * (num_goals ** 0.85) * 10000


def get_goal_cost(year: int) -> float:
    """Calculate goal cost at a given year: 100000 * 1.08^year (matches environment config)"""
    return 100000 * (1.08 ** year)


def get_terminal_goal_cost(num_goals: int, time_horizon: int = 16) -> float:
    """
    Get the goal cost at the terminal time (last goal year).

    For DP which solves single-goal problem, we use the last goal's cost.
    """
    if num_goals == 1:
        goal_year = time_horizon  # 16
    elif num_goals == 2:
        goal_year = time_horizon  # 16 (last goal)
    elif num_goals == 4:
        goal_year = time_horizon  # 16 (last goal)
    elif num_goals == 8:
        goal_year = time_horizon  # 16 (last goal)
    else:  # 16 goals
        goal_year = time_horizon  # 16 (last goal)

    return get_goal_cost(goal_year)


def create_env_for_goals(num_goals: int, data_mode: str = "simulation",
                         sentiment_provider=None, vix_simulator=None,
                         use_sentiment_adjusted_returns: bool = False,
                         use_monthly_steps: bool = False,
                         vix_params: Optional[VIXModelParams] = None) -> GBWMEnvironment:
    """Create environment with proper initial wealth

    Args:
        num_goals: Number of financial goals
        data_mode: 'simulation' or 'historical'
        sentiment_provider: Optional SentimentProvider for VIX features
        vix_simulator: Optional VIX simulator for COUPLED dynamics where
                      VIX and market returns share the same shock Z
        use_sentiment_adjusted_returns: If True (Option B), sentiment adjusts mu/sigma.
                      If False (Option A, default), use base mu/sigma for all methods.
        use_monthly_steps: If True, use monthly environment (192 steps) with correct VIX causality
        vix_params: VIX model parameters (learned or default)
    """
    initial_wealth = get_initial_wealth(num_goals)

    # Use monthly environment with correct VIX causality
    if use_monthly_steps:
        use_sentiment = sentiment_provider is not None or use_sentiment_adjusted_returns
        return make_gbwm_env_monthly(
            num_goals=num_goals,
            use_sentiment=use_sentiment,
            vix_params=vix_params
        )

    # Legacy: yearly environment
    if sentiment_provider is not None:
        return make_sentiment_gbwm_env(
            num_goals=num_goals,
            initial_wealth=initial_wealth,
            data_mode=data_mode,
            sentiment_provider=sentiment_provider,
            sentiment_start_date="2005-01-01",
            vix_simulator=vix_simulator,
            use_sentiment_adjusted_returns=use_sentiment_adjusted_returns
        )
    else:
        return make_gbwm_env(
            num_goals=num_goals,
            initial_wealth=initial_wealth,
            data_mode=data_mode
        )


def generate_shared_random_state(num_simulations: int, time_horizon: int,
                                  base_seed: int = 42,
                                  use_monthly_steps: bool = False,
                                  months_per_year: int = 12) -> SharedRandomState:
    """
    Generate shared random state for fair comparison across all methods.

    Args:
        num_simulations: Number of Monte Carlo simulations
        time_horizon: Time horizon in years
        base_seed: Random seed for reproducibility
        use_monthly_steps: If True, generate 12× more shocks for monthly steps
        months_per_year: Months per year (typically 12)

    Returns:
        SharedRandomState with market_samples shaped:
        - (num_simulations, time_horizon) if use_monthly_steps=False
        - (num_simulations, time_horizon * months_per_year) if use_monthly_steps=True
    """
    np.random.seed(base_seed)

    random_seeds = np.random.randint(0, 2**31-1, size=num_simulations)

    # For monthly steps, generate 12× more shocks (192 instead of 16)
    num_steps = time_horizon * months_per_year if use_monthly_steps else time_horizon
    market_samples = np.random.standard_normal(size=(num_simulations, num_steps))

    return SharedRandomState(random_seeds=random_seeds, market_samples=market_samples)


def apply_config_preset(config: ComparisonConfig, preset: str) -> ComparisonConfig:
    """Apply configuration preset"""
    if preset == "conservative":
        config.encoder_type = "simple"
        config.policy_type = "standard"
        config.value_type = "standard"
        config.hidden_dim = 64
        config.dropout_rate = 0.1
    elif preset == "aggressive":
        config.encoder_type = "attention"
        config.policy_type = "hierarchical"
        config.value_type = "dual_head"
        config.hidden_dim = 64
    elif preset == "research":
        config.encoder_type = "feature"
        config.policy_type = "standard"
        config.value_type = "standard"
        config.hidden_dim = 128
        config.use_batch_norm = True
    # default uses existing values
    return config


# =============================================================================
# INTEGRATED COMPARISON CLASS
# =============================================================================

class IntegratedComparison:
    """
    Integrated GBWM comparison following paper methodology.

    Combines training, simulation, hybrid analysis, and visualization.
    """

    def __init__(self, config: ComparisonConfig, output_dir: str = None):
        self.config = config

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "data" / "comparisons" / f"integrated_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        file_handler = logging.FileHandler(self.output_dir / "comparison.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Generate shared random state
        # For monthly steps: 192 shocks (16 years × 12 months)
        # For yearly steps: 16 shocks
        logger.info(f"Generating shared random state for fair comparison...")
        logger.info(f"  use_monthly_steps={config.use_monthly_steps}")
        self.shared_state = generate_shared_random_state(
            config.num_simulations, config.time_horizon, config.base_seed,
            use_monthly_steps=config.use_monthly_steps,
            months_per_year=config.months_per_year
        )

        # Storage for trained models
        # Using MultiGoalGBWMDP for proper multi-goal optimization
        self.dp_policies: Dict[int, MultiGoalGBWMDP] = {}
        self.rl_agents: Dict[int, PPOAgent] = {}
        self.sentiment_agents: Dict[int, SentimentAwarePPOAgent] = {}

        # Sentiment provider
        self.sentiment_provider = None

        # Results storage
        self.results: Dict[str, Dict[int, SimulationResult]] = {}
        self.hybrid_results: Dict[str, Dict[int, SimulationResult]] = {}

        # Initialize benchmark strategies
        self.benchmark_strategies = [
            RandomStrategy(),
            BuyAndHoldStrategy(),
            GreedyGoalStrategy(),
            ConservativeStrategy(),
            AggressiveStrategy()
        ]

        # Initialize VIX regime simulator for realistic sentiment dynamics (legacy)
        self.vix_simulator = VIXRegimeSimulator(base_seed=config.base_seed, dt=1.0)

        # Initialize coupled VIX simulator for training (VIX-return correlation) (legacy)
        self.coupled_vix_simulator = CoupledVIXSimulator(base_seed=config.base_seed)

        # ═══════════════════════════════════════════════════════════════════════
        # PRE-TRAINING: Load Efficient Frontier and β/δ from cache
        # ═══════════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 60)
        logger.info("LOADING PRE-TRAINED PARAMETERS")
        logger.info("=" * 60)

        # Load efficient frontier from cache (or compute if not available)
        logger.info("\n[Stage B] Loading efficient frontier...")
        self.efficient_frontier = compute_efficient_frontier(
            start_date=config.ef_start_date,
            end_date=config.ef_end_date,
            num_portfolios=config.num_portfolios,
            use_cache=config.use_pretrain_cache,
            force_recompute=config.force_pretrain_recompute
        )
        logger.info(f"  Portfolios: {len(self.efficient_frontier['mean_returns'])}")
        logger.info(f"  Return range: {self.efficient_frontier['mean_returns'][0]:.2%} to {self.efficient_frontier['mean_returns'][-1]:.2%}")
        logger.info(f"  Volatility range: {self.efficient_frontier['volatilities'][0]:.2%} to {self.efficient_frontier['volatilities'][-1]:.2%}")

        # Store portfolio parameters for Monte Carlo simulation
        self.portfolio_means = np.array(self.efficient_frontier['mean_returns'])
        self.portfolio_stds = np.array(self.efficient_frontier['volatilities'])

        # Learn β/δ from historical data (VIX → Returns relationship)
        logger.info("\n[Stage A] Loading β/δ parameters...")
        if config.use_monthly_steps:
            self.beta_delta_params = learn_beta_delta(
                start_date=config.beta_delta_start_date,
                end_date=config.beta_delta_end_date,
                use_cache=config.use_pretrain_cache,
                force_recompute=config.force_pretrain_recompute
            )

            # Check if we have portfolio-specific β/δ or single values
            if 'portfolio_betas' in self.beta_delta_params:
                # Use moderate portfolio (index 7) for VIXModelParams
                learned_beta = self.beta_delta_params['portfolio_betas'][7]
                learned_delta = self.beta_delta_params['portfolio_deltas'][7]
                logger.info(f"  β (drift adjustment, portfolio 7): {learned_beta:.6f}")
                logger.info(f"  δ (volatility adjustment, portfolio 7): {learned_delta:.6f}")
                logger.info(f"  Asset βs: bonds={self.beta_delta_params['asset_betas'][0]:.4f}, "
                           f"US={self.beta_delta_params['asset_betas'][1]:.4f}, "
                           f"intl={self.beta_delta_params['asset_betas'][2]:.4f}")
                logger.info(f"  Asset δs: bonds={self.beta_delta_params['asset_deltas'][0]:.4f}, "
                           f"US={self.beta_delta_params['asset_deltas'][1]:.4f}, "
                           f"intl={self.beta_delta_params['asset_deltas'][2]:.4f}")
            else:
                # Single β/δ values (legacy format)
                learned_beta = self.beta_delta_params.get('beta', config.vix_beta_sensitivity)
                learned_delta = self.beta_delta_params.get('delta', config.vix_delta_sensitivity)
                logger.info(f"  β (drift adjustment): {learned_beta:.6f}")
                logger.info(f"  δ (volatility adjustment): {learned_delta:.6f}")
        else:
            self.beta_delta_params = get_default_beta_delta()
            learned_beta = config.vix_beta_sensitivity
            learned_delta = config.vix_delta_sensitivity
            logger.info(f"  Using default β={learned_beta:.4f}, δ={learned_delta:.4f}")

        logger.info("=" * 60)

        # Initialize VIX Market Model with correct causality
        # VIX PREDICTS returns via β(VIX) and δ(VIX)
        self.vix_params = VIXModelParams(
            kappa=config.vix_kappa,
            theta=config.vix_theta,
            beta_sensitivity=learned_beta,
            delta_sensitivity=learned_delta
        )
        self.vix_market_model = VIXMarketModel(params=self.vix_params, seed=config.base_seed)

        # Store SentimentGBWMTrainer instances (one per goal count)
        self.sentiment_trainers: Dict[int, SentimentGBWMTrainer] = {}

        # Initialize historical backtester for evaluation on real data
        # Pass rolling EF settings from config
        self.backtester = HistoricalBacktester(
            use_rolling_ef=config.use_rolling_ef,
            ef_lookback_years=config.ef_lookback_years,
            ef_min_years=config.ef_min_years
        )

        # Initialize crisis year analyzer if enabled
        self.crisis_analyzer = None
        if config.crisis_analysis:
            self.crisis_analyzer = CrisisYearAnalyzer(output_dir=self.output_dir)
            logger.info("Crisis year analysis enabled")

        logger.info(f"IntegratedComparison initialized:")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Simulations: {config.num_simulations:,}")
        logger.info(f"  Goal counts: {config.goal_counts}")
        logger.info(f"  Training iterations: {config.num_iterations}")
        logger.info(f"  Architecture: policy={config.policy_type}, value={config.value_type}, encoder={config.encoder_type}")
        ef_mode = "Rolling" if config.use_rolling_ef else "Fixed"
        logger.info(f"  Efficient frontier: {ef_mode}")

    def _initialize_sentiment_provider(self) -> bool:
        """Initialize sentiment provider with historical VIX data"""
        if self.config.skip_sentiment:
            return False

        try:
            self.sentiment_provider = SentimentProvider(
                cache_dir=str(project_root / "data" / "sentiment"),
                lookback_days=365
            )
            # Use initialize_historical to fetch VIX data from 1990 to present
            # This ensures we have VIX data for all historical simulation periods
            success = self.sentiment_provider.initialize_historical(
                start_date='1990-01-02',  # VIX inception date
                force_refresh=False  # Use cache if available and covers the range
            )
            if success:
                logger.info("Sentiment provider initialized with historical VIX data")
            return success
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment provider: {e}")
            return False

    # =========================================================================
    # TRAINING METHODS
    # =========================================================================

    def train_all_models(self):
        """Train all models (DP, RL, Sentiment RL)"""
        if self.config.skip_training:
            logger.info("Skipping training (--skip_training flag set)")
            return self._load_existing_models()

        self._train_dp_policies()
        self._train_rl_agents()

        if not self.config.skip_sentiment:
            self._initialize_sentiment_provider()
            self._train_sentiment_agents()

    def _train_dp_policies(self):
        """Train Multi-Goal DP policies for all goal counts.

        Uses MultiGoalGBWMDP which optimizes expected total utility from goals,
        making BOTH goal-taking AND portfolio decisions optimally.

        Key difference from single-goal DP:
        - Single-goal DP: max P[W(T) >= G] (probability of reaching terminal wealth)
        - Multi-goal DP: max E[Σ utilities from fulfilled goals] (expected utility)
        """
        if self.config.skip_dp:
            logger.info("Skipping DP training")
            return

        logger.info("=" * 60)
        logger.info("TRAINING MULTI-GOAL DYNAMIC PROGRAMMING POLICIES")
        logger.info("(Optimizes expected utility from multiple goals)")
        logger.info("=" * 60)

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            logger.info(f"\nMulti-Goal DP for {num_goals} goals (W0=${initial_wealth:,.0f})...")

            # Set goal years based on num_goals
            if num_goals == 1:
                goal_years = [16]
            elif num_goals == 2:
                goal_years = [8, 16]
            elif num_goals == 4:
                goal_years = [4, 8, 12, 16]
            elif num_goals == 8:
                goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
            else:
                goal_years = list(range(1, 17))

            # Configure multi-goal DP
            # grid_density controls wealth grid resolution (higher = more points = more accurate)
            # Typical values: 0.5 (fast), 1.0 (balanced), 3.0 (paper default)
            dp_config = MultiGoalDPConfig(
                initial_wealth=initial_wealth,
                goal_years=goal_years,
                time_horizon=self.config.time_horizon,
                num_portfolios=self.config.num_portfolios,
                grid_density=self.config.dp_grid_density,  # Use directly from config
                base_goal_cost=100000.0,
                goal_cost_growth_rate=1.08,
                base_utility=10.0,
                utility_time_bonus=1.0
            )

            logger.info(f"  Goal years: {goal_years}")
            logger.info(f"  Grid density: {dp_config.grid_density}")

            dp = MultiGoalGBWMDP(dp_config)

            start_time = time.time()
            dp.solve()
            solve_time = time.time() - start_time

            self.dp_policies[num_goals] = dp

            # Save DP results
            dp_dir = self.output_dir / "models" / "dp" / f"goals_{num_goals}"
            dp_dir.mkdir(parents=True, exist_ok=True)

            np.save(dp_dir / "value_function.npy", dp.value_function)
            np.save(dp_dir / "goal_policy.npy", dp.goal_policy)
            np.save(dp_dir / "portfolio_policy.npy", dp.portfolio_policy)
            np.save(dp_dir / "wealth_grid.npy", dp.wealth_grid)

            # Save config (convert to serializable dict)
            config_dict = {
                'initial_wealth': dp_config.initial_wealth,
                'goal_years': dp_config.goal_years,
                'time_horizon': dp_config.time_horizon,
                'num_portfolios': dp_config.num_portfolios,
                'grid_density': dp_config.grid_density,
                'base_goal_cost': dp_config.base_goal_cost,
                'goal_cost_growth_rate': dp_config.goal_cost_growth_rate,
                'base_utility': dp_config.base_utility,
                'utility_time_bonus': dp_config.utility_time_bonus
            }
            with open(dp_dir / "config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)

            expected_utility = dp.get_expected_utility()
            logger.info(f"  Solved in {solve_time:.2f}s, expected utility: {expected_utility:.2f}")

    def _train_rl_agents(self):
        """Train pure RL agents"""
        logger.info("=" * 60)
        logger.info("TRAINING PURE RL AGENTS")
        logger.info("=" * 60)

        # Calculate time horizon: 192 steps for monthly, 16 for yearly
        effective_time_horizon = (
            self.config.time_horizon * self.config.months_per_year
            if self.config.use_monthly_steps
            else self.config.time_horizon
        )
        logger.info(f"  Time horizon: {effective_time_horizon} steps "
                    f"({'monthly' if self.config.use_monthly_steps else 'yearly'})")

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            logger.info(f"\nRL for {num_goals} goals (W0=${initial_wealth:,.0f})...")

            # Create environment with monthly steps if configured
            env = create_env_for_goals(
                num_goals,
                self.config.data_mode,
                use_monthly_steps=self.config.use_monthly_steps,
                vix_params=self.vix_params
            )

            training_config = TrainingConfig(
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                n_neurons=self.config.hidden_dim,
                ppo_epochs=4,
                mini_batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.5,
                entropy_coeff=0.01,
                max_grad_norm=0.5,
                time_horizon=effective_time_horizon,  # Use effective time horizon
                device=self.config.device
            )

            agent = PPOAgent(env=env, config=training_config)

            # Total timesteps = iterations × batch_size × time_horizon
            total_timesteps = self.config.num_iterations * self.config.batch_size * effective_time_horizon

            start_time = time.time()
            history = agent.train(total_timesteps=total_timesteps)
            train_time = time.time() - start_time

            self.rl_agents[num_goals] = agent

            # Save model
            model_dir = self.output_dir / "models" / "rl" / f"goals_{num_goals}"
            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                'policy_state_dict': agent.policy_net.state_dict(),
                'value_state_dict': agent.value_net.state_dict(),
                'config': asdict(training_config)
            }, model_dir / "model.pth")

            mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
            logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")

            env.close()

    def _train_sentiment_agents(self):
        """Train sentiment-aware RL agents with correct VIX causality.

        Uses SentimentGBWMTrainer which:
        - Loads pre-trained β/δ from BetaDeltaLearner cache
        - Loads efficient frontier from EfficientFrontierCalculator cache
        - Uses monthly time steps (192 months = 16 years × 12)
        - Applies professor's formulas: μ_adj = μ + β×(θ-VIX)/θ, σ_adj = σ - δ×(θ-VIX)/θ
        - VIX evolves via Mean-Reverting Jump-Diffusion
        """
        logger.info("=" * 60)
        logger.info("TRAINING SENTIMENT-AWARE RL AGENTS")
        logger.info("=" * 60)

        if self.config.use_sentiment_trainer and self.config.use_monthly_steps:
            # Use new SentimentGBWMTrainer with pre-trained parameters
            logger.info("  Using SentimentGBWMTrainer with pre-trained β/δ and efficient frontier")
            logger.info(f"  β={self.vix_params.beta_sensitivity:.4f}, δ={self.vix_params.delta_sensitivity:.4f}")
            logger.info(f"  Time horizon: {self.config.time_horizon * self.config.months_per_year} monthly steps")

            for num_goals in self.config.goal_counts:
                initial_wealth = get_initial_wealth(num_goals)
                logger.info(f"\nSentiment RL for {num_goals} goals (W0=${initial_wealth:,.0f})...")

                # Create SentimentTrainingConfig from ComparisonConfig
                sentiment_config = SentimentTrainingConfig(
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    n_neurons=self.config.hidden_dim,
                    years_horizon=self.config.time_horizon,
                    months_per_year=self.config.months_per_year,
                    policy_type=self.config.policy_type,
                    value_type=self.config.value_type,
                    encoder_type=self.config.encoder_type,
                    use_sentiment_adjusted_returns=self.config.use_sentiment_adjusted_returns,
                    n_traj=self.config.batch_size * self.config.num_iterations,
                    # Pre-training cache settings
                    efficient_frontier_start_date=self.config.ef_start_date,
                    efficient_frontier_end_date=self.config.ef_end_date,
                    beta_delta_start_date=self.config.beta_delta_start_date,
                    beta_delta_end_date=self.config.beta_delta_end_date,
                    use_cache=self.config.use_pretrain_cache,
                    force_recompute=self.config.force_pretrain_recompute,
                    # VIX model parameters
                    vix_kappa=self.config.vix_kappa,
                    vix_theta=self.config.vix_theta,
                    device=self.config.device
                )

                # Create trainer with experiment name
                experiment_name = f"sentiment_goals_{num_goals}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trainer = SentimentGBWMTrainer(
                    config=sentiment_config,
                    experiment_name=experiment_name
                )

                # Train
                start_time = time.time()
                history = trainer.train(
                    num_goals=num_goals,
                    num_iterations=self.config.num_iterations
                )
                train_time = time.time() - start_time

                # Store trainer and agent
                self.sentiment_trainers[num_goals] = trainer
                self.sentiment_agents[num_goals] = trainer.agent

                # Save model to comparison output directory
                model_dir = self.output_dir / "models" / "sentiment_rl" / f"goals_{num_goals}"
                model_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'policy_state_dict': trainer.agent.policy_net.state_dict(),
                    'value_state_dict': trainer.agent.value_net.state_dict(),
                    'beta': trainer.beta,
                    'delta': trainer.delta,
                    'policy_type': self.config.policy_type,
                    'value_type': self.config.value_type,
                    'encoder_type': self.config.encoder_type
                }, model_dir / "model.pth")

                mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
                logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")
                logger.info(f"  Model saved to: {model_dir}")

        else:
            # Legacy approach: Manual SentimentAwarePPOAgent setup
            logger.info("  Using LEGACY training approach")
            if self.config.use_monthly_steps:
                logger.info("  Using MONTHLY time steps with VIX → Returns causality")
                logger.info(f"  β={self.vix_params.beta_sensitivity:.4f}, δ={self.vix_params.delta_sensitivity:.4f}")
            else:
                logger.info("  Using YEARLY time steps with Cholesky coupling")
            if self.config.use_sentiment_adjusted_returns:
                logger.info("  Option B: VIX adjusts μ/σ during training")
            else:
                logger.info("  Option A: Base μ/σ (sentiment only affects decisions)")

            # Calculate time horizon: 192 steps for monthly, 16 for yearly
            effective_time_horizon = (
                self.config.time_horizon * self.config.months_per_year
                if self.config.use_monthly_steps
                else self.config.time_horizon
            )
            logger.info(f"  Time horizon: {effective_time_horizon} steps")

            for num_goals in self.config.goal_counts:
                initial_wealth = get_initial_wealth(num_goals)
                logger.info(f"\nSentiment RL for {num_goals} goals (W0=${initial_wealth:,.0f})...")

                if self.config.use_monthly_steps:
                    env = create_env_for_goals(
                        num_goals,
                        self.config.data_mode,
                        sentiment_provider=self.sentiment_provider,
                        use_monthly_steps=True,
                        vix_params=self.vix_params,
                        use_sentiment_adjusted_returns=self.config.use_sentiment_adjusted_returns
                    )
                else:
                    vix_simulator = VIXRegimeSimulator(
                        base_seed=self.config.base_seed + num_goals * 1000,
                        use_cholesky_coupling=True
                    )
                    env = create_env_for_goals(
                        num_goals, self.config.data_mode,
                        self.sentiment_provider, vix_simulator=vix_simulator,
                        use_sentiment_adjusted_returns=self.config.use_sentiment_adjusted_returns
                    )

                training_config = TrainingConfig(
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    n_neurons=self.config.hidden_dim,
                    ppo_epochs=4,
                    mini_batch_size=256,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_epsilon=0.5,
                    entropy_coeff=0.01,
                    max_grad_norm=0.5,
                    time_horizon=effective_time_horizon,
                    device=self.config.device
                )

                agent = SentimentAwarePPOAgent(
                    env=env,
                    config=training_config,
                    policy_type=self.config.policy_type,
                    value_type=self.config.value_type,
                    encoder_type=self.config.encoder_type,
                    sentiment_enabled=True
                )

                start_time = time.time()
                history = []
                for iteration in range(self.config.num_iterations):
                    metrics = agent.train_iteration()
                    history.append(metrics)
                train_time = time.time() - start_time

                self.sentiment_agents[num_goals] = agent

                model_dir = self.output_dir / "models" / "sentiment_rl" / f"goals_{num_goals}"
                model_dir.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'policy_state_dict': agent.policy_net.state_dict(),
                    'value_state_dict': agent.value_net.state_dict(),
                    'config': asdict(training_config),
                    'policy_type': self.config.policy_type,
                    'value_type': self.config.value_type,
                    'encoder_type': self.config.encoder_type
                }, model_dir / "model.pth")

                mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
                logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")

                env.close()

    def _load_existing_models(self):
        """Load existing trained models"""
        logger.info("Loading existing models...")

        for num_goals in self.config.goal_counts:
            # Load Multi-Goal DP
            dp_dir = self.output_dir / "models" / "dp" / f"goals_{num_goals}"
            if (dp_dir / "config.json").exists():
                with open(dp_dir / "config.json", 'r') as f:
                    dp_config_dict = json.load(f)
                dp_config = MultiGoalDPConfig(**dp_config_dict)
                dp = MultiGoalGBWMDP(dp_config)
                dp.value_function = np.load(dp_dir / "value_function.npy")
                dp.goal_policy = np.load(dp_dir / "goal_policy.npy")
                dp.portfolio_policy = np.load(dp_dir / "portfolio_policy.npy")
                dp.wealth_grid = np.load(dp_dir / "wealth_grid.npy")
                # Pre-compute transition matrices for simulation
                dp._precompute_transition_matrices()
                self.dp_policies[num_goals] = dp
                logger.info(f"  Loaded Multi-Goal DP for {num_goals} goals")

            # Load RL
            rl_path = self.output_dir / "models" / "rl" / f"goals_{num_goals}" / "model.pth"
            if rl_path.exists():
                env = create_env_for_goals(num_goals, self.config.data_mode)
                checkpoint = torch.load(rl_path, map_location=self.config.device)

                training_config = TrainingConfig(**checkpoint['config'])
                agent = PPOAgent(env=env, config=training_config)
                agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                agent.value_net.load_state_dict(checkpoint['value_state_dict'])

                self.rl_agents[num_goals] = agent
                logger.info(f"  Loaded RL for {num_goals} goals")
                env.close()

    # =========================================================================
    # SIMULATION METHODS
    # =========================================================================

    def run_all_simulations(self):
        """Run Monte Carlo simulations for all methods"""
        logger.info("=" * 60)
        logger.info(f"RUNNING {self.config.num_simulations:,} MONTE CARLO SIMULATIONS")
        logger.info("=" * 60)

        # Simulate all methods
        self._simulate_dp()
        self._simulate_rl()
        self._simulate_sentiment_rl()
        self._simulate_benchmarks()
        self._simulate_hybrid_strategies()

        # Calculate efficiencies relative to DP
        self._calculate_efficiencies()

    def run_historical_evaluation(self):
        """
        Run historical backtesting evaluation for all methods.

        This is DETERMINISTIC (not Monte Carlo):
        - Uses 39 overlapping 16-year windows from 1970-2023
        - All methods evaluated on SAME historical returns
        - NO sentiment adjustment in evaluation (real returns already reflect market conditions)

        IMPORTANT: Option A vs Option B (--use_sentiment_adjusted_returns) does NOT affect
        historical backtesting. Real historical returns are used as-is for ALL methods.
        The only difference between methods is:
        - DP/RL/Benchmarks: Cannot see VIX, make decisions based on time/wealth only
        - Sentiment RL: Can see VIX features and adjust decisions accordingly

        Use this instead of run_all_simulations for realistic out-of-sample evaluation.
        """
        logger.info("=" * 60)
        logger.info("RUNNING HISTORICAL BACKTESTING EVALUATION")
        logger.info(f"  Windows: {self.backtester.get_num_windows()} (1970-2023)")
        logger.info("  Note: All methods use REAL historical returns (no sentiment adjustment)")
        logger.info("=" * 60)

        # Store historical results separately
        self.historical_results = {}

        for num_goals in self.config.goal_counts:
            logger.info(f"\n--- Evaluating {num_goals} goals on historical data ---")

            # Evaluate all methods on historical windows
            goal_results = self.evaluate_on_historical(num_goals)

            for method_name, result in goal_results.items():
                if method_name not in self.historical_results:
                    self.historical_results[method_name] = {}
                self.historical_results[method_name][num_goals] = result

        # Calculate efficiencies relative to DP for historical results
        self._calculate_historical_efficiencies()

        # Generate crisis analysis plots if enabled
        if self.crisis_analyzer is not None:
            logger.info("\n--- Generating Crisis Year Analysis ---")
            self.crisis_analyzer.generate_all_plots()

        logger.info("\n" + "=" * 60)
        logger.info("Historical Evaluation Complete")
        logger.info("=" * 60)

    def _calculate_historical_efficiencies(self):
        """Calculate efficiencies for historical backtest results"""
        if 'DP' not in self.historical_results:
            logger.warning("No DP results for efficiency calculation")
            return

        for method_name, goal_results in self.historical_results.items():
            for num_goals, result in goal_results.items():
                if num_goals in self.historical_results.get('DP', {}):
                    dp_reward = self.historical_results['DP'][num_goals]['mean_reward']
                    if dp_reward > 0:
                        result['efficiency'] = result['mean_reward'] / dp_reward
                    else:
                        result['efficiency'] = 0.0
                else:
                    result['efficiency'] = 0.0

                logger.info(f"  {method_name} {num_goals}g: efficiency={result.get('efficiency', 0):.3f}")

    def _simulate_dp(self):
        """Simulate DP policy"""
        if not self.dp_policies:
            logger.warning("No DP policies to simulate")
            return

        logger.info("\nSimulating DP policies...")
        self.results["DP"] = {}

        for num_goals in self.config.goal_counts:
            if num_goals not in self.dp_policies:
                continue

            dp = self.dp_policies[num_goals]
            initial_wealth = get_initial_wealth(num_goals)

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in range(self.config.num_simulations):
                if sim_idx % 10000 == 0 and sim_idx > 0:
                    logger.info(f"  DP {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                seed = int(self.shared_state.random_seeds[sim_idx])
                market_shocks = self.shared_state.market_samples[sim_idx]

                reward, goal_taken, final_wealth = self._simulate_dp_trajectory(
                    dp, initial_wealth, market_shocks, num_goals, sim_idx
                )

                rewards.append(reward)
                goal_successes.append(1.0 if goal_taken else 0.0)
                final_wealths.append(final_wealth)

            self.results["DP"][num_goals] = SimulationResult(
                method_name="DP",
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=1.0,  # DP is baseline
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths)
            )

            logger.info(f"  DP {num_goals} goals: reward={np.mean(rewards):.2f}, success={np.mean(goal_successes):.3f}")

    def _simulate_dp_trajectory(self, dp, initial_wealth, market_shocks, num_goals, sim_idx=0):
        """Simulate single trajectory using Multi-Goal DP optimal policy.

        The Multi-Goal DP provides BOTH:
        1. Optimal goal-taking decisions (take/skip at each goal year)
        2. Optimal portfolio selection

        This is a key difference from the previous single-goal DP which only
        provided portfolio selection (goal-taking was done heuristically).

        Option B: When use_sentiment_adjusted_returns=True, DP also faces
        regime-switching dynamics (same as Sentiment RL for fair comparison).
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Get goal schedule from DP config
        goal_years = dp.config.goal_years

        # Option B: Initialize VIX simulator for regime-switching dynamics
        if self.config.use_sentiment_adjusted_returns:
            self.vix_simulator.reset(sim_idx)

        for t in range(self.config.time_horizon):
            # Get optimal strategy from Multi-Goal DP (includes BOTH goal and portfolio decisions)
            goal_action, portfolio_idx, mu, sigma = dp.get_optimal_strategy(wealth, t)

            # Option B: Simulate VIX and adjust mu/sigma
            if self.config.use_sentiment_adjusted_returns:
                shock = market_shocks[t] if t < len(market_shocks) else 0
                vix_level, vix_sentiment, vix_momentum = self.vix_simulator.simulate_step(shock)
                mu, sigma = self._compute_sentiment_adjusted_params(
                    mu, sigma, vix_sentiment, vix_momentum, vix_level
                )

            # Check if goal year and apply optimal goal decision
            # Aligns with RL environment's _is_goal_available() logic
            is_goal_year = t in goal_years
            effective_year = t

            # Handle terminal goal: goal at year T is processed at t=T-1
            if not is_goal_year and t == self.config.time_horizon - 1 and self.config.time_horizon in goal_years:
                is_goal_year = True
                effective_year = self.config.time_horizon

            if is_goal_year:
                goal_cost = dp.config.get_goal_cost(effective_year)
                goal_utility = dp.config.get_goal_utility(effective_year)

                # Use DP's optimal goal decision (not heuristic "if affordable")
                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Wealth evolution with market shock
            if t < self.config.time_horizon - 1:
                shock = market_shocks[t]
                wealth = wealth * np.exp(mu - 0.5 * sigma**2 + sigma * shock)
                wealth = max(0, wealth)

        return total_reward, goal_taken, wealth

    def _simulate_rl(self):
        """Simulate RL agents using shared market shocks for fair comparison"""
        if not self.rl_agents:
            logger.warning("No RL agents to simulate")
            return

        logger.info("\nSimulating RL agents...")
        self.results["RL"] = {}

        for num_goals in self.config.goal_counts:
            if num_goals not in self.rl_agents:
                continue

            agent = self.rl_agents[num_goals]
            agent.policy_net.eval()

            initial_wealth = get_initial_wealth(num_goals)

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in range(self.config.num_simulations):
                if sim_idx % 10000 == 0 and sim_idx > 0:
                    logger.info(f"  RL {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                # Use shared market shocks for fair comparison with DP
                market_shocks = self.shared_state.market_samples[sim_idx]

                total_reward, goal_taken, final_wealth = self._simulate_rl_trajectory(
                    agent, initial_wealth, market_shocks, num_goals, sim_idx
                )

                rewards.append(total_reward)
                goal_successes.append(1.0 if goal_taken else 0.0)
                final_wealths.append(final_wealth)

            self.results["RL"][num_goals] = SimulationResult(
                method_name="RL",
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=0.0,  # Will be calculated later
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths)
            )

            logger.info(f"  RL {num_goals} goals: reward={np.mean(rewards):.2f}, success={np.mean(goal_successes):.3f}")

    def _get_portfolio_adjustments(self, portfolio_idx: int, vix_avg: float) -> Tuple[float, float]:
        """
        Get portfolio-specific β and δ adjustments based on VIX level.

        Args:
            portfolio_idx: Portfolio index (0-14)
            vix_avg: Current VIX average

        Returns:
            (beta, delta) adjusted for this portfolio and VIX level
        """
        vix_theta = self.beta_delta_params.get('vix_theta', 20.0)

        # Get portfolio-specific sensitivities
        portfolio_betas = self.beta_delta_params['portfolio_betas']
        portfolio_deltas = self.beta_delta_params['portfolio_deltas']

        beta_sensitivity = portfolio_betas[portfolio_idx]
        delta_sensitivity = portfolio_deltas[portfolio_idx]

        # Apply VIX scaling: β = β_sens × (θ - VIX_avg) / θ
        # When VIX < θ (calm markets): positive adjustment (higher returns, lower vol)
        # When VIX > θ (stressed markets): negative adjustment (lower returns, higher vol)
        vix_factor = (vix_theta - vix_avg) / vix_theta

        beta = beta_sensitivity * vix_factor
        delta = delta_sensitivity * vix_factor

        return beta, delta

    def _simulate_rl_trajectory(self, agent, initial_wealth, market_shocks, num_goals, sim_idx=0):
        """
        Simulate single RL trajectory with shared market shocks.

        Supports both yearly (16 steps) and monthly (192 steps) time resolution.
        Uses same time resolution as Sentiment RL for fair comparison.

        Key difference from Sentiment RL:
        - Pure RL does NOT see VIX features (state is just [time, wealth])
        - In Option B, RL faces SAME VIX-adjusted market dynamics but can't adapt

        Option A: Base μ/σ (same market for all methods)
        Option B: VIX-adjusted μ/σ but RL can't see VIX to adapt portfolio
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Determine time resolution (same as Sentiment RL)
        use_monthly = self.config.use_monthly_steps
        months_per_year = self.config.months_per_year
        total_steps = self.config.time_horizon * months_per_year if use_monthly else self.config.time_horizon
        dt = 1.0 / months_per_year if use_monthly else 1.0

        # Get goal schedule
        if num_goals == 1:
            goal_years = [16]
        elif num_goals == 2:
            goal_years = [8, 16]
        elif num_goals == 4:
            goal_years = [4, 8, 12, 16]
        elif num_goals == 8:
            goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            goal_years = list(range(1, 17))

        # Convert to goal steps if using monthly
        goal_steps = [y * months_per_year for y in goal_years] if use_monthly else goal_years

        # Get base ANNUAL portfolio parameters from efficient frontier (pre-computed)
        # These are loaded from cache via compute_efficient_frontier() in __init__
        annual_means = self.portfolio_means
        annual_stds = self.portfolio_stds

        # Option B: Initialize VIX model for same market dynamics as Sentiment RL
        if self.config.use_sentiment_adjusted_returns:
            self.vix_market_model.reset(episode_idx=sim_idx)

        for step in range(total_steps):
            # Calculate normalized time and current year
            if use_monthly:
                current_year = (step + 1) // months_per_year
                normalized_time = step / total_steps
            else:
                current_year = step + 1
                normalized_time = step / self.config.time_horizon

            # Create normalized state for RL agent (NO VIX features)
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
            state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

            # Get action from RL agent
            with torch.no_grad():
                action = agent.predict(state, deterministic=True)

            goal_action = int(action[0])
            portfolio_action = int(action[1])

            # Check for goal
            if (step + 1) in goal_steps:
                goal_cost = get_goal_cost(current_year)
                goal_utility = 10 + current_year

                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Get market shock for this step
            shock = market_shocks[step] if step < len(market_shocks) else 0

            # Wealth evolution
            if step < total_steps - 1:
                base_mu = annual_means[portfolio_action]
                base_sigma = annual_stds[portfolio_action]

                if self.config.use_sentiment_adjusted_returns:
                    # Option B: RL faces SAME VIX-adjusted dynamics as Sentiment RL
                    # But RL can't see VIX, so it can't adapt portfolio to it
                    # Use PORTFOLIO-SPECIFIC β and δ
                    vix_avg = self.vix_market_model.get_vix_average(lookback=4 if use_monthly else 1)
                    beta, delta = self._get_portfolio_adjustments(portfolio_action, vix_avg)

                    mu_adj = base_mu + beta
                    sigma_adj = base_sigma - delta

                    mu_adj = np.clip(mu_adj, -0.15, 0.30)
                    sigma_adj = np.clip(sigma_adj, 0.02, 0.50)

                    # Advance VIX
                    self.vix_market_model.step_vix(dt=dt)
                else:
                    # Option A: Use base parameters
                    mu_adj = base_mu
                    sigma_adj = base_sigma

                # GBM formula: R = (μ - ½σ²)dt + σ√dt × Z
                drift = (mu_adj - 0.5 * sigma_adj**2) * dt
                diffusion = sigma_adj * np.sqrt(dt) * shock

                log_return = drift + diffusion
                wealth = wealth * np.exp(log_return)
                wealth = max(0, wealth)

        return total_reward, goal_taken, wealth

    def _simulate_sentiment_rl(self):
        """Simulate sentiment-aware RL agents using shared market shocks"""
        if not self.sentiment_agents:
            logger.info("No sentiment agents to simulate")
            return

        logger.info("\nSimulating Sentiment RL agents...")
        self.results["Sentiment RL"] = {}

        for num_goals in self.config.goal_counts:
            if num_goals not in self.sentiment_agents:
                continue

            agent = self.sentiment_agents[num_goals]
            agent.policy_net.eval()

            initial_wealth = get_initial_wealth(num_goals)

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in range(self.config.num_simulations):
                if sim_idx % 10000 == 0 and sim_idx > 0:
                    logger.info(f"  Sentiment RL {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                # Use shared market shocks for fair comparison
                market_shocks = self.shared_state.market_samples[sim_idx]

                total_reward, goal_taken, final_wealth = self._simulate_sentiment_trajectory(
                    agent, initial_wealth, market_shocks, num_goals, sim_idx
                )

                rewards.append(total_reward)
                goal_successes.append(1.0 if goal_taken else 0.0)
                final_wealths.append(final_wealth)

            self.results["Sentiment RL"][num_goals] = SimulationResult(
                method_name="Sentiment RL",
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=0.0,
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths)
            )

            logger.info(f"  Sentiment RL {num_goals} goals: reward={np.mean(rewards):.2f}, success={np.mean(goal_successes):.3f}")

    def _simulate_sentiment_trajectory(self, agent, initial_wealth, market_shocks, num_goals, sim_idx=0):
        """
        Simulate single sentiment RL trajectory with shared market shocks.

        NEW: Uses correct VIX → Returns causality with β/δ adjustments:
        - VIX follows mean-reverting jump-diffusion (independent process)
        - VIX PREDICTS returns via β(VIX) and δ(VIX)
        - μ_adj = μ + β, σ_adj = σ - δ (professor's formula)

        Supports both yearly (16 steps) and monthly (192 steps) time resolution.

        Timeline for each step:
        1. Agent sees VIX at BEGINNING of period
        2. β, δ calculated from VIX average
        3. Agent chooses action based on observation (including VIX features)
        4. Execute goal decision (if goal period)
        5. Generate return with VIX-adjusted μ, σ (if use_sentiment_adjusted_returns)
        6. Update wealth
        7. Advance VIX for next period
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Determine time resolution
        use_monthly = self.config.use_monthly_steps
        months_per_year = self.config.months_per_year
        total_steps = self.config.time_horizon * months_per_year if use_monthly else self.config.time_horizon
        dt = 1.0 / months_per_year if use_monthly else 1.0

        # Get goal schedule
        if num_goals == 1:
            goal_years = [16]
        elif num_goals == 2:
            goal_years = [8, 16]
        elif num_goals == 4:
            goal_years = [4, 8, 12, 16]
        elif num_goals == 8:
            goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            goal_years = list(range(1, 17))

        # Convert to goal months if using monthly steps
        goal_steps = [y * months_per_year for y in goal_years] if use_monthly else goal_years

        # Get base ANNUAL portfolio parameters from efficient frontier (pre-computed)
        # These are loaded from cache via compute_efficient_frontier() in __init__
        annual_means = self.portfolio_means
        annual_stds = self.portfolio_stds

        # Reset VIX market model for this trajectory
        self.vix_market_model.reset(episode_idx=sim_idx)

        for step in range(total_steps):
            # Calculate normalized time and current year
            if use_monthly:
                current_year = (step + 1) // months_per_year
                normalized_time = step / total_steps
            else:
                current_year = step + 1
                normalized_time = step / self.config.time_horizon

            # Get VIX features for observation
            vix_features = self.vix_market_model.get_sentiment_features()
            vix_level = vix_features[0] * self.config.vix_theta  # Denormalize
            vix_avg = self.vix_market_model.get_vix_average(lookback=4 if use_monthly else 1)

            # Create state for sentiment agent
            # State: [normalized_time, normalized_wealth, vix_level_norm, vix_avg_norm, vix_momentum]
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
            state = np.array([
                normalized_time,
                normalized_wealth,
                vix_features[0],  # vix_level normalized
                vix_features[1],  # vix_avg normalized
                vix_features[2]   # vix_momentum
            ], dtype=np.float32)

            # Get action from sentiment agent
            with torch.no_grad():
                action = agent.predict(state, deterministic=True)

            goal_action = int(action[0])
            portfolio_action = int(action[1])

            # Check for goal (only at specific steps)
            if (step + 1) in goal_steps:
                goal_cost = get_goal_cost(current_year)
                goal_utility = 10 + current_year

                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Get market shock for this time step
            shock = market_shocks[step] if step < len(market_shocks) else 0

            # Wealth evolution with correct VIX → Returns causality
            if step < total_steps - 1:
                # Use ANNUAL parameters and apply dt scaling
                base_mu = annual_means[portfolio_action]
                base_sigma = annual_stds[portfolio_action]

                if self.config.use_sentiment_adjusted_returns:
                    # Option B: Apply PORTFOLIO-SPECIFIC β/δ adjustments
                    # μ_adj = μ + β, σ_adj = σ - δ (professor's formula)
                    beta, delta = self._get_portfolio_adjustments(portfolio_action, vix_avg)
                    mu_adj = base_mu + beta
                    sigma_adj = base_sigma - delta

                    # Bounds (annual)
                    mu_adj = np.clip(mu_adj, -0.15, 0.30)
                    sigma_adj = np.clip(sigma_adj, 0.02, 0.50)
                else:
                    # Option A: Use base parameters (sentiment only affects decisions)
                    mu_adj = base_mu
                    sigma_adj = base_sigma

                # GBM formula: R = (μ - ½σ²)dt + σ√dt × Z
                drift = (mu_adj - 0.5 * sigma_adj**2) * dt
                diffusion = sigma_adj * np.sqrt(dt) * shock

                log_return = drift + diffusion
                wealth = wealth * np.exp(log_return)
                wealth = max(0, wealth)

            # Advance VIX for next step
            self.vix_market_model.step_vix(dt=dt)

        return total_reward, goal_taken, wealth

    def _simulate_sentiment_trajectory_coupled(self, agent, initial_wealth, num_goals, sim_idx=0):
        """
        Simulate sentiment RL trajectory with COUPLED VIX-return dynamics for training.

        Key difference from _simulate_sentiment_trajectory:
        - Uses CoupledVIXSimulator where the SAME market shock Z drives both:
          1. VIX evolution (negatively correlated: market down -> VIX up)
          2. Wealth evolution (directly affected by the shock)

        This coupling teaches the model the TRUE relationship between VIX and returns.
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Get goal schedule
        goal_years = self._get_goal_years_list(num_goals)

        # Get base portfolio parameters from efficient frontier (pre-computed)
        portfolio_means = self.portfolio_means
        portfolio_stds = self.portfolio_stds

        # Reset coupled VIX simulator for this trajectory
        self.coupled_vix_simulator.reset(sim_idx)

        for t in range(self.config.time_horizon):
            # Generate SINGLE market shock that drives BOTH VIX and returns
            z_shock = np.random.normal(0, 1)

            # Get VIX features using the SAME shock (creates coupling)
            vix_level, vix_sentiment, vix_momentum = self.coupled_vix_simulator.simulate_step(z_shock)

            # Create state for sentiment agent
            normalized_time = t / self.config.time_horizon
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
            state = np.array([normalized_time, normalized_wealth, vix_sentiment, vix_momentum], dtype=np.float32)

            # Get action from sentiment agent
            with torch.no_grad():
                action = agent.predict(state, deterministic=True)

            goal_action = int(action[0])
            portfolio_action = int(action[1])

            # Check for goal
            if (t + 1) in goal_years:
                goal_cost = get_goal_cost(t + 1)
                goal_utility = 10 + (t + 1)

                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Wealth evolution: use SAME z_shock that drove VIX
            # Option A (default): Use BASE mu/sigma (same as DP/RL) - sentiment only affects decisions
            # Option B: Use sentiment-adjusted mu/sigma (set use_sentiment_adjusted_returns=True)
            if t < self.config.time_horizon - 1:
                base_mu = portfolio_means[portfolio_action]
                base_sigma = portfolio_stds[portfolio_action]

                if getattr(self.config, 'use_sentiment_adjusted_returns', False):
                    # Option B: Apply sentiment adjustments to market dynamics
                    mu_adj, sigma_adj = self._compute_sentiment_adjusted_params(
                        base_mu, base_sigma, vix_sentiment, vix_momentum, vix_level
                    )
                else:
                    # Option A: Use base parameters (fair comparison with DP/RL)
                    mu_adj, sigma_adj = base_mu, base_sigma

                # Use the SAME z_shock for wealth evolution
                wealth = wealth * np.exp(mu_adj - 0.5 * sigma_adj**2 + sigma_adj * z_shock)
                wealth = max(0, wealth)

        return total_reward, goal_taken, wealth

    def _get_goal_years_list(self, num_goals: int) -> List[int]:
        """Get goal years based on number of goals"""
        if num_goals == 1:
            return [16]
        elif num_goals == 2:
            return [8, 16]
        elif num_goals == 4:
            return [4, 8, 12, 16]
        elif num_goals == 8:
            return [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            return list(range(1, 17))

    # =========================================================================
    # HISTORICAL BACKTESTING METHODS (Evaluation on Real Data)
    # =========================================================================

    def evaluate_on_historical(self, num_goals: int) -> Dict[str, Dict]:
        """
        Evaluate all methods on historical data using deterministic backtesting.

        This is NOT Monte Carlo - it replays actual historical 16-year windows.
        All methods use the SAME historical returns for fair comparison.

        Returns:
            Dict with results for each method
        """
        logger.info(f"\nEvaluating on historical data for {num_goals} goals...")

        initial_wealth = get_initial_wealth(num_goals)
        n_windows = self.backtester.get_num_windows()

        results = {}

        # Evaluate DP
        if num_goals in self.dp_policies:
            dp_results = self._backtest_dp_on_historical(num_goals, initial_wealth)
            results['DP'] = self._aggregate_backtest_results(dp_results, 'DP', num_goals)
            logger.info(f"  DP: mean_reward={results['DP']['mean_reward']:.2f}, "
                       f"windows={len(dp_results)}")

        # Evaluate Pure RL
        if num_goals in self.rl_agents:
            rl_results = self._backtest_rl_on_historical(num_goals, initial_wealth)
            results['RL'] = self._aggregate_backtest_results(rl_results, 'RL', num_goals)
            logger.info(f"  RL: mean_reward={results['RL']['mean_reward']:.2f}")

        # Evaluate Sentiment RL
        if num_goals in self.sentiment_agents:
            sentiment_results = self._backtest_sentiment_on_historical(num_goals, initial_wealth)
            results['Sentiment RL'] = self._aggregate_backtest_results(sentiment_results, 'Sentiment RL', num_goals)
            logger.info(f"  Sentiment RL: mean_reward={results['Sentiment RL']['mean_reward']:.2f}")

        # Evaluate benchmarks
        for strategy in self.benchmark_strategies:
            bench_results = self._backtest_benchmark_on_historical(strategy, num_goals, initial_wealth)
            results[strategy.name] = self._aggregate_backtest_results(bench_results, strategy.name, num_goals)
            logger.info(f"  {strategy.name}: mean_reward={results[strategy.name]['mean_reward']:.2f}")

        return results

    def _backtest_dp_on_historical(self, num_goals: int, initial_wealth: float) -> List[Dict]:
        """Backtest DP policy on historical windows"""
        dp_policy = self.dp_policies[num_goals]
        goal_years = self._get_goal_years_list(num_goals)
        results = []
        all_yearly_metrics: List[YearlyMetrics] = []

        for window_idx in range(self.backtester.get_num_windows()):
            window = self.backtester.get_window(window_idx)
            returns = window['returns']  # (16, 15) real returns
            vix_features = window.get('vix_features', None)

            wealth = initial_wealth
            total_reward = 0.0
            goals_taken = 0

            for t in range(self.config.time_horizon):
                wealth_start = wealth

                # Get DP action
                wealth_idx = np.argmin(np.abs(dp_policy.wealth_grid - wealth))
                goal_action = dp_policy.goal_policy[wealth_idx, t]
                portfolio_action = dp_policy.portfolio_policy[wealth_idx, t]

                # Check for goal
                goal_available = (t + 1) in goal_years
                goal_taken = False
                goal_reward = 0.0
                if goal_available:
                    goal_cost = get_goal_cost(t + 1)
                    if goal_action == 1 and wealth >= goal_cost:
                        goal_reward = 10 + (t + 1)
                        total_reward += goal_reward
                        wealth -= goal_cost
                        goals_taken += 1
                        goal_taken = True

                # Wealth evolution using REAL returns (NO simulation adjustment)
                if t < self.config.time_horizon - 1:
                    real_return = returns[t, portfolio_action]
                    wealth = wealth * (1 + real_return)
                    wealth = max(0, wealth)

                # Track yearly metrics for crisis analysis
                if self.crisis_analyzer is not None and vix_features is not None:
                    vix = vix_features[t]
                    vix_level = vix.get('vix_level', 20.0)
                    calendar_year = window['start_year'] + t
                    wealth_change_pct = ((wealth - wealth_start) / wealth_start * 100) if wealth_start > 0 else 0.0

                    all_yearly_metrics.append(YearlyMetrics(
                        calendar_year=calendar_year,
                        vix_level=vix_level,
                        vix_regime=self.crisis_analyzer.classify_vix_regime(vix_level),
                        portfolio_choice=int(portfolio_action),
                        wealth_start=wealth_start,
                        wealth_end=wealth,
                        wealth_change_pct=wealth_change_pct,
                        goal_available=goal_available,
                        goal_taken=goal_taken,
                        goal_reward=goal_reward,
                        cumulative_reward=total_reward
                    ))

            results.append({
                'window_idx': window_idx,
                'start_year': window['start_year'],
                'end_year': window['end_year'],
                'total_reward': total_reward,
                'final_wealth': wealth,
                'goals_taken': goals_taken
            })

        # Add yearly metrics to crisis analyzer
        if self.crisis_analyzer is not None and all_yearly_metrics:
            self.crisis_analyzer.add_yearly_metrics('DP', all_yearly_metrics)

        return results

    def _backtest_rl_on_historical(self, num_goals: int, initial_wealth: float) -> List[Dict]:
        """Backtest RL agent on historical windows"""
        agent = self.rl_agents[num_goals]
        goal_years = self._get_goal_years_list(num_goals)
        results = []
        all_yearly_metrics: List[YearlyMetrics] = []

        for window_idx in range(self.backtester.get_num_windows()):
            window = self.backtester.get_window(window_idx)
            returns = window['returns']
            vix_features = window.get('vix_features', None)

            wealth = initial_wealth
            total_reward = 0.0
            goals_taken = 0

            for t in range(self.config.time_horizon):
                wealth_start = wealth

                # Build state (2D for pure RL)
                normalized_time = t / self.config.time_horizon
                normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

                # Get action
                with torch.no_grad():
                    action = agent.predict(state, deterministic=True)

                goal_action = int(action[0])
                portfolio_action = int(action[1])

                # Check for goal
                goal_available = (t + 1) in goal_years
                goal_taken = False
                goal_reward = 0.0
                if goal_available:
                    goal_cost = get_goal_cost(t + 1)
                    if goal_action == 1 and wealth >= goal_cost:
                        goal_reward = 10 + (t + 1)
                        total_reward += goal_reward
                        wealth -= goal_cost
                        goals_taken += 1
                        goal_taken = True

                # Wealth evolution using REAL returns
                if t < self.config.time_horizon - 1:
                    real_return = returns[t, portfolio_action]
                    wealth = wealth * (1 + real_return)
                    wealth = max(0, wealth)

                # Track yearly metrics for crisis analysis
                if self.crisis_analyzer is not None and vix_features is not None:
                    vix = vix_features[t]
                    vix_level = vix.get('vix_level', 20.0)
                    calendar_year = window['start_year'] + t
                    wealth_change_pct = ((wealth - wealth_start) / wealth_start * 100) if wealth_start > 0 else 0.0

                    all_yearly_metrics.append(YearlyMetrics(
                        calendar_year=calendar_year,
                        vix_level=vix_level,
                        vix_regime=self.crisis_analyzer.classify_vix_regime(vix_level),
                        portfolio_choice=int(portfolio_action),
                        wealth_start=wealth_start,
                        wealth_end=wealth,
                        wealth_change_pct=wealth_change_pct,
                        goal_available=goal_available,
                        goal_taken=goal_taken,
                        goal_reward=goal_reward,
                        cumulative_reward=total_reward
                    ))

            results.append({
                'window_idx': window_idx,
                'start_year': window['start_year'],
                'end_year': window['end_year'],
                'total_reward': total_reward,
                'final_wealth': wealth,
                'goals_taken': goals_taken
            })

        # Add yearly metrics to crisis analyzer
        if self.crisis_analyzer is not None and all_yearly_metrics:
            self.crisis_analyzer.add_yearly_metrics('RL', all_yearly_metrics)

        return results

    def _backtest_sentiment_on_historical(self, num_goals: int, initial_wealth: float) -> List[Dict]:
        """
        Backtest Sentiment RL on historical windows.

        Uses REAL VIX scores and REAL portfolio returns.
        NO sentiment adjustment on returns - they're already real historical returns.
        """
        agent = self.sentiment_agents[num_goals]
        goal_years = self._get_goal_years_list(num_goals)
        results = []
        all_yearly_metrics: List[YearlyMetrics] = []

        for window_idx in range(self.backtester.get_num_windows()):
            window = self.backtester.get_window(window_idx)
            returns = window['returns']  # (16, 15) REAL returns
            vix_features = window['vix_features']  # REAL VIX

            wealth = initial_wealth
            total_reward = 0.0
            goals_taken = 0

            for t in range(self.config.time_horizon):
                wealth_start = wealth

                # Get VIX features for this year
                vix = vix_features[t]

                # Build state (4D for sentiment RL)
                normalized_time = t / self.config.time_horizon
                normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                state = np.array([
                    normalized_time,
                    normalized_wealth,
                    vix['sentiment'],
                    vix['momentum']
                ], dtype=np.float32)

                # Get action
                with torch.no_grad():
                    action = agent.predict(state, deterministic=True)

                goal_action = int(action[0])
                portfolio_action = int(action[1])

                # Check for goal
                goal_available = (t + 1) in goal_years
                goal_taken = False
                goal_reward = 0.0
                if goal_available:
                    goal_cost = get_goal_cost(t + 1)
                    if goal_action == 1 and wealth >= goal_cost:
                        goal_reward = 10 + (t + 1)
                        total_reward += goal_reward
                        wealth -= goal_cost
                        goals_taken += 1
                        goal_taken = True

                # Wealth evolution using REAL returns (NO sentiment adjustment!)
                # The real returns already reflect market conditions
                if t < self.config.time_horizon - 1:
                    real_return = returns[t, portfolio_action]
                    wealth = wealth * (1 + real_return)
                    wealth = max(0, wealth)

                # Track yearly metrics for crisis analysis
                if self.crisis_analyzer is not None:
                    vix_level = vix.get('vix_level', 20.0)
                    calendar_year = window['start_year'] + t
                    wealth_change_pct = ((wealth - wealth_start) / wealth_start * 100) if wealth_start > 0 else 0.0

                    all_yearly_metrics.append(YearlyMetrics(
                        calendar_year=calendar_year,
                        vix_level=vix_level,
                        vix_regime=self.crisis_analyzer.classify_vix_regime(vix_level),
                        portfolio_choice=int(portfolio_action),
                        wealth_start=wealth_start,
                        wealth_end=wealth,
                        wealth_change_pct=wealth_change_pct,
                        goal_available=goal_available,
                        goal_taken=goal_taken,
                        goal_reward=goal_reward,
                        cumulative_reward=total_reward
                    ))

            results.append({
                'window_idx': window_idx,
                'start_year': window['start_year'],
                'end_year': window['end_year'],
                'total_reward': total_reward,
                'final_wealth': wealth,
                'goals_taken': goals_taken
            })

        # Add yearly metrics to crisis analyzer
        if self.crisis_analyzer is not None and all_yearly_metrics:
            self.crisis_analyzer.add_yearly_metrics('Sentiment RL', all_yearly_metrics)

        return results

    def _backtest_benchmark_on_historical(self, strategy: BenchmarkStrategy,
                                          num_goals: int, initial_wealth: float) -> List[Dict]:
        """Backtest a benchmark strategy on historical windows"""
        goal_years = self._get_goal_years_list(num_goals)
        results = []
        all_yearly_metrics: List[YearlyMetrics] = []

        # Create deterministic RNG for benchmark
        rng = np.random.RandomState(self.config.base_seed)

        for window_idx in range(self.backtester.get_num_windows()):
            window = self.backtester.get_window(window_idx)
            returns = window['returns']
            vix_features = window.get('vix_features', None)

            wealth = initial_wealth
            total_reward = 0.0
            goals_taken = 0

            strategy.reset()

            for t in range(self.config.time_horizon):
                wealth_start = wealth

                # Build state
                normalized_time = t / self.config.time_horizon
                normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                state = np.array([normalized_time, normalized_wealth])

                # Build info dict with all fields strategies might need
                is_goal_year = (t + 1) in goal_years
                goal_cost = get_goal_cost(t + 1) if is_goal_year else 0
                info = {
                    'goal_cost': goal_cost,
                    'goal_available': is_goal_year,
                    'current_wealth': wealth
                }

                # Get action from strategy
                goal_action, portfolio_action = strategy.get_action(state, info, rng)

                # Check for goal
                goal_available = (t + 1) in goal_years
                goal_taken_flag = False
                goal_reward = 0.0
                if goal_available:
                    goal_cost = get_goal_cost(t + 1)
                    if goal_action == 1 and wealth >= goal_cost:
                        goal_reward = 10 + (t + 1)
                        total_reward += goal_reward
                        wealth -= goal_cost
                        goals_taken += 1
                        goal_taken_flag = True

                # Wealth evolution using REAL returns
                if t < self.config.time_horizon - 1:
                    real_return = returns[t, portfolio_action]
                    wealth = wealth * (1 + real_return)
                    wealth = max(0, wealth)

                # Track yearly metrics for crisis analysis (only for Buy & Hold as representative benchmark)
                if self.crisis_analyzer is not None and vix_features is not None and strategy.name == 'Buy & Hold':
                    vix = vix_features[t]
                    vix_level = vix.get('vix_level', 20.0)
                    calendar_year = window['start_year'] + t
                    wealth_change_pct = ((wealth - wealth_start) / wealth_start * 100) if wealth_start > 0 else 0.0

                    all_yearly_metrics.append(YearlyMetrics(
                        calendar_year=calendar_year,
                        vix_level=vix_level,
                        vix_regime=self.crisis_analyzer.classify_vix_regime(vix_level),
                        portfolio_choice=int(portfolio_action),
                        wealth_start=wealth_start,
                        wealth_end=wealth,
                        wealth_change_pct=wealth_change_pct,
                        goal_available=goal_available,
                        goal_taken=goal_taken_flag,
                        goal_reward=goal_reward,
                        cumulative_reward=total_reward
                    ))

            results.append({
                'window_idx': window_idx,
                'start_year': window['start_year'],
                'end_year': window['end_year'],
                'total_reward': total_reward,
                'final_wealth': wealth,
                'goals_taken': goals_taken
            })

        # Add yearly metrics to crisis analyzer (only for Buy & Hold)
        if self.crisis_analyzer is not None and all_yearly_metrics and strategy.name == 'Buy & Hold':
            self.crisis_analyzer.add_yearly_metrics('Buy & Hold', all_yearly_metrics)

        return results

    def _aggregate_backtest_results(self, results: List[Dict], method_name: str,
                                    num_goals: int) -> Dict:
        """Aggregate backtest results into summary statistics"""
        rewards = [r['total_reward'] for r in results]
        final_wealths = [r['final_wealth'] for r in results]
        goals_taken = [r['goals_taken'] for r in results]

        return {
            'method_name': method_name,
            'num_goals': num_goals,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_final_wealth': np.mean(final_wealths),
            'std_final_wealth': np.std(final_wealths),
            'mean_goals_taken': np.mean(goals_taken),
            'num_windows': len(results),
            'window_results': results  # Keep detailed results
        }

    def _compute_sentiment_adjusted_params_beta_delta(self, base_mu, base_sigma, vix_avg, vix_model=None):
        """
        Compute sentiment-adjusted mu and sigma using correct VIX → Returns causality.

        Professor's formula (Ito's lemma):
            μ_adj = μ + β(VIX)
            σ_adj = σ - δ(VIX)

        Where:
            β(VIX) = β_sensitivity × (θ - VIX_avg) / θ
            δ(VIX) = δ_sensitivity × (θ - VIX_avg) / θ

        Key insight:
        - When VIX > θ (high fear): β < 0 (lower expected return), δ < 0 (higher volatility)
        - When VIX < θ (low fear): β > 0 (higher expected return), δ > 0 (lower volatility)
        - Agent sees μ_adj, σ_adj at BEGINNING of period and can adapt portfolio choice
        """
        # Get VIX model parameters
        theta = self.config.vix_theta
        beta_sens = self.config.vix_beta_sensitivity
        delta_sens = self.config.vix_delta_sensitivity

        # Calculate β and δ adjustments
        # β = β_sensitivity × (θ - VIX_avg) / θ
        # δ = δ_sensitivity × (θ - VIX_avg) / θ
        vix_deviation = (theta - vix_avg) / theta  # Normalized deviation from mean
        beta = beta_sens * vix_deviation
        delta = delta_sens * vix_deviation

        # Apply adjustments (professor's formula)
        mu_adj = base_mu + beta
        sigma_adj = base_sigma - delta

        # Bounds to prevent extreme values
        mu_adj = np.clip(mu_adj, -0.15, 0.30)
        sigma_adj = np.clip(sigma_adj, 0.02, 0.50)

        return mu_adj, sigma_adj, beta, delta

    def _compute_sentiment_adjusted_params(self, base_mu, base_sigma, vix_sentiment, vix_momentum, vix_level):
        """
        Legacy method for backward compatibility.
        Converts to new β/δ formula using VIX level as proxy for VIX_avg.
        """
        # Use vix_level as vix_avg (backward compatible)
        mu_adj, sigma_adj, _, _ = self._compute_sentiment_adjusted_params_beta_delta(
            base_mu, base_sigma, vix_level
        )
        return mu_adj, sigma_adj

    def _simulate_benchmarks(self):
        """Simulate benchmark strategies using shared market shocks"""
        logger.info("\nSimulating benchmark strategies...")

        for strategy in self.benchmark_strategies:
            self.results[strategy.name] = {}

            for num_goals in self.config.goal_counts:
                initial_wealth = get_initial_wealth(num_goals)

                rewards = []
                goal_successes = []
                final_wealths = []

                for sim_idx in range(self.config.num_simulations):
                    if sim_idx % 20000 == 0 and sim_idx > 0:
                        logger.info(f"  {strategy.name} {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                    seed = int(self.shared_state.random_seeds[sim_idx])
                    rng = np.random.RandomState(seed)
                    market_shocks = self.shared_state.market_samples[sim_idx]

                    total_reward, goal_taken, final_wealth = self._simulate_benchmark_trajectory(
                        strategy, initial_wealth, market_shocks, num_goals, rng, sim_idx
                    )

                    rewards.append(total_reward)
                    goal_successes.append(1.0 if goal_taken else 0.0)
                    final_wealths.append(final_wealth)

                self.results[strategy.name][num_goals] = SimulationResult(
                    method_name=strategy.name,
                    num_goals=num_goals,
                    mean_reward=np.mean(rewards),
                    std_reward=np.std(rewards),
                    efficiency=0.0,
                    num_simulations=self.config.num_simulations,
                    goal_success_rate=np.mean(goal_successes),
                    mean_final_wealth=np.mean(final_wealths),
                    std_final_wealth=np.std(final_wealths)
                )

                logger.info(f"  {strategy.name} {num_goals} goals: reward={np.mean(rewards):.2f}")

    def _simulate_benchmark_trajectory(self, strategy, initial_wealth, market_shocks, num_goals, rng, sim_idx=0):
        """Simulate single benchmark trajectory with shared market shocks.

        Option B: When use_sentiment_adjusted_returns=True, benchmarks also face
        regime-switching dynamics (same as Sentiment RL for fair comparison).
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Get goal schedule
        if num_goals == 1:
            goal_years = [16]
        elif num_goals == 2:
            goal_years = [8, 16]
        elif num_goals == 4:
            goal_years = [4, 8, 12, 16]
        elif num_goals == 8:
            goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            goal_years = list(range(1, 17))

        # Get portfolio parameters from efficient frontier (pre-computed)
        portfolio_means = self.portfolio_means
        portfolio_stds = self.portfolio_stds

        # Option B: Initialize VIX simulator for regime-switching dynamics
        if self.config.use_sentiment_adjusted_returns:
            self.vix_simulator.reset(sim_idx)

        for t in range(self.config.time_horizon):
            # Create state for strategy
            normalized_time = t / self.config.time_horizon
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
            state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

            # Create info dict for strategy
            goal_available = (t + 1) in goal_years
            goal_cost = get_goal_cost(t + 1) if goal_available else float('inf')
            info = {
                'goal_available': goal_available,
                'current_wealth': wealth,
                'goal_cost': goal_cost
            }

            # Get action from strategy
            goal_action, portfolio_action = strategy.get_action(state, info, rng)

            # Check for goal
            if (t + 1) in goal_years:
                goal_utility = 10 + (t + 1)

                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Wealth evolution with shared market shock
            if t < self.config.time_horizon - 1:
                mu = portfolio_means[portfolio_action]
                sigma = portfolio_stds[portfolio_action]
                shock = market_shocks[t]

                # Option B: Simulate VIX and adjust mu/sigma
                if self.config.use_sentiment_adjusted_returns:
                    vix_level, vix_sentiment, vix_momentum = self.vix_simulator.simulate_step(shock)
                    mu, sigma = self._compute_sentiment_adjusted_params(
                        mu, sigma, vix_sentiment, vix_momentum, vix_level
                    )

                wealth = wealth * np.exp(mu - 0.5 * sigma**2 + sigma * shock)
                wealth = max(0, wealth)

        return total_reward, goal_taken, wealth

    def _simulate_hybrid_strategies(self):
        """Simulate hybrid strategies using shared market shocks"""
        logger.info("\nSimulating hybrid strategies...")

        hybrid_configs = [
            ("DP-Port + RL-Goal", "dp", "rl"),
            ("RL-Port + DP-Goal", "rl", "dp"),
        ]

        # Add sentiment hybrids if available
        if self.sentiment_agents:
            hybrid_configs.extend([
                ("DP-Port + Sent-Goal", "dp", "sentiment"),
                ("Sent-Port + DP-Goal", "sentiment", "dp"),
                ("RL-Port + Sent-Goal", "rl", "sentiment"),
                ("Sent-Port + RL-Goal", "sentiment", "rl"),
            ])

        for hybrid_name, port_source, goal_source in hybrid_configs:
            self.hybrid_results[hybrid_name] = {}

            for num_goals in self.config.goal_counts:
                # Check if we have all required components
                if port_source == "dp" or goal_source == "dp":
                    if num_goals not in self.dp_policies:
                        continue
                if port_source == "rl" or goal_source == "rl":
                    if num_goals not in self.rl_agents:
                        continue
                if port_source == "sentiment" or goal_source == "sentiment":
                    if num_goals not in self.sentiment_agents:
                        continue

                initial_wealth = get_initial_wealth(num_goals)

                rewards = []
                goal_successes = []
                final_wealths = []

                for sim_idx in range(self.config.num_simulations):
                    if sim_idx % 20000 == 0 and sim_idx > 0:
                        logger.info(f"  {hybrid_name} {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                    market_shocks = self.shared_state.market_samples[sim_idx]

                    total_reward, goal_taken, final_wealth = self._simulate_hybrid_trajectory(
                        port_source, goal_source, initial_wealth, market_shocks, num_goals, sim_idx
                    )

                    rewards.append(total_reward)
                    goal_successes.append(1.0 if goal_taken else 0.0)
                    final_wealths.append(final_wealth)

                self.hybrid_results[hybrid_name][num_goals] = SimulationResult(
                    method_name=hybrid_name,
                    num_goals=num_goals,
                    mean_reward=np.mean(rewards),
                    std_reward=np.std(rewards),
                    efficiency=0.0,
                    num_simulations=self.config.num_simulations,
                    goal_success_rate=np.mean(goal_successes),
                    mean_final_wealth=np.mean(final_wealths),
                    std_final_wealth=np.std(final_wealths)
                )

                logger.info(f"  {hybrid_name} {num_goals} goals: reward={np.mean(rewards):.2f}")

    def _simulate_hybrid_trajectory(self, port_source, goal_source, initial_wealth, market_shocks, num_goals, sim_idx=0):
        """
        Simulate single hybrid trajectory with shared market shocks.

        Uses regime-switching VIX simulation for realistic sentiment dynamics
        when sentiment components are involved.

        Option B: When use_sentiment_adjusted_returns=True, ALL hybrid strategies
        face regime-switching dynamics (even non-sentiment hybrids).
        """
        wealth = initial_wealth
        total_reward = 0
        goal_taken = False

        # Get goal schedule
        if num_goals == 1:
            goal_years = [16]
        elif num_goals == 2:
            goal_years = [8, 16]
        elif num_goals == 4:
            goal_years = [4, 8, 12, 16]
        elif num_goals == 8:
            goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            goal_years = list(range(1, 17))

        # Get portfolio parameters from efficient frontier (pre-computed)
        portfolio_means = self.portfolio_means
        portfolio_stds = self.portfolio_stds

        # Check if sentiment is involved (affects state representation)
        uses_sentiment_state = (port_source == "sentiment" or goal_source == "sentiment")

        # Option B: ALL methods face regime-switching dynamics
        # Option A: Only sentiment-involved hybrids use adjusted dynamics
        use_adjusted_dynamics = uses_sentiment_state or self.config.use_sentiment_adjusted_returns

        # Initialize VIX regime simulator if needed
        if use_adjusted_dynamics:
            self.vix_simulator.reset(sim_idx)

        for t in range(self.config.time_horizon):
            # Create state
            normalized_time = t / self.config.time_horizon
            normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
            state_2d = np.array([normalized_time, normalized_wealth], dtype=np.float32)

            # Compute VIX-based sentiment for 4D state using regime-switching simulator
            if use_adjusted_dynamics:
                # Get market shock for this time step
                shock = market_shocks[t] if t < len(market_shocks) else 0

                # Simulate VIX using regime-switching model
                # Returns: (raw_vix, normalized_sentiment, momentum)
                vix_level, vix_sentiment, vix_momentum = self.vix_simulator.simulate_step(shock)
            else:
                vix_sentiment = 0.0
                vix_momentum = 0.0
                vix_level = 18.0

            state_4d = np.array([normalized_time, normalized_wealth, vix_sentiment, vix_momentum], dtype=np.float32)

            # Create info dict
            goal_available = (t + 1) in goal_years
            goal_cost = get_goal_cost(t + 1) if goal_available else float('inf')
            info = {
                'goal_available': goal_available,
                'current_wealth': wealth,
                'goal_cost': goal_cost
            }

            # Get portfolio action from port_source
            portfolio_action = self._get_portfolio_action_for_hybrid(
                port_source, num_goals, state_2d, state_4d, wealth, t
            )

            # Get goal action from goal_source
            goal_action = self._get_goal_action_for_hybrid(
                goal_source, num_goals, state_2d, state_4d, wealth, goal_cost, goal_available, t
            )

            # Check for goal
            if (t + 1) in goal_years:
                goal_utility = 10 + (t + 1)

                if goal_action == 1 and wealth >= goal_cost:
                    total_reward += goal_utility
                    wealth -= goal_cost
                    goal_taken = True

            # Wealth evolution with shared market shock
            if t < self.config.time_horizon - 1:
                base_mu = portfolio_means[portfolio_action]
                base_sigma = portfolio_stds[portfolio_action]
                shock = market_shocks[t]

                # Use sentiment-adjusted dynamics when:
                # - Option B is enabled (all methods face regime-switching)
                # - Or sentiment components are involved (even in Option A)
                if use_adjusted_dynamics:
                    mu, sigma = self._compute_sentiment_adjusted_params(
                        base_mu, base_sigma, vix_sentiment, vix_momentum, vix_level
                    )
                else:
                    mu = base_mu
                    sigma = base_sigma

                wealth = wealth * np.exp(mu - 0.5 * sigma**2 + sigma * shock)
                wealth = max(0, wealth)

        return total_reward, goal_taken, wealth

    def _get_portfolio_action_for_hybrid(self, source, num_goals, state_2d, state_4d, wealth, t):
        """Get portfolio action for hybrid simulation"""
        if source == "dp":
            dp = self.dp_policies.get(num_goals)
            if dp:
                # Multi-Goal DP returns (goal_action, portfolio_idx, mu, sigma)
                _, portfolio_idx, _, _ = dp.get_optimal_strategy(wealth, t)
                return int(portfolio_idx)
            return 7
        elif source == "rl":
            agent = self.rl_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state_2d, deterministic=True)
                    return int(action[1])
            return 7
        elif source == "sentiment":
            agent = self.sentiment_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state_4d, deterministic=True)
                    return int(action[1])
            return 7
        return 7

    def _get_goal_action_for_hybrid(self, source, num_goals, state_2d, state_4d, wealth, goal_cost, goal_available, t=0):
        """Get goal action for hybrid simulation"""
        if source == "dp":
            # Multi-Goal DP provides optimal goal decisions
            dp = self.dp_policies.get(num_goals)
            if dp and goal_available:
                goal_action, _, _, _ = dp.get_optimal_strategy(wealth, t)
                # Only take if optimal AND affordable
                if goal_action == 1 and wealth >= goal_cost:
                    return 1
            return 0
        elif source == "rl":
            agent = self.rl_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state_2d, deterministic=True)
                    return int(action[0])
            return 0
        elif source == "sentiment":
            agent = self.sentiment_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state_4d, deterministic=True)
                    return int(action[0])
            return 0
        return 0

    def _get_goal_action(self, source: str, num_goals: int, state: np.ndarray, info: Dict) -> int:
        """Get goal action from specified source"""
        if source == "dp":
            # Multi-Goal DP provides optimal goal decisions
            dp = self.dp_policies.get(num_goals)
            if dp and info.get('goal_available', False):
                wealth = info.get('current_wealth', 0)
                goal_cost = info.get('goal_cost', float('inf'))
                t = int(state[0] * self.config.time_horizon)
                goal_action, _, _, _ = dp.get_optimal_strategy(wealth, t)
                # Only take if optimal AND affordable
                if goal_action == 1 and wealth >= goal_cost:
                    return 1
            return 0
        elif source == "rl":
            agent = self.rl_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state[:2], deterministic=True)  # RL uses 2D state
                    return int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action // 15)
            return 0
        elif source == "sentiment":
            agent = self.sentiment_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state, deterministic=True)
                    return int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action // 15)
            return 0
        return 0

    def _get_portfolio_action(self, source: str, num_goals: int, state: np.ndarray, info: Dict) -> int:
        """Get portfolio action from specified source"""
        if source == "dp":
            dp = self.dp_policies.get(num_goals)
            if dp:
                wealth = info.get('current_wealth', state[1] * get_initial_wealth(num_goals))
                t = int(state[0] * self.config.time_horizon)
                # Multi-Goal DP returns (goal_action, portfolio_idx, mu, sigma)
                _, portfolio_idx, _, _ = dp.get_optimal_strategy(wealth, t)
                return int(portfolio_idx)
            return 7
        elif source == "rl":
            agent = self.rl_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state[:2], deterministic=True)
                    return int(action[1]) if isinstance(action, (list, np.ndarray)) else int(action % 15)
            return 7
        elif source == "sentiment":
            agent = self.sentiment_agents.get(num_goals)
            if agent:
                with torch.no_grad():
                    action = agent.predict(state, deterministic=True)
                    return int(action[1]) if isinstance(action, (list, np.ndarray)) else int(action % 15)
            return 7
        return 7

    def _calculate_efficiencies(self):
        """Calculate efficiencies relative to DP baseline"""
        logger.info("\nCalculating efficiencies...")

        for method_name, method_results in self.results.items():
            for num_goals, result in method_results.items():
                if "DP" in self.results and num_goals in self.results["DP"]:
                    dp_reward = self.results["DP"][num_goals].mean_reward
                    if dp_reward > 0:
                        result.efficiency = result.mean_reward / dp_reward

        for hybrid_name, hybrid_results in self.hybrid_results.items():
            for num_goals, result in hybrid_results.items():
                if "DP" in self.results and num_goals in self.results["DP"]:
                    dp_reward = self.results["DP"][num_goals].mean_reward
                    if dp_reward > 0:
                        result.efficiency = result.mean_reward / dp_reward

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def generate_paper_figures(self, use_historical: bool = False):
        """
        Generate paper-style figures.

        Args:
            use_historical: If True, use historical backtest results; otherwise use simulation results.
        """
        logger.info("=" * 60)
        logger.info("GENERATING PAPER FIGURES")
        if use_historical:
            logger.info("  (Using historical backtesting results)")
        logger.info("=" * 60)

        self._generate_figure1_efficiency_vs_goals(use_historical=use_historical)
        self._generate_figure2_hybrid_analysis(use_historical=use_historical)
        self._generate_summary_dashboard(use_historical=use_historical)

    def _generate_figure1_efficiency_vs_goals(self, use_historical: bool = False):
        """Generate Figure 1: Efficiency vs Number of Goals for all strategies"""
        logger.info("\nGenerating Figure 1: Efficiency vs Number of Goals...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define colors and markers for each method
        method_styles = {
            "DP": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.5},
            "RL": {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "linewidth": 2.5},
            "Sentiment RL": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "linewidth": 2.5},
            "Random": {"color": "#d62728", "marker": "x", "linestyle": "--", "linewidth": 1.5},
            "Buy & Hold": {"color": "#9467bd", "marker": "d", "linestyle": "--", "linewidth": 1.5},
            "Greedy Goal": {"color": "#8c564b", "marker": "v", "linestyle": "--", "linewidth": 1.5},
            "Conservative": {"color": "#e377c2", "marker": "<", "linestyle": ":", "linewidth": 1.5},
            "Aggressive": {"color": "#7f7f7f", "marker": ">", "linestyle": ":", "linewidth": 1.5},
        }

        # Select the appropriate results source
        if use_historical and hasattr(self, 'historical_results') and self.historical_results:
            results_source = self.historical_results
            eval_type = "Historical Backtesting"
        else:
            results_source = self.results
            eval_type = "Monte Carlo Simulation"

        # Track max efficiency for dynamic y-axis
        max_efficiency = 100.0

        # Plot each method
        for method_name, method_results in results_source.items():
            if not method_results:
                continue

            goals = sorted(method_results.keys())

            # Extract efficiencies (handle both dict and SimulationResult)
            efficiencies = []
            for g in goals:
                result = method_results[g]
                if hasattr(result, 'efficiency'):
                    eff = result.efficiency
                elif isinstance(result, dict) and 'efficiency' in result:
                    eff = result['efficiency']
                else:
                    eff = 0.0
                efficiencies.append(eff * 100)

            # Update max efficiency
            if efficiencies:
                max_efficiency = max(max_efficiency, max(efficiencies))

            style = method_styles.get(method_name, {"color": "gray", "marker": ".", "linestyle": "-", "linewidth": 1})

            ax.plot(goals, efficiencies,
                   color=style["color"],
                   marker=style["marker"],
                   linestyle=style["linestyle"],
                   linewidth=style["linewidth"],
                   markersize=10,
                   label=method_name)

        # Formatting
        ax.set_xlabel("Number of Goals", fontsize=14, fontweight='bold')
        ax.set_ylabel("Efficiency (% of DP Optimal)", fontsize=14, fontweight='bold')
        title_suffix = f"\n({eval_type}: 39 historical 16-year windows)" if use_historical else "\n(Paper Section III.D)"
        ax.set_title(f"Figure 1: GBWM Method Efficiency vs Number of Goals{title_suffix}",
                    fontsize=16, fontweight='bold')

        ax.set_xticks(self.config.goal_counts)

        # Dynamic y-axis: add 10% padding above max value
        y_max = max(110, max_efficiency * 1.1)
        ax.set_ylim(0, y_max)

        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3, label='DP Optimal')
        ax.axhline(y=94, color='orange', linestyle=':', alpha=0.5, label='Paper RL Baseline (94%)')

        ax.legend(loc='lower right', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        suffix = "_historical" if use_historical else ""
        fig_path = self.output_dir / f"figure1_efficiency_vs_goals{suffix}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {fig_path}")

    def _generate_figure2_hybrid_analysis(self, use_historical: bool = False):
        """Generate Figure 2: Hybrid Strategy Analysis"""
        logger.info("\nGenerating Figure 2: Hybrid Strategy Analysis...")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define styles for hybrid methods
        hybrid_styles = {
            "DP": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
            "RL": {"color": "#ff7f0e", "marker": "s", "linestyle": "-"},
            "Sentiment RL": {"color": "#2ca02c", "marker": "^", "linestyle": "-"},
            "DP-Port + RL-Goal": {"color": "#17becf", "marker": "D", "linestyle": "--"},
            "RL-Port + DP-Goal": {"color": "#bcbd22", "marker": "p", "linestyle": "--"},
            "DP-Port + Sent-Goal": {"color": "#e377c2", "marker": "h", "linestyle": ":"},
            "Sent-Port + DP-Goal": {"color": "#7f7f7f", "marker": "H", "linestyle": ":"},
            "RL-Port + Sent-Goal": {"color": "#8c564b", "marker": "*", "linestyle": "-."},
            "Sent-Port + RL-Goal": {"color": "#9467bd", "marker": "P", "linestyle": "-."},
        }

        # Select appropriate results source
        if use_historical and hasattr(self, 'historical_results') and self.historical_results:
            results_source = self.historical_results
            eval_type = "Historical Backtesting"
        else:
            results_source = self.results
            eval_type = "Monte Carlo Simulation"

        # Track max efficiency for dynamic y-axis
        max_efficiency = 100.0

        # Plot main methods
        for method_name in ["DP", "RL", "Sentiment RL"]:
            if method_name not in results_source:
                continue

            method_results = results_source[method_name]
            if not method_results:
                continue

            goals = sorted(method_results.keys())

            # Extract efficiencies (handle both dict and SimulationResult)
            efficiencies = []
            for g in goals:
                result = method_results[g]
                if hasattr(result, 'efficiency'):
                    eff = result.efficiency
                elif isinstance(result, dict) and 'efficiency' in result:
                    eff = result['efficiency']
                else:
                    eff = 0.0
                efficiencies.append(eff * 100)

            # Update max efficiency
            if efficiencies:
                max_efficiency = max(max_efficiency, max(efficiencies))

            style = hybrid_styles.get(method_name, {"color": "gray", "marker": ".", "linestyle": "-"})

            ax.plot(goals, efficiencies,
                   color=style["color"],
                   marker=style["marker"],
                   linestyle=style["linestyle"],
                   linewidth=2.5,
                   markersize=10,
                   label=method_name)

        # Plot hybrid methods (only available for simulation results)
        if not use_historical:
            for hybrid_name, hybrid_results in self.hybrid_results.items():
                if not hybrid_results:
                    continue

                goals = sorted(hybrid_results.keys())
                efficiencies = [hybrid_results[g].efficiency * 100 for g in goals]

                # Update max efficiency
                if efficiencies:
                    max_efficiency = max(max_efficiency, max(efficiencies))

                style = hybrid_styles.get(hybrid_name, {"color": "gray", "marker": ".", "linestyle": "--"})

                ax.plot(goals, efficiencies,
                       color=style["color"],
                       marker=style["marker"],
                       linestyle=style["linestyle"],
                       linewidth=1.5,
                       markersize=8,
                       label=hybrid_name)

        # Formatting
        ax.set_xlabel("Number of Goals", fontsize=14, fontweight='bold')
        ax.set_ylabel("Efficiency (% of DP Optimal)", fontsize=14, fontweight='bold')
        title_suffix = f"\n({eval_type})" if use_historical else "\n(DP/RL/Sentiment Component Mixing)"
        ax.set_title(f"Figure 2: Hybrid Strategy Analysis{title_suffix}",
                    fontsize=16, fontweight='bold')

        ax.set_xticks(self.config.goal_counts)

        # Dynamic y-axis: add 10% padding above max value
        y_max = max(110, max_efficiency * 1.1)
        ax.set_ylim(0, y_max)

        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3)

        ax.legend(loc='lower right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        suffix = "_historical" if use_historical else ""
        fig_path = self.output_dir / f"figure2_hybrid_analysis{suffix}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {fig_path}")

    def _generate_summary_dashboard(self, use_historical: bool = False):
        """Generate summary dashboard with all results"""
        logger.info("\nGenerating Summary Dashboard...")

        # Select the appropriate results source
        if use_historical and hasattr(self, 'historical_results') and self.historical_results:
            results_source = self.historical_results
            eval_type = "Historical Backtesting (39 windows)"
        else:
            results_source = self.results
            eval_type = f"{self.config.num_simulations:,} Monte Carlo Simulations"

        # Calculate number of rows needed for table
        num_rows = sum(len(results) for results in results_source.values())
        num_rows = min(num_rows, 20)  # Cap at 20 rows

        # Adjust figure height based on table size
        table_height_ratio = max(1.5, num_rows * 0.1)
        fig = plt.figure(figsize=(20, 14 + table_height_ratio * 2))

        # Create grid with more vertical spacing
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, table_height_ratio],
                             hspace=0.45, wspace=0.3,
                             top=0.92, bottom=0.05, left=0.08, right=0.95)

        # Subplot 1: Efficiency comparison (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_efficiency_bars(ax1, results_source)

        # Subplot 2: Success rates
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_success_rates(ax2, results_source)

        # Subplot 3: Mean rewards
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_mean_rewards(ax3, results_source)

        # Subplot 4: Summary table
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_summary_table(ax4, results_source)

        fig.suptitle(f"GBWM Comprehensive Comparison Dashboard\n({eval_type})",
                    fontsize=18, fontweight='bold', y=0.97)

        suffix = "_historical" if use_historical else ""
        fig_path = self.output_dir / f"summary_dashboard{suffix}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {fig_path}")

    def _plot_efficiency_bars(self, ax, results_source=None):
        """Plot efficiency bar chart for 4-goal case"""
        if results_source is None:
            results_source = self.results

        num_goals = 4  # Focus on 4-goal case

        methods = []
        efficiencies = []
        colors = []

        color_map = {
            "DP": "#1f77b4",
            "RL": "#ff7f0e",
            "Sentiment RL": "#2ca02c",
            "Random": "#d62728",
            "Buy & Hold": "#9467bd",
            "Greedy Goal": "#8c564b",
            "Conservative": "#e377c2",
            "Aggressive": "#7f7f7f"
        }

        for method_name, method_results in results_source.items():
            if num_goals in method_results:
                result = method_results[num_goals]
                # Handle both SimulationResult and dict
                if hasattr(result, 'efficiency'):
                    eff = result.efficiency
                elif isinstance(result, dict) and 'efficiency' in result:
                    eff = result['efficiency']
                else:
                    eff = 0.0
                methods.append(method_name)
                efficiencies.append(eff * 100)
                colors.append(color_map.get(method_name, "gray"))

        if not methods:
            ax.text(0.5, 0.5, "No results for 4 goals", ha='center', va='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_title("Method Efficiency Comparison (4 Goals)")
            return

        bars = ax.bar(range(len(methods)), efficiencies, color=colors, alpha=0.8)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel("Efficiency (%)")
        ax.set_title(f"Method Efficiency Comparison ({num_goals} Goals)")
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3)

        # Dynamic y-axis: add padding above max value for labels
        max_eff = max(efficiencies) if efficiencies else 100
        y_max = max(110, max_eff * 1.15)  # 15% padding for text labels
        ax.set_ylim(0, y_max)

        for bar, eff in zip(bars, efficiencies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)

    def _plot_success_rates(self, ax, results_source=None):
        """Plot goal success rates"""
        if results_source is None:
            results_source = self.results

        plotted = False
        for method_name in ["DP", "RL", "Sentiment RL"]:
            if method_name not in results_source:
                continue

            method_results = results_source[method_name]
            if not method_results:
                continue

            goals = sorted(method_results.keys())
            success_rates = []
            for g in goals:
                result = method_results[g]
                # Handle both SimulationResult and dict
                if hasattr(result, 'goal_success_rate'):
                    rate = result.goal_success_rate
                elif isinstance(result, dict) and 'mean_goals_taken' in result:
                    # For historical results, use mean_goals_taken / num_goals as proxy
                    rate = result.get('mean_goals_taken', 0) / g if g > 0 else 0
                else:
                    rate = 0.0
                success_rates.append(rate * 100)

            if goals:
                ax.plot(goals, success_rates, marker='o', label=method_name, linewidth=2)
                plotted = True

        ax.set_xlabel("Number of Goals")
        ax.set_ylabel("Goal Success Rate (%)")
        ax.set_title("Goal Achievement Rates")
        if plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_mean_rewards(self, ax, results_source=None):
        """Plot mean rewards"""
        if results_source is None:
            results_source = self.results

        plotted = False
        for method_name in ["DP", "RL", "Sentiment RL"]:
            if method_name not in results_source:
                continue

            method_results = results_source[method_name]
            if not method_results:
                continue

            goals = sorted(method_results.keys())
            rewards = []
            for g in goals:
                result = method_results[g]
                # Handle both SimulationResult and dict
                if hasattr(result, 'mean_reward'):
                    reward = result.mean_reward
                elif isinstance(result, dict) and 'mean_reward' in result:
                    reward = result['mean_reward']
                else:
                    reward = 0.0
                rewards.append(reward)

            if goals:
                ax.plot(goals, rewards, marker='s', label=method_name, linewidth=2)
                plotted = True

        ax.set_xlabel("Number of Goals")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward by Method")
        if plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_summary_table(self, ax, results_source=None):
        """Plot summary statistics table"""
        if results_source is None:
            results_source = self.results

        ax.axis('off')

        # Create table data
        headers = ["Method", "Goals", "Mean Reward", "Std", "Efficiency", "Success Rate"]
        rows = []

        for method_name, method_results in results_source.items():
            for num_goals, result in sorted(method_results.items()):
                # Handle both SimulationResult and dict
                if hasattr(result, 'mean_reward'):
                    mean_reward = result.mean_reward
                    std_reward = result.std_reward
                    efficiency = result.efficiency
                    success_rate = result.goal_success_rate
                elif isinstance(result, dict):
                    mean_reward = result.get('mean_reward', 0)
                    std_reward = result.get('std_reward', 0)
                    efficiency = result.get('efficiency', 0)
                    # For historical results, calculate success rate from mean_goals_taken
                    if 'goal_success_rate' in result:
                        success_rate = result['goal_success_rate']
                    elif 'mean_goals_taken' in result:
                        success_rate = result['mean_goals_taken'] / num_goals if num_goals > 0 else 0
                    else:
                        success_rate = 0
                else:
                    continue

                rows.append([
                    method_name,
                    str(num_goals),
                    f"{mean_reward:.2f}",
                    f"{std_reward:.2f}",
                    f"{efficiency*100:.1f}%",
                    f"{success_rate*100:.1f}%"
                ])

        # Limit to first 20 rows for readability
        if len(rows) > 20:
            rows = rows[:20]
            rows.append(["...", "...", "...", "...", "...", "..."])

        # Handle empty results case
        if len(rows) == 0:
            ax.text(0.5, 0.5, "No results available", ha='center', va='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_title("Summary Statistics", fontsize=12, fontweight='bold', pad=20)
            return

        table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        ax.set_title("Summary Statistics", fontsize=12, fontweight='bold', pad=20)

    # =========================================================================
    # SAVE AND REPORT
    # =========================================================================

    def save_results(self):
        """Save all results to files"""
        logger.info("\nSaving results...")

        # Save main results
        results_dict = {}
        for method_name, method_results in self.results.items():
            results_dict[method_name] = {
                str(k): asdict(v) for k, v in method_results.items()
            }

        with open(self.output_dir / "results.json", 'w') as f:
            json.dump(results_dict, f, indent=2, default=float)

        # Save hybrid results
        hybrid_dict = {}
        for hybrid_name, hybrid_results in self.hybrid_results.items():
            hybrid_dict[hybrid_name] = {
                str(k): asdict(v) for k, v in hybrid_results.items()
            }

        with open(self.output_dir / "hybrid_results.json", 'w') as f:
            json.dump(hybrid_dict, f, indent=2, default=float)

        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"  Results saved to: {self.output_dir}")

    def print_summary(self):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("GBWM INTEGRATED COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\nSimulations: {self.config.num_simulations:,}")
        print(f"Goal counts: {self.config.goal_counts}")

        print("\n" + "-" * 80)
        print(f"{'Method':<20} {'Goals':<8} {'Reward':<12} {'Efficiency':<12} {'Success':<10}")
        print("-" * 80)

        for method_name, method_results in self.results.items():
            for num_goals, result in sorted(method_results.items()):
                print(f"{method_name:<20} {num_goals:<8} {result.mean_reward:<12.2f} "
                      f"{result.efficiency*100:<12.1f}% {result.goal_success_rate*100:<10.1f}%")

        if self.hybrid_results:
            print("\n" + "-" * 80)
            print("HYBRID STRATEGIES")
            print("-" * 80)

            for hybrid_name, hybrid_results in self.hybrid_results.items():
                for num_goals, result in sorted(hybrid_results.items()):
                    print(f"{hybrid_name:<25} {num_goals:<8} {result.mean_reward:<12.2f} "
                          f"{result.efficiency*100:<12.1f}%")

        print("\n" + "=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)

    def run(self, use_historical_eval: bool = True):
        """
        Run complete comparison pipeline.

        Args:
            use_historical_eval: If True, use historical backtesting for evaluation (recommended).
                               If False, use Monte Carlo simulation (legacy behavior).
        """
        start_time = time.time()

        try:
            # Train all models
            self.train_all_models()

            # Evaluate models
            if use_historical_eval:
                # Historical backtesting (deterministic, real data)
                self.run_historical_evaluation()
            else:
                # Monte Carlo simulation (legacy)
                self.run_all_simulations()

            # Generate figures and save
            self.generate_paper_figures(use_historical=use_historical_eval)
            self.save_results()
            self.print_summary()

            total_time = time.time() - start_time
            logger.info(f"\nTotal execution time: {total_time/60:.1f} minutes")

            return True

        except Exception as e:
            logger.error(f"Comparison failed: {e}", exc_info=True)
            return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Integrated GBWM Comparison with Paper Figures")

    # Simulation settings
    parser.add_argument('--num_simulations', type=int, default=100000,
                       help='Number of Monte Carlo simulations (default: 100000)')
    parser.add_argument('--goal_counts', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                       help='Goal counts to evaluate')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base random seed')

    # Training settings
    parser.add_argument('--num_iterations', type=int, default=10,
                       help='Training iterations per model')
    parser.add_argument('--batch_size', type=int, default=4800,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')

    # Architecture settings
    parser.add_argument('--config_preset', type=str, default='default',
                       choices=['default', 'conservative', 'aggressive', 'research'],
                       help='Configuration preset')
    parser.add_argument('--policy_type', type=str, default='standard',
                       choices=['standard', 'hierarchical'])
    parser.add_argument('--value_type', type=str, default='standard',
                       choices=['standard', 'dual_head', 'ensemble'])
    parser.add_argument('--encoder_type', type=str, default='feature',
                       choices=['simple', 'feature', 'adaptive', 'attention'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dp_grid_density', type=float, default=None,
                       help='DP wealth grid density (0.5=fast, 1.5=default, 3.0=paper accurate)')

    # Data mode
    parser.add_argument('--data_mode', type=str, default='simulation',
                       choices=['simulation', 'historical'])

    # Flags
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, use existing models')
    parser.add_argument('--skip_dp', action='store_true',
                       help='Skip DP computation')
    parser.add_argument('--skip_sentiment', action='store_true',
                       help='Skip sentiment-aware methods')
    # Evaluation mode selection
    parser.add_argument('--eval_mode', type=str, default='monte_carlo',
                       choices=['monte_carlo', 'historical'],
                       help='Evaluation mode: '
                            '"monte_carlo" = Simulated training + Monte Carlo evaluation with shared seeds (original paper approach). '
                            '"historical" = Simulated training + Real historical data evaluation (39 windows from 1970-2023). '
                            'Default: monte_carlo')
    # Legacy flags (kept for backward compatibility)
    parser.add_argument('--use_historical_eval', action='store_true',
                       help='[DEPRECATED] Use --eval_mode=historical instead. Forces historical evaluation.')
    parser.add_argument('--use_monte_carlo', action='store_true',
                       help='[DEPRECATED] Use --eval_mode=monte_carlo instead. Forces Monte Carlo evaluation.')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer simulations, smaller models)')
    parser.add_argument('--use_sentiment_adjusted_returns', action='store_true',
                       help='Option B: Apply sentiment adjustments to mu/sigma for wealth evolution. '
                            'Uses VIX → Returns causality with β/δ formula. '
                            'Default (off) = Option A: All methods use base mu/sigma, sentiment only affects decisions.')
    parser.add_argument('--use_sentiment_trainer', action='store_true', default=True,
                       help='Use SentimentGBWMTrainer (monthly steps, pre-trained β/δ, efficient frontier). '
                            'Default: True. Use --no_sentiment_trainer to disable.')
    parser.add_argument('--no_sentiment_trainer', action='store_true',
                       help='Disable SentimentGBWMTrainer and use legacy training approach.')
    parser.add_argument('--ultra_quick', action='store_true',
                       help='Ultra-quick pipeline test (skips DP, minimal simulations)')

    # Monthly time step settings (NEW: correct VIX causality)
    parser.add_argument('--use_monthly_steps', action='store_true', default=True,
                       help='Use monthly time steps (192 steps) instead of yearly (16 steps). '
                            'Provides more granular VIX learning opportunities. Default: True.')
    parser.add_argument('--use_yearly_steps', action='store_true',
                       help='Force yearly time steps (16 steps). Faster but less VIX dynamics.')

    # VIX model parameters
    parser.add_argument('--vix_beta_sensitivity', type=float, default=0.03,
                       help='How much VIX affects μ (drift adjustment). Default: 0.03')
    parser.add_argument('--vix_delta_sensitivity', type=float, default=0.05,
                       help='How much VIX affects σ (volatility adjustment). Default: 0.05')
    parser.add_argument('--vix_theta', type=float, default=20.0,
                       help='VIX long-term mean. Default: 20.0')
    parser.add_argument('--vix_kappa', type=float, default=3.0,
                       help='VIX mean reversion speed. Default: 3.0')

    # Efficient Frontier settings
    parser.add_argument('--use_rolling_ef', action='store_true',
                       help='Use rolling efficient frontier (recompute weights using only past data). '
                            'More realistic but slower. Default: fixed weights.')
    parser.add_argument('--ef_lookback_years', type=int, default=20,
                       help='Years of historical data for rolling EF calculation (default: 20)')
    parser.add_argument('--ef_min_years', type=int, default=10,
                       help='Minimum years required for rolling EF calculation (default: 10)')

    # Crisis analysis
    parser.add_argument('--crisis_analysis', action='store_true',
                       help='Enable year-by-year crisis analysis. Generates visualizations comparing '
                            'DP, RL, Sentiment RL during crisis years (2008, 2020). '
                            'Requires --eval_mode=historical.')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda', 'auto'])

    args = parser.parse_args()

    # Apply ultra-quick mode settings (pipeline test only)
    if args.ultra_quick:
        logger.info("ULTRA-QUICK MODE: Pipeline test (skipping DP)")
        args.num_simulations = 20
        args.num_iterations = 1
        args.batch_size = 100
        args.goal_counts = [4]
        args.skip_dp = True  # Skip DP training entirely
        args.skip_sentiment = True
        args.quick = True  # Also enable quick mode

    # Apply quick mode settings
    elif args.quick:
        logger.info("QUICK MODE: Using reduced settings for fast testing")
        args.num_simulations = min(args.num_simulations, 100)
        args.num_iterations = 1
        args.batch_size = 480
        args.goal_counts = [4]  # Only test 4 goals
        args.skip_sentiment = True

    # Setup device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create config
    # dp_grid_density controls Multi-Goal DP wealth grid resolution
    # Higher values = more accurate but slower (0.5=fast, 1.5=default, 3.0=paper)
    if args.dp_grid_density is not None:
        dp_grid_density = args.dp_grid_density
    else:
        dp_grid_density = 0.5 if args.quick else 1.5

    # Determine monthly vs yearly steps
    use_monthly = args.use_monthly_steps and not args.use_yearly_steps

    config = ComparisonConfig(
        num_simulations=args.num_simulations,
        goal_counts=args.goal_counts,
        base_seed=args.base_seed,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_type=args.policy_type,
        value_type=args.value_type,
        encoder_type=args.encoder_type,
        hidden_dim=args.hidden_dim,
        dp_grid_density=dp_grid_density,
        data_mode=args.data_mode,
        skip_training=args.skip_training,
        skip_dp=args.skip_dp,
        skip_sentiment=args.skip_sentiment,
        quick_mode=args.quick,
        use_rolling_ef=args.use_rolling_ef,
        ef_lookback_years=args.ef_lookback_years,
        ef_min_years=args.ef_min_years,
        crisis_analysis=args.crisis_analysis,
        use_sentiment_adjusted_returns=args.use_sentiment_adjusted_returns,
        # Monthly time step settings
        use_monthly_steps=use_monthly,
        # VIX model parameters
        vix_beta_sensitivity=args.vix_beta_sensitivity,
        vix_delta_sensitivity=args.vix_delta_sensitivity,
        vix_theta=args.vix_theta,
        vix_kappa=args.vix_kappa,
        device=args.device,
        # Sentiment trainer settings
        use_sentiment_trainer=args.use_sentiment_trainer and not args.no_sentiment_trainer
    )

    # Apply preset
    config = apply_config_preset(config, args.config_preset)

    logger.info(f"Configuration preset: {args.config_preset}")
    logger.info(f"Architecture: policy={config.policy_type}, value={config.value_type}, encoder={config.encoder_type}")

    # Log sentiment adjustment mode
    if config.use_sentiment_adjusted_returns:
        logger.info("Sentiment mode: Option B - VIX adjusts mu/sigma via β/δ formula")
        logger.info(f"  VIX → Returns: μ_adj = μ + β(VIX), σ_adj = σ - δ(VIX)")
        logger.info(f"  β_sensitivity={config.vix_beta_sensitivity}, δ_sensitivity={config.vix_delta_sensitivity}")
        logger.info("  (Note: Only affects Monte Carlo simulation. Historical backtesting uses real returns.)")
    else:
        logger.info("Sentiment mode: Option A - Sentiment as pure information (base mu/sigma for all)")

    # Log time resolution
    if config.use_monthly_steps:
        total_steps = config.time_horizon * config.months_per_year
        logger.info(f"Time resolution: MONTHLY ({total_steps} steps = {config.time_horizon} years × {config.months_per_year} months)")
    else:
        logger.info(f"Time resolution: YEARLY ({config.time_horizon} steps)")

    # Log sentiment trainer mode
    if config.use_sentiment_trainer:
        logger.info("Sentiment training: SentimentGBWMTrainer (monthly steps, pre-trained β/δ, efficient frontier)")
    else:
        logger.info("Sentiment training: Legacy approach (yearly steps)")

    # Determine evaluation mode
    # Priority: legacy flags > --eval_mode > default
    if args.use_historical_eval:
        # Legacy flag takes precedence
        logger.warning("--use_historical_eval is deprecated. Use --eval_mode=historical instead.")
        use_historical = True
    elif args.use_monte_carlo:
        # Legacy flag takes precedence
        logger.warning("--use_monte_carlo is deprecated. Use --eval_mode=monte_carlo instead.")
        use_historical = False
    else:
        # Use new --eval_mode argument
        use_historical = (args.eval_mode == 'historical')

    # Check crisis analysis requirements
    if args.crisis_analysis and not use_historical:
        logger.warning("--crisis_analysis requires --eval_mode=historical. Enabling historical evaluation.")
        use_historical = True

    # Log evaluation mode details
    if use_historical:
        ef_mode = "Rolling EF" if config.use_rolling_ef else "Fixed weights"
        logger.info("=" * 60)
        logger.info("EVALUATION MODE: HISTORICAL BACKTESTING")
        logger.info("=" * 60)
        logger.info("  Training: Simulated data (GBM)")
        logger.info("  Evaluation: Real historical data (1970-2023)")
        logger.info(f"  Windows: 39 overlapping 16-year periods")
        logger.info(f"  Efficient frontier: {ef_mode}")
        if config.use_rolling_ef:
            logger.info(f"  - Lookback: {config.ef_lookback_years} years, min: {config.ef_min_years} years")
        if config.crisis_analysis:
            logger.info("  Crisis analysis: ENABLED")
    else:
        logger.info("=" * 60)
        logger.info("EVALUATION MODE: MONTE CARLO SIMULATION")
        logger.info("=" * 60)
        logger.info("  Training: Simulated data (GBM)")
        logger.info("  Evaluation: Simulated data with shared seeds")
        logger.info(f"  Simulations: {args.num_simulations:,}")
        logger.info("  Note: All methods use identical market shocks for fair comparison")

    # Run comparison
    comparison = IntegratedComparison(config, args.output_dir)
    success = comparison.run(use_historical_eval=use_historical)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
