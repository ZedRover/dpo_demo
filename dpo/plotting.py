"""Plotting utilities for visualizing DPO training results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import json


def plot_reward_kl_frontier(
    tracker_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "DPO Reward vs KL Divergence",
    comparison_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """
    Plot the reward-KL frontier (Figure 2 left from the paper).

    Args:
        tracker_data: Data from RewardKLTracker
        save_path: Path to save the plot
        title: Plot title
        comparison_data: Optional dict of {method_name: tracker_data} for comparison
    """
    plt.figure(figsize=(10, 6))

    # Plot main data
    plt.scatter(
        tracker_data['kls'],
        tracker_data['rewards'],
        alpha=0.6,
        s=100,
        label='DPO',
        color='blue',
        marker='o',
    )

    # Plot comparison methods if provided
    if comparison_data:
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        markers = ['s', '^', 'D', 'v', 'p']

        for idx, (method_name, data) in enumerate(comparison_data.items()):
            plt.scatter(
                data['kls'],
                data['rewards'],
                alpha=0.6,
                s=100,
                label=method_name,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
            )

    plt.xlabel('KL Divergence from Reference Policy', fontsize=12)
    plt.ylabel('Expected Reward', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_training_curves(
    tracker_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training curves showing reward and KL over time.

    Args:
        tracker_data: Data from RewardKLTracker
        save_path: Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    steps = tracker_data['steps']

    # Plot 1: Rewards over time
    ax1.plot(steps, tracker_data['chosen_rewards'], label='Chosen Rewards', color='green', linewidth=2)
    ax1.plot(steps, tracker_data['rejected_rewards'], label='Rejected Rewards', color='red', linewidth=2)
    ax1.plot(steps, tracker_data['rewards'], label='Average Reward', color='blue', linewidth=2, linestyle='--')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Reward')
    ax1.set_title('Rewards over Training', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: KL divergence over time
    ax2.plot(steps, tracker_data['kls'], label='KL Divergence', color='purple', linewidth=2)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence from Reference over Training', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Reward margin (chosen - rejected)
    reward_margins = [c - r for c, r in zip(tracker_data['chosen_rewards'], tracker_data['rejected_rewards'])]
    ax3.plot(steps, reward_margins, label='Reward Margin', color='orange', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Reward Margin (Chosen - Rejected)')
    ax3.set_title('Reward Margin over Training', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_win_rate_by_temperature(
    temperatures: List[float],
    win_rates: List[float],
    method_name: str = "DPO",
    save_path: Optional[str] = None,
    comparison_methods: Optional[Dict[str, List[float]]] = None,
):
    """
    Plot win rate vs sampling temperature (Figure 2 right and Figure 3 from the paper).

    Args:
        temperatures: List of temperature values
        win_rates: List of win rates corresponding to each temperature
        method_name: Name of the method being plotted
        save_path: Path to save the plot
        comparison_methods: Dict of {method_name: win_rates_list} for comparison
    """
    plt.figure(figsize=(10, 6))

    # Plot main method
    plt.plot(
        temperatures,
        win_rates,
        marker='o',
        linewidth=2,
        markersize=8,
        label=method_name,
        color='blue',
    )

    # Plot comparison methods if provided
    if comparison_methods:
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        markers = ['s', '^', 'D', 'v', 'p']

        for idx, (method, rates) in enumerate(comparison_methods.items()):
            plt.plot(
                temperatures,
                rates,
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=8,
                label=method,
                color=colors[idx % len(colors)],
            )

    # Add 50% reference line
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random (50%)')

    plt.xlabel('Sampling Temperature', fontsize=12)
    plt.ylabel('Win Rate vs Reference', fontsize=12)
    plt.title('Win Rate vs Sampling Temperature', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def create_paper_style_plots(
    tracker_path: str,
    output_dir: str = "./plots",
):
    """
    Create all plots in the style of the DPO paper.

    Args:
        tracker_path: Path to saved RewardKLTracker data
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load tracker data
    with open(tracker_path, 'r') as f:
        tracker_data = json.load(f)

    # Plot 1: Reward-KL Frontier
    plot_reward_kl_frontier(
        tracker_data,
        save_path=os.path.join(output_dir, 'reward_kl_frontier.png'),
    )

    # Plot 2: Training Curves
    plot_training_curves(
        tracker_data,
        save_path=os.path.join(output_dir, 'training_curves.png'),
    )

    print(f"All plots saved to {output_dir}")


def load_and_compare_methods(
    tracker_paths: Dict[str, str],
    output_dir: str = "./plots",
):
    """
    Load data from multiple methods and create comparison plots.

    Args:
        tracker_paths: Dict of {method_name: tracker_path}
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_data = {}
    for method_name, path in tracker_paths.items():
        with open(path, 'r') as f:
            all_data[method_name] = json.load(f)

    # Get the first method as main, rest as comparison
    methods = list(all_data.keys())
    main_method = methods[0]
    main_data = all_data[main_method]

    comparison_data = {m: all_data[m] for m in methods[1:]} if len(methods) > 1 else None

    # Create comparison plot
    plot_reward_kl_frontier(
        main_data,
        save_path=os.path.join(output_dir, 'method_comparison.png'),
        title="Reward-KL Frontier Comparison",
        comparison_data=comparison_data,
    )

    print(f"Comparison plot saved to {output_dir}")
