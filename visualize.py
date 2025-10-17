"""Visualization script for creating paper-style plots."""

import argparse
import os

from dpo.plotting import create_paper_style_plots, load_and_compare_methods


def main():
    parser = argparse.ArgumentParser(
        description="Create paper-style visualizations from training data"
    )
    parser.add_argument(
        "--tracker_path",
        type=str,
        required=True,
        help="Path to reward_kl_tracker.json file from training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs='+',
        default=None,
        help="Additional tracker files to compare (format: method_name:path)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare:
        # Parse comparison files
        tracker_paths = {"DPO": args.tracker_path}

        for comp in args.compare:
            if ':' in comp:
                name, path = comp.split(':', 1)
                tracker_paths[name] = path
            else:
                # Use filename as method name
                name = os.path.basename(comp).replace('_tracker.json', '').replace('reward_kl_', '')
                tracker_paths[name] = comp

        print(f"Comparing methods: {list(tracker_paths.keys())}")
        load_and_compare_methods(tracker_paths, args.output_dir)

    else:
        # Single method visualization
        print(f"Creating plots from: {args.tracker_path}")
        create_paper_style_plots(args.tracker_path, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
