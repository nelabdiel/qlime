import argparse
from highlighting import run_q_lime_example_with_html
from comparison import run_comparison
from benchmark import benchmark_test

def main():
    parser = argparse.ArgumentParser(description="Run Q-LIME experiments.")
    parser.add_argument(
        "task",
        choices=["example", "comparison", "benchmark"],
        help="Choose the task to execute: example (Q-LIME visualization), comparison (Classical vs Q-LIME), benchmark (performance evaluation)."
    )
    args = parser.parse_args()

    if args.task == "example":
        print("Running Q-LIME example visualization...\n")
        run_q_lime_example_with_html()
    elif args.task == "comparison":
        print("Running Classical LIME vs Q-LIME comparison...\n")
        run_comparison()
    elif args.task == "benchmark":
        print("Running benchmark tests for Q-LIME and Classical LIME...\n")
        benchmark_test()
    else:
        print("Invalid task. Please choose from 'example', 'comparison', or 'benchmark'.")

if __name__ == "__main__":
    main()
