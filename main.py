"""
Main Entry Point — Drone Altitude Stabilization Project
========================================================
Interactive menu to run:
  1. Stabilizer simulation  (with configurable time steps & initial height)
  2. Classical vs Quantum comparison
  3. Large-scale benchmark
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def print_banner():
    print("\n" + "=" * 60)
    print("  Drone Altitude Stabilization — QFIE Fuzzy Controller")
    print("=" * 60)
    print("  [1]  Stabilizer Simulation")
    print("  [2]  Classical vs Quantum Comparison")
    print("  [3]  Large-Scale Benchmark")
    print("  [0]  Exit")
    print("=" * 60)


def get_float(prompt, default):
    """Prompt for a float value with a default."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid input, using default ({default})")
        return default


def get_int(prompt, default):
    """Prompt for an integer value with a default."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid input, using default ({default})")
        return default


def run_stabilizer():
    """Run the stabilizer simulation with user-configured parameters."""
    from stablizer import DroneAltitudeController

    print("\n--- Stabilizer Configuration ---")
    initial_error = get_float("Initial altitude error in cm (positive = below target)", -15.0)
    velocity      = get_float("Initial velocity (cm/s)", 0.0)
    steps         = get_int("Number of time steps", 150)

    controller = DroneAltitudeController()
    controller.run_simulation(error=initial_error, velocity=velocity, steps=steps)


def run_comparison():
    """Run the classical vs quantum comparison."""
    from comparison import main as comparison_main
    comparison_main()


def run_benchmark():
    """Run the large-scale benchmark."""
    from large_scale_benchmark import main as benchmark_main
    benchmark_main()


def main():
    while True:
        print_banner()
        choice = input("  Enter your choice: ").strip()

        if choice == "1":
            run_stabilizer()
        elif choice == "2":
            run_comparison()
        elif choice == "3":
            run_benchmark()
        elif choice == "0":
            print("\n  Goodbye!\n")
            break
        else:
            print("\n  Invalid choice. Please enter 0–3.")


if __name__ == "__main__":
    main()
