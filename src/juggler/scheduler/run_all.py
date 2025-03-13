#!/usr/bin/env python3
"""
Run the accuracy-enhanced scheduler simulation.

This script launches the scheduler simulation with accuracy metrics tracking
to evaluate different scheduling strategies for machine learning inference.
"""
import os
import sys
from datetime import datetime
import logging

# Create log directory
timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M')
version = "v1.0.4-accuracy-sim"
log_dir = f"log-images/{timestamp}-{version}"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/simulation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the simulator
from scheduler_sim_trend import (
    run_all_scenarios_with_accuracy,
    generate_summary_report,
    MODEL_NAME_MAPPING
)

def main():
    """Main function to run the simulation"""
    print("=" * 80)
    print("ACCURACY-ENHANCED SCHEDULER SIMULATION")
    print("=" * 80)
    print("\nThis simulation evaluates different scheduling strategies for ML inference workloads")
    print("with a focus on both system efficiency and inference accuracy.")
    print("\nModel mapping:")
    for code_name, accuracy_name in MODEL_NAME_MAPPING.items():
        print(f"  {code_name} -> {accuracy_name}")
    
    print("\nRunning all scenarios. This may take a few minutes...")
    print("Results will be saved to:", log_dir)
    
    # Run all scenarios
    all_metrics = run_all_scenarios_with_accuracy()
    
    # Generate summary report
    generate_summary_report(all_metrics)
    
    print("\nSimulation complete!")
    print(f"Check {log_dir} for detailed visualization of results")
    print("=" * 80)

if __name__ == "__main__":
    main()


# # Run all scenarios and compare with accuracy metrics
# def run_all_scenarios_with_accuracy():
#     """Run all scenarios with accuracy tracking and generate comparative visualizations"""
#     # First, generate reference accuracy matrix
#     plot_model_image_accuracy_matrix()
    
#     # Set up containers for all metrics
#     all_metrics = {}
    
#     # Run Scenario 1 with accuracy tracking
#     print("\n=== Running Scenario 1 with accuracy tracking: Scheduler decides both device and model ===")
#     scenario1_metrics = run_scenario_scheduler_decides_all_modified()
#     if scenario1_metrics is not None:
#         all_metrics["Scenario 1"] = scenario1_metrics
#         plot_scenario_results(scenario1_metrics, "Scenario 1: Scheduler Decides Both")
#         plot_accuracy_metrics(scenario1_metrics, "Scenario 1: Scheduler Decides Both")
    
#     # Run Scenario 2 with accuracy tracking
#     print("\n=== Running Scenario 2 with accuracy tracking: User decides device, scheduler decides model ===")
#     scenario2_metrics = run_scenario_user_decides_device_modified()
#     if scenario2_metrics is not None:
#         all_metrics["Scenario 2"] = scenario2_metrics
#         plot_scenario_results(scenario2_metrics, "Scenario 2: User Decides Device")
#         plot_accuracy_metrics(scenario2_metrics, "Scenario 2: User Decides Device")
    
#     # Run Scenario 3 with accuracy tracking
#     print("\n=== Running Scenario 3 with accuracy tracking: User decides model, scheduler decides device ===")
#     scenario3_metrics = run_scenario_user_decides_model_modified()
#     if scenario3_metrics is not None:
#         all_metrics["Scenario 3"] = scenario3_metrics
#         plot_scenario_results(scenario3_metrics, "Scenario 3: User Decides Model")
#         plot_accuracy_metrics(scenario3_metrics, "Scenario 3: User Decides Model")
    
#     # Run Scenario 4 with accuracy tracking
#     print("\n=== Running Scenario 4 with accuracy tracking: Adaptive Model Switching ===")
#     scenario4_metrics = run_scenario_adaptive_model_switching()
#     if scenario4_metrics is not None:
#         all_metrics["Scenario 4"] = scenario4_metrics
#         plot_scenario_results(scenario4_metrics, "Scenario 4: Adaptive Model Switching")
#         plot_accuracy_metrics(scenario4_metrics, "Scenario 4: Adaptive Model Switching")
    
#     # Run Scenario 5 with accuracy tracking
#     print("\n=== Running Scenario 5 with accuracy tracking: Cross-Device Model Consistency ===")
#     scenario5_metrics = run_scenario_cross_device_model_consistency()
#     if scenario5_metrics is not None:
#         all_metrics["Scenario 5"] = scenario5_metrics
#         plot_scenario_results(scenario5_metrics, "Scenario 5: Cross-Device Model Consistency")
#         plot_accuracy_metrics(scenario5_metrics, "Scenario 5: Cross-Device Model Consistency")

#     # Run Scenario 6 with accuracy tracking
#     print("\n=== Running Scenario 6 with accuracy tracking: Accuracy-Optimized Scheduler ===")
#     scenario6_metrics = run_scenario_accuracy_optimized()
#     if scenario6_metrics is not None:
#         all_metrics["Scenario 6"] = scenario6_metrics
#         plot_scenario_results(scenario6_metrics, "Scenario 6: Accuracy-Optimized Scheduler")
#         plot_accuracy_metrics(scenario6_metrics, "Scenario 6: Accuracy-Optimized Scheduler")

#     # Run Scenario 7 with power tracking
#     print("\n=== Running Scenario 7 with accuracy tracking: Power-Efficiency-Optimized Scheduler ===")
#     scenario7_metrics = run_scenario_power_optimized()
#     if scenario7_metrics is not None:
#         all_metrics["Scenario 7"] = scenario7_metrics
#         plot_scenario_results(scenario7_metrics, "Scenario 7: Power-Efficiency-Optimized Scheduler")
#         plot_accuracy_metrics(scenario7_metrics, "Scenario 7: Power-Efficiency-Optimized Scheduler")
    
#     # Generate comparative visualization
#     if len(all_metrics) > 1:
#         create_comparative_plots(all_metrics)
#         plot_comparative_accuracy(all_metrics)
    
#     return all_metrics