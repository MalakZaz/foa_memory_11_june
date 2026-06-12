"""
===========================================================
 FILE: show_experiment_info.py
 PURPOSE: Display complete experiment information
===========================================================
"""

from src.utils.experiment_info import (
    print_execution_info,
    print_scenario,
    print_experiment_description,
    print_problem_formulation,
    print_metrics
)


if __name__ == "__main__":

    print_execution_info()
    print_scenario()
    print_experiment_description()
    print_problem_formulation()
    print_metrics()