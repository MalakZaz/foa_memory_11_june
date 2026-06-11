"""
===========================================================
 FILE: logger.py
 MODULE: Scientific Logging Utility (FOA-CRN V0)
===========================================================
"""

import json
import os
from datetime import datetime


# ===========================================================
# LOGGING ENGINE
# ===========================================================
class ExperimentLogger:
    """
    Lightweight logger for FOA experiments.
    Stores results in JSON format for reproducibility.
    """

    def __init__(self, save_path="results/v0"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.log_data = {
            "timestamp": str(datetime.now()),
            "iterations": [],
            "best_fitness": None,
            "final_population_stats": {}
        }

    # -------------------------------------------------------
    def log_iteration(self, iteration, best_fit, avg_fit):
        self.log_data["iterations"].append({
            "iter": iteration,
            "best_fitness": float(best_fit),
            "avg_fitness": float(avg_fit)
        })

    # -------------------------------------------------------
    def set_final_best(self, best_fit):
        self.log_data["best_fitness"] = float(best_fit)

    # -------------------------------------------------------
    def save(self, filename="run.json"):
        path = os.path.join(self.save_path, filename)

        with open(path, "w") as f:
            json.dump(self.log_data, f, indent=4)

        return path