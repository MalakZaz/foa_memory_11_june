"""
===========================================================
 FILE: generate_tables.py
 MODULE: IEEE / Elsevier Result Tables Generator
===========================================================

PURPOSE
-------
Generate publication-ready tables from experimental results:

    - Mean ± Std
    - Ranking
    - Significance summary

===========================================================
"""

import numpy as np
import pandas as pd


# ===========================================================
# TABLE 1: PERFORMANCE SUMMARY
# ===========================================================
def create_summary_table(results_dict):

    table = []

    for algo, values in results_dict.items():

        values = np.array(values)

        table.append({
            "Algorithm": algo,
            "Mean Fitness": np.mean(values),
            "Std Dev": np.std(values),
            "Min": np.min(values),
            "Max": np.max(values)
        })

    df = pd.DataFrame(table)

    df = df.sort_values(by="Mean Fitness", ascending=False)

    return df


# ===========================================================
# TABLE 2: STATISTICAL SIGNIFICANCE VS FOA
# ===========================================================
def create_significance_table(report):

    rows = []

    for algo, test in report["tests"].items():

        rows.append({
            "Comparison": f"FOA vs {algo}",
            "t-stat": test["t_stat"],
            "p-value": test["p_value"],
            "Significant (p<0.05)": test["significant"]
        })

    return pd.DataFrame(rows)


# ===========================================================
# EXPORT LATEX (IEEE READY)
# ===========================================================
def export_latex(df, path):

    with open(path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))


# ===========================================================
# MAIN WRAPPER
# ===========================================================
def generate_all_tables(results_dict, report):

    summary = create_summary_table(results_dict)
    sig = create_significance_table(report)

    summary.to_csv("results/v0/table_summary.csv", index=False)
    sig.to_csv("results/v0/table_significance.csv", index=False)

    export_latex(summary, "results/v0/table_summary.tex")
    export_latex(sig, "results/v0/table_significance.tex")

    return summary, sig