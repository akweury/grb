# Created by jing at 03.03.25

from scripts import config
import ace_tools_open as tools
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch


def analysis_llava(principle, model_name):
    path = config.results / principle
    json_path = path / f"{model_name}_evaluation_results.json"
    data = json.load(open(json_path, "r"))
    accs = torch.tensor([v["accuracy"] / 100 for k, v in data.items()])
    f1s = torch.tensor([v["f1_score"] for k, v in data.items()])
    # Convert JSON to DataFrame

    # Calculate performance statistics
    mean_accuracy = accs.mean()
    mean_f1 = f1s.mean()
    std_accuracy = accs.std()
    std_f1 = f1s.std()

    # Compute confidence intervals
    confidence = 0.95
    n = len(accs)
    ci_accuracy = stats.t.interval(confidence, n - 1, loc=mean_accuracy, scale=std_accuracy / np.sqrt(n))
    ci_f1 = stats.t.interval(confidence, n - 1, loc=mean_f1, scale=std_f1 / np.sqrt(n))

    # Redraw the bar chart without std lines and save as PDF
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Define colors
    accuracy_color = "#1f77b4"  # Blue
    f1_color = "#ff7f0e"  # Orange

    # Plot accuracy distribution with mean
    axes[0].hist(accs, bins=10, edgecolor="black", alpha=0.7, color=accuracy_color)
    axes[0].axvline(mean_accuracy, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_accuracy:.3f}")
    axes[0].set_title(f"Accuracy Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()

    # Plot F1-score distribution with mean
    axes[1].hist(f1s, bins=10, edgecolor="black", alpha=0.7, color=f1_color)
    axes[1].axvline(mean_f1, color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {mean_f1:.3f}")
    axes[1].set_title(f"F1 Score Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("F1 Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()

    # Save the figure as PDF
    pdf_filename = f"{principle}_{model_name}_model_performance_charts.pdf"
    plt.tight_layout()
    plt.savefig(path / pdf_filename, format="pdf")

    # Format confidence intervals using ± notation
    accuracy_conf_interval = (mean_accuracy - ci_accuracy[0], ci_accuracy[1] - mean_accuracy)
    f1_conf_interval = (mean_f1 - ci_f1[0], ci_f1[1] - mean_f1)

    # Create a formatted table
    formatted_performance_table = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score"],
        "Mean ± Std": [f"{mean_accuracy:.3f} ± {std_accuracy:.3f}", f"{mean_f1:.3f} ± {std_f1:.3f}"],
        "95% Confidence Interval": [f"{mean_accuracy:.3f} ± {accuracy_conf_interval[0]:.3f}",
                                    f"{mean_f1:.3f} ± {f1_conf_interval[0]:.3f}"]
    })

    # Display the formatted table
    tools.display_dataframe_to_user(name=f"Formatted Performance Table {principle}",
                                    dataframe=formatted_performance_table)


def analysis_vit(principle, model_name, num):
    path = config.results / principle
    json_path = path / f"{model_name}_{num}_evaluation_results.json"
    data = json.load(open(json_path, "r"))

    # Convert JSON to DataFrame
    df = pd.DataFrame(data[principle]).T
    df["accuracy"] /= 100  # Convert accuracy to percentage

    # Calculate performance statistics
    mean_accuracy = df["accuracy"].mean()
    precision = df["precision"].values
    recall = df["recall"].values
    f1_score = 2 * (precision * recall) / ((precision + recall)+1e-20)

    mean_f1 = f1_score.mean()
    mean_precision = df["precision"].mean()
    mean_recall = df["recall"].mean()

    std_accuracy = df["accuracy"].std()
    std_f1 = df["f1_score"].std()
    std_precision = df["precision"].std()
    std_recall = df["recall"].std()

    # Redraw the bar chart without std lines and save as PDF
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Define colors
    accuracy_color = "#1f77b4"  # Blue
    f1_color = "#ff7f0e"  # Orange
    precision_color = "#2ca02c"  # Green
    recall_color = "#d62728"  # Red

    # Plot accuracy distribution with mean
    axes[0].hist(df["accuracy"], bins=10, edgecolor="black", alpha=0.7, color=accuracy_color)
    axes[0].axvline(mean_accuracy, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_accuracy:.2f}")
    axes[0].set_title(f"Accuracy Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()

    # Plot F1-score distribution with mean
    axes[1].hist(df["f1_score"], bins=10, edgecolor="black", alpha=0.7, color=f1_color)
    axes[1].axvline(mean_f1, color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {mean_f1:.2f}")
    axes[1].set_title(f"F1 Score Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("F1 Score", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()

    # Plot Precision distribution with mean
    axes[2].hist(df["precision"], bins=10, edgecolor="black", alpha=0.7, color=precision_color)
    axes[2].axvline(mean_precision, color='black', linestyle='dashed', linewidth=1, label=f"Mean: {mean_precision:.2f}")
    axes[2].set_title(f"Precision Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Precision", fontsize=12)
    axes[2].set_ylabel("Frequency", fontsize=12)
    axes[2].legend()

    # Plot Recall distribution with mean
    axes[3].hist(df["recall"], bins=10, edgecolor="black", alpha=0.7, color=recall_color)
    axes[3].axvline(mean_recall, color='purple', linestyle='dashed', linewidth=1, label=f"Mean: {mean_recall:.2f}")
    axes[3].set_title(f"Recall Distribution ({principle})", fontsize=14, fontweight='bold')
    axes[3].set_xlabel("Recall", fontsize=12)
    axes[3].set_ylabel("Frequency", fontsize=12)
    axes[3].legend()

    # Save the figure as PDF
    pdf_filename = f"{principle}_{model_name}_model_performance_charts.pdf"
    plt.tight_layout()
    plt.savefig(path / pdf_filename, format="pdf")

    # Create Line Chart for Precision & Recall without Standard Deviation
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    x = np.arange(len(df))  # X-axis index

    axes[0].plot(x, df["precision"], label="Precision", color=precision_color)
    axes[0].plot(x, df["recall"], label="Recall", color=recall_color)
    axes[0].set_title(f"Precision & Recall Over Samples ({principle})", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Sample Index", fontsize=12)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot Precision - Recall Difference
    diff = np.abs(df["precision"] - df["recall"])
    axes[1].plot(x, diff, label="Precision - Recall", color='black')
    axes[1].set_title(f"Precision - Recall Difference ({principle})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Sample Index", fontsize=12)
    axes[1].set_ylabel("Difference", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Save the performance line chart as PDF
    line_chart_filename = f"performance_line_chart.pdf"
    plt.tight_layout()
    plt.savefig(path / line_chart_filename, format="pdf")

    # Create a formatted table
    formatted_performance_table = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
        "Mean ± Std": [
            f"{mean_accuracy:.2f} ± {std_accuracy:.2f}",
            f"{mean_f1:.2f} ± {std_f1:.2f}",
            f"{mean_precision:.2f} ± {std_precision:.2f}",
            f"{mean_recall:.2f} ± {std_recall:.2f}"
        ]
    })

    # Display the formatted table
    tools.display_dataframe_to_user(name=f"Formatted Performance Table {principle}",
                                    dataframe=formatted_performance_table)


if __name__ == "__main__":
    # principle = "proximity"
    # principle = "similarity"
    # principle = "closure"
    # principle = "symmetry"
    principle = "continuity"

    # model_name = "Llava"
    model_name = "vit_base_patch16_224"
    if model_name == "Llava":
        analysis_llava(principle, model_name)
    else:
        analysis_vit(principle, model_name, num=5)
