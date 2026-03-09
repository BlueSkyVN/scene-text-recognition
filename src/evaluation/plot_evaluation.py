import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


RESULTS_DIR = "results"
EVAL_FILE = os.path.join(RESULTS_DIR, "evaluation.json")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_evaluation():
    """
    Load evaluation results from evaluation.json
    """
    if not os.path.exists(EVAL_FILE):
        raise FileNotFoundError(
            "evaluation.json not found. Please export evaluation results first."
        )

    with open(EVAL_FILE, "r") as f:
        data = json.load(f)

    return data


def plot_metrics_bar(precision, recall, f1):
    metrics = ["Precision", "Recall", "F1-score"]
    values = [precision, recall, f1]

    plt.figure(figsize=(6,4))
    plt.bar(metrics, values)

    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0,1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_bar_chart.png"))
    plt.close()


def plot_detection_distribution(tp, fp, fn):

    labels = ["True Positives", "False Positives", "False Negatives"]
    values = [tp, fp, fn]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)

    plt.title("Detection Result Distribution")
    plt.ylabel("Number of detections")

    for i, v in enumerate(values):
        plt.text(i, v + 50, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "detection_distribution.png"))
    plt.close()


def plot_pie(tp, fp, fn):

    sizes = [tp, fp, fn]
    labels = ["True Positives", "False Positives", "False Negatives"]

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

    plt.title("Detection Results Breakdown")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "detection_pie_chart.png"))
    plt.close()


def plot_confidence_histogram(confidence_scores):

    if confidence_scores is None:
        return

    plt.figure(figsize=(6,4))
    plt.hist(confidence_scores, bins=20)

    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Detection Confidence")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_histogram.png"))
    plt.close()


def plot_pr_curve(y_true, y_scores):

    if y_true is None or y_scores is None:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
    plt.close()


def main():

    data = load_evaluation()

    tp = data["TP"]
    fp = data["FP"]
    fn = data["FN"]

    precision = data["precision"]
    recall = data["recall"]
    f1 = data["f1"]

    confidence_scores = data.get("confidence_scores", None)
    y_true = data.get("y_true", None)
    y_scores = data.get("y_scores", None)

    if confidence_scores is not None:
        confidence_scores = np.array(confidence_scores)

    if y_true is not None:
        y_true = np.array(y_true)

    if y_scores is not None:
        y_scores = np.array(y_scores)

    print("Generating evaluation plots...")

    plot_metrics_bar(precision, recall, f1)
    plot_detection_distribution(tp, fp, fn)
    plot_pie(tp, fp, fn)
    plot_confidence_histogram(confidence_scores)
    plot_pr_curve(y_true, y_scores)

    print("All plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()