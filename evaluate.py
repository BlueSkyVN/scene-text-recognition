import os
import json

from src.evaluation.evaluate_icdar import evaluate_dataset


image_dir = "data/icdar2015/train_images"
gt_dir = "data/icdar2015/train_gt"

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "evaluation.json")


def main():

    print("Running evaluation...")

    # Run evaluation
    results = evaluate_dataset(image_dir, gt_dir)

    # Expected results format
    tp = results["TP"]
    fp = results["FP"]
    fn = results["FN"]
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save to JSON
    output_data = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    print("\nEvaluation results:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()