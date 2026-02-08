import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True)
    parser.add_argument("--gt_csv", type=str, required=True)
    args = parser.parse_args()

    raise SystemExit(
        "eval.py is a placeholder. Provide evaluation logic (e.g., ROC-AUC) if required by the organizer."
    )


if __name__ == "__main__":
    main()
