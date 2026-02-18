import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


def extract_steps_from_filename(filepath: str):
    filename = os.path.basename(filepath)
    match = re.search(r"--(\d+)\.log$", filename)
    return int(match.group(1)) if match else None


def find_last_percentage_in_log(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError as e:
        print(f"skip unreadable file: {filepath} ({e})", flush=True)
        return None

    # Prefer explicit success-rate lines, avoid progress bars such as "Loading ... 33%".
    patterns = [
        r"# successes:\s*\d+\s*\((\d+\.?\d*)%\)",
        r"Current total success\s*rate:\s*(\d+\.?\d*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            return float(matches[-1])
    return None


def collect_results(log_dir: Path):
    results = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if not file.endswith(".log"):
                continue
            filepath = os.path.join(root, file)
            steps = extract_steps_from_filename(filepath)
            if steps is None:
                continue
            acc = find_last_percentage_in_log(filepath)
            if acc is None:
                continue
            results.append((steps, acc, filepath))
    results.sort(key=lambda x: x[0])
    return results


def plot_results(rows, out_png: Path):
    steps = [r["Training Steps"] for r in rows]
    success_rates = [r["Success Rate (%)"] for r in rows]

    plt.figure(figsize=(14, 8))
    plt.plot(
        steps,
        success_rates,
        "b-o",
        linewidth=2.5,
        markersize=8,
        markerfacecolor="lightblue",
        markeredgecolor="blue",
        markeredgewidth=1.5,
    )
    plt.title("Training Steps vs Success Rate", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Training Steps", fontsize=14, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")

    for step, rate in zip(steps, success_rates):
        plt.annotate(
            f"{rate}%",
            (step, rate),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    min_rate = min(success_rates)
    max_rate = max(success_rates)
    margin = (max_rate - min_rate) * 0.15 if max_rate != min_rate else max(1.0, max_rate * 0.15)
    plt.ylim(min_rate - margin, max_rate + margin)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"chart saved: {out_png}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Plot current LIBERO eval accuracy from batch logs.")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path(".tmp/session/eval_manual"),
        help="Directory containing batch eval per-checkpoint subfolders and .log files.",
    )
    parser.add_argument(
        "--out_png",
        type=Path,
        default=Path(".tmp/session/eval_manual/current_acc_plot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(".tmp/session/eval_manual/current_acc_table.csv"),
        help="Output CSV path with steps and success rate.",
    )
    args = parser.parse_args()

    results = collect_results(args.log_dir)
    if not results:
        print(f"no valid percentage data found under: {args.log_dir}", flush=True)
        return

    dedup = {}
    for step, rate, src in sorted(results, key=lambda x: x[0]):
        dedup[step] = {"Training Steps": step, "Success Rate (%)": rate, "Source Log": src}
    rows = [dedup[s] for s in sorted(dedup.keys())]

    print("current parsed results:", flush=True)
    for r in rows:
        print(f"{r['Training Steps']:>8}  {r['Success Rate (%)']:>6.2f}%", flush=True)

    rates = [r["Success Rate (%)"] for r in rows]
    print(f"min: {min(rates):.2f}%", flush=True)
    print(f"max: {max(rates):.2f}%", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Training Steps", "Success Rate (%)", "Source Log"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"table saved: {args.out_csv}", flush=True)

    plot_results(rows, args.out_png)


if __name__ == "__main__":
    main()
