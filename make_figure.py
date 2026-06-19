"""
Regenerate the results figure for the README from the saved analysis artifacts.

Reads analysis_full.json (bucket accuracies, overall SC accuracy, routing AUC) and
routing_curve_full.csv (error-coverage vs budget), and writes figures/routing_results.png.
All numbers come from the artifacts - nothing is hardcoded.

Requires matplotlib (figure-only dependency, not needed to run the pipeline or analyze.py):
    pip install matplotlib
    python make_figure.py
"""
import os
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANALYSIS = "analysis_full.json"
CURVE = "routing_curve_full.csv"
OUT = os.path.join("figures", "routing_results.png")

TEAL, AMBER, RED, BLUE, GRAY = "#1D9E75", "#BA7517", "#E24B4A", "#378ADD", "#888780"


def main() -> None:
    with open(ANALYSIS) as f:
        a = json.load(f)
    n = a["n_valid"]
    sc_acc = a["self_consistency_accuracy"] * 100
    auc = a["routing_auc"]
    buckets = a["buckets"]
    order = ["unanimous", "strong-majority", "split"]
    accs = [buckets[b]["sc_accuracy"] * 100 for b in order]
    counts = [buckets[b]["count"] for b in order]

    budgets, coverage = [], []
    with open(CURVE) as f:
        for row in csv.DictReader(f):
            budgets.append(float(row["budget_frac"]) * 100)
            coverage.append(float(row["frac_errors_covered"]) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(
        f"BIRD Mini-Dev (n={n}): candidate disagreement vs self-consistency errors",
        fontsize=13, y=1.02,
    )

    # Left: SC accuracy by agreement bucket
    bars = ax1.bar(order, accs, color=[TEAL, AMBER, RED], width=0.62, zorder=3)
    ax1.axhline(sc_acc, color=GRAY, ls="--", lw=1.3, zorder=2,
                label=f"overall {sc_acc:.1f}%")
    for bar, acc, c in zip(bars, accs, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, acc + 1.5,
                 f"{acc:.1f}%\n(n={c})", ha="center", va="bottom", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("self-consistency accuracy (%)")
    ax1.set_title("accuracy falls as agreement drops", fontsize=11)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    ax1.set_axisbelow(True)

    # Right: routing curve vs random baseline
    ax2.plot(budgets, coverage, color=BLUE, lw=2.2, zorder=3,
             label=f"route by disagreement (AUC {auc:.3f})")
    ax2.fill_between(budgets, coverage, color=BLUE, alpha=0.08, zorder=1)
    ax2.plot([0, 100], [0, 100], color=GRAY, ls="--", lw=1.3, zorder=2,
             label="random baseline (AUC 0.500)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("routing budget (% of queries, most-disagreeing first)")
    ax2.set_ylabel("% of self-consistency errors covered")
    ax2.set_title("errors concentrate at low agreement", fontsize=11)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    ax2.grid(alpha=0.25, zorder=0)
    ax2.set_axisbelow(True)

    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}  (n={n}, SC acc {sc_acc:.1f}%, routing AUC {auc:.3f})")


if __name__ == "__main__":
    main()
