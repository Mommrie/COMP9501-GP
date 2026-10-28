"""Standalone, reproducible scatter plots for the six overlapping task pairs."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace plots from this same study")
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    target = Path(args.output)
    if not args.overwrite and (target.with_suffix(".png").exists() or target.with_suffix(".pdf").exists()):
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.8),sharey=True,layout="constrained")
    pairs = summary["pairs"]
    for ax,signal,title in zip(axes,["delta_cosine","raw_cosine","weight_cosine"],
                               ["Delta activation cosine","Raw activation cosine","Weight delta cosine"]):
        x = np.asarray([p[signal] for p in pairs])
        y = np.asarray([100*p["mean_accuracy_drop"] for p in pairs])
        for pair,a,b in zip(pairs,x,y):
            same = pair["same_input_format_family"]
            ax.scatter(a,b,color="#c45436" if same else "#21618c",marker="s" if same else "o",s=55)
            offset,align = ((-10,-17),"right") if signal == "weight_cosine" and pair["tasks"] == ["sst2","cola"] else ((4,5),"left")
            ax.annotate("/".join(t.upper() for t in pair["tasks"]),(a,b),xytext=offset,
                        textcoords="offset points",fontsize=7.5,ha=align)
        ax.axhline(0,color="0.6",lw=.7)
        ax.grid(alpha=.2)
        rho = summary["correlations"][signal]["spearman"]
        ax.set_title(f"{title}\nSpearman rho = {rho:.3f}",fontsize=11)
        ax.set_xlabel("Cosine similarity")
        ax.margins(x=.28,y=.25)
    axes[0].set_ylabel("Mean pair accuracy loss (percentage points)")
    fig.suptitle("Equal-weight pair merges, protocol v2, seed 42\n6 overlapping pairs from 4 tasks; descriptive, no significance claim",fontsize=12)
    fig.supxlabel("Squares: same sentence-field family; circles: different families. Positive loss means merging hurts.",fontsize=9)
    fig.savefig(target.with_suffix(".png"),dpi=180)
    fig.savefig(target.with_suffix(".pdf"))
    plt.close(fig)
    print("STUDY_PLOTS_SAVED",target)


if __name__ == "__main__":
    main()
