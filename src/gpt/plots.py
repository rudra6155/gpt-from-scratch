"""Plots + ablation aggregation (CSV in -> PNG/CSV out). Nothing here invents data: every
point drawn comes from a CSV written by gpt.train.

    python -m gpt.plots loss --csv results/tiny_loss.csv --out results/tiny_loss.png
    python -m gpt.plots ablation --runs_dir results/ablation_runs --out_prefix results/ablation_pos_enc
    python -m gpt.plots breaks --runs_dir results/breaks --out results/breaks.png
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Palette: first three categorical slots of a CVD-validated palette (all-pairs safe for 3).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _label_end(ax, x, y, text) -> None:
    """Direct label at the end of a line: text in ink, identity carried by the line color."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK)


def read_csv(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols: dict[str, np.ndarray] = {}
    for k in rows[0]:
        cols[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    return cols


def plot_loss(csv_path: str | Path, out_path: str | Path, title: str = "Tiny Shakespeare") -> None:
    d = read_csv(csv_path)
    fig, (ax, ax_lr) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE)
    for ax_ in (ax, ax_lr):
        _style(ax_)
    for key, name, color in (("train_loss", "train", SERIES[0]), ("val_loss", "val", SERIES[1])):
        ax.plot(d["step"], d[key], color=color, linewidth=1.8, label=name, solid_capstyle="round")
        _label_end(ax, d["step"][-1], d[key][-1], f"{name} {d[key][-1]:.3f}")
    best = int(np.nanargmin(d["val_loss"]))
    if d["val_loss"][-1] - d["val_loss"][best] > 0.005:  # mark best only if visibly != final
        ax.scatter([d["step"][best]], [d["val_loss"][best]], s=36, color=SERIES[1], zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"best val {d['val_loss'][best]:.3f}", (d["step"][best], d["val_loss"][best]),
                    xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set_ylabel("cross-entropy (nats / token)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    ax.set_title(f"{title}  |  final val ppl {math.exp(d['val_loss'][-1]):.2f}", loc="left",
                 fontsize=11, color=INK)
    ax.set_xlim(0, d["step"][-1] * 1.16)  # right-hand room for the direct labels
    ax_lr.plot(d["step"], d["lr"], color=INK_2, linewidth=1.5)
    ax_lr.set_ylabel("learning rate", color=INK_2, fontsize=9)
    ax_lr.set_xlabel("optimizer step", color=INK_2, fontsize=9)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


RUN_RE = re.compile(r"(?P<cfg>[a-z]+)_seed(?P<seed>\d+)\.csv$")


def summarize_ablation(runs_dir: str | Path, out_prefix: str | Path,
                       order: tuple[str, ...] = ("learned", "sinusoidal", "none")) -> list[dict]:
    runs: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)
    for p in sorted(Path(runs_dir).glob("*_seed*.csv")):
        m = RUN_RE.search(p.name)
        if m:
            runs[m["cfg"]][int(m["seed"])] = read_csv(p)
    cfgs = [c for c in order if c in runs]
    if not cfgs:
        raise SystemExit(f"no runs found in {runs_dir}")

    summary = []
    for c in cfgs:
        seeds = sorted(runs[c])
        final = np.array([runs[c][s]["val_loss"][-1] for s in seeds])
        best = np.array([np.nanmin(runs[c][s]["val_loss"]) for s in seeds])
        final_train = np.array([runs[c][s]["train_loss"][-1] for s in seeds])
        ddof = 1 if len(seeds) > 1 else 0  # sample std over seeds
        summary.append({
            "pos_encoding": c, "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds)),
            "steps": int(runs[c][seeds[0]]["step"][-1]),
            "final_val_loss_mean": final.mean(), "final_val_loss_std": final.std(ddof=ddof),
            "final_val_ppl_mean": np.exp(final).mean(), "final_val_ppl_std": np.exp(final).std(ddof=ddof),
            "best_val_loss_mean": best.mean(), "best_val_loss_std": best.std(ddof=ddof),
            "final_train_loss_mean": final_train.mean(), "final_train_loss_std": final_train.std(ddof=ddof),
            "per_seed_final_val_loss": " ".join(f"{v:.4f}" for v in final),
        })
    out_csv = Path(f"{out_prefix}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[plots] wrote {out_csv}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    for a in (ax, ax2):
        _style(a)
    for i, c in enumerate(cfgs):
        seeds = sorted(runs[c])
        steps = runs[c][seeds[0]]["step"]
        curves = np.stack([runs[c][s]["val_loss"] for s in seeds])
        mu, sd = curves.mean(0), curves.std(0, ddof=1 if len(seeds) > 1 else 0)
        ax.plot(steps, mu, color=SERIES[i], linewidth=1.8, label=c)
        ax.fill_between(steps, mu - sd, mu + sd, color=SERIES[i], alpha=0.18, linewidth=0)
        _label_end(ax, steps[-1], mu[-1], c)
    ax.set_title("Validation loss, mean ± std over seeds (y zoomed)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("optimizer step", color=INK_2, fontsize=9)
    ax.set_ylabel("val cross-entropy (nats / token)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlim(0, steps[-1] * 1.16)
    lo = min(r["final_val_loss_mean"] - r["final_val_loss_std"] for r in summary)
    hi = max(np.nanmax(runs[c][s]["val_loss"][len(runs[c][s]["val_loss"]) // 3:])
             for c in cfgs for s in runs[c])
    ax.set_ylim(lo - 0.05, hi + 0.05)  # zoom past the initial ln(V) drop so the gap is visible

    for i, r in enumerate(summary):
        c = r["pos_encoding"]
        pts = [runs[c][s]["val_loss"][-1] for s in sorted(runs[c])]
        ax2.scatter([i] * len(pts), pts, s=18, color=SERIES[i], alpha=0.55, zorder=2,
                    edgecolor=SURFACE, linewidth=1)
        ax2.errorbar([i], [r["final_val_loss_mean"]], yerr=[r["final_val_loss_std"]], fmt="o",
                     color=SERIES[i], ecolor=INK_2, elinewidth=1.2, capsize=5, markersize=8,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax2.annotate(f"{r['final_val_loss_mean']:.3f}\n± {r['final_val_loss_std']:.3f}",
                     (i, r["final_val_loss_mean"]), xytext=(14, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK)
    ax2.set_xticks(range(len(summary)), [r["pos_encoding"] for r in summary])
    ax2.set_xlim(-0.5, len(summary) - 0.1)
    ax2.set_title(f"Final val loss (step {summary[0]['steps']}, n={summary[0]['n_seeds']})",
                  loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    out_png = Path(f"{out_prefix}.png")
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plots] wrote {out_png}")
    return summary


def plot_breaks(runs_dir: str | Path, out_path: str | Path) -> None:
    """One small panel per deliberate-breakage run, each against the healthy baseline."""
    runs_dir = Path(runs_dir)
    base = read_csv(runs_dir / "baseline.csv")
    broken = sorted(p for p in runs_dir.glob("*.csv") if p.stem != "baseline")
    fig, axes = plt.subplots(1, len(broken), figsize=(4.2 * len(broken), 3.6), facecolor=SURFACE,
                             squeeze=False)
    for ax, p in zip(axes[0], broken):
        _style(ax)
        d = read_csv(p)
        ax.plot(base["step"], base["val_loss"], color=SERIES[0], linewidth=1.8, label="baseline val")
        ax.plot(d["step"], d["train_loss"], color=SERIES[1], linewidth=1.8, linestyle="--",
                label="broken train")
        ax.plot(d["step"], d["val_loss"], color=SERIES[1], linewidth=1.8, label="broken val")
        ax.set_title(p.stem.replace("_", " "), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("optimizer step", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0][0].set_ylabel("cross-entropy (nats / token)", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plots] wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("loss")
    a.add_argument("--csv", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--title", default="Tiny Shakespeare")
    b = sub.add_parser("ablation")
    b.add_argument("--runs_dir", default="results/ablation_runs")
    b.add_argument("--out_prefix", default="results/ablation_pos_enc")
    c = sub.add_parser("breaks")
    c.add_argument("--runs_dir", default="results/breaks")
    c.add_argument("--out", default="results/breaks.png")
    args = p.parse_args(argv)
    if args.cmd == "loss":
        plot_loss(args.csv, args.out, args.title)
    elif args.cmd == "ablation":
        summarize_ablation(args.runs_dir, args.out_prefix)
    else:
        plot_breaks(args.runs_dir, args.out)


if __name__ == "__main__":
    main()
