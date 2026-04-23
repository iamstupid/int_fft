"""Plot fft-cleaned vs GMP mpn_mul_n from fft_bench_sweep CSV.

Usage: python plot_sweep.py <sweep.csv> [-o sweep.png]

Top panel: ns/limb vs limbs (both algos, log-x).
Bottom panel: speedup (mpn / fft) vs limbs (log-x, log-y).
Markers colored by fft_M (1=pow2, 3/5/7 = PFA).
"""
import argparse
import csv
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


M_COLORS = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e", 7: "#d62728"}
M_LABELS = {1: "pow-2", 3: "PFA-3", 5: "PFA-5", 7: "PFA-7"}


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "limbs": int(row["limbs"]),
                "fft_nsl": float(row["fft_ns_per_limb"]),
                "mpn_nsl": float(row["mpn_ns_per_limb"]),
                "fft_ns": float(row["fft_ns"]),
                "mpn_ns": float(row["mpn_ns"]),
                "N": int(row["fft_N"]),
                "M": int(row["fft_M"]),
            })
    return rows


def plot(rows, out_path):
    limbs = [r["limbs"] for r in rows]
    fft = [r["fft_nsl"] for r in rows]
    mpn = [r["mpn_nsl"] for r in rows]
    speedup = [r["mpn_ns"] / r["fft_ns"] for r in rows]
    Ms = [r["M"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        sharex=True,
    )

    # Top: ns/limb lines + M-colored markers on fft line.
    ax1.plot(limbs, mpn, "-", color="#555555", lw=1.3, label="mpn_mul_n (GMP dispatch)", zorder=2)
    ax1.plot(limbs, fft, "-", color="#888888", lw=0.8, alpha=0.6, zorder=1)
    for m_key in (1, 3, 5, 7):
        xs = [l for l, mm in zip(limbs, Ms) if mm == m_key]
        ys = [f for f, mm in zip(fft, Ms) if mm == m_key]
        if xs:
            ax1.plot(xs, ys, "o", color=M_COLORS[m_key], ms=5,
                     label=f"fft-cleaned ({M_LABELS[m_key]})", zorder=3)

    ax1.set_ylabel("ns per limb")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_title("fft-cleaned vs mpn_mul_n — balanced n×n multiply")

    # Bottom: speedup ratio (mpn / fft; >1 means fft-cleaned wins).
    ax2.axhline(1.0, color="#888", lw=0.7, ls="--")
    ax2.plot(limbs, speedup, "-", color="#444444", lw=0.8, alpha=0.5, zorder=1)
    for m_key in (1, 3, 5, 7):
        xs = [l for l, mm in zip(limbs, Ms) if mm == m_key]
        ys = [s for s, mm in zip(speedup, Ms) if mm == m_key]
        if xs:
            ax2.plot(xs, ys, "o", color=M_COLORS[m_key], ms=5, zorder=2)

    ax2.set_xlabel("limbs (per operand, log₂)")
    ax2.set_ylabel("speedup\n(mpn / fft-cleaned)")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV from fft_bench_sweep")
    ap.add_argument("-o", "--out", default="sweep.png")
    args = ap.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        sys.exit("no rows in CSV")
    plot(rows, args.out)


if __name__ == "__main__":
    main()
