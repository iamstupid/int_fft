"""Plot fft::mul_auto trunk delegate sweep CSV.

Usage: python plot_trunk_delegate.py bench/trunk_delegate.csv -o bench/trunk_delegate.png
"""
import argparse
import csv
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


BIT_COLORS = {
    16: "#1f77b4",
    15: "#2ca02c",
    14: "#ff7f0e",
    13: "#d62728",
}


def maybe_float(value):
    return None if value == "" else float(value)


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "target_N": int(row["target_N"]),
                "limbs": int(row["limbs"]),
                "bits": int(row["trunk_bits"]),
                "fft_N": int(row["fft_N"]),
                "M": int(row["fft_M"]),
                "fft_ns": float(row["fft_ns"]),
                "fft_nsl": float(row["fft_ns_per_limb"]),
                "mpn_ns": maybe_float(row["mpn_ns"]),
                "mpn_nsl": maybe_float(row["mpn_ns_per_limb"]),
                "speedup": maybe_float(row["speedup"]),
            })
    return rows


def add_thresholds(ax):
    for x, label in [
        (2**16, "16->15"),
        (2**19, "15->14"),
        (2**21, "14->13"),
        (2**23, "2^23"),
    ]:
        ax.axvline(x, color="#777777", lw=0.7, ls="--", alpha=0.45)
        ax.text(x, 0.97, label, transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=8, color="#555555",
                rotation=90)


def plot(rows, out_path):
    if not rows:
        sys.exit("no rows in CSV")

    x = [r["fft_N"] for r in rows]
    fft = [r["fft_nsl"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )

    ax1.plot(x, fft, "-", color="#555555", lw=0.8, alpha=0.6, zorder=1)
    for bits in (16, 15, 14, 13):
        xs = [r["fft_N"] for r in rows if r["bits"] == bits]
        ys = [r["fft_nsl"] for r in rows if r["bits"] == bits]
        if xs:
            ax1.plot(xs, ys, "o", color=BIT_COLORS[bits], ms=4,
                     label=f"{bits}-bit trunks", zorder=3)

    mpn_rows = [r for r in rows if r["mpn_nsl"] is not None]
    if mpn_rows:
        ax1.plot([r["fft_N"] for r in mpn_rows],
                 [r["mpn_nsl"] for r in mpn_rows],
                 "-", color="#111111", lw=1.1, label="GMP mpn_mul_n")

    ax1.set_ylabel("ns per limb")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.set_title("fft::mul_auto trunk delegate sweep")
    add_thresholds(ax1)

    speed_rows = [r for r in rows if r["speedup"] is not None]
    ax2.axhline(1.0, color="#777777", lw=0.7, ls="--")
    if speed_rows:
        ax2.plot([r["fft_N"] for r in speed_rows],
                 [r["speedup"] for r in speed_rows],
                 "-", color="#444444", lw=0.8, alpha=0.55)
        for bits in (16, 15, 14, 13):
            xs = [r["fft_N"] for r in speed_rows if r["bits"] == bits]
            ys = [r["speedup"] for r in speed_rows if r["bits"] == bits]
            if xs:
                ax2.plot(xs, ys, "o", color=BIT_COLORS[bits], ms=4)
    else:
        ax2.text(0.5, 0.5, "GMP speedup not measured",
                 transform=ax2.transAxes, ha="center", va="center")

    ax2.set_xlabel("actual transform length N (log2)")
    ax2.set_ylabel("speedup\nGMP / auto")
    ax2.set_xscale("log", base=2)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.xaxis.set_major_locator(mticker.LogLocator(base=2))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"2^{int(math.log2(v))}" if v > 0 and abs(math.log2(v) - round(math.log2(v))) < 1e-9 else ""
    ))
    add_thresholds(ax2)

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("-o", "--out", default="trunk_delegate.png")
    args = parser.parse_args()
    plot(load_csv(args.csv), args.out)


if __name__ == "__main__":
    main()
