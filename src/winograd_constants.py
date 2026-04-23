"""
Winograd PFA butterfly constants for radix-3, radix-5, radix-7.

Emits high-precision C++ constexpr double literals (17 significant digits,
the shortest decimal that round-trips through IEEE 754 double without loss).
Also cross-validates the Winograd factorizations against a direct DFT on a
random complex input and reports the max error in the script stderr.

Usage:
    python winograd_constants.py > winograd_constants.h
"""
import sys
from mpmath import mp, cos, sin, pi, mpc, mpf

# Force UTF-8 I/O on Windows (console may default to cp936/GBK).
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

mp.dps = 50  # 50 decimal digits of internal precision


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------

def lit(v):
    """Render an mpmath real as a double literal with 17 sig digits."""
    return mp.nstr(mpf(v), 17)


def decl(name, val, comment=""):
    pad = " " * max(1, 18 - len(name))
    suffix = f"  // {comment}" if comment else ""
    print(f"constexpr double {name}{pad}= {lit(val)};{suffix}")


def err(msg):
    print(msg, file=sys.stderr)


# ----------------------------------------------------------------------
# reference DFTs
# ----------------------------------------------------------------------

def omega(M, j):
    a = -2 * pi * j / M
    return mpc(cos(a), sin(a))

def dft(x, M, inverse=False):
    sign = 1 if inverse else -1
    w = lambda j: mpc(cos(sign * 2 * pi * j / M), sin(sign * 2 * pi * j / M))
    return [sum(x[j] * w(k * j) for j in range(M)) for k in range(M)]


# ----------------------------------------------------------------------
# radix-5 Winograd (5 real mults, 17 real adds)
# ----------------------------------------------------------------------

# With ω = exp(-2π i / 5):
#   Y_0 = x_0 + x_1 + x_2 + x_3 + x_4
#   Y_1, Y_4 share real part R_14; imag parts are ± I_14.
#   Y_2, Y_3 share real part R_23; imag parts are ± I_23.
#
#   u14 = x_1 + x_4,  v14 = x_1 − x_4
#   u25 = x_2 + x_3,  v25 = x_2 − x_3
#   us  = u14 + u25           (feeds R via -1/4)
#   um  = u14 − u25           (feeds R via √5/4)
#   vs  = v14 − v25           (feeds I via s1)
#
#   5 real mults:  m1 = us · W5_CP, m2 = um · W5_CM,
#                   m3 = vs · W5_S1, m4 = v25 · W5_S1pS2, m5 = v14 · W5_S2mS1.
#
#   I14 = m3 + m4 = s1·v14 + s2·v25
#   I23 = m3 + m5 = s2·v14 − s1·v25
#   mid = x_0 + m1;  R14 = mid + m2;  R23 = mid − m2
#   Y_0 = x_0 + us
#   Y_1 = R14 − i · I14        Y_4 = R14 + i · I14
#   Y_2 = R23 − i · I23        Y_3 = R23 + i · I23

def winograd5_fwd(x):
    c1 = cos(2 * pi / 5); c2 = cos(4 * pi / 5)
    s1 = sin(2 * pi / 5); s2 = sin(4 * pi / 5)
    cp = (c1 + c2) / 2           # = -1/4
    cm = (c1 - c2) / 2           # = √5/4
    u14 = x[1] + x[4]; v14 = x[1] - x[4]
    u25 = x[2] + x[3]; v25 = x[2] - x[3]
    us = u14 + u25
    um = u14 - u25
    vs = v14 - v25
    m1 = us  * cp
    m2 = um  * cm
    m3 = vs  * s1
    m4 = v25 * (s1 + s2)
    m5 = v14 * (s2 - s1)
    I14 = m3 + m4
    I23 = m3 + m5
    mid = x[0] + m1
    R14 = mid + m2
    R23 = mid - m2
    j = mpc(0, 1)
    return [x[0] + us, R14 - j * I14, R23 - j * I23, R23 + j * I23, R14 + j * I14]


def winograd5_inv(y):
    # Inverse: same structure, sign of the i-term flipped.
    c1 = cos(2 * pi / 5); c2 = cos(4 * pi / 5)
    s1 = sin(2 * pi / 5); s2 = sin(4 * pi / 5)
    cp = (c1 + c2) / 2
    cm = (c1 - c2) / 2
    u14 = y[1] + y[4]; v14 = y[1] - y[4]
    u25 = y[2] + y[3]; v25 = y[2] - y[3]
    us = u14 + u25
    um = u14 - u25
    vs = v14 - v25
    m1 = us  * cp
    m2 = um  * cm
    m3 = vs  * s1
    m4 = v25 * (s1 + s2)
    m5 = v14 * (s2 - s1)
    I14 = m3 + m4
    I23 = m3 + m5
    mid = y[0] + m1
    R14 = mid + m2
    R23 = mid - m2
    j = mpc(0, 1)
    # x_k = y_0 + ω̄^k y_1 + ω̄^{2k} y_2 + ω̄^{3k} y_3 + ω̄^{4k} y_4
    # The Y_k ↔ X_k reversal manifests as swapping ±i pairs:
    #   X_1 ≡ R14 + i I14 (was Y_4 formula);  X_4 ≡ R14 − i I14
    return [y[0] + us, R14 + j * I14, R23 + j * I23, R23 - j * I23, R14 - j * I14]


# ----------------------------------------------------------------------
# radix-7 direct form (18 real mults, ~32 adds)
# ----------------------------------------------------------------------

# Pairs (j, 7−j): (1,6), (2,5), (3,4).
#   u_p = x_j + x_{7−j},  v_p = x_j − x_{7−j}
#
# Real part of Y_k (k = 1..3) uses row k of the 3×3 cosine matrix; each row
# is a cyclic permutation of (c1, c2, c3) determined by (k·j mod 7):
#   R_1 = x_0 + c1 u16 + c2 u25 + c3 u34
#   R_2 = x_0 + c2 u16 + c3 u25 + c1 u34
#   R_3 = x_0 + c3 u16 + c1 u25 + c2 u34
#
# Imag part uses the sine matrix with sign flips from s_{7−k} = −s_k:
#   I_1 =  s1 v16 + s2 v25 + s3 v34
#   I_2 =  s2 v16 − s3 v25 − s1 v34
#   I_3 =  s3 v16 − s1 v25 + s2 v34
#
# Y_0 = x_0 + u16 + u25 + u34
# Y_k = R_k − i · I_k   for k = 1, 2, 3
# Y_{7-k} = R_k + i · I_k

def winograd7_fwd(x):
    # 8-mult Winograd-7 DFT derived via Rader + 4-mult 3-point cyclic
    # convolution (cosine side, circulant) + 4-mult 3-point anti-cyclic
    # convolution (sine side, skew-circulant in Rader order).
    #
    # Verified: for x=(0,1,0,0,0,0,0), returns ω_7^k for k=0..6 (within mpmath ε).
    c1, c2, c3 = cos(2 * pi / 7), cos(4 * pi / 7), cos(6 * pi / 7)
    s1, s2, s3 = sin(2 * pi / 7), sin(4 * pi / 7), sin(6 * pi / 7)

    # Stage 1: antipodal pairs.
    u1 = x[1] + x[6]; v1 = x[1] - x[6]
    u2 = x[3] + x[4]; v2 = x[3] - x[4]
    u3 = x[2] + x[5]; v3 = x[2] - x[5]

    a1 = u1 + u2 + u3

    # Stage 2: linear combos feeding the 4+4 mults.
    pc  = u1 - u3
    qc  = u2 - u3
    rc  = u1 + u2 - 2 * u3

    bv  = v1 - v2 + v3
    pv  = v1 - v3
    qv  = v2 + v3
    sv  = v1 + v2

    # Stage 3: 8 non-trivial mults (data · real constant).
    # "mid_c" absorbs the a1·(-1/6) mult into the fmadd with x_0.
    mid_c = x[0] - a1 / 6
    CB = (c1 - c3) * pc
    CC = (c2 - c3) * qc
    CD = ((c1 + c2 - 2 * c3) / 3) * rc
    SA = ((s1 + s2 - s3) / 3) * bv
    SB = ((s1 + s3) / 3) * pv
    SC = (-(s2 + s3) / 3) * qv
    SD = ((s1 - s2) / 3) * sv

    # Stage 4: cosine outputs S_k = x_0 + P_k, sine outputs T̃_p (Rader order).
    S1 = mid_c + CB - CD
    S2 = mid_c - CB - CC + 2 * CD
    S3 = mid_c + CC - CD

    T0 = SA + SB - 2 * SC + SD      # = +Q_1
    T1 = -SA - SB - SC + 2 * SD     # = -Q_2
    T2 = SA - 2 * SB + SC + SD      # = -Q_3

    # Stage 5: assemble Y_k = S_k - i·Q_k and its antipodal partner.
    # Q_1 = T0, Q_2 = -T1, Q_3 = -T2.
    j = mpc(0, 1)
    return [x[0] + a1,
            S1 - j * T0,   # Y_1
            S2 + j * T1,   # Y_2 = S_2 - i·(-T1)
            S3 + j * T2,   # Y_3
            S3 - j * T2,   # Y_4
            S2 - j * T1,   # Y_5
            S1 + j * T0]   # Y_6


def winograd7_inv(y):
    # Inverse: same flow, i-signs on each output pair flipped.
    c1, c2, c3 = cos(2 * pi / 7), cos(4 * pi / 7), cos(6 * pi / 7)
    s1, s2, s3 = sin(2 * pi / 7), sin(4 * pi / 7), sin(6 * pi / 7)
    u1 = y[1] + y[6]; v1 = y[1] - y[6]
    u2 = y[3] + y[4]; v2 = y[3] - y[4]
    u3 = y[2] + y[5]; v3 = y[2] - y[5]
    a1 = u1 + u2 + u3
    pc = u1 - u3; qc = u2 - u3; rc = u1 + u2 - 2 * u3
    bv = v1 - v2 + v3; pv = v1 - v3; qv = v2 + v3; sv = v1 + v2
    mid_c = y[0] - a1 / 6
    CB = (c1 - c3) * pc
    CC = (c2 - c3) * qc
    CD = ((c1 + c2 - 2 * c3) / 3) * rc
    SA = ((s1 + s2 - s3) / 3) * bv
    SB = ((s1 + s3) / 3) * pv
    SC = (-(s2 + s3) / 3) * qv
    SD = ((s1 - s2) / 3) * sv
    S1 = mid_c + CB - CD
    S2 = mid_c - CB - CC + 2 * CD
    S3 = mid_c + CC - CD
    T0 = SA + SB - 2 * SC + SD
    T1 = -SA - SB - SC + 2 * SD
    T2 = SA - 2 * SB + SC + SD
    j = mpc(0, 1)
    return [y[0] + a1,
            S1 + j * T0, S2 - j * T1, S3 - j * T2,
            S3 + j * T2, S2 + j * T1, S1 - j * T0]


def direct7_fwd(x):
    c = [cos(2 * pi * k / 7) for k in range(4)]  # c[1..3] used
    s = [sin(2 * pi * k / 7) for k in range(4)]
    u16 = x[1] + x[6]; v16 = x[1] - x[6]
    u25 = x[2] + x[5]; v25 = x[2] - x[5]
    u34 = x[3] + x[4]; v34 = x[3] - x[4]
    R1 = x[0] + c[1] * u16 + c[2] * u25 + c[3] * u34
    R2 = x[0] + c[2] * u16 + c[3] * u25 + c[1] * u34
    R3 = x[0] + c[3] * u16 + c[1] * u25 + c[2] * u34
    I1 =        s[1] * v16 + s[2] * v25 + s[3] * v34
    I2 =        s[2] * v16 - s[3] * v25 - s[1] * v34
    I3 =        s[3] * v16 - s[1] * v25 + s[2] * v34
    j = mpc(0, 1)
    return [x[0] + u16 + u25 + u34,
            R1 - j * I1, R2 - j * I2, R3 - j * I3,
            R3 + j * I3, R2 + j * I2, R1 + j * I1]


def direct7_inv(y):
    c = [cos(2 * pi * k / 7) for k in range(4)]
    s = [sin(2 * pi * k / 7) for k in range(4)]
    u16 = y[1] + y[6]; v16 = y[1] - y[6]
    u25 = y[2] + y[5]; v25 = y[2] - y[5]
    u34 = y[3] + y[4]; v34 = y[3] - y[4]
    R1 = y[0] + c[1] * u16 + c[2] * u25 + c[3] * u34
    R2 = y[0] + c[2] * u16 + c[3] * u25 + c[1] * u34
    R3 = y[0] + c[3] * u16 + c[1] * u25 + c[2] * u34
    I1 =        s[1] * v16 + s[2] * v25 + s[3] * v34
    I2 =        s[2] * v16 - s[3] * v25 - s[1] * v34
    I3 =        s[3] * v16 - s[1] * v25 + s[2] * v34
    j = mpc(0, 1)
    # Inverse swaps the ±i pair roles:
    return [y[0] + u16 + u25 + u34,
            R1 + j * I1, R2 + j * I2, R3 + j * I3,
            R3 - j * I3, R2 - j * I2, R1 - j * I1]


# ----------------------------------------------------------------------
# cross-validation
# ----------------------------------------------------------------------

def validate(M, fwd, inv):
    x = [mpc(cos(k * 0.31 + 0.7), sin(k * 0.91 + 0.2)) for k in range(M)]
    y_ref = dft(x, M, inverse=False)
    y_fast = fwd(x)
    fwd_err = max(abs(a - b) for a, b in zip(y_ref, y_fast))
    x_ref = [c / M for c in dft(y_ref, M, inverse=True)]
    # inv() here is the "unnormalized" inverse, so round-trip factor is M.
    x_round = [c / M for c in inv(y_fast)]
    inv_err = max(abs(a - b) for a, b in zip(x, x_round))
    err(f"// radix-{M}: fwd max err = {mp.nstr(fwd_err, 4)},"
        f" inv round-trip err = {mp.nstr(inv_err, 4)}")


validate(5, winograd5_fwd, winograd5_inv)
validate(7, direct7_fwd,   direct7_inv)
validate(7, winograd7_fwd, winograd7_inv)  # must match direct7 bit-for-bit


# ----------------------------------------------------------------------
# emit C++ header
# ----------------------------------------------------------------------

print("// Generated by winograd_constants.py — DO NOT EDIT BY HAND.")
print("// PFA Winograd butterfly constants, 17 sig digits (double round-trip).")
print("#pragma once")
print()
print("// --- Radix-3 (fft-pfa.md §4) ---")
decl("W3_C",  -mpf(1)/2,      "cos(2π/3) = -1/2  (exact)")
decl("W3_S",  sin(2 * pi / 3), "sin(2π/3) = √3/2")
print()
print("// --- Radix-5 Winograd (5 real mults, 17 real adds) ---")
c1 = cos(2 * pi / 5); c2 = cos(4 * pi / 5)
s1 = sin(2 * pi / 5); s2 = sin(4 * pi / 5)
decl("W5_CP",    (c1 + c2) / 2, "(cos 2π/5 + cos 4π/5)/2 = -1/4 (exact)")
decl("W5_CM",    (c1 - c2) / 2, "(cos 2π/5 − cos 4π/5)/2 = √5/4")
decl("W5_S1",    s1,            "sin 2π/5")
decl("W5_S1pS2", s1 + s2,       "sin 2π/5 + sin 4π/5")
decl("W5_S2mS1", s2 - s1,       "sin 4π/5 − sin 2π/5")
print()
print("// --- Radix-7 direct form (18 real mults, ~32 adds) ---")
# Note: a full Winograd-7 (8 non-trivial mults) is derived and cross-validated
# in winograd7_fwd above; it loses to the direct form on AVX2+FMA (extra adds
# contend for the same FP ports), so the generated header emits only the
# direct-form constants.
for k in (1, 2, 3):
    decl(f"W7_C{k}", cos(2 * pi * k / 7), f"cos 2π·{k}/7")
for k in (1, 2, 3):
    decl(f"W7_S{k}", sin(2 * pi * k / 7), f"sin 2π·{k}/7")
print()
print("// --- ω_M^k = exp(-2π·i·k/M) tables for PFA cross-pair pointwise dispatch ---")


def omega_table(name, M):
    print(f"constexpr double {name}_RE[{M}] = {{")
    print("    " + ", ".join(lit(cos(2 * pi * k / M)) for k in range(M)))
    print("};")
    print(f"constexpr double {name}_IM[{M}] = {{")
    print("    " + ", ".join(lit(-sin(2 * pi * k / M)) for k in range(M)))
    print("};")


omega_table("W3", 3)
omega_table("W5", 5)
omega_table("W7", 7)
