#!/usr/bin/env python3
# s4b_schur_check.py  --  RUN 114 reproducibility script (cal-record item iii, C2)
#
# WHAT THIS IS.  A numerical witness for the S4b discrete Schur summation of
# Theorem an17:S4b-D5 (appendix_e2a_omega.tex).  It builds the dyadic Schur kernel
#
#     K(M,Lambda) = (M/Lambda)^gamma * 1[ M/Lambda <= c_geo ],   gamma = a - 1/2 + beta
#
# (l.2148-2150), one-sided in the octave difference because the (D4) support cut
# omega=0 for u>c_geo makes it one-sided (l.2177-2179).  At the proved beta=1/2 the
# exponent is gamma = a (l.2151), and the hypothesis floor a>2 gives gamma>3/2>0.
# The theorem's finite Schur constant is
#
#     c_a := ( c_geo^gamma / (1 - 2^{-gamma}) )^2       (eq an17:eq-D5, l.2160)
#
# and Schur's test (row sum R_M and column sum S_Lambda both <= c_a^{1/2}=:R,
# l.2166-2176) bounds the operator ||K||_{l2->l2} <= R, so ||K||^2 <= c_a.
#
# WHAT IS CONFIRMED (cal-record iii, l.2532-2537):
#   (a) the row/column Schur sums are FINITE (built and summed, not asserted);
#   (b) the empirical Schur ratio is NON-INCREASING in the octave count -- the
#       per-octave increment of the Schur estimate shrinks monotonically to 0 --
#       and is DOMINATED by the proved c_a;
#   (c) hence Theorem an17:S4b-D5 generates NO hidden (Lambda*rho)-log: the Schur
#       estimate does NOT grow with log(Lambda) (slope ~ 0 vs octave count), in
#       contrast to the gamma->0 borderline where it grows linearly (a genuine log).
#
# HONESTY.  This is a WITNESS, not a proof (cal-record: "a witness, not a proof";
# "they do not prove any bound and they do not saturate any bound").  Every number
# below is COMPUTED from the explicitly built kernel matrix (numpy row/col sums and
# SVD); the closed forms R_inf and c_a^{1/2} are printed only as the theoretical
# TARGETS the measurements are checked against -- they are never substituted in for
# a measured quantity.  The gamma=0 control is included precisely to show the test
# has discriminating power (it WOULD flag a log if one were present).
#
# Runnable as:  python3 s4b_schur_check.py
#
# BINDING SPEC (grep anchors in appendix_e2a_omega.tex):
#   an17:S4b intro (l.2142-2152)   K(M,L), gamma=a-1/2+beta, a>2 floor, gamma>3/2
#   an17:S4b-D5 / eq-D5 (l.2154-2180)  c_a, R_M<=R, S_L<=R, "no (Lambda rho)-log"
#   an17:eq-D4 (l.2071)            omega support cut at c_geo => one-sided kernel
#   c_geo := 1 + kappa_2 rho_1/4 <= 39/32 (l.321),  c_geo in [1, 39/32]

import numpy as np

# -----------------------------------------------------------------------------
# Kernel builder.  Rows m -> output freq M=2^m ; cols l -> input freq Lambda=2^l.
# K[m,l] = (M/Lambda)^gamma on support {M/Lambda<=c_geo} i.e. {m - l <= log2(c_geo)},
#          = 2^{gamma(m-l)} * 1[m - l <= log2(c_geo)].
# For c_geo in [1,39/32) (log2 c_geo in [0,0.286)) and integer octaves this is the
# one-sided (upper) kernel {m<=l}.  Nothing here is hardcoded to a target value.
# -----------------------------------------------------------------------------
def build_K(N, gamma, c_geo):
    m = np.arange(N + 1)
    D = m[:, None] - m[None, :]                      # m - l  (integer octave diff)
    supp = D <= np.log2(c_geo) + 1e-12               # one-sided support cut
    with np.errstate(over='ignore'):
        K = np.where(supp, np.power(2.0, gamma * D), 0.0)
    return K

def max_row_sum(K):  return float(K.sum(axis=1).max())   # sup_M R_M
def max_col_sum(K):  return float(K.sum(axis=0).max())   # sup_Lambda S_Lambda
def op_norm(K):      return float(np.linalg.svd(K, compute_uv=False)[0])  # ||K||_{l2->l2}

def R_inf(gamma):    return 1.0 / (1.0 - 2.0 ** (-gamma))          # plateau (c_geo=1)
def ca_sqrt(gamma, c_geo): return (c_geo ** gamma) / (1.0 - 2.0 ** (-gamma))  # proved

TOL = 1e-9
checks = []   # (name, passed, detail)

# -----------------------------------------------------------------------------
# Main witness loop over admissible (a, c_geo).  beta = 1/2 (proved, D4) so gamma=a.
# a>2 floor; consuming loci have a>3.  c_geo in {1, 39/32}.
# -----------------------------------------------------------------------------
BETA = 0.5
cases = [(a, cg) for a in (2.5, 3.0, 4.0, 5.0) for cg in (1.0, 39.0 / 32.0)]
octaves = list(range(1, 21))     # N = 1..20 octaves ; N = log2(Lambda_max)

print("=" * 78)
print("s4b_schur_check.py -- S4b discrete Schur summation (Thm an17:S4b-D5) WITNESS")
print("  K(M,L)=(M/L)^gamma 1[M/L<=c_geo], gamma=a-1/2+beta, beta=1/2 => gamma=a")
print("  Schur test: sup_M R_M, sup_L S_L  <=  c_a^{1/2}=c_geo^gamma/(1-2^-gamma)")
print("=" * 78)

for a, cg in cases:
    gamma = a - 0.5 + BETA           # = a at beta=1/2
    Rinf = R_inf(gamma)
    cas = ca_sqrt(gamma, cg)         # proved bound c_a^{1/2}
    ca = cas ** 2                    # proved c_a

    # ---- measured quantities from the built matrices, per octave count ----
    Rmax = np.array([max_row_sum(build_K(N, gamma, cg)) for N in octaves])
    Smax = np.array([max_col_sum(build_K(N, gamma, cg)) for N in octaves])
    # empirical Schur estimate for ||K|| and for c_a:
    schur_est = np.sqrt(Rmax * Smax)          # <= c_a^{1/2}, the Schur bound on ||K||
    ca_emp = Rmax * Smax                       # empirical c_a = (sup R)(sup S)

    # (a) FINITENESS + DOMINATION: every row/col sum finite and <= c_a^{1/2}.
    finite = bool(np.all(np.isfinite(Rmax)) and np.all(np.isfinite(Smax)))
    dominated = bool(np.all(schur_est <= cas + 1e-9) and np.all(ca_emp <= ca + 1e-9))

    # (b) NON-INCREASING per-octave increment (the "Schur ratio non-increasing").
    incr = np.diff(Rmax)                        # Delta(N) = R(N)-R(N-1) = 2^{-gamma N}
    non_incr = bool(np.all(incr[1:] <= incr[:-1] + 1e-15))   # monotone non-increasing
    incr_to_zero = bool(incr[-1] < 1e-6)        # increments vanish (saturation)

    # (c) NO (Lambda rho)-LOG: slope of Schur estimate vs octave count over top
    #     octaves must be ~0.  N = log2(Lambda_max), so this IS d/dlog2(Lambda).
    top = slice(-6, None)                        # top 6 octaves
    slope = float(np.polyfit(np.array(octaves)[top], schur_est[top], 1)[0])
    no_log = bool(abs(slope) < 1e-6)

    # (d) TRUE operator norm <= Schur bound (Schur's test holds numerically) and
    #     stays bounded (dominated by c_a^{1/2}) as octaves grow -- no growth.
    opn = np.array([op_norm(build_K(N, gamma, cg)) for N in (8, 12, 16, 20)])
    schur_dominates_true = bool(np.all(opn <= schur_est[-1] + 1e-9)
                                and np.all(opn <= cas + 1e-9))

    # plateau value the measurement reached (should equal R_inf to machine eps):
    plateau = Rmax[-1]
    plateau_ok = bool(abs(plateau - Rinf) < 1e-9)

    passed = (finite and dominated and non_incr and incr_to_zero and no_log
              and schur_dominates_true and plateau_ok)
    checks.append((f"a={a} gamma={gamma:.2f} c_geo={cg:.4f}", passed))

    print(f"\n-- a={a}  gamma={gamma:.3f}  c_geo={cg:.5f} "
          f"(c_geo^gamma slack={cg**gamma:.4f}) --")
    print(f"   sup_M R_M (measured, N=20) = {Rmax[-1]:.8f}   target R_inf = {Rinf:.8f}"
          f"   [{'PASS' if plateau_ok else 'FAIL'}]")
    print(f"   sup_L S_L (measured, N=20) = {Smax[-1]:.8f}   (= sup_M R_M, symmetric)")
    print(f"   proved bound  c_a^{{1/2}}     = {cas:.8f}   c_a = {ca:.8f}")
    print(f"   empirical Schur est ||K||<= = {schur_est[-1]:.8f}  <= c_a^1/2 ? "
          f"{'yes' if schur_est[-1] <= cas + 1e-9 else 'NO'}   "
          f"[dominated: {'PASS' if dominated else 'FAIL'}]")
    print(f"   per-octave increment Delta(N): {incr[0]:.3e} -> {incr[4]:.3e} -> "
          f"{incr[-1]:.3e}  non-increasing? {'PASS' if non_incr else 'FAIL'}")
    print(f"   Schur est slope vs log2(Lambda) over top 6 octaves = {slope:+.3e}   "
          f"target ~0  [no-log: {'PASS' if no_log else 'FAIL'}]")
    print(f"   true ||K|| (SVD) at N=8,12,16,20 = "
          f"[{opn[0]:.5f} {opn[1]:.5f} {opn[2]:.5f} {opn[3]:.5f}]  <= c_a^1/2 & Schur? "
          f"[{'PASS' if schur_dominates_true else 'FAIL'}]")

# -----------------------------------------------------------------------------
# Physical Lambda sweep {4000,...,32000} ~ 2^{12}..2^{15}: Schur estimate is flat.
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Physical Lambda sweep {4000,8000,16000,32000} ~ 2^{12..15}  (a=4, c_geo=39/32)")
gamma = 4.0
lam_list = [4000, 8000, 16000, 32000]
prev = None
flat = True
for lam in lam_list:
    N = int(np.ceil(np.log2(lam)))
    R = max_row_sum(build_K(N, gamma, 39.0 / 32.0))
    tag = "" if prev is None else f"  d(Schur est)={abs(R-prev):.2e}"
    if prev is not None and abs(R - prev) > 1e-9:
        flat = False
    print(f"   Lambda_max={lam:6d} (N={N} octaves): sup_M R_M = {R:.10f}{tag}")
    prev = R
print(f"   -> Schur estimate flat across the physical Lambda range (no log-growth): "
      f"[{'PASS' if flat else 'FAIL'}]")
checks.append(("physical Lambda sweep flat (no log)", flat))

# -----------------------------------------------------------------------------
# CONTROL: gamma -> 0 borderline.  This is what a HIDDEN (Lambda rho)-LOG looks
# like: the row sum = N+1 grows LINEARLY in the octave count = log2(Lambda_max).
# Its presence here (slope ~1) and ABSENCE above (slope ~0) is the honest witness
# that gamma = a > 2 (not gamma = 0) is what kills the log.
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("CONTROL gamma=0 (borderline): what a hidden (Lambda*rho)-log WOULD look like")
Ns = np.array(octaves)
R0 = np.array([max_row_sum(build_K(N, 0.0, 1.0)) for N in octaves])
slope0 = float(np.polyfit(Ns[-6:], R0[-6:], 1)[0])
log_detected = bool(abs(slope0 - 1.0) < 1e-6)     # linear growth, slope 1 = the log
print(f"   sup_M R_M at N=5,10,15,20 = "
      f"[{R0[4]:.1f} {R0[9]:.1f} {R0[14]:.1f} {R0[19]:.1f}]  (= N+1: grows with log Lambda)")
print(f"   slope vs log2(Lambda) = {slope0:+.4f}   target 1.0 (a genuine log)   "
      f"[control detects log: {'PASS' if log_detected else 'FAIL'}]")
print(f"   => contrast: gamma=a>2 gave slope ~0 (NO log); gamma=0 gives slope ~1 (log).")
checks.append(("control: gamma=0 log detected (test has discriminating power)",
               log_detected))

# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
allpass = all(p for _, p in checks)
for name, p in checks:
    print(f"[{'PASS' if p else 'FAIL'}]  {name}")
print("-" * 78)
print("measured: row/col Schur sums finite; Schur estimate <= c_a^{1/2}; per-octave")
print("          increment non-increasing -> 0; slope vs log(Lambda) ~ 0 for gamma=a>2")
print("target:   finite & dominated by c_a; no growth with log(Lambda)  =>  NO hidden")
print("          (Lambda*rho)-log  (control gamma=0 confirms a log would show slope 1)")
print("RESULT:", "S4b Schur summation CONFIRMED -- no hidden (Lambda rho)-log"
      if allpass else "SCHUR CHECK FAILED -- report as finding (possible hidden log)")
raise SystemExit(0 if allpass else 1)
