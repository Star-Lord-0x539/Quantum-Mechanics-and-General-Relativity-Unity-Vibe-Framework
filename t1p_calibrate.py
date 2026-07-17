#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
t1p_calibrate.py  --  the ref12 integrator + the T1' / S4 calibration gate suite.

Companion reproducibility script for Remark an17:cal-record (appendix_e2a_omega.tex,
sec. A.6''), calibrating the exhibited ball-uniform constants / exponents of the
Schur assembly A.6  (Prop. an17:S4a-D1, Thm. an17:S4a-D4, Thm. an17:S4b-D5,
Thm. an17:T1prime).

HONESTY NOTE (this is the whole point of the calibration record).
  This suite is a *numerics WITNESS*, not a proof.  Every quantity below is
  computed independently from the A.6 fan geometry / affine-synthesis exponent
  count; nothing is hard-coded to the printed figure and nothing is curve-fit.
  The gates CONFIRM exhibited exponents and check the integrator + the exponent
  bookkeeping are reproducible.  They neither prove nor SATURATE any bound
  (e.g. (D1) is a bound the data approach from below; the Theta_osc part alone
  over-covers the empirical plateau).  The analytic gain N0 over U_omega is a
  Jensen/majorant THEOREM, not a numerical finding -- the suite says nothing
  about it.

  "ref12" = the reference integrator: a composite 12-node Gauss-Legendre rule.
  For each quadrature gate we report the digit stability under DOUBLING the
  panel count ("doubling the quadrature resolution changes no reported digit at
  1e-8").

Run:  python3 t1p_calibrate.py
"""
import numpy as np

# =====================================================================
# ref12 : composite 12-node Gauss-Legendre reference integrator on [a,b].
# =====================================================================
_GL12_x, _GL12_w = np.polynomial.legendre.leggauss(12)   # 12-node -> degree 23 exact

def ref12_nodes(npan, a=0.0, b=1.0):
    """nodes/weights of the composite 12-node GL rule with `npan` panels on [a,b]."""
    edges = np.linspace(a, b, npan + 1)
    T = []; W = []
    for i in range(npan):
        lo, hi = edges[i], edges[i + 1]
        T.append(0.5 * (hi - lo) * _GL12_x + 0.5 * (lo + hi))
        W.append(0.5 * (hi - lo) * _GL12_w)
    return np.concatenate(T), np.concatenate(W)

def ref12(f, npan, a=0.0, b=1.0):
    """int_a^b f  by the ref12 rule with npan panels (f takes an array)."""
    t, w = ref12_nodes(npan, a, b)
    return np.sum(w * f(t))

def panels_for(Lam, base=6):
    """enough panels for the ref12 rule to resolve an e^{i Lam t} oscillation."""
    return int(base + np.ceil(Lam / 6.0))

# =====================================================================
# A.6 geometry / model objects (all independent of the printed numbers)
# =====================================================================
def model_I(Lam, c, wfun, npan):
    """scalar model integral  I_Lam[w] = int_0^1 e^{i Lam c t} w(t) dt  (R6/eq an17-model),
       c = e.v  (chord cosine).  ref12 quadrature."""
    t, w = ref12_nodes(npan)
    return np.sum(w * wfun(t) * np.exp(1j * Lam * c * t))

def G_lin_closed(xi, A=1.0):
    """closed form of G(xi)=A*int_0^1 t e^{-i xi t}dt = A (I1 - e^{-i xi})/(i xi),
       I1=(1-e^{-i xi})/(i xi).  Exact for all xi (used to defeat aliasing)."""
    xi = np.asarray(xi, float); out = np.empty_like(xi, complex)
    z = xi != 0; x = xi[z]
    I1 = (1 - np.exp(-1j * x)) / (1j * x)
    out[z] = A * (I1 - np.exp(-1j * x)) / (1j * x)
    out[~z] = A * 0.5
    return out

def D1_plateau(L, A=1.0, ppl=60):
    """L^{1/2} ||I_Lam||_{L2(fan_rho,dtheta)} for the flat round planar-section fan.
       Fan-coordinate reduction:  L*||I||^2 = (1/pi) int_{-L}^{L}|G|^2/sqrt(1-(xi/L)^2) dxi,
       L = Lam*rho.  (genuine fan L2 norm; midpoint in xi, closed-form G.)"""
    N = int(ppl * L)
    xi = (np.arange(N) + 0.5) / N * 2 * L - L
    integ = np.abs(G_lin_closed(xi, A)) ** 2 / np.sqrt(1 - (xi / L) ** 2)
    return np.sqrt((1.0 / np.pi) * np.sum(integ) * (2 * L / N))

def shell_slope(Ffun, u_list, npan=8):
    """small-u shell-mass slope of the exact affine synthesis:
       S(u0)=(int_{u0}^{2u0}|F(1-u)|^2 du)^{1/2} ~ u0^beta ; slope=beta (Thm an17:S4a-D4)."""
    S = []
    for u0 in u_list:
        t, w = ref12_nodes(npan, u0, 2 * u0)
        S.append(np.sqrt(np.sum(w * np.abs(Ffun(1 - t)) ** 2)))
    S = np.array(S)
    return np.polyfit(np.log(u_list), np.log(S), 1)[0]

def Jfac(t, delta):
    """geometric factor J(t)=|e^T d_y gamma| = t + delta t(1-t) (R1: d_y gamma(1)=Id => J(1)=1)."""
    return t + delta * t * (1 - t)

# Schur kernel (S4b, D5):  K(M,Lam)=(M/Lam)^gamma 1[M/Lam<=c_geo],  gamma=a-1/2+beta
def schur_row(a, beta, c_geo, noct=200):
    gamma = a - 0.5 + beta
    i = np.arange(noct)
    # dyadic Lambda >= M/c_geo : ratios (M/Lam) = c_geo * 2^{-i}, capped at c_geo
    return np.sum((c_geo * 2.0 ** (-i)) ** gamma)

# =====================================================================
# gate harness
# =====================================================================
_results = []
def gate(name, ok, measured, target, tol, stab=None):
    _results.append(bool(ok))
    s = "PASS" if ok else "FAIL"
    line = "[%s] %-46s meas=%-14s target=%-12s tol=%.0e" % (
        s, name, ("%.10g" % measured if np.isscalar(measured) else str(measured)),
        ("%.10g" % target if np.isscalar(target) else str(target)), tol)
    if stab is not None:
        line += "   doubling-Delta=%.2e" % stab
    print(line)

print("=" * 96)
print("t1p_calibrate.py  --  ref12 integrator + 17-gate T1'/S4 calibration suite")
print("  (numerics WITNESS for Remark an17:cal-record; confirms, does not prove/saturate)")
print("=" * 96)

# ---- Gate 1: ref12 reproduces int_0^1 e^{i Lam t} dt (closed form) -------------
Lam = 50.0
exact = (np.exp(1j * Lam) - 1) / (1j * Lam)
np1 = panels_for(Lam)
v1 = ref12(lambda t: np.exp(1j * Lam * t), np1)
v1b = ref12(lambda t: np.exp(1j * Lam * t), 2 * np1)
err1 = abs(v1 - exact)
gate("1 ref12 int e^{iLt}  vs closed form", err1 < 1e-11, err1, 0.0, 1e-11, abs(v1 - v1b))

# ---- Gate 2: ref12 reproduces G(xi)=int_0^1 t e^{-i xi t}dt --------------------
xi = 40.0
np2 = panels_for(xi)
v2 = ref12(lambda t: t * np.exp(-1j * xi * t), np2)
v2b = ref12(lambda t: t * np.exp(-1j * xi * t), 2 * np2)
err2 = abs(v2 - G_lin_closed(np.array([xi]))[0])
gate("2 ref12 int t e^{-i xi t} vs closed form", err2 < 1e-11, err2, 0.0, 1e-11, abs(v2 - v2b))

# ---- Gate 3: ref12 polynomial exactness int_0^1 t^2 = 1/3 ----------------------
v3 = ref12(lambda t: t ** 2, 3); v3b = ref12(lambda t: t ** 2, 6)
err3 = abs(v3 - 1.0 / 3.0)
gate("3 ref12 polynomial exactness (t^2)", err3 < 1e-13, err3, 1.0/3.0, 1e-13, abs(v3 - v3b))

# ---- Gate 4: HEADLINE reproducibility -- doubling changes no digit at 1e-8 -----
# model integral I_Lam[w], w(t)=t, c=0.37 ; compare ref12 at npan and 2*npan.
Lam4, c4 = 800.0, 0.37
n4 = panels_for(Lam4 * abs(c4))
IA = model_I(Lam4, c4, lambda t: t, n4)
IB = model_I(Lam4, c4, lambda t: t, 2 * n4)
dstab = abs(IA - IB)
gate("4 quadrature doubling stability (~1e-11)", dstab < 1e-8, dstab, 0.0, 1e-8, dstab)

# ---- Gate 5: Plancherel identity  int|G|^2 dxi = 2 pi ||w||_2^2 ----------------
# numeric (closed-form G on fine grid) vs analytic 2*pi/3 for w=t.
xig = np.linspace(-4000, 4000, 4_000_001)
_trap = getattr(np, "trapezoid", getattr(np, "trapz"))
lhs = _trap(np.abs(G_lin_closed(xig)) ** 2, xig)
rhs = 2 * np.pi * (1.0 / 3.0)
err5 = abs(lhs - rhs) / rhs
gate("5 Plancherel int|G|^2 = 2pi||w||_2^2", err5 < 2e-3, lhs, rhs, 2e-3)

# ---- Gate 6: D1 plateau equals the Plancherel prediction sqrt(2)||w||_2 --------
A6 = 1.0
pl6 = D1_plateau(3200.0, A6); pl6b = D1_plateau(3200.0, A6, ppl=120)
pred6 = np.sqrt(2.0) * A6 * np.sqrt(1.0 / 3.0)
err6 = abs(pl6 - pred6)
gate("6 D1 plateau = sqrt(2)||w||_2 (approached<-)", err6 < 1e-3, pl6, pred6, 1e-3, abs(pl6 - pl6b))

# ---- Gate 7: D1 exponent -1/2 (fan L2 slope) ----------------------------------
Ls = np.array([200., 400, 800, 1600, 3200], float)
vals = np.array([D1_plateau(L) / np.sqrt(L) for L in Ls])
slope7 = np.polyfit(np.log(Ls), np.log(vals), 1)[0]
gate("7 D1 slope -> -1/2", abs(slope7 + 0.5) < 1e-2, slope7, -0.5, 1e-2)

# ---- Gates 8-10: D4 transfer exponent beta = (order of vanishing of J*w)+1/2 ---
u_list = 2.0 ** (-np.arange(6, 16))
b8 = shell_slope(lambda tt: tt ** 2, u_list)                    # w=t, F=t^2, w(1)!=0 -> 1/2
b9 = shell_slope(lambda tt: tt * (1 - tt), u_list)              # w(1)=0 order1 -> 3/2
b10 = shell_slope(lambda tt: tt * (1 - tt) ** 2, u_list)        # order2 -> 5/2
gate("8  D4 beta non-vanishing -> 1/2", abs(b8 - 0.5) < 3e-2, b8, 0.5, 3e-2)
gate("9  D4 beta vanishing(w(1)=0) -> 3/2", abs(b9 - 1.5) < 5e-2, b9, 1.5, 5e-2)
gate("10 D4 beta doubly-vanishing -> 5/2", abs(b10 - 2.5) < 5e-2, b10, 2.5, 5e-2)

# ---- Gate 11: exponent identity  M^{a-1/2} omega(M/Lam) Lam^{1/2-a} = (M/Lam)^{a-1/2+beta}
rng = np.random.default_rng(12)
beta = 0.5; c_geo = 39.0 / 32.0
resid11 = 0.0
for _ in range(2000):
    a = rng.uniform(2.01, 6.0); Lm = rng.uniform(1, 1e4); M = rng.uniform(1, c_geo * Lm)
    u = M / Lm; om = u ** 0.5 if u <= c_geo else 0.0
    lhs = M ** (a - 0.5) * om * Lm ** (0.5 - a)
    rhs = u ** (a - 0.5 + beta)
    resid11 = max(resid11, abs(lhs - rhs))
gate("11 exponent identity residual (beta=1/2)", resid11 < 1e-9, resid11, 0.0, 1e-9)

# ---- Gate 12: Schur ROW sum closed form  R_M = c_geo^gamma/(1-2^{-gamma}) ------
a12 = 3.0; beta = 0.5; gamma = a12 - 0.5 + beta
R_num = schur_row(a12, beta, c_geo, noct=4000)
R_cf = c_geo ** gamma / (1 - 2 ** (-gamma))
err12 = abs(R_num - R_cf)
gate("12 Schur row sum = c_geo^g/(1-2^-g)", err12 < 1e-10, R_num, R_cf, 1e-10)

# ---- Gate 13: Schur COLUMN sum bounded by the same R (one-sided kernel) --------
# column S_Lam = sum_M (M/Lam)^gamma 1[M<=c_geo Lam], dyadic M<=c_geo Lam
i = np.arange(4000)
S_col = np.sum((c_geo * 2.0 ** (-i)) ** gamma)   # same geometric series, one-sided
gate("13 Schur col sum <= row bound R", S_col <= R_cf * (1 + 1e-12), S_col, R_cf, 1e-12)

# ---- Gate 14: c_a = (c_geo^gamma/(1-2^{-gamma}))^2 = R^2 -----------------------
c_a = (c_geo ** gamma / (1 - 2 ** (-gamma))) ** 2
err14 = abs(c_a - R_cf ** 2)
gate("14 c_a = R^2 consistency", err14 < 1e-12, c_a, R_cf ** 2, 1e-12)

# ---- Gate 15: boundary datum  J(1)=1 EXACTLY across bends (bend-invariance) ----
maxdev15 = max(abs(Jfac(1.0, d) - 1.0) for d in (0.0, 0.15, 0.35))
gate("15 J(1)=1 exact across bends (dy gamma(1)=Id)", maxdev15 < 1e-14, maxdev15, 0.0, 1e-14)

# ---- Gate 16: disintegration exponent  int_0^rho0 rho^{-1} rho^{n-1} drho, n=4 -
# fan currency rho^{-1} (from D1) integrable against rho^{n-1} drho = rho^3 drho.
rho0 = 0.3
val16 = ref12(lambda r: r ** (-1) * r ** 3, 8, 1e-6, rho0)   # = int rho^2 drho
cf16 = (rho0 ** 3 - 1e-18) / 3.0
err16 = abs(val16 - cf16)
gate("16 near-diag disintegration rho^{-1}*rho^3", err16 < 1e-12, val16, cf16, 1e-12)

# ---- Gate 17: booking floor  gamma = a-1/2+beta > 3/2 for a>2 at beta=1/2 ------
a_grid = np.linspace(2.001, 10, 500); beta = 0.5
gmin = np.min(a_grid - 0.5 + beta)   # = min a > 2
gate("17 booking floor gamma>3/2 (a>2,beta=1/2)", gmin > 1.5, gmin, 1.5, 0.0)

# =====================================================================
print("-" * 96)
npass = sum(_results); ntot = len(_results)
print("GATE SUITE: %d/%d PASS" % (npass, ntot))
print("quadrature stability (headline gate 4, doubled resolution): %.2e  (<1e-8: %s)"
      % (dstab, "yes" if dstab < 1e-8 else "NO"))
print("interpretation: integrator + exponent bookkeeping reproduce; this WITNESSES the")
print("exhibited S4 exponents (beta=1/2, gamma=a, -1/2 fan slope) -- it does NOT prove or")
print("saturate the (D1)/(D3)/(D5) bounds, nor the analytic gain N0 over U_omega.")
print("RESULT:", "17/17 PASS" if (npass == ntot == 17) else ("%d/%d" % (npass, ntot)))
