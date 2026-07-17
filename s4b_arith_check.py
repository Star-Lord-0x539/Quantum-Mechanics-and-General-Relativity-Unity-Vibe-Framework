#!/usr/bin/env python3
# s4b_arith_check.py  --  RUN 114 reproducibility script (cal-record item iii, C1)
#
# WHAT THIS IS.  A machine check of the S4b exponent-bookkeeping identity cited by
# Remark an17:cal-record (appendix_e2a_omega.tex, l.2532-2534):
#
#     M^{a-1/2} * omega(M/Lambda) * Lambda^{1/2-a}  =  (M/Lambda)^{a-1/2+beta}
#
# with the weight omega read verbatim from (D4), Theorem an17:S4a-D4, eq an17:eq-D4
# (l.2071):   omega(u) = u^{1/2}  on the support 0 < u <= c_geo,  0 otherwise,
# where the boundary order 1/2 IS the transfer exponent beta (l.2074).  Thus on the
# support omega(u) = u^beta, and at the proved value beta = 1/2 the RHS collapses to
# exactly (M/Lambda)^a (l.2147-2148).  The downstream Schur exponent is
# gamma := a - 1/2 + beta (l.2150), and the finite Schur constant is
# c_a := ( c_geo^gamma / (1 - 2^{-gamma}) )^2  (eq an17:eq-D5, l.2160).
#
# HONESTY.  This is a *witness*, not a proof (cal-record: "a witness, not a proof").
# The identity itself is a pure power-law bookkeeping fact; the script only CONFIRMS
# that the exponents printed in A.6 are mutually consistent and reduce to an
# IDENTICALLY-ZERO symbolic residual (sympy.simplify(...) == 0), i.e. that no hidden
# fractional power was dropped anywhere in the (D4)->(D5) exponent chain.  It proves
# nothing about the analytic estimates that carry those exponents.
#
# NO HARDCODING.  Every residual below is formed from the symbolic definitions and
# reduced by sympy.simplify; the target is literally the integer 0.  A random-point
# numeric cross-check re-evaluates LHS-RHS at floating-point arguments to rule out a
# simplify() artifact.  Runnable as:  python3 s4b_arith_check.py
#
# BINDING SPEC (grep anchors in appendix_e2a_omega.tex):
#   an17:eq-D4  (l.2068-2072)   omega(u)=u^{1/2}, cutoff at c_geo, beta=1/2
#   an17:S4b intro (l.2146-2150) exponent identity, gamma=a-1/2+beta, kernel K
#   an17:eq-D5  (l.2157-2163)   c_a=(c_geo^gamma/(1-2^-gamma))^2; beta=1/2 form
#   an17:cal-record (iii) (l.2532-2537)  "identically-zero symbolic residual"
#   c_geo := 1 + kappa_2 rho_1/4 <= 39/32   (l.321),  c_geo >= 1

import sympy as sp
import random

# Positive frequency symbols so that non-integer powers of the same base combine
# (x^p x^q -> x^{p+q} requires a positive base in sympy).  a,beta kept symbolic.
M, Lam = sp.symbols('M Lambda', positive=True)
a, beta = sp.symbols('a beta', real=True)
c_geo = sp.symbols('c_geo', positive=True)
half = sp.Rational(1, 2)
u = M / Lam                     # dyadic octave ratio, u = M/Lambda

results = []   # (name, residual_expr, passed)

def check(name, residual):
    r = sp.simplify(residual)
    passed = (r == 0)
    results.append((name, r, passed))
    return passed

# --- (1) the core (D4)->(D5) exponent identity, GENERAL beta -----------------
# omega read from (D4): on its support omega(u) = u^{1/2} = u^beta.  We keep beta
# symbolic to show the identity is beta-uniform (a pure power-law fact), then pin
# beta=1/2 in (2).  This is the exact line the remark machine-checks.
omega_support = u**beta
lhs = M**(a - half) * omega_support * Lam**(half - a)
rhs = u**(a - half + beta)
check("(1) M^{a-1/2} omega(M/L) L^{1/2-a} - (M/L)^{a-1/2+beta}  [general beta]",
      lhs - rhs)

# --- (2) the beta = 1/2 collapse: RHS is exactly (M/Lambda)^a -----------------
# "which at beta=1/2 is exactly (M/Lambda)^a" (l.2147-2148).  omega(u)=u^{1/2}.
lhs_half = M**(a - half) * (u**half) * Lam**(half - a)
check("(2) beta=1/2:  LHS - (M/L)^a  (RHS collapses to (M/L)^a)",
      lhs_half - u**a)

# --- (3) the Schur exponent gamma := a - 1/2 + beta ; at beta=1/2, gamma = a ---
gamma = a - half + beta
check("(3a) gamma - (a - 1/2 + beta)             [definition, l.2150]",
      gamma - (a - half + beta))
check("(3b) gamma|_{beta=1/2} - a                 [gamma=a at target]",
      gamma.subs(beta, half) - a)

# --- (4) kernel exponent consistency: RHS of (1) is exactly (M/L)^gamma -------
# The summand becomes the dyadic Schur kernel K(M,L)=(M/L)^gamma on support
# (l.2148-2150).  So the extracted power equals the kernel power, no slack.
check("(4) (M/L)^{a-1/2+beta} - (M/L)^gamma        [kernel power = extracted power]",
      u**(a - half + beta) - u**gamma)

# --- (5) c_a substitution identity: general-gamma form vs beta=1/2 form -------
# c_a := (c_geo^gamma/(1-2^{-gamma}))^2 ; at beta=1/2 this is (c_geo^a/(1-2^{-a}))^2
# (eq an17:eq-D5, l.2160-2163).  Substituting gamma|_{beta=1/2}=a must be exact.
c_a_gamma = (c_geo**gamma / (1 - 2**(-gamma)))**2
c_a_half = (c_geo**a / (1 - 2**(-a)))**2
check("(5) c_a(gamma)|_{beta=1/2} - (c_geo^a/(1-2^-a))^2   [D5 constant collapse]",
      c_a_gamma.subs(beta, half) - c_a_half)

# --- numeric cross-check (guards against a simplify() false-zero) -------------
# Re-evaluate residuals (1) and (2) at random positive floating-point arguments.
# We use a RELATIVE residual |LHS/RHS - 1|: with a up to ~12 and frequencies up to
# 5e4, LHS is a product of separately-huge factors (M^{a-1/2} ~ 1e54, L^{1/2-a} ~
# 1e-52), so an *absolute* LHS-RHS would only measure floating cancellation, not the
# identity.  RHS = (M/L)^{a-1/2+beta} is never zero, so the ratio is well posed.
random.seed(114)
max_num_res = 0.0
for _ in range(2000):
    subs = {M: random.uniform(1.0, 5e4), Lam: random.uniform(1.0, 5e4),
            a: random.uniform(2.01, 12.0), beta: random.uniform(0.0, 2.0)}
    v_lhs = complex(lhs.subs(subs)); v_rhs = complex(rhs.subs(subs))
    r1 = abs(v_lhs / v_rhs - 1.0)
    subs2 = {k: v for k, v in subs.items() if k is not beta}
    v_lhs2 = complex(lhs_half.subs(subs2)); v_rhs2 = complex((u**a).subs(subs2))
    r2 = abs(v_lhs2 / v_rhs2 - 1.0)
    max_num_res = max(max_num_res, r1, r2)
numeric_ok = max_num_res < 1e-9

# --- report -------------------------------------------------------------------
print("=" * 74)
print("s4b_arith_check.py  --  S4b (D4)->(D5) exponent-bookkeeping identity")
print("  omega(u) = u^{1/2} = u^beta on 0<u<=c_geo   (eq an17:eq-D4, beta=1/2)")
print("  target residual for every identity:  the integer 0  (sympy.simplify)")
print("=" * 74)
all_zero = True
for name, r, passed in results:
    all_zero = all_zero and passed
    print(f"[{'PASS' if passed else 'FAIL'}]  simplify(residual) = {r!s:<6}  {name}")
print("-" * 74)
print(f"[{'PASS' if numeric_ok else 'FAIL'}]  numeric cross-check (2000 random pts):"
      f"  max|LHS/RHS - 1| = {max_num_res:.3e}   target < 1e-9")
print("-" * 74)
ok = all_zero and numeric_ok
print("RESULT:", "ALL RESIDUALS IDENTICALLY ZERO -- exponent identity confirmed"
      if ok else "SOME RESIDUAL NONZERO -- exponent identity FAILS (report as finding)")
print("measured:  {} symbolic residuals, all == 0;  numeric max = {:.2e}"
      .format(len(results), max_num_res))
print("target:    all symbolic residuals == 0 (identically);  numeric == 0")
raise SystemExit(0 if ok else 1)
