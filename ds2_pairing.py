# ============================================================================
# ds2_pairing.py -- RUN 56 ARM A2 companion checks for ds2_pairing.tex
# P1 exact chart = dS_4 (Ric = 3/l^2 g, all components, symbolic)
# P2 horizon facts: Gamma^mu_vv|_N = 0; h_AB v-indep on N; theta_l = 4v/l^2;
#    g_vv quadratic in u
# P3 linearised Raychaudhuri on N: trace perturbation dR_vv|_N = -f''(v);
#    TT perturbation dR_vv|_N = 0; extension robustness (dg_vv = eps u^2 q(v))
# P4 pairing identity per generator (quadrature): int (v-V) F = G(inf)-G(V)
#    with far boundary term (v-V)*dtheta -> 0; normalization independence
# P5 R8 witness: F = 1/(1+v^2): ANEC finite, first moment divergent,
#    area deficit log-divergent (the premise-(b) consumption located)
# P6 improvement/Wald telescoping (F3): int (v-V)(-Phi'') = Phi(V)-Phi(inf);
#    G_eff algebra; equivalence of pairings (a_W) <=> (a_A)
# P7 second cut-variation: d^2/dV^2 of both sides -> pointwise density
# ============================================================================
import sympy as sp
import numpy as np
from scipy.integrate import quad

results = []

def report(tag, ok, detail=""):
    results.append((tag, ok))
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} {detail}")

# ---------------------------------------------------------------- symbolic
u, v, th, ph, l, eps = sp.symbols('u v theta phi ell epsilon', real=True)
f = sp.Function('f')(v)   # trace perturbation profile
s = sp.Function('s')(v)   # TT perturbation profile
q = sp.Function('q')(v)   # extension-probe profile

def christoffel(g, x):
    ginv = g.inv()
    n = len(x)
    Gam = [[[sp.S(0)]*n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                expr = sp.S(0)
                for d in range(n):
                    expr += ginv[a, d]*(sp.diff(g[d, b], x[c])
                                        + sp.diff(g[d, c], x[b])
                                        - sp.diff(g[b, c], x[d]))
                Gam[a][b][c] = sp.simplify(expr/2)
    return Gam

def ricci(g, x):
    n = len(x)
    Gam = christoffel(g, x)
    Ric = sp.zeros(n, n)
    for b in range(n):
        for c in range(n):
            expr = sp.S(0)
            for a in range(n):
                expr += sp.diff(Gam[a][b][c], x[a]) - sp.diff(Gam[a][b][a], x[c])
                for d in range(n):
                    expr += Gam[a][a][d]*Gam[d][b][c] - Gam[a][c][d]*Gam[d][b][a]
            Ric[b, c] = sp.simplify(expr)
    return Ric

x = (u, v, th, ph)
r2 = (l + 2*u*v/l)**2
g0 = sp.diag(0, 0, r2, r2*sp.sin(th)**2)
g0[0, 1] = g0[1, 0] = -2
g0[1, 1] = 4*u**2/l**2

# P1: exact chart is dS_4
Ric0 = ricci(g0, x)
dev = sp.simplify(Ric0 - 3/l**2*g0)
report("P1 chart=dS4: Ric-3/l^2 g == 0", dev == sp.zeros(4, 4))

# P2: horizon facts on N = {u=0}
Gam0 = christoffel(g0, x)
gam_vv_onN = [sp.simplify(Gam0[a][1][1].subs(u, 0)) for a in range(4)]
report("P2a Gamma^mu_vv|_N = 0 (affine generators)",
       all(t == 0 for t in gam_vv_onN), str(gam_vv_onN))
hAB_v_indep = (sp.simplify(sp.diff(g0[2, 2], v).subs(u, 0)) == 0 and
               sp.simplify(sp.diff(g0[3, 3], v).subs(u, 0)) == 0)
report("P2b h_AB v-independent on N (theta=sigma=0 exactly)", hAB_v_indep)
sqrth = sp.sqrt(sp.simplify(g0[2, 2]*g0[3, 3]))
theta_l = sp.simplify(sp.diff(sp.log(sqrth), u).subs(u, 0))
report("P2c theta_l = 4v/l^2 (cuts v!=0 not totally geodesic)",
       sp.simplify(theta_l - 4*v/l**2) == 0, f"theta_l={theta_l}")
report("P2d g_vv = 4u^2/l^2 (quadratic vanishing at N)",
       sp.simplify(g0[1, 1] - 4*u**2/l**2) == 0)
report("P2e R_vv|_N = 0 = G_vv|_N (background)",
       sp.simplify(Ric0[1, 1].subs(u, 0)) == 0)

# P3: linearised Raychaudhuri on N
def dRvv_onN(gpert):
    Ric = ricci(gpert, x)
    series = sp.series(Ric[1, 1].subs(u, 0), eps, 0, 2).removeO()
    return sp.simplify(sp.expand(series).coeff(eps, 1))

# (a) trace perturbation dh_AB = eps f(v) h_AB  => expect -f''(v)
ga = g0.copy()
ga[2, 2] = g0[2, 2]*(1 + eps*f)
ga[3, 3] = g0[3, 3]*(1 + eps*f)
dRa = dRvv_onN(ga)
report("P3a trace pert: dR_vv|_N = -f''(v) (linearised Raychaudhuri sign)",
       sp.simplify(dRa + sp.diff(f, v, 2)) == 0, f"dR_vv={dRa}")

# (b) TT perturbation dh_thth = +eps s, dh_phph = -eps s (traceless) => 0
gb = g0.copy()
gb[2, 2] = g0[2, 2]*(1 + eps*s)
gb[3, 3] = g0[3, 3]*(1 - eps*s)
dRb = dRvv_onN(gb)
report("P3b TT pert: dR_vv|_N = 0 (no shear at linear order)",
       sp.simplify(dRb) == 0, f"dR_vv={dRb}")

# (c) extension robustness: add dg_vv = eps u^2 q(v) to (a) => unchanged
gc = ga.copy()
gc[1, 1] = g0[1, 1] + eps*u**2*q
dRc = dRvv_onN(gc)
report("P3c extension probe dg_vv=eps u^2 q(v): dR_vv|_N unchanged",
       sp.simplify(dRc - dRa) == 0, f"diff={sp.simplify(dRc-dRa)}")

# ---------------------------------------------------------------- numeric
INF = np.inf

def dtheta(F, vv):          # dtheta(v) = int_v^inf F  (FN: dtheta(inf)=0)
    return quad(F, vv, INF, limit=400)[0]

def Gfun(F, vv, Ginf=0.0):  # G(v) = Ginf - int_v^inf dtheta(s) ds
    return Ginf - quad(lambda s2: dtheta(F, s2), vv, INF, limit=400)[0]

def wflux(F, V):            # int_V^inf (v-V) F dv
    return quad(lambda t: (t - V)*F(t), V, INF, limit=400)[0]

# P4: pairing identity, smooth + power-law profiles, several cuts,
#     normalization (Ginf) independence, far-boundary decay.
#     deficit G(inf)-G(V) = int_V^inf dtheta(s) ds computed with an ANALYTIC
#     inner tail for the power-law profile (single-level quadrature).
F1 = lambda t: np.exp(-(t - 3.0)**2)*(1.0 + 0.5*np.sin(t))
dtheta1 = lambda s2: dtheta(F1, s2)
F2 = lambda t: 0.8/(1.0 + abs(t))**3.2
def dtheta2(s2):            # exact int_s^inf F2
    tail0 = 0.8/2.2
    if s2 >= 0:
        return 0.8/(2.2*(1.0 + s2)**2.2)
    # int_s^0 0.8 (1-t)^{-3.2} dt = (0.8/2.2)[1 - (1-s)^{-2.2}] ... derivative check:
    # d/dt [ (0.8/2.2)(1-t)^{-2.2} ] = 0.8 (1-t)^{-3.2}  => int_s^0 = (0.8/2.2)(1-(1-s)^{-2.2})
    return tail0 + (0.8/2.2)*(1.0 - (1.0 - s2)**(-2.2))
ok4, det4, fails4 = True, [], []
for F, dth, nm in ((F1, dtheta1, "gauss"), (F2, dtheta2, "pow3.2")):
    for V in (-1.0, 0.0, 2.5):
        dfc = quad(dth, V, INF, limit=800, epsabs=1e-11)[0]   # Ginf-indep deficit
        rhs = wflux(F, V)
        ok = abs(dfc - rhs) < 1e-6*(1 + abs(rhs))
        ok4 &= ok
        (det4 if ok else fails4).append(
            f"{nm},V={V}: dfct={dfc:.8f} flux={rhs:.8f}")
bnd = [(vv - 0.0)*dtheta2(vv) for vv in (10, 100, 1000)]
ok4 &= bnd[0] > bnd[1] > bnd[2] > 0 and bnd[2] < 2e-3
report("P4 pairing: area deficit == (v-V)-weighted flux; boundary->0",
       ok4, "; ".join(det4[:3]) + f" ... bdry(v-V)dtheta @10/100/1000: "
       f"{bnd[0]:.2e},{bnd[1]:.2e},{bnd[2]:.2e}"
       + ("" if not fails4 else " FAILS: " + "; ".join(fails4)))

# P5: R8 witness F = 1/(1+v^2): ANEC finite, first moment divergent,
#     deficit log-divergent => display has no finite LHS (premise (b) locus)
F8 = lambda t: 1.0/(1.0 + t*t)
anec = quad(F8, 0, INF)[0]
moments = [0.5*np.log(1.0 + L*L) for L in (1e2, 1e4, 1e6)]   # exact int_0^L t F8
# dtheta(v) = int_v^inf F8 = arctan(1/v);  G(L)-G(0) = int_0^L arctan(1/v) dv
#           = L arctan(1/L) + (1/2) log(1+L^2)   (exact; -> 1 + log L)
deficits = [L*np.arctan(1.0/L) + 0.5*np.log(1.0 + L*L) for L in (1e2, 1e4, 1e6)]
ratios = [moments[i+1]/moments[i] for i in range(2)]   # log-growth: 2.0, 1.5
ok5 = (abs(anec - np.pi/2) < 1e-9 and moments[2] > 12 and
       abs(ratios[0] - 2.0) < 1e-3 and abs(ratios[1] - 1.5) < 1e-3 and
       deficits[2] > 13 and abs(deficits[2] - (1.0 + np.log(1e6))) < 1e-3)
report("P5 R8 witness: ANEC=pi/2 finite; first moment ~ log L (divergent); "
       "area deficit ~ log L (no finite LHS)", ok5,
       f"anec={anec:.9f} mom@1e2/1e4/1e6={moments[0]:.3f}/{moments[1]:.3f}/"
       f"{moments[2]:.3f} dfct={deficits[0]:.3f}/{deficits[1]:.3f}/{deficits[2]:.3f}")

# P6: improvement/Wald telescoping (F3): int_V^inf (v-V)(-Phi'') = Phi(V)-Phi(inf)
Phi  = lambda t: 1.0/(1.0 + t*t)                      # Phi(inf)=0
Phi2 = lambda t: (6*t*t - 2)/(1.0 + t*t)**3           # Phi''
ok6a, det6 = True, []
for V in (0.0, 1.5, -0.5):
    # run-53 R6 form: int_V^inf (v-V) Phi'' dv = Phi(V) - Phi(inf)   [tex (II)
    # equivalently: int (v-V)(-Phi'') = Phi(inf) - Phi(V)]
    lhs = quad(lambda t: (t - V)*Phi2(t), V, INF, limit=400)[0]
    rhs = Phi(V) - 0.0
    ok6a &= abs(lhs - rhs) < 1e-8*(1 + abs(rhs))
    det6.append(f"V={V}: {lhs:.9f} vs {rhs:.9f}")
report("P6a F3 telescoping (R6): int (v-V) Phi'' = Phi(V)-Phi(inf)", ok6a,
       "; ".join(det6))

# P6b: G_eff algebra (symbolic): dR = 8 pi G (Tmin - xi Phi'' + xi c dR)
#      <=> dR = 8 pi G_eff (Tmin - xi Phi''), 1/G_eff = 1/G - 8 pi xi c
G_, xi_, c_, Tm_, P2_, dR_ = sp.symbols('G xi c Tmin Phipp dR', positive=False)
sol = sp.solve(sp.Eq(dR_, 8*sp.pi*G_*(Tm_ - xi_*P2_ + xi_*c_*dR_)), dR_)[0]
Geff = 1/(1/G_ - 8*sp.pi*xi_*c_)
ok6b = sp.simplify(sol - 8*sp.pi*Geff*(Tm_ - xi_*P2_)) == 0
report("P6b condensate => G_eff: dR = 8 pi G_eff (Tmin - xi Phi'')", ok6b)

# P6c: equivalence of pairings per generator (numbers):
#      (a_A): dfct/4G_eff = 2 pi int (v-V)[Tmin - xi Phi'']
#      (a_W): dfct/4G_eff + 2 pi xi (Phi(V)-Phi(inf)) = 2 pi int (v-V) Tmin
GN, xi = 1.7, 0.31          # arbitrary
Geffn = 1.0/(1.0/GN - 8*np.pi*xi*0.05)  # condensate c=0.05 folded in
Tmin = lambda t: np.exp(-(t - 2.0)**2)
dRvv = lambda t: 8*np.pi*Geffn*(Tmin(t) - xi*Phi2(t))   # field eq. on N
ok6c, det6c = True, []
for V in (0.0, 1.2):
    dfct = wflux(dRvv, V)/(4*Geffn)                     # Thm: geometric side
    aA = 2*np.pi*quad(lambda t: (t-V)*(Tmin(t) - xi*Phi2(t)), V, INF, limit=400)[0]
    aW = 2*np.pi*quad(lambda t: (t-V)*Tmin(t), V, INF, limit=400)[0]
    okA = abs(dfct - aA) < 1e-7*(1 + abs(aA))
    okW = abs(dfct + 2*np.pi*xi*Phi(V) - aW) < 1e-7*(1 + abs(aW))
    ok6c &= okA and okW
    det6c.append(f"V={V}: dfct={dfct:.7f} aA={aA:.7f} aW-shift ok={okW}")
report("P6c pairings equivalent: (a_A) and (a_W) both hold, differ by F3 term",
       ok6c, "; ".join(det6c))

# P7: second cut-variation: d^2/dV^2 [weighted flux] = Phi(V) and
#     d^2/dV^2 [deficit] = -G''(V) = F(V) (finite differences)
ok7, det7 = True, []
Ftest = lambda t: np.exp(-(t - 2.0)**2)*(1.3 + np.cos(t))
for V in (0.5, 2.0):
    h = 1e-3
    d2flux = (wflux(Ftest, V + h) - 2*wflux(Ftest, V) + wflux(Ftest, V - h))/h**2
    dfc = lambda W: -Gfun(Ftest, W)          # deficit with Ginf=0
    d2def = (dfc(V + h) - 2*dfc(V) + dfc(V - h))/h**2
    ok7 &= abs(d2flux - Ftest(V)) < 5e-5 and abs(d2def - Ftest(V)) < 5e-5
    det7.append(f"V={V}: flux''={d2flux:.7f} dfct''={d2def:.7f} F(V)={Ftest(V):.7f}")
report("P7 second variation: d^2/dV^2 of BOTH sides = pointwise density F(V)",
       ok7, "; ".join(det7))

n_ok = sum(1 for _, o in results if o)
print(f"\n==== {n_ok}/{len(results)} PASS ====")
