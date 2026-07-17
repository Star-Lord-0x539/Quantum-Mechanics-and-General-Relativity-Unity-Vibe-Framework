# RUN 56 A1 — H1 re-derivation checks (independent of run-53 ds_adv.py)
# C1: u affine + Z u-independence on common horizon (symbolic, embedding)
# C2: BD pole residue at Z=1 is mass-independent (exact 2F1 closed form)
# C3: distributional limit: pole -> chiral kernel -(1/4pi)d^2/(u-u'-i0)^2;
#     log term -> 0 (rates), smooth term -> 0 identically
# C4: large-Z asymptotics W,E ~ Z^{-h_minus} (first-moment falloff input, H4)
import sympy as sp
import mpmath as mp

ok = []
# ---- C1: embedding geometry ----
u, up, s = sp.symbols('u uprime s', real=True)
H = sp.symbols('Hc', positive=True)
n1, n2, n3, m1, m2, m3 = sp.symbols('n1 n2 n3 m1 m2 m3', real=True)
X = sp.Matrix([u, u, n1/H, n2/H, n3/H])       # X(u, nhat)
Xp = sp.Matrix([up, up, m1/H, m2/H, m3/H])
eta = sp.diag(-1, 1, 1, 1, 1)
XX = (X.T*eta*X)[0, 0]
ok.append(('C1a X.X=1/H^2 iff |n|=1', sp.simplify(XX - 1/H**2)
           == sp.simplify((n1**2+n2**2+n3**2-1)/H**2)))
Z = sp.simplify(H**2*(X.T*eta*Xp)[0, 0])
ok.append(('C1b Z = n.m (u,up-free)', Z == n1*m1+n2*m2+n3*m3))
# straight null line in R^{1,4} lying on hyperboloid => geodesic, u affine:
dX = sp.Matrix([1, 1, 0, 0, 0])
ok.append(('C1c generator null + straight', sp.simplify((dX.T*eta*dX)[0, 0]) == 0))
# boost in (X0,X1): X -> (X0 ch + X1 sh, X0 sh + X1 ch, ...) on horizon: u -> e^s u
B = sp.eye(5)
B[0, 0] = B[1, 1] = sp.cosh(s); B[0, 1] = B[1, 0] = sp.sinh(s)
BX = B*X
ok.append(('C1d boost|_H = dilation u->e^s u',
           sp.simplify(BX[0]-sp.exp(s)*u) == 0 and sp.simplify(BX[1]-sp.exp(s)*u) == 0))

# ---- C2: exact BD two-point fn, pole residue mass-independent ----
# W(Z) = (H^2/16pi^2) G(hp)G(hm) 2F1(hp,hm;2;(1+Z)/2), h± = 3/2 ± nu
# 2F1(a,b;c;w), c-a-b = -1: leading sing = G(c)G(a+b-c)/(G(a)G(b)) (1-w)^{-1}
# => residue of W at w=1: (H^2/16pi^2)*G(2)G(1) = H^2/16pi^2, m-independent
# check numerically: (1-w)*W -> H^2/16pi^2 *2/... careful: 1-w=(1-Z)/2
mp.mp.dps = 30
def W_BD(Z, m2overH2):
    nu = mp.sqrt(mp.mpf(9)/4 - m2overH2)  # complementary if real, principal if imag
    hp, hm = mp.mpf(3)/2 + nu, mp.mpf(3)/2 - nu
    return (mp.gamma(hp)*mp.gamma(hm)/(16*mp.pi**2))*mp.hyp2f1(hp, hm, 2, (1+Z)/2)
res = []
for m2 in [mp.mpf('0.5'), mp.mpf(4), mp.mpf(100)]:   # compl., principal, m>>H
    vals = [(1-(1+Zv)/2)*W_BD(Zv, m2) for Zv in [mp.mpf('0.999999'), mp.mpf('0.99999999')]]
    res.append(vals[-1])
target = 1/(16*mp.pi**2)
ok.append(('C2 pole residue univ. (3 masses)',
           all(abs(r-target) < 1e-4*target for r in res)))

# ---- C3: distributional limits (transverse integral, epsilon->0) ----
# pole: -d^2/dDu^2 [C/(r^2+i e Du)], C=1/4pi^2; smeared vs transverse f=1 near 0:
# I(e) = int_0^R 2pi r dr 2C e^2 (r^2+ieDu)^-3 -> -1/(4 pi Du^2)
Du = mp.mpf('0.7'); R = mp.mpf(2); C = 1/(4*mp.pi**2)
def Ipole(e):
    f = lambda r: 2*mp.pi*r*2*C*e**2*(r**2+1j*e*Du)**-3
    return mp.quad(f, [0, R])
ok.append(('C3a pole -> -1/(4pi Du^2)',
           abs(Ipole(mp.mpf('1e-6')) - (-1/(4*mp.pi*Du**2))) < 1e-4))
def Ilog(e):  # -d^2/dDu^2 log(r^2+ie Du) = e^2 (r^2+ie Du)^-2
    f = lambda r: 2*mp.pi*r*e**2*(r**2+1j*e*Du)**-2
    return mp.quad(f, [0, R])
ok.append(('C3b log-term -> 0 (rate ~e)',
           abs(Ilog(mp.mpf('1e-4'))) < 1e-3 and
           abs(Ilog(mp.mpf('1e-5'))) < 1e-4))

# ---- C4: large-Z falloff of mode/commutator ~ |Z|^{-hm} (principal: |Z|^{-3/2}) ----
def decay_exp(m2):
    Zs = [mp.mpf(10)**k for k in [3, 4]]
    ws = [abs(W_BD(-Zv, m2)) for Zv in Zs]   # spacelike far side, real branch
    return mp.log(ws[0]/ws[1])/mp.log(Zs[1]/Zs[0])
d_c = decay_exp(mp.mpf('0.5'))    # hm = 3/2 - sqrt(9/4-0.5) ~ 0.17712
ok.append(('C4a compl. decay ~ hm', abs(d_c - (mp.mpf(3)/2 - mp.sqrt(mp.mpf(9)/4-mp.mpf('0.5')))) < 0.02))
# principal series: W ~ |Z|^{-3/2} x oscillatory Z^{±i rho}; test envelope:
# |W| |Z|^{3/2} bounded above and away from 0 in mean over 2 decades
env = [abs(W_BD(-mp.mpf(10)**k, mp.mpf(4)))*mp.mpf(10)**(mp.mpf(3)*k/2)
       for k in [mp.mpf(3)+mp.mpf(j)/4 for j in range(9)]]
ok.append(('C4b principal envelope |Z|^{-3/2} (bounded, nonzero mean)',
           max(env) < 1 and sum(env)/len(env) > 1e-6))

for name, passed in ok:
    print(('PASS' if passed else 'FAIL'), name)
print('ALL PASS' if all(p for _, p in ok) else 'SOME FAIL')
