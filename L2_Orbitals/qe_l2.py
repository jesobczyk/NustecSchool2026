"""Physics engine and interactive panels for the Lecture 2 demo (quasielastic electron scattering school).
Everything the notebook shows is defined here: the Fermi gas and lepton side (Lecture 1), the Woods-Saxon orbitals,
spectral functions and the inclusive response (Lecture 2), the data loaders, and one show_X() function per panel."""
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import ipywidgets as W
from IPython.display import display

# ---------- visual language: thin marks, recessive axes, fixed categorical order ----------
C = dict(data='#2b2b2b', data_light='#9a9a9a', rfg='#2a78d6', L='#eb6834', T='#1baf7a',
         eqs1='#2a78d6', eqs2='#eb6834', accent='#6250d6', warn='#e34948', grid='#e6e6e3')
mpl.rcParams.update({
    'figure.dpi': 110, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.7,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False, 'lines.linewidth': 2.0,
    'mathtext.fontset': 'cm',
})
# If sliders show "Error displaying widget", your Jupyter front-end lacks the ipywidgets extension.
# Fix: `pip install -U ipywidgets widgetsnbextension` (+ `jupyter nbextension enable --py widgetsnbextension` for classic Notebook < 7),
# restart the server.  Or set USE_WIDGETS = False to get static versions of every interactive panel.
USE_WIDGETS = True

def show_panel(fn, controls, widget_kwargs, static_calls):
    """Interactive panel when widgets work; otherwise the static small-multiple calls."""
    if USE_WIDGETS:
        display(controls, W.interactive_output(fn, widget_kwargs))
    else:
        for kw in static_calls: fn(**kw)

def polish(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=2, color='#999'); ax.set_axisbelow(True)

M = 0.93827; ALPHA = 1/137.036; HBARC = 0.197327; HBARC2_NB = 0.389379e6   # GeV, GeV fm, GeV^-2 -> nb
MU_P, MU_N = 2.793, -1.913
trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

def form_factors(Q2):
    GD = 1.0/(1+Q2/0.71)**2; tau = Q2/(4*M*M)
    return GD, MU_P*GD, -MU_N*tau*GD/(1+5.6*tau), MU_N*GD          # GEp, GMp, GEn (Galster), GMn

def mott(E, th):
    s2 = np.sin(th/2)**2
    return ALPHA**2*(1-s2)/(4*E*E*s2*s2)

def lepton_factors(E, th, omega):
    Q2 = 4*E*(E-omega)*np.sin(th/2)**2
    q = np.sqrt(Q2 + omega**2)
    return Q2, q, (Q2/q**2)**2, Q2/(2*q**2) + np.tan(th/2)**2

def rfg_responses(omega, q, kF, eB, Z=6, N=6, npts=300):
    """R_L, R_T [GeV^-1] for arrays omega, q (same shape)."""
    omega = np.atleast_1d(omega).astype(float); q = np.broadcast_to(q, omega.shape).astype(float)
    w = (omega - eB)[:, None]; qq = q[:, None]
    p = np.linspace(1e-4, kF, npts)[None, :]
    Ep = np.sqrt(p*p + M*M); Epp = Ep + w
    ok = Epp > M
    pp2 = np.where(ok, Epp**2 - M*M, 0.0)
    cos = np.clip((pp2 - p*p - qq*qq)/(2*p*qq), -1, 1)
    ok &= (np.abs((pp2 - p*p - qq*qq)/(2*p*qq)) <= 1) & (pp2 > kF*kF)      # kinematics & Pauli blocking
    Q2 = np.clip(qq*qq - w*w, 1e-6, None)
    pdotq = Ep*w - p*qq*cos
    pt0 = Ep + pdotq*w/Q2                       # ptilde^0 = E - (p.q) w / q_mu q^mu,  q_mu q^mu = -Q2
    pperp2 = p*p*(1-cos**2)
    jac = 2*np.pi*p*Epp/qq * (M*M/(Ep*Epp))     # 2 pi p^2 dp * E'/(p q) * M^2/(E E')
    dp = p[0,1]-p[0,0]
    GEp, GMp, GEn, GMn = form_factors(Q2)
    RL = 0; RT = 0
    for nN, GE, GM in ((Z, GEp, GMp), (N, GEn, GMn)):
        tau = Q2/(4*M*M); W1 = tau*GM**2; W2 = (GE**2+tau*GM**2)/(1+tau)
        w00 = -W1*(qq*qq/Q2) + W2*pt0**2/(M*M)
        wT  = 2*W1 + W2*pperp2/(M*M)
        norm = 3*nN/(4*np.pi*kF**3)*dp
        RL = RL + norm*np.sum(np.where(ok, jac*w00, 0.0), axis=1)
        RT = RT + norm*np.sum(np.where(ok, jac*wT , 0.0), axis=1)
    return RL, RT

def rfg_xsec(E, th, omega, kF, eB, Z=6, N=6):
    """(total, L part, T part) of d2sigma/dOmega dE' in nb/sr/GeV."""
    Q2, q, vL, vT = lepton_factors(E, th, omega)
    RL, RT = rfg_responses(omega, q, kF, eB, Z, N)
    sM = mott(E, th)*HBARC2_NB
    return sM*(vL*RL + vT*RT), sM*vL*RL, sM*vT*RT

def qe_peak_position(E, th, eB, Mn=M):
    """omega at which a nucleon at rest with removal energy eB is knocked out."""
    s2 = np.sin(th/2)**2
    return (4*E*E*s2 + 2*Mn*eB - eB*eB)/(2*Mn - 2*eB + 4*E*s2)

# ---------- scaling (Donnelly–Sick) ----------
def scaling_pieces(E, th, omega, kF, Eshift, Z=6, N=6):
    """psi', and the single-nucleon factors G_L, G_T so that f(psi') = sigma / (sigma_Mott (vL G_L + vT G_T)) and the RFG gives exactly 3/4 (1 - psi'^2)."""
    Q2, q, vL, vT = lepton_factors(E, th, omega)
    w = omega - Eshift
    lam = w/(2*M); kap = q/(2*M); tau = np.clip(kap**2 - lam**2, 1e-9, None)
    etaF = kF/M; xiF = np.sqrt(1+etaF**2)-1
    psi = (lam-tau)/np.sqrt(xiF*((1+lam)*tau + kap*np.sqrt(tau*(1+tau))))
    GEp, GMp, GEn, GMn = form_factors(4*M*M*tau)
    Delta = xiF*(1-psi**2)*(kap*np.sqrt(1+1/tau) + xiF*(1-psi**2)/3)
    UL = lambda GE, GM: (kap**2/tau)*(GE**2 + (GE**2+tau*GM**2)/(1+tau)*Delta)
    UT = lambda GE, GM: 2*tau*GM**2 + (GE**2+tau*GM**2)/(1+tau)*Delta
    pref = xiF/(M*etaF**3*kap)
    GL = pref*(Z*UL(GEp,GMp) + N*UL(GEn,GMn)); GT = pref*(Z*UT(GEp,GMp) + N*UT(GEn,GMn))
    return psi, GL, GT, vL, vT, q

f_RFG = lambda psi: 0.75*(1-psi**2)*(np.abs(psi) < 1)

# ---------- GiBUU mean field: Welke et al. parametrisation, Teis fits (Leitner, Eq. 7.17, Table 7.3) ----------
RHO0 = 0.168   # fm^-3
EQS = {'EQS 1  (p-independent, K = 215 MeV)': dict(A=-0.2870, B=0.2337, C=0.0,     tau=1.23, Lam=None),
       'EQS 2  (p-dependent,   K = 290 MeV)': dict(A=-0.0293, B=0.0572, C=-0.0635, tau=1.76, Lam=2.13)}

def V_gibuu(p, rho, par):
    """V_N(p, rho) in GeV for symmetric matter; p in GeV, rho in fm^-3."""
    p = np.clip(np.atleast_1d(p).astype(float), 1e-6, None); x = rho/RHO0
    V = par['A']*x + par['B']*x**par['tau'] + 0*p
    if par['C']:
        pF = HBARC*(1.5*np.pi**2*rho)**(1/3); Lam = par['Lam']*HBARC
        pp = np.linspace(1e-5, pF, 300); P, PP = np.meshgrid(p, pp, indexing='ij')
        ang = Lam**2/(2*P*PP)*np.log((Lam**2+(P+PP)**2)/(Lam**2+(P-PP)**2))   # angular integral of the Yukawa kernel
        integ = trapz(2*np.pi*PP**2*ang, pp, axis=1)                        # GeV^3
        V = V + 2*par['C']/RHO0 * 4*integ/(2*np.pi)**3/HBARC**3                # g=2 spin, n+p
    return V

def rho_C12(r, R=1.692, a=1.082):
    """Harmonic-oscillator density of 12C (Leitner Table 7.4), normalised to A=12; r in fm."""
    f = (1 + a*(r/R)**2)*np.exp(-(r/R)**2)
    rr = np.linspace(0, 12, 2000); norm = trapz(4*np.pi*rr**2*(1 + a*(rr/R)**2)*np.exp(-(rr/R)**2), rr)
    return 12*f/norm

# units in this engine: MeV and fm (hbar c = 197.327 MeV fm); momenta in MeV
trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')   # numpy < 2.0 compatibility
from scipy.special import spherical_jn
HBARC_MEVFM = 197.327; M_MEV = 938.27; ALPHA = 1/137.036

# ------------------------------------------------------------------ Woods-Saxon single-particle problem
def ws_potential(r, V0=55.0, r0=1.32, a=0.55, Vso=9.0, A=12, Z=6, proton=True, l=0, j=0.5):
    """Mougey-type potential: central WS of depth -V0, Thomas spin-orbit with strength Vso (MeV fm^2 units via (hbar/m_pi c)^2),
    Coulomb of a uniform sphere of radius 1.3 A^{1/3} with charge (Z-1)e for protons."""
    R = r0*A**(1/3); f = 1/(1+np.exp((r-R)/a))
    V = -V0*f
    # spin-orbit: V_so (hbar/m_pi c)^2 (1/r) df/dr  l.s ,  (hbar/m_pi c)^2 = 2.0 fm^2
    dfdr = -f*(1-f)/a
    ls = 0.5*(j*(j+1) - l*(l+1) - 0.75)
    V = V + Vso*2.0*(1/np.maximum(r, 1e-6))*dfdr*ls
    if proton:
        Rc = 1.3*A**(1/3); zz = (Z-1)*ALPHA*HBARC_MEVFM
        V = V + np.where(r < Rc, zz*(3 - (r/Rc)**2)/(2*Rc), zz/np.maximum(r, 1e-6))
    return V

def solve_ws(l, j, V0=55.0, r0=1.32, a=0.55, Vso=9.0, A=12, Z=6, proton=True, rmax=15.0, n=600):
    """Bound states of the radial equation for given (l, j): returns energies (MeV) and radial functions u(r) (normalised)."""
    mu = M_MEV*(A-1)/A
    r = np.linspace(rmax/n, rmax, n); h = r[1]-r[0]
    V = ws_potential(r, V0, r0, a, Vso, A, Z, proton, l, j) + HBARC_MEVFM**2*l*(l+1)/(2*mu*r*r)
    k = HBARC_MEVFM**2/(2*mu*h*h)
    diag = 2*k + V; off = -k*np.ones(n-1)
    from scipy.linalg import eigh_tridiagonal
    E, U = eigh_tridiagonal(diag, off, select='v', select_range=(-200, 0))
    us = []
    for i in range(len(E)):
        u = U[:, i]; u = u/np.sqrt(np.sum(u*u)*h); us.append(u*np.sign(u[np.argmax(np.abs(u[:n//3]))]))
    return r, E, us

def momentum_distribution(r, u, l, p):
    """n(p) [MeV^-3] for p in MeV, normalised so that 4 pi int p^2 n(p) dp = 1.
    phi(k) = sqrt(2/pi) int u(r) j_l(k r) r dr with k = p/(hbar c) in fm^-1; n = |phi|^2/(4 pi) in fm^3 -> MeV^-3."""
    k = p/HBARC_MEVFM
    kr = np.outer(k, r)
    phi = np.sqrt(2/np.pi)*trapz(u[None, :]*spherical_jn(l, kr)*r[None, :], r, axis=1)
    return phi**2/(4*np.pi)/HBARC_MEVFM**3

def c12_orbitals(V0p=55.0, V0s=66.0, r0=1.32, a=0.55, Vso=9.0, proton=True):
    """The two occupied proton (or neutron) orbitals of 12C with Mougey's shell-dependent depths. Returns dict per shell."""
    out = {}
    for name, l, j, V0 in (('1s1/2', 0, 0.5, V0s), ('1p3/2', 1, 1.5, V0p)):
        r, E, us = solve_ws(l, j, V0=V0, r0=r0, a=a, Vso=Vso, proton=proton)
        out[name] = dict(l=l, j=j, E=E[0], u=us[0], r=r, degeneracy=int(2*j+1))
    return out

# ------------------------------------------------------------------ spectral functions
def ipm_spectral_function(orbs, p, E, widths={'1s1/2': 20.0, '1p3/2': 3.0}, occ={'1s1/2': 1.0, '1p3/2': 1.0}):
    """P(p, E) [fm^3 MeV^-1] for Z protons: sum over shells of occ * (2j+1) * n_alpha(p) * Lorentzian(E; -E_alpha, Gamma)."""
    P = np.zeros((len(p), len(E)))
    for name, o in orbs.items():
        n = momentum_distribution(o['r'], o['u'], o['l'], p)
        G = widths[name]; Em = -o['E']
        L = (G/2/np.pi)/((E-Em)**2 + (G/2)**2)
        P += occ[name]*o['degeneracy']*np.outer(n, L)
    return P

def lfg_spectral_function(p, E, rho_fn, S=16.0, rmax=8.0, nr=300):
    """Local-Fermi-gas P(p,E) [MeV^-3 MeV^-1] for one nucleon species with density rho_fn(r) [fm^-3] (integrating to Z):
    at each r a Fermi sphere with kF(r), removal energy E_rem = S + T_F(r) - T(p); normalised so that
    sum 4 pi p^2 dp dE P = Z."""
    r = np.linspace(1e-3, rmax, nr); dr = r[1]-r[0]; rho = rho_fn(r); kF = HBARC_MEVFM*(3*np.pi**2*rho)**(1/3)
    P = np.zeros((len(p), len(E))); dE = E[1]-E[0]
    for ri, kfi in zip(r, kF):
        if kfi < 1: continue
        T = p**2/(2*M_MEV); Erem = S + kfi**2/(2*M_MEV) - T
        wgt = 4*np.pi*ri*ri*dr*2/(2*np.pi*HBARC_MEVFM)**3          # nucleons per MeV^3 of momentum space from this shell
        idx = (Erem - E[0])/dE
        for pi in np.where(p < kfi)[0]:
            k = int(np.floor(idx[pi]))
            if 0 <= k < len(E)-1:
                f = idx[pi]-k; P[pi, k] += (1-f)*wgt/dE; P[pi, k+1] += f*wgt/dE
    return P

def rho_c12_proton(r, R=1.692, a=1.082, Z=6):
    f = (1 + a*(r/R)**2)*np.exp(-(r/R)**2)
    rr = np.linspace(0, 12, 2000); norm = trapz(4*np.pi*rr**2*(1 + a*(rr/R)**2)*np.exp(-(rr/R)**2), rr)
    return Z*f/norm

def lfg_momentum_distribution(p, rho_fn, rmax=8.0, nr=400):
    """n(p) [fm^3] of the local Fermi gas, normalised to 4 pi int p^2 n dp = Z (p in MeV): n(p) = int d^3r 2 theta(kF(r)-p)/(2 pi hbar c)^3."""
    r = np.linspace(1e-3, rmax, nr); rho = rho_fn(r); kF = HBARC_MEVFM*(3*np.pi**2*rho)**(1/3)
    n = np.array([trapz(4*np.pi*r*r*2*(kF > pp)/(2*np.pi*HBARC_MEVFM)**3, r) for pp in p])   # 1/MeV^3
    return n

def lfg_removal_energy_band(p, rho_fn, S=16.0, rmax=8.0, nr=200):
    """For each p: the range of removal energies S + T_F(r) - T(p) over the r where the nucleon exists (kF(r) > p)."""
    r = np.linspace(1e-3, rmax, nr); rho = rho_fn(r); kF = HBARC_MEVFM*(3*np.pi**2*rho)**(1/3)
    lo, hi = [], []
    for pp in p:
        ok = kF > pp
        if ok.any():
            Erem = S + kF[ok]**2/(2*M_MEV) - pp**2/(2*M_MEV); lo.append(Erem.min()); hi.append(Erem.max())
        else: lo.append(np.nan); hi.append(np.nan)
    return np.array(lo), np.array(hi)

# ------------------------------------------------------------------ inclusive (e,e') from a spectral function (PWIA, factorised)
MU_P, MU_N = 2.793, -1.913
def form_factors_mev(Q2):   # Q2 in MeV^2
    GD = 1/(1+Q2/710000.0)**2; tau = Q2/(4*M_MEV*M_MEV)
    return GD, MU_P*GD, -MU_N*tau*GD/(1+5.6*tau), MU_N*GD

def responses_from_sf(omega, q, p, E, P, Z=6, N=6, npts_cos=None):
    """R_L, R_T [MeV^-1] for a nucleus whose protons and neutrons share the (normalised-per-nucleon) spectral function P(p,E)
    (grid p [MeV], E [MeV], P in MeV^-3 MeV^-1 with sum_i,j 4 pi p^2 dp dE P = 1).
    omega: array, q: scalar or array. Energy balance: omega = E + E_{p+q} - M_MEV (A-1 recoil neglected).
    The nucleon tensor is the free one evaluated with the effective energy transfer omega - E."""
    omega = np.atleast_1d(omega).astype(float); q = np.broadcast_to(q, omega.shape).astype(float)
    dp = p[1]-p[0]; dE = E[1]-E[0]
    RL = np.zeros_like(omega); RT = np.zeros_like(omega)
    Ep = np.sqrt(M_MEV*M_MEV + p*p)
    for iw, (w, qq) in enumerate(zip(omega, q)):
        # for each (p, E): final energy fixed -> cos theta fixed
        wt = w - E[None, :]                                    # (np, nE) energy given to the nucleon
        Epp = Ep[:, None] + wt
        ok = Epp > M_MEV
        pp2 = np.where(ok, Epp**2 - M_MEV*M_MEV, 0.0)
        cosr = (pp2 - p[:, None]**2 - qq*qq)/(2*p[:, None]*qq); ok &= np.abs(cosr) <= 1
        cos = np.clip(cosr, -1, 1)
        Q2 = np.clip(qq*qq - wt*wt, 1.0, None)
        GEp, GMp, GEn, GMn = form_factors_mev(Q2)
        pdotq = Ep[:, None]*wt - p[:, None]*qq*cos; pt0 = Ep[:, None] + pdotq*wt/Q2; pperp2 = p[:, None]**2*(1-cos**2)
        jac = 2*np.pi*p[:, None]*Epp/qq*(M_MEV*M_MEV/(Ep[:, None]*Epp))     # d^3p delta(...) with the relativistic normalisation
        for nN, GE, GM in ((Z, GEp, GMp), (N, GEn, GMn)):
            tau = Q2/(4*M_MEV*M_MEV); W1 = tau*GM**2; W2 = (GE**2+tau*GM**2)/(1+tau)
            w00 = -W1*(qq*qq/Q2) + W2*pt0**2/(M_MEV*M_MEV); wT = 2*W1 + W2*pperp2/(M_MEV*M_MEV)
            RL[iw] += nN*np.sum(np.where(ok, jac*w00*P, 0.0))*dp*dE
            RT[iw] += nN*np.sum(np.where(ok, jac*wT*P, 0.0))*dp*dE
    return RL, RT


raw = np.genfromtxt(os.path.join(DATA, '12C.dat'), usecols=(2,3,4,5,6), invalid_raise=False)
cite = np.genfromtxt(os.path.join(DATA, '12C.dat'), usecols=(-1), dtype=str, invalid_raise=False)
datasets = {}
for E, th in sorted({(round(r[0],4), round(r[1],3)) for r in raw}):
    m = (np.abs(raw[:,0]-E)<1e-4) & (np.abs(raw[:,1]-th)<1e-3)
    datasets[f'E = {E:.3f} GeV,  θ = {th:g}°   ({cite[m][0].split(":")[0]})'] = dict(E=E, th=np.radians(th), om=raw[m,2], sig=raw[m,3], err=raw[m,4])
def get(E, th):
    return next(v for v in datasets.values() if abs(v['E']-E)<1e-3 and abs(np.degrees(v['th'])-th)<0.05)
import os
def load(fn): return np.loadtxt(os.path.join(DATA, fn))
mougey = {k: load(f) for k, f in [('Em_0_36', 'mougey_fig9a_0-36MeVc.txt'), ('Em_80_180', 'mougey_fig9b_80-180MeVc.txt'), ('n_1p', 'mougey_fig10a_1p_15-21.5MeV.txt'), ('n_1s', 'mougey_fig10b_1s_30-50MeV.txt')]}
dutta = {k: load(f) for k, f in [('Em', 'dutta_fig9_Em_spectral_function.txt'), ('n_p', 'dutta_fig10a_p_shell_10-25MeV.txt'), ('n_s', 'dutta_fig10b_s_shell_30-50MeV.txt')]}

def missing(E_e, E_ep, th_e, T_p, th_p, Mp=938.27, MA1=10255.1):
    """Missing momentum and energy for in-plane (e,e'p) on 12C: angles in degrees; theta_p measured from the beam on the other side of q."""
    om = E_e - E_ep; th = np.radians(th_e); q = np.sqrt(E_e**2 + E_ep**2 - 2*E_e*E_ep*np.cos(th))
    thq = np.arctan2(E_ep*np.sin(th), E_e - E_ep*np.cos(th))      # angle of q w.r.t. beam
    pp = np.sqrt((T_p+Mp)**2 - Mp**2); thp = np.radians(th_p)
    pmx = q*np.cos(thq) - pp*np.cos(thp); pmy = q*np.sin(thq) - pp*np.sin(thp)    # q and the proton on the same side of the beam (opposite to the electron)
    pm = np.hypot(pmx, pmy); Em = om - T_p - (np.sqrt(MA1**2 + pm**2) - MA1)
    return pm, Em, q, np.degrees(thq)

def eep_sketch(E_e=497, E_ep=394, th_e=52.9, T_p=87, th_p=70.0):
    """Schematic of the in-plane (e,e'p) geometry and the missing-momentum triangle, to scale in momentum."""
    pm, Em, q, thq = missing(E_e, E_ep, th_e, T_p, th_p); om = E_e - E_ep
    pp = np.sqrt((T_p+938.27)**2 - 938.27**2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(wspace=0.05, width_ratios=[1.25, 1]))
    def arrow(ax, x0, y0, x1, y1, color, lw=2.2, ls='-'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=16))
    # --- left: the set-up in the scattering plane (electron up, hadron side down)
    ax = ax1; s = 1/600.
    the, thq_, thp = np.radians(th_e), np.radians(thq), np.radians(th_p)
    arrow(ax, -1.25, 0, -0.02, 0, C['data'])                                              # beam
    ax.text(-0.7, 0.07, r"beam  $e$,  $E_e$ = %d MeV" % E_e, ha='center', fontsize=10.5)
    arrow(ax, 0, 0, E_ep*s*np.cos(the), E_ep*s*np.sin(the), C['data'])                     # e'
    ax.text(E_ep*s*np.cos(the)+0.05, E_ep*s*np.sin(the)+0.02, r"$e'$:  $E_{e'}$ = %d MeV" % E_ep, fontsize=10.5)
    arrow(ax, 0, 0, q*s*np.cos(thq_), -q*s*np.sin(thq_), C['rfg'], ls='--')               # q
    ax.text(q*s*np.cos(thq_)+0.05, -q*s*np.sin(thq_), r"$\mathbf{q}$:  |q| = %d MeV, ω = %d MeV" % (q, om), color=C['rfg'], fontsize=10.5, va='center', ha='left')
    arrow(ax, 0, 0, pp*s*np.cos(thp), -pp*s*np.sin(thp), C['L'])                            # p'
    ax.text(pp*s*np.cos(thp)-0.04, -pp*s*np.sin(thp)-0.06, r"$p'$:  $T_{p'}$ = %d MeV" % T_p, color=C['L'], fontsize=10.5, ha='right', va='top')
    # angle arcs
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Arc((0, 0), 0.5, 0.5, theta1=0, theta2=th_e, color=C['data'], lw=1)); ax.text(0.3*np.cos(the/2), 0.3*np.sin(the/2), r'$\theta_e$', fontsize=11)
    ax.add_patch(mpatches.Arc((0, 0), 0.7, 0.7, theta1=-th_p, theta2=0, color=C['L'], lw=1)); ax.text(0.42*np.cos(thp/2), -0.42*np.sin(thp/2), r'$\theta_p$', color=C['L'], fontsize=11)
    ax.plot(0, 0, 'o', ms=14, color='#bbb', mec='#777'); ax.text(-0.04, -0.12, r'$^{12}$C', fontsize=10, ha='right')
    ax.set(xlim=(-1.3, 1.35), ylim=(-1.0, 0.85), title='the (e,e′p) set-up in the scattering plane (momenta to scale)'); ax.set_aspect('equal'); ax.axis('off')
    # --- right: the missing-momentum triangle
    ax = ax2; s2 = 1/650.
    qx, qy = q*s2*np.cos(thq_), -q*s2*np.sin(thq_); px, py = pp*s2*np.cos(thp), -pp*s2*np.sin(thp)
    arrow(ax, 0, 0, qx, qy, C['rfg'], ls='--'); arrow(ax, 0, 0, px, py, C['L']); arrow(ax, px, py, qx, qy, C['accent'], lw=2.6)
    ax.text(qx/2+0.06, qy/2+0.06, r'$\mathbf{q}$', color=C['rfg'], fontsize=13, ha='left', va='bottom'); ax.text(px/2-0.06, py/2-0.02, r"$\mathbf{p}'$", color=C['L'], fontsize=13, ha='right', va='top')
    ax.text((px+qx)/2+0.02, (py+qy)/2-0.09, r"$\mathbf{p}_m = \mathbf{q} - \mathbf{p}'$", color=C['accent'], fontsize=13, ha='left', va='top')
    ax.plot(0, 0, 'o', ms=6, color='#777')
    ax.text(0.98, 0.97, "\n".join([r"$|\mathbf{p}_m|$ = %.0f MeV/c" % pm, r"$E_m = \omega - T_{p'} - T_{A-1}$ = %.1f MeV" % Em, '',
                                   r"PWIA:  $\mathbf{p}_m = -\mathbf{p}_{\rm initial}$,", r"           $E_m$ = removal energy"]),
            transform=ax.transAxes, fontsize=11, va='top', ha='right', bbox=dict(boxstyle='round', fc='#fafafa', ec='#ccc'))
    ax.set(xlim=(-0.1, 1.35), ylim=(-0.95, 0.5), title='the missing-momentum triangle'); ax.set_aspect('equal'); ax.axis('off')
    plt.show()


w_Ee = W.FloatSlider(497, min=300, max=3000, step=1, description='$E_e$ [MeV]', readout_format='.0f', layout=W.Layout(width='380px'))
w_Eep = W.FloatSlider(394, min=100, max=2900, step=1, description="$E_{e'}$ [MeV]", readout_format='.0f', layout=W.Layout(width='380px'))
w_the = W.FloatSlider(52.9, min=5, max=120, step=0.1, description='$\\theta_e$ [deg]', readout_format='.1f', layout=W.Layout(width='380px'))
w_Tp = W.FloatSlider(87, min=20, max=1500, step=1, description='$T_p$ [MeV]', readout_format='.0f', layout=W.Layout(width='380px'))
w_thp = W.FloatSlider(48.5, min=0, max=120, step=0.1, description='$\\theta_p$ [deg]', readout_format='.1f', layout=W.Layout(width='380px'))

pgrid = np.linspace(1, 350, 120)
lo, hi = lfg_removal_energy_band(pgrid, rho_c12_proton, S=16.0)
def panel_A(E_e, E_ep, th_e, T_p, th_p):
    pm, Em, q, thq = missing(E_e, E_ep, th_e, T_p, th_p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.25, width_ratios=[1.1, 1]))
    # (p_m, E_m) landscape
    ax1.fill_between(pgrid, lo, hi, color=C['rfg'], alpha=0.25, lw=0, label='local Fermi gas: $E_{rem} = S + T_F(r) - T(p)$, all r')
    ax1.axhspan(15, 19, color=C['L'], alpha=0.35, lw=0); ax1.text(300, 17, '1p shell (data)', color=C['L'], va='center', ha='right', fontsize=11)
    ax1.axhspan(30, 50, color=C['accent'], alpha=0.25, lw=0); ax1.text(300, 40, '1s shell (data, broad)', color=C['accent'], va='center', ha='right', fontsize=11)
    ax1.plot([pm], [Em], 'o', ms=11, color=C['data'], mfc='white', mew=2.5, label=f'this kinematics: $p_m$ = {pm:.0f} MeV/c, $E_m$ = {Em:.1f} MeV')
    ax1.set(xlabel=r'missing momentum  $p_m$  [MeV/c]', ylabel=r'missing energy  $E_m$  [MeV]', xlim=(0, 350), ylim=(0, 80), title='where the strength is: model vs data')
    ax1.legend(loc='upper left', fontsize=9.5); polish(ax1)
    # missing-energy spectra
    d = dutta['Em']; ax2.bar(d[:, 0], d[:, 1], width=4.6, color=C['data'], alpha=0.8, label='Dutta 2003, $Q^2$ = 1.28 GeV$^2$ (5 MeV bins)')
    m = mougey['Em_0_36']; sc = d[:, 1].max()/m[:, 1].max()
    ax2.plot(m[:, 0], m[:, 1]*sc, '-', color=C['L'], lw=1.4, label='Mougey 1976, $p_m$ < 36 MeV/c (scaled to the peak)')
    ax2.set(xlabel=r'$E_m$  [MeV]', ylabel=r'$\int S\, d^3p_m$  [MeV$^{-1}$]', xlim=(0, 80), title=r'$^{12}$C(e,e′p): the missing-energy spectrum'); ax2.legend(fontsize=9.5); polish(ax2)
    fig.text(0.5, -0.04, f'|q| = {q:.0f} MeV,  θ_q = {thq:.1f}°  —  the local Fermi gas has no structure in E_m; the data have two shells', ha='center', fontsize=11, color='#555')
    plt.show()

def show_A():
    """Panel A: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_A, W.VBox([W.HBox([w_Ee, w_Eep, w_the]), W.HBox([w_Tp, w_thp])]), dict(E_e=w_Ee, E_ep=w_Eep, th_e=w_the, T_p=w_Tp, th_p=w_thp),
               static_calls=[dict(E_e=497, E_ep=394, th_e=52.9, T_p=87, th_p=48.5)])

w_V0p = W.FloatSlider(57, min=35, max=80, step=0.5, description='$V_0$(1p) [MeV]', readout_format='.1f', layout=W.Layout(width='340px'))
w_V0s = W.FloatSlider(66, min=35, max=90, step=0.5, description='$V_0$(1s) [MeV]', readout_format='.1f', layout=W.Layout(width='340px'))
w_r0 = W.FloatSlider(1.32, min=1.0, max=1.7, step=0.01, description='$r_0$ [fm]', readout_format='.2f', layout=W.Layout(width='340px'))
w_a = W.FloatSlider(0.55, min=0.3, max=0.9, step=0.01, description='a [fm]', readout_format='.2f', layout=W.Layout(width='340px'))
w_Vso = W.FloatSlider(9, min=0, max=20, step=0.5, description='$V_{so}$ [MeV]', readout_format='.1f', layout=W.Layout(width='340px'))
SHELLCOL = {'1s1/2': C['accent'], '1p3/2': C['L']}
pB = np.linspace(0, 500, 251)

def panel_B(V0p, V0s, r0, a, Vso):
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(wspace=0.3))
    r = orbs['1s1/2']['r']
    ax = axs[0]; ax.plot(r, ws_potential(r, V0p, r0, a, Vso, l=1, j=1.5), color=C['data'], lw=2, label='well (1p, with Coulomb + spin–orbit)')
    for name, o in orbs.items():
        ax.axhline(o['E'], color=SHELLCOL[name], lw=1.8, xmax=0.55); ax.text(8.5, o['E']+1.2, f'{name}:  {o["E"]:.1f} MeV', color=SHELLCOL[name], fontsize=11)
    ax.axhline(0, color='#999', lw=0.8); ax.set(xlabel='r [fm]', ylabel='energy [MeV]', ylim=(-75, 15), xlim=(0, 12), title='the well and its occupied levels'); ax.legend(fontsize=9, loc='lower right'); polish(ax)
    ax = axs[1]
    for name, o in orbs.items(): ax.plot(o['r'], o['u'], color=SHELLCOL[name], lw=2.2, label=f'{name}')
    ax.axhline(0, color='#999', lw=0.8); ax.set(xlabel='r [fm]', ylabel='radial function u(r)', xlim=(0, 8), title='the orbitals in r'); ax.legend(); polish(ax)
    ax = axs[2]
    for name, o in orbs.items():
        n = momentum_distribution(o['r'], o['u'], o['l'], pB); ax.plot(pB, 4*np.pi*pB**2*n*1e3, color=SHELLCOL[name], lw=2.2, label=f'{name}')
    nFG = np.where(pB <= 221, 1/(4*np.pi/3*221**3), 0.0); ax.plot(pB, 4*np.pi*pB**2*nFG*1e3, color=C['rfg'], lw=1.6, ls='--', label='Fermi gas, $k_F$ = 221 MeV')
    ax.set(xlabel='p [MeV]', ylabel=r'$4\pi p^2 n(p)$  [$10^{-3}$ MeV$^{-1}$], per nucleon', xlim=(0, 500), title='the orbitals in p'); ax.legend(fontsize=9.5); polish(ax)
    fig.text(0.5, -0.04, 'Mougey 1976 mean removal energies: 1p 17.5 ± 0.4 MeV, 1s 38.1 ± 1.0 MeV', ha='center', fontsize=11, color='#555')
    plt.show()

def show_B():
    """Panel B: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_B, W.VBox([W.HBox([w_V0p, w_V0s, w_Vso]), W.HBox([w_r0, w_a])]), dict(V0p=w_V0p, V0s=w_V0s, r0=w_r0, a=w_a, Vso=w_Vso),
               static_calls=[dict(V0p=57, V0s=66, r0=1.32, a=0.55, Vso=9), dict(V0p=45, V0s=55, r0=1.32, a=0.55, Vso=9)])

w_Sp = W.FloatSlider(1.0, min=0.2, max=1.0, step=0.02, description='$S$(1p)', readout_format='.2f', layout=W.Layout(width='340px'))
w_Ss = W.FloatSlider(1.0, min=0.2, max=1.0, step=0.02, description='$S$(1s)', readout_format='.2f', layout=W.Layout(width='340px'))
w_Gs = W.FloatSlider(20, min=5, max=35, step=1, description='$\\Gamma$(1s) [MeV]', readout_format='.0f', layout=W.Layout(width='340px'))
w_T = W.FloatSlider(0.65, min=0.4, max=1.0, step=0.05, description='transparency T', readout_format='.2f', layout=W.Layout(width='340px'))
w_model = W.ToggleButtons(options=['orbitals (IPM)', 'local Fermi gas', 'Fermi gas'], description='model')
def _dutta_protons(key):
    d = dutta[key]; m = d[:, 0] > 0; p = np.r_[0, d[m, 0]]; v = np.r_[d[m, 1][0], d[m, 1]]; return trapz(4*np.pi*p**2*v, p)
DUTTA_TOTAL = _dutta_protons('n_p') + _dutta_protons('n_s')   # protons seen in Fig. 10 (distorted): ~2.45
pC = np.linspace(1, 400, 200); EC = np.linspace(0, 80, 321)

def panel_C(Sp, Ss, Gs, T, model, V0p, V0s, r0, a, Vso):
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    occ = {'1s1/2': Ss, '1p3/2': Sp}
    if model == 'orbitals (IPM)': P = ipm_spectral_function(orbs, pC, EC, widths={'1s1/2': Gs, '1p3/2': 4.0}, occ=occ)
    elif model == 'local Fermi gas': P = lfg_spectral_function(pC, EC, rho_c12_proton, S=16.0)
    else:
        P = np.zeros((len(pC), len(EC))); iE = np.argmin(np.abs(EC-25)); P[pC < 221, iE] = 6*3/(4*np.pi*221**3)/(EC[1]-EC[0])
    dp = pC[1]-pC[0]; dE = EC[1]-EC[0]
    fig = plt.figure(figsize=(14, 8.6)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2]); Z = 4*np.pi*pC[:, None]**2*P
    ax.pcolormesh(pC, EC, Z.T, cmap='Blues', shading='auto', vmax=np.percentile(Z, 99.5)); ax.invert_yaxis()
    ax.set(xlabel='p [MeV]', ylabel='E (removal) [MeV]', title=f'{model}: the spectral function  $4\\pi p^2 P(p,E)$'); ax.grid(False)
    # projection on E vs Dutta / Mougey
    ax = fig.add_subplot(gs[0, 2]); projE = (4*np.pi*pC[:, None]**2*P).sum(0)*dp       # per MeV, total protons
    d = dutta['Em']; bars = d[:, 1]/(d[:, 1].sum()*5)*DUTTA_TOTAL                        # scaled to the protons seen in Fig. 10
    ax.bar(d[:, 0], bars, width=4.6, color=C['data'], alpha=0.8, label=f'Dutta 2003 (scaled to {DUTTA_TOTAL:.2f} protons)')
    ax.plot(EC, projE*T, color=C['rfg'] if model != 'orbitals (IPM)' else C['L'], lw=2.2, label=f'{model} × T')
    ax.set(xlabel=r'$E_m$ [MeV]', ylabel=r'$\int P\,dp$  [MeV$^{-1}$]', xlim=(0, 80), title='projection on E'); ax.legend(fontsize=9, loc='center right'); polish(ax)
    # momentum distributions per shell vs Dutta fig 10 (shape-normalised at S=1 to the proton number of each shell)
    for k, (name, key, nshell, col) in enumerate([('1p3/2', 'n_p', 4, C['L']), ('1s1/2', 'n_s', 2, C['accent'])]):
        ax = fig.add_subplot(gs[1, k])
        dd = dutta[key]; pm = np.abs(dd[:, 0]); v = dd[:, 1]
        if model == 'orbitals (IPM)':
            n = momentum_distribution(orbs[name]['r'], orbs[name]['u'], orbs[name]['l'], pC)*nshell
            ax.plot(pC, n*occ[name]*T*1e8, color=col, lw=2.2, label=f'{name}: S = {occ[name]:.2f}, T = {T:.2f}')
        else:
            n = (4*np.pi*pC[:, None]**2*P)[:, (EC > (10 if k == 0 else 30)) & (EC < (25 if k == 0 else 50))].sum(1)*dE/(4*np.pi*pC**2)
            ax.plot(pC, n*T*1e8, color=C['rfg'], lw=2.2, label=f'{model} × T')
        ax.plot(pm, v*1e8, 'o', ms=4.5, color=C['data'], label=f'Dutta 2003, {"10 < E_m < 25" if k == 0 else "30 < E_m < 50"} MeV (absolute)')
        mm = mougey['n_1p' if k == 0 else 'n_1s']; ax.plot(mm[:, 0], mm[:, 1]/mm[:, 1].max()*(v*1e8).max(), 's', ms=4, mfc='none', color='#777', label='Mougey 1976 (shape)')
        ax.set(xlabel=r'$p_m$ [MeV/c]', ylabel=r'$n(p)$  [$10^{-8}$ MeV$^{-3}$]', xlim=(0, 320), title=f'{name} momentum distribution'); ax.legend(fontsize=8.5); polish(ax)
    # strength bookkeeping
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    tot = (Z.sum()*dp*dE)
    txt = (f'protons in the orbitals: {tot:.2f} of 6\n\n' + (f'1p: {4*Sp:.2f} of 4   (Mougey: 2.5)\n1s: {2*Ss:.2f} of 2   (Mougey: 1.0)\n\nDutta sees {_dutta_protons("n_p"):.2f} and {_dutta_protons("n_s"):.2f}\n= S x T x (4, 2)  with T = {T:.2f}:\n  S(1p) = {_dutta_protons("n_p")/4/T:.2f},  S(1s) = {_dutta_protons("n_s")/2/T:.2f}\n\nmissing: {6-4*Sp-2*Ss:.2f} protons\n→ where are they?  (Lecture 3)' if model == 'orbitals (IPM)' else 'no shells, no spectroscopic factors:\nthe strength is a band'))
    ax.text(0.02, 0.95, txt, va='top', fontsize=13, family='monospace', bbox=dict(boxstyle='round', fc='#F3F4F7', ec='none'))
    plt.show()

WELL = dict(V0p=w_V0p, V0s=w_V0s, r0=w_r0, a=w_a, Vso=w_Vso)      # Panel B's sliders: C and D listen to them too
WELL0 = dict(V0p=57, V0s=66, r0=1.32, a=0.55, Vso=9)
def show_C():
    """Panel C: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_C, W.VBox([w_model, W.HBox([w_Sp, w_Ss]), W.HBox([w_Gs, w_T])]), dict(Sp=w_Sp, Ss=w_Ss, Gs=w_Gs, T=w_T, model=w_model, **WELL),
               static_calls=[dict(Sp=1.0, Ss=1.0, Gs=20, T=0.65, model='orbitals (IPM)', **WELL0), dict(Sp=0.68, Ss=0.5, Gs=20, T=0.65, model='orbitals (IPM)', **WELL0), dict(Sp=1.0, Ss=1.0, Gs=20, T=0.65, model='local Fermi gas', **WELL0)])

def xsec_from_sf(E, th, omega, P, pgrid, Egrid, Z=6, N=6):
    """d2sigma/dOmega domega [nb/sr/GeV] from a per-nucleon spectral function P (normalised to 1)."""
    om = np.atleast_1d(omega)*1e3
    Q2, q, vL, vT = lepton_factors(E, th, np.atleast_1d(omega))
    RL, RT = responses_from_sf(om, q*1e3, pgrid, Egrid, P, Z, N)     # MeV^-1
    sM = mott(E, th)*HBARC2_NB
    return sM*(vL*RL + vT*RT)*1e3, RL*1e3, RT*1e3

names = list(datasets)
w_setD = W.Dropdown(options=[n for n in names if n.startswith(('E = 0.961', 'E = 0.560 GeV,  θ = 60', 'E = 1.299', 'E = 2.000'))], value=next(n for n in names if n.startswith('E = 0.961')), description='dataset', layout=W.Layout(width='480px'))
pD = np.linspace(1, 500, 170); ED = np.linspace(0, 90, 91)

def panel_D(Sp, Ss, name, Gs, V0p, V0s, r0, a, Vso):
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    # occupation = the spectroscopic factors of Panel C; the overall S is the fraction of the 6 protons left in the orbitals
    P = ipm_spectral_function(orbs, pD, ED, widths={'1s1/2': Gs, '1p3/2': 4.0}, occ={'1s1/2': Ss, '1p3/2': Sp})/6.0
    S = (2*Ss + 4*Sp)/6.0
    d = datasets[name]; om = np.linspace(max(d['om'].min()*0.6, 0.01), min(d['om'].max()*1.05, d['E']*0.97), 160)
    sig_sf, RLs, RTs = xsec_from_sf(d['E'], d['th'], om, P, pD, ED)
    sig_fg, _, _ = rfg_xsec(d['E'], d['th'], om, 0.221, 0.025)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw=dict(wspace=0.25))
    ax1.errorbar(d['om']*1e3, d['sig'], d['err'], fmt='o', ms=3.5, color=C['data'], ecolor=C['data_light'], elinewidth=1, capsize=0, zorder=3, label=name)
    ax1.plot(om*1e3, sig_fg, color=C['rfg'], lw=2, ls='--', label='Fermi gas ($k_F$ = 221, $\\varepsilon_B$ = 25 MeV)')
    ax1.plot(om*1e3, sig_sf, color=C['L'], lw=2.4, label=f'orbitals: $S$(1p) = {Sp:.2f}, $S$(1s) = {Ss:.2f}  →  {S*6:.1f} of 6 protons')
    ax1.set(xlabel=r'$\omega$ [MeV]', ylabel=r'$d^2\sigma/d\Omega\,d\omega$  [nb sr$^{-1}$ GeV$^{-1}$]', ylim=(0, max(d['sig'].max(), sig_fg.max())*1.2), title='inclusive (e,e′): same current, different nucleus')
    ax1.legend(fontsize=9.5); polish(ax1)
    # scaling function of the IPM vs data (only this dataset) and the parabola
    psi, GL, GT, vL, vT, q = scaling_pieces(d['E'], d['th'], om, 0.228, 0.020)
    f_sf = sig_sf/HBARC2_NB/(mott(d['E'], d['th'])*(vL*GL + vT*GT))
    psid, GLd, GTd, vLd, vTd, qd = scaling_pieces(d['E'], d['th'], d['om'], 0.228, 0.020)
    f_d = d['sig']/HBARC2_NB/(mott(d['E'], d['th'])*(vLd*GLd + vTd*GTd))
    ax2.plot(psid, f_d, 'o', ms=4, color=C['data'], label='data'); ax2.plot(psi, f_sf, color=C['L'], lw=2.4, label='orbitals × S')
    ps = np.linspace(-2, 2, 300); ax2.plot(ps, f_RFG(ps), color=C['rfg'], lw=1.8, ls='--', label='Fermi gas parabola')
    ax2.set(xlabel=r"$\psi'$", ylabel=r"$f(\psi')$", xlim=(-2, 2), ylim=(0, 0.9), title='the scaling function: shape'); ax2.legend(fontsize=9.5); polish(ax2)
    plt.show()

def show_D():
    """Panel D: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_D, W.VBox([w_setD, W.HTML('<i>S(1p), S(1s), Γ(1s) and the well are taken live from Panels B and C.</i>')]), dict(Sp=w_Sp, Ss=w_Ss, name=w_setD, Gs=w_Gs, **WELL),
               static_calls=[dict(Sp=1.0, Ss=1.0, name=next(n for n in names if n.startswith('E = 0.961')), Gs=20, **WELL0), dict(Sp=0.68, Ss=0.5, name=next(n for n in names if n.startswith('E = 0.961')), Gs=20, **WELL0)])

# ---- Rosenbluth-separated world data (R_L, R_T in MeV^-1 at fixed q)
def _ld(fn, scale=1.0, cols=(0, 1, 2), mask_zero=False):
    d = np.loadtxt(os.path.join(DATA, 'LT', fn)); m = d[:, 1] > 0 if mask_zero else np.ones(len(d), bool)
    err = d[m, cols[2]] if len(cols) > 2 and d.shape[1] > cols[2] else 0.0*d[m, 1]
    return d[m, cols[0]], d[m, cols[1]]*scale, err*scale
LT = {}
for q in (300, 380, 570):
    LT[f'12C   q = {q} MeV   (Jourdan, world data)'] = dict(A=12, Z=6, N=6, q=q, L=_ld(f'c12_q{q}_L.dat'), T=_ld(f'c12_q{q}_T.dat'))
for q in (300, 380, 570):
    dL = np.loadtxt(os.path.join(DATA, 'LT', f'JRD_CA40_q{q}_RL_wrld.dat')); dT = np.loadtxt(os.path.join(DATA, 'LT', f'JRD_CA40_q{q}_RT_wrld.dat'))
    LT[f'40Ca   q = {q} MeV   (Jourdan, world data)'] = dict(A=40, Z=20, N=20, q=q, L=(dL[:, 0], dL[:, 1]*1e-2, (dL[:, 3]-dL[:, 2])*0.5e-2),
                                                              T=(dT[:, 0], dT[:, 1]*1e-2, 0.08*dT[:, 1]*1e-2))
for q in (300, 350, 400, 450, 500, 550):
    LT[f'40Ca   q = {q} MeV   (Williamson et al., Bates)'] = dict(A=40, Z=20, N=20, q=q, L=_ld(f'ca40_q{q}.lon.graph', cols=(0, 1, 3), mask_zero=True), T=_ld(f'ca40_q{q}.trn.graph', cols=(0, 1, 3), mask_zero=True))
for q in (410, 550):
    LT[f'40Ca   q = {q} MeV   (Meziani et al., Saclay)'] = dict(A=40, Z=20, N=20, q=q, L=_ld(f'Meziani_Ca40_q{q}_RL.dat', cols=(0, 1)), T=_ld(f'Meziani_Ca40_q{q}_RT.dat', cols=(0, 1)))

def ca40_orbitals(V0=55.0, r0=1.25, a=0.65, Vso=9.0):
    """The six occupied proton shells of 40Ca in one Woods-Saxon well."""
    out = {}
    for name, l, j in (('1s1/2', 0, 0.5), ('1p3/2', 1, 1.5), ('1p1/2', 1, 0.5), ('1d5/2', 2, 2.5), ('2s1/2', 0, 0.5), ('1d3/2', 2, 1.5)):
        r, E, us = solve_ws(l, j, V0=V0, r0=r0, a=a, Vso=Vso, A=40, Z=20); n = 1 if name[0] == '2' else 0
        out[name] = dict(l=l, j=j, E=E[n], u=us[n], r=r, degeneracy=int(2*j+1))
    return out
CA40 = ca40_orbitals(); CA40_W = {'1s1/2': 25, '1p3/2': 15, '1p1/2': 12, '1d5/2': 6, '2s1/2': 3, '1d3/2': 2}
MONIZ = {12: (0.221, 0.025), 40: (0.235, 0.028)}      # k_F, eps_B [GeV] from the Moniz fits
pE = np.linspace(1, 500, 170); EE = np.linspace(0, 100, 101)

def _peak_ratio(om_d, R_d, om_m, R_m, R_ref):
    """integral of data / integral of model over the quasielastic peak region, defined as the omega window where the orbital response R_ref exceeds 20 % of its maximum"""
    win = om_m[R_ref > 0.2*R_ref.max()]; sel = (om_d >= win.min()) & (om_d <= win.max())
    if sel.sum() < 3: return np.nan
    m = np.interp(om_d[sel], om_m, R_m); return trapz(R_d[sel], om_d[sel])/max(trapz(m, om_d[sel]), 1e-12)

w_LT = W.Dropdown(options=list(LT), value=list(LT)[1], description='data set', layout=W.Layout(width='520px'))
w_rfgE = W.Checkbox(True, description='show Fermi gas')

def panel_E(name, show_rfg, Sp, Ss, Gs, V0p, V0s, r0, a, Vso):
    d = LT[name]; A, Z, N, q = d['A'], d['Z'], d['N'], d['q']
    if A == 12:
        orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
        P = ipm_spectral_function(orbs, pE, EE, widths={'1s1/2': Gs, '1p3/2': 4.0}, occ={'1s1/2': Ss, '1p3/2': Sp})/6.0; Slab = f'$S$(1p) = {Sp:.2f}, $S$(1s) = {Ss:.2f}'
    else:
        Sbar = (2*Ss + 4*Sp)/6.0
        P = ipm_spectral_function(CA40, pE, EE, widths=CA40_W, occ={k: Sbar for k in CA40_W})/20.0; Slab = f'$S$ = {Sbar:.2f} (all shells)'
    om = np.linspace(5, max(d['L'][0].max(), d['T'][0].max())*1.15, 110)
    RLs, RTs = responses_from_sf(om, float(q), pE, EE, P, Z, N)
    kF, eB = MONIZ[A]; RLf, RTf = rfg_responses(om*1e-3, q*1e-3, kF, eB, Z, N); RLf, RTf = RLf*1e-3, RTf*1e-3
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw=dict(wspace=0.25))
    for ax, c, Rs, Rf in ((axs[0], 'L', RLs, RLf), (axs[1], 'T', RTs, RTf)):
        o, R, e = d[c]
        ax.errorbar(o, R, e, fmt='o', ms=4, color=C['data'], ecolor=C['data_light'], elinewidth=1, capsize=0, zorder=3, label='data')
        if show_rfg: ax.plot(om, Rf, color=C['rfg'], lw=2, ls='--', label=f'Fermi gas  ($k_F$ = {kF*1e3:.0f}, $\\varepsilon_B$ = {eB*1e3:.0f} MeV), {Z} protons')
        ax.plot(om, Rs, color=C['L'] if c == 'L' else C['T'], lw=2.4, label=f'orbitals, {Slab}')
        rs = _peak_ratio(o, R, om, Rs, Rs); rf = _peak_ratio(o, R, om, Rf, Rs)
        ax.set(xlabel=r'$\omega$ [MeV]', ylabel=f'$R_{c}$  [MeV$^{{-1}}$]', ylim=(0, max(R.max(), Rf.max() if show_rfg else 0, Rs.max())*1.3), xlim=(0, om.max()),
               title=f'$R_{c}$ at $q$ = {q} MeV,  $^{{{A}}}${"C" if A == 12 else "Ca"}\n' + f'peak region:  ∫data / ∫orbitals = {rs:.2f}' + (f',   ∫data / ∫Fermi gas = {rf:.2f}' if show_rfg else ''))
        ax.legend(fontsize=9, loc='upper right'); polish(ax); ax.title.set_fontsize(11)
    plt.show()

def show_E():
    """Panel E: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_E, W.VBox([w_LT, w_rfgE, W.HTML('<i>S(1p), S(1s), Γ(1s) and the carbon well are taken live from Panels B and C.</i>')]),
               dict(name=w_LT, show_rfg=w_rfgE, Sp=w_Sp, Ss=w_Ss, Gs=w_Gs, **WELL),
               static_calls=[dict(name=list(LT)[1], show_rfg=True, Sp=0.68, Ss=0.5, Gs=20, **WELL0), dict(name=list(LT)[2], show_rfg=True, Sp=0.68, Ss=0.5, Gs=20, **WELL0),
                             dict(name=list(LT)[4], show_rfg=True, Sp=0.68, Ss=0.5, Gs=20, **WELL0)])
