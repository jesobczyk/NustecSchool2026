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
S_P_12C = 15.96   # MeV, proton separation energy of 12C (threshold of the missing-energy spectrum)

def ipm_spectral_function(orbs, p, E, widths={'1s1/2': 20.0, '1p3/2': 3.0}, occ={'1s1/2': 1.0, '1p3/2': 1.0}):
    """P(p, E) [MeV^-4] for Z protons: sum over shells of occ * (2j+1) * n_alpha(p) * line(E), the line being a Gaussian at
    E = -eps_alpha with full width at half maximum `widths` (a Lorentzian would put long tails far outside the peak)."""
    P = np.zeros((len(p), len(E)))
    for name, o in orbs.items():
        n = momentum_distribution(o['r'], o['u'], o['l'], p)
        sg = widths[name]/2.3548; Em = -o['E']
        L = np.exp(-0.5*((E-Em)/sg)**2)*(E >= S_P_12C)              # no strength below the proton separation energy
        dE = np.gradient(E) if len(E) > 1 else np.ones_like(E)
        L = L/max(np.sum(L*dE), 1e-300)
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
    """Missing momentum and energy for in-plane (e,e'p) (default recoil: 11B of 12C; pass MA1 for another nucleus): angles in degrees; theta_p measured from the beam on the other side of q."""
    om = E_e - E_ep; th = np.radians(th_e); q = np.sqrt(E_e**2 + E_ep**2 - 2*E_e*E_ep*np.cos(th))
    thq = np.arctan2(E_ep*np.sin(th), E_e - E_ep*np.cos(th))      # angle of q w.r.t. beam
    pp = np.sqrt((T_p+Mp)**2 - Mp**2); thp = np.radians(th_p)
    pmx = q*np.cos(thq) - pp*np.cos(thp); pmy = q*np.sin(thq) - pp*np.sin(thp)    # q and the proton on the same side of the beam (opposite to the electron)
    pm = np.hypot(pmx, pmy); Em = om - T_p - (np.sqrt(MA1**2 + pm**2) - MA1)
    return pm, Em, q, np.degrees(thq)

def eep_sketch(E_e=2222, E_ep=1716, th_e=20.0, T_p=455, th_p=54.0, MA1=36290.0, target=r'$^{40}$Ar'):
    """Schematic of the in-plane (e,e'p) geometry and the missing-momentum triangle, to scale in momentum."""
    pm, Em, q, thq = missing(E_e, E_ep, th_e, T_p, th_p, MA1=MA1); om = E_e - E_ep
    pp = np.sqrt((T_p+938.27)**2 - 938.27**2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(wspace=0.05, width_ratios=[1.25, 1]))
    def arrow(ax, x0, y0, x1, y1, color, lw=2.2, ls='-'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=16))
    # --- left: the set-up in the scattering plane (electron up, hadron side down)
    ax = ax1; s = 1.15/max(E_e, pp, q)
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
    ax.plot(0, 0, 'o', ms=14, color='#bbb', mec='#777'); ax.text(-0.04, -0.12, target, fontsize=10, ha='right')
    ax.set(xlim=(-1.3, 1.35), ylim=(-1.0, 0.85), title='the (e,e′p) set-up in the scattering plane (momenta to scale)'); ax.set_aspect('equal'); ax.axis('off')
    # --- right: the missing-momentum triangle
    ax = ax2; s2 = 1.0/max(q, pp)
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


WIDE = {'description_width': 'initial'}   # long plain-text labels (Colab does not render LaTeX in widget labels)

# ================================================================== Panel A: design a JLab spectral-function measurement
# ---- a model spectral function of carbon (protons): Woods-Saxon shells + a correlated tail
A_C = 12; Z_C = 6
M_A1 = 10255.1                    # MeV, mass of the residual 11B
E_THR2 = 27.0                     # MeV, threshold for removing a correlated pair (two-nucleon separation energy of 12C)
SIG_CM = 140.0                    # MeV/c, width (per component) of the pair centre-of-mass momentum distribution
F_SHELL = 0.80                    # fraction of the protons in the shells; the rest (20 %) sits in correlated pairs
KAPPA_TAIL = 105.0                # MeV/c, fall-off of the correlated momentum distribution  n_corr(k) ~ exp(-k/kappa)
pA = np.arange(2.5, 900, 5.0); EA = np.arange(0.5, 350, 1.0)            # grid of the model (MeV/c, MeV)

def n_corr(k):
    """correlated momentum distribution [MeV^-3], normalised to 4 pi int k^2 n dk = 1."""
    return np.exp(-k/KAPPA_TAIL)/(8*np.pi*KAPPA_TAIL**3)

def correlated_sf(p, E):
    """Ciofi degli Atti - Simula two-nucleon-correlation model: the struck proton (momentum k) was in a pair whose partner
    carries -k + K (K = pair c.m. momentum, Gaussian of width SIG_CM). The residual nucleus is left with the partner moving
    relative to the A-2 core:  E = E_thr + |K - a k|^2 / (2 mu),  a = (A-2)/(A-1),  mu = M (A-2)/(A-1).
    Returns P_corr(p, E) [MeV^-4] normalised to 1 (per proton)."""
    a = (A_C-2)/(A_C-1); mu = M_MEV*(A_C-2)/(A_C-1)
    Es = np.clip(E - E_THR2, 0, None)[None, :]; r = np.sqrt(2*mu*Es); m = a*p[:, None]
    s = SIG_CM
    fr = r/(m*s*np.sqrt(2*np.pi))*(np.exp(-(r-m)**2/(2*s*s)) - np.exp(-(r+m)**2/(2*s*s)))   # density of |K - a k|
    fE = np.where(Es > 0, fr*mu/np.maximum(r, 1e-9), 0.0)                                     # density in E
    fE /= np.maximum(fE.sum(1, keepdims=True)*(E[1]-E[0]), 1e-300)
    return n_corr(p)[:, None]*fE

def carbon_sf(V0p=57.0, V0s=66.0):
    """Model proton spectral function of 12C on (pA, EA) [MeV^-4], normalised to Z = 6:
    80 % in the 1p and 1s shells (Woods-Saxon orbitals, Gaussian lines of FWHM 6 and 20 MeV, nothing below the separation energy), 20 % correlated."""
    orbs = c12_orbitals(V0p, V0s)
    P = ipm_spectral_function(orbs, pA, EA, widths={'1s1/2': 20.0, '1p3/2': 6.0}, occ={'1s1/2': F_SHELL, '1p3/2': F_SHELL})
    P *= np.where(EA > 60, np.exp(-(EA - 60)/12), 1.0)[None, :]   # no Lorentzian tails deep into the continuum
    P *= Z_C*F_SHELL/(4*np.pi*np.sum(pA[:, None]**2*P)*5.0*1.0)
    P += Z_C*(1-F_SHELL)*correlated_sf(pA, EA)
    return P

P_CARBON = carbon_sf()
from scipy.interpolate import RegularGridInterpolator
_P_interp = RegularGridInterpolator((pA, EA), P_CARBON, bounds_error=False, fill_value=0.0)

# ---- the panel works with ARGON (the target of the JLab experiment E12-14-012): Woods-Saxon orbitals for the momentum
#      dependence, energies, widths and strengths of each shell from the fit of Jiang et al., PRD 107, 012005 (2023), Tables VII-VIII
A_AR = 40; Z_AR = 18
M_A1_AR = 36290.0                 # MeV, mass of the residual 39Cl
E_THR2_AR = 24.1                  # MeV, threshold of the correlated part (fitted by Jiang et al.)
AR_SHELLS = {                     # name: (l, j, radial node, peak energy [MeV], Gaussian sigma [MeV], protons in the peak)
    '1d3/2': (2, 1.5, 0, 12.53, 1.9, 0.89),
    '2s1/2': (0, 0.5, 1, 12.92, 3.8, 1.72),
    '1d5/2': (2, 2.5, 0, 18.23, 9.2, 3.52),
    '1p1/2': (1, 0.5, 0, 28.8, 12.1, 1.53),
    '1p3/2': (1, 1.5, 0, 33.0, 9.3, 3.07),
    '1s1/2': (0, 0.5, 0, 53.4, 28.3, 2.51)}
AR_CORR = 3.77                    # protons in the correlated part (Jiang et al.)
S_P_AR = 0.0                      # MeV: the Gaussian lines are not cut at the separation energy (12.53 MeV), as in the fit

def ar40_orbitals(V0=55.0, r0=1.25, a=0.65, Vso=9.0):
    """The six occupied proton shells of 40Ar in one Woods-Saxon well (used only for their momentum distributions)."""
    out = {}
    for name, (l, j, node, *_r) in AR_SHELLS.items():
        r, E, us = solve_ws(l, j, V0=V0, r0=r0, a=a, Vso=Vso, A=A_AR, Z=Z_AR)
        out[name] = dict(l=l, j=j, E=E[node], u=us[node], r=r)
    return out

def correlated_sf_A(p, E, A, Ethr):
    """the correlated_sf model above for a nucleus of mass number A and pair threshold Ethr; normalised to 1."""
    a = (A-2)/(A-1); mu = M_MEV*(A-2)/(A-1)
    Es = np.clip(E - Ethr, 0, None)[None, :]; r = np.sqrt(2*mu*Es); m = a*p[:, None]; s = SIG_CM
    fr = r/(m*s*np.sqrt(2*np.pi))*(np.exp(-(r-m)**2/(2*s*s)) - np.exp(-(r+m)**2/(2*s*s)))
    fE = np.where(Es > 0, fr*mu/np.maximum(r, 1e-9), 0.0)
    fE /= np.maximum(fE.sum(1, keepdims=True)*(E[1]-E[0]), 1e-300)
    return n_corr(p)[:, None]*fE

def argon_sf():
    """Model proton spectral function of 40Ar on (pA, EA) [MeV^-4]: Gaussian lines with the fitted energies, widths and
    strengths of Jiang et al. (13.2 protons in the shells), plus 3.8 protons in correlated pairs."""
    orbs = ar40_orbitals(); P = np.zeros((len(pA), len(EA))); dE = EA[1]-EA[0]; parts = {}
    for name, (l, j, node, Em, sg, Np) in AR_SHELLS.items():
        n = momentum_distribution(orbs[name]['r'], orbs[name]['u'], l, pA)
        G = np.exp(-0.5*((EA - Em)/sg)**2)*(EA >= S_P_AR); G /= G.sum()*dE
        n /= 4*np.pi*np.sum(pA**2*n)*(pA[1]-pA[0])
        parts[name] = Np*np.outer(n, G); P += parts[name]
    parts['correlated'] = AR_CORR*correlated_sf_A(pA, EA, A_AR, E_THR2_AR); P += parts['correlated']
    return P, parts

P_ARGON, AR_PARTS = argon_sf()
_P_interp = RegularGridInterpolator((pA, EA), P_ARGON, bounds_error=False, fill_value=0.0)    # the panel samples argon

# ---- the elementary cross section: de Forest's cc1 off-shell prescription (parallel kinematics: no LT, TT terms)
def proton_ff(Q2):                                   # Q2 in MeV^2 -> F1, F2 (F2(0) = 1.793)
    GE, GM = 1/(1+Q2/710000.0)**2, MU_P/(1+Q2/710000.0)**2
    tau = Q2/(4*M_MEV**2)
    return (GE + tau*GM)/(1+tau), (GM - GE)/(1+tau)

def sigma_cc1(E_e, E_ep, th_e, pp, pm, sin_gam=0.0):
    """de Forest sigma_cc1 [cm^2/sr] (all energies in MeV): electron (E_e -> E_ep, angle th_e [rad]), proton momentum pp,
    missing momentum pm, sin of the angle between p' and q."""
    om = E_e - E_ep; Q2 = 4*E_e*E_ep*np.sin(th_e/2)**2; q2 = Q2 + om**2
    Ebar = np.sqrt(M_MEV**2 + pm**2); Ep = np.sqrt(M_MEV**2 + pp**2); ombar = Ep - Ebar; Q2bar = q2 - ombar**2
    F1, F2 = proton_ff(Q2bar); t = Q2bar/(4*M_MEV**2)
    mott = (ALPHA*np.cos(th_e/2))**2/(4*E_e**2*np.sin(th_e/2)**4)*0.389379e-21        # cm^2/sr
    wL = 0.25*((Ebar+Ep)**2*(F1**2 + t*F2**2) - q2*(F1+F2)**2)
    wT = 0.5*Q2bar*(F1+F2)**2 + pp**2*sin_gam**2*(F1**2 + t*F2**2)
    return mott/(Ebar*Ep)*((Q2/q2)**2*wL + (Q2/(2*q2) + np.tan(th_e/2)**2)*wT)

# ---- the experiment: Hall A-like spectrometers, parallel kinematics
E_BEAM = 2222.0                   # MeV (the JLab spectral-function experiment E12-14-012)
DP = 0.045                        # momentum acceptance +-4.5 % (both spectrometers)
DTH, DPH = 0.030, 0.060           # rad: in-plane +-30 mrad, out-of-plane +-60 mrad  ->  about 7 msr
LUMI = 1.0e36                     # cm^-2 s^-1 per argon nucleus (about 20 uA on the 0.56 g/cm^2 gas target of E12-14-012); an assumption
_BOXCOL = ['#eb6834', '#d1495b', '#6a4c93', '#1f5fa8', '#2a9d8f']
BOXES = {f'p_m {lo}–{lo+100}': dict(p=(lo, lo+100), E=(0, 80), col=_BOXCOL[k]) for k, lo in enumerate(range(0, 500, 100))}   # 0 < E_m < 80 MeV

def setting_for(pm_target, Em_target, th_e_deg):
    """central kinematics with p' parallel to q and |p'| = |q| + p_m, that put (|p_m|, E_m) at the target.
    p_m > 0: the proton is faster than q (the choice of the JLab experiment); p_m < 0: slower.
    Returns dict or None if no electron energy does it."""
    th = np.radians(th_e_deg)
    def Em_of(Eep):
        om = E_BEAM - Eep; q = np.sqrt(E_BEAM**2 + Eep**2 - 2*E_BEAM*Eep*np.cos(th)); pp = q + pm_target
        if pp <= 50: return np.nan
        Tp = np.sqrt(pp**2 + M_MEV**2) - M_MEV; Trec = np.sqrt(M_A1_AR**2 + pm_target**2) - M_A1_AR
        return om - Tp - Trec - Em_target
    grid = np.linspace(200, E_BEAM - 5, 600); vals = np.array([Em_of(x) for x in grid])
    sgn = [i for i in np.where(np.diff(np.sign(vals)) != 0)[0] if np.isfinite(vals[i]) and np.isfinite(vals[i+1])]
    if len(sgn) == 0: return None
    from scipy.optimize import brentq
    Eep = brentq(Em_of, grid[sgn[-1]], grid[sgn[-1]+1])                 # the larger E' (smaller Q^2) solution
    om = E_BEAM - Eep; q = np.sqrt(E_BEAM**2 + Eep**2 - 2*E_BEAM*Eep*np.cos(th)); pp = q + pm_target
    thq = np.arctan2(Eep*np.sin(th), E_BEAM - Eep*np.cos(th)); Q2 = q*q - om*om
    return dict(Eep=Eep, om=om, q=q, pp=pp, Tp=np.sqrt(pp**2 + M_MEV**2) - M_MEV, thq=thq, Q2=Q2, x=Q2/(2*M_MEV*om), th=th)

_RNG = np.random.default_rng(1)
_U = _RNG.random((6, 120000))      # fixed random numbers: the same sample for every slider position (smooth response)

def simulate(s):
    """Sample the acceptance of both spectrometers around the central setting s; return p_m, E_m and the rate weight
    of each event [events/s]. No final-state interactions (plane-wave impulse approximation)."""
    u = _U
    Eep = s['Eep']*(1 + DP*(2*u[0]-1)); pp = s['pp']*(1 + DP*(2*u[1]-1))
    the = s['th'] + DTH*(2*u[2]-1); phe = DPH*(2*u[3]-1)
    thp = s['thq'] + DTH*(2*u[4]-1); php = DPH*(2*u[5]-1)
    # electron on one side of the beam (+x), proton on the other (-x); z = beam
    kp = Eep[:, None]*np.stack([np.sin(the)*np.cos(phe), np.sin(phe), np.cos(the)*np.cos(phe)], 1)
    k = np.array([0, 0, E_BEAM]); qv = k - kp
    pv = pp[:, None]*np.stack([-np.sin(thp)*np.cos(php), np.sin(php), np.cos(thp)*np.cos(php)], 1)
    pmv = pv - qv; pm = np.linalg.norm(pmv, axis=1)
    om = E_BEAM - Eep; Ep = np.sqrt(pp**2 + M_MEV**2)
    Em = om - (Ep - M_MEV) - (np.sqrt(M_A1_AR**2 + pm**2) - M_A1_AR)
    qn = np.linalg.norm(qv, axis=1); cosg = np.einsum('ij,ij->i', pv, qv)/(pp*qn); sing = np.sqrt(np.clip(1-cosg**2, 0, 1))
    sig = sigma_cc1(E_BEAM, Eep, np.arccos(kp[:, 2]/Eep), pp, pm, sing)
    S = _P_interp(np.stack([pm, Em], 1))
    d6 = pp*Ep*sig*S                                       # cm^2 / (MeV^2 sr^2)
    vol = (2*DP*s['Eep'])*(2*DP*s['pp'])*(pp/Ep)*(4*DTH*DPH)**2     # dE' dE_p dOmega_e dOmega_p of the sample
    w = d6*vol/len(pm)*LUMI
    return pm, Em, w

def panel_A(pm_target, Em_target, th_e):
    s = setting_for(pm_target, Em_target, th_e)
    fig = plt.figure(figsize=(14, 6.0)); gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1.15], hspace=0.45, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    Z = 4*np.pi*pA[:, None]**2*P_ARGON*1e3                                   # linear scale
    im = ax.pcolormesh(pA, EA, Z.T, cmap='Greys', shading='auto', vmin=0, vmax=Z[:, EA < 150].max())
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04); cb.set_label(r'$4\pi p^2 P(p,E)$  [$10^{-3}$ MeV$^{-2}$]')
    pr = np.linspace(0, 700, 100); ax.plot(pr, E_THR2_AR + (A_AR-2)/(A_AR-1)*pr**2/(2*M_MEV), color=C['warn'], lw=1, ls=':')
    ax.text(455, 118, 'correlated\npairs', color=C['warn'], fontsize=9, ha='right', va='center')
    ax.axhspan(140, 150, color='#f1d9d9', alpha=0.5, lw=0); ax.text(20, 148.5, 'pion production', color='#a33', fontsize=9, va='bottom')
    ax.text(520, 15, '1d, 2s', color='#333', fontsize=9.5, va='center'); ax.text(520, 31, '1p', color='#333', fontsize=9.5, va='center'); ax.text(520, 55, '1s (broad)', color='#333', fontsize=9.5, va='center')
    lines = []
    if s is None:
        ax.text(0.5, 0.5, 'no electron energy reaches this point\nat this angle', transform=ax.transAxes, ha='center', fontsize=13, color=C['warn'])
    else:
        pm, Em, w = simulate(s)
        H, xe, ye = np.histogram2d(pm, Em, bins=(150, 75), range=((0, 750), (0, 150)))
        ax.contourf(0.5*(xe[1:]+xe[:-1]), 0.5*(ye[1:]+ye[:-1]), (H.T > 0).astype(float), levels=[0.5, 1.5], colors=[C['rfg']], alpha=0.35)
        ax.plot(abs(pm_target), Em_target, '+', ms=14, mew=2, color=C['rfg'])
        for name, b in BOXES.items():
            inb = (pm > b['p'][0]) & (pm < b['p'][1]) & (Em > b['E'][0]) & (Em < b['E'][1]); rate = w[inb].sum()*3600
            hours = 100/rate if rate > 0 else np.inf
            lines.append((name, b['col'], rate, hours))
        total = w.sum()*3600
    for name, b in BOXES.items():
        ax.add_patch(plt.Rectangle((b['p'][0], b['E'][0]), b['p'][1]-b['p'][0], b['E'][1]-b['E'][0], fill=False, ec=b['col'], lw=2))
    ax.set(xlim=(0, 700), ylim=(150, 0), xlabel=r'missing momentum $p_m$ [MeV/c]', ylabel=r'missing energy $E_m$ [MeV]',
           title='argon: where the protons are; blue = what your setting sees'); ax.grid(False)
    # right top: the model in missing energy, shell by shell (Jiang et al. 2023 fit), in the p_m range of the experiment
    ax2 = fig.add_subplot(gs[0, 1]); mp = pA < 900; dp = pA[1]-pA[0]
    proj = lambda Pp: (4*np.pi*pA[mp, None]**2*Pp[mp]).sum(0)*dp
    tot = proj(P_ARGON); ax2.fill_between(EA, 0, tot, color=C['rfg'], alpha=0.18, lw=0, label='all shells + pairs')
    for name, col in (('1d3/2', C['L']), ('2s1/2', C['L']), ('1d5/2', C['L']), ('1p1/2', C['accent']), ('1p3/2', C['accent']), ('1s1/2', '#666')):
        ax2.plot(EA, proj(AR_PARTS[name]), color=col, lw=1.2, ls='-' if name in ('1d5/2', '1p3/2', '1s1/2') else '--')
    ax2.plot(EA, proj(AR_PARTS['correlated']), color=C['warn'], lw=1.2, ls=':', label='correlated pairs')
    ax2.set(xlim=(0, 100), ylim=(0, tot.max()*1.15), xlabel=r'$E_m$ [MeV]', ylabel=r'protons per MeV', title='argon: the shells in missing energy (all $p_m$)')
    ax2.text(14, tot.max()*1.02, '1d, 2s', color=C['L'], fontsize=9); ax2.text(32, tot.max()*0.45, '1p', color=C['accent'], fontsize=9); ax2.text(55, tot.max()*0.25, '1s', color='#666', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right'); polish(ax2)
    # right bottom: the numbers
    ax3 = fig.add_subplot(gs[1, 1]); ax3.axis('off')
    if s is not None:
        head = (f"electron:  E' = {s['Eep']/1e3:.3f} GeV at {th_e:.1f}°     Q² = {s['Q2']/1e6:.2f} GeV²\n"
                f"proton:    p' = {s['pp']/1e3:.3f} GeV/c at {np.degrees(s['thq']):.1f}°   (along q)\n"
                f"           T_p = {s['Tp']:.0f} MeV\n")
        ax3.text(0.0, 1.0, head, va='top', fontsize=10.5, family='monospace', transform=ax3.transAxes)
        flags = []
        if s['Tp'] < 300: flags.append('slow proton (T_p < 300 MeV): final-state interactions (neglected here) are strongest')
        if Em_target > 140: flags.append('above pion threshold: pion production fills this region')
        if flags: ax3.text(0.0, 0.68, '\n'.join('⚠ ' + f for f in flags), va='top', fontsize=9, color='#b5651d', transform=ax3.transAxes)
        for i, (name, col, rate, hours) in enumerate(lines):
            txt = f'{name:<14s} {rate:9.3g} counts/hour   ' + ('   —' if not np.isfinite(hours) else (f'{hours*3600:5.0f} s' if hours < 1/60 else f'{hours*60:5.1f} min' if hours < 1 else (f'{hours:5.1f} hours' if hours < 48 else f'{hours/24:5.0f} days')))
            ax3.text(0.0, 0.50 - 0.11*i, txt, va='top', fontsize=10.5, family='monospace', color=col, transform=ax3.transAxes, fontweight='bold')
        ax3.text(0.0, -0.12, 'time for 100 counts in the box (0 < E_m < 80 MeV)  (assumed luminosity 10³⁶ cm⁻² s⁻¹, no final-state interactions)', fontsize=8.5, color='#666', transform=ax3.transAxes)
    plt.show()

def show_A():
    """Panel A: design a spectral-function measurement on argon at 2.222 GeV (parallel kinematics)."""
    w_pm = W.FloatSlider(100, min=-600, max=600, step=10, description='aim at missing momentum [MeV/c]', style=WIDE, continuous_update=False, layout=W.Layout(width='520px'))
    w_Em = W.FloatSlider(16, min=5, max=200, step=1, description='aim at missing energy [MeV]', style=WIDE, continuous_update=False, layout=W.Layout(width='520px'))
    w_th = W.FloatSlider(21.5, min=12, max=40, step=0.5, description='electron angle [deg]', style=WIDE, continuous_update=False, layout=W.Layout(width='520px'))
    show_panel(panel_A, W.VBox([w_pm, w_Em, w_th]), dict(pm_target=w_pm, Em_target=w_Em, th_e=w_th),
               static_calls=[dict(pm_target=100, Em_target=16, th_e=21.5), dict(pm_target=250, Em_target=40, th_e=21.5),
                             dict(pm_target=450, Em_target=60, th_e=21.5), dict(pm_target=450, Em_target=60, th_e=15)])

# ================================================================== Panel B: the well, compared with data
R_CH_EXP = 2.470                  # fm, charge radius of 12C (elastic electron scattering)
V0_DEFAULT = 57.0                 # MeV: one depth for both shells (the Lecture 2 model)
w_V0p = W.FloatSlider(V0_DEFAULT, min=40, max=80, step=0.5, description='depth for the 1p shell [MeV]', style=WIDE, readout_format='.1f', continuous_update=False, layout=W.Layout(width='480px'))
w_V0s = W.FloatSlider(V0_DEFAULT, min=40, max=80, step=0.5, description='depth for the 1s shell [MeV]', style=WIDE, readout_format='.1f', continuous_update=False, layout=W.Layout(width='480px'))
SHELLCOL = {'1s1/2': C['accent'], '1p3/2': C['L']}
pB = np.linspace(0, 400, 201)
EXP_E = {'1p3/2': (17.5, 0.4, 4.0), '1s1/2': (38.1, 1.0, 20.0)}     # Mougey: mean removal energy, error, and peak width (FWHM, rough)

# ---- (e,e'p) momentum distributions with the final-state interactions removed (as in Lecture 2): divide by the published factors
T_DUTTA = 0.60                                    # Dutta et al. 2003, Table III: transparency of carbon at Q^2 = 1.28 GeV^2
ETA_MOUGEY = {'1p3/2': 0.66, '1s1/2': 0.52}       # Mougey et al. 1976, Table 5: distortion factors (DWIA / PWIA)
MOUGEY_WIN = {'1p3/2': '15 < E_m < 21.5', '1s1/2': '30 < E_m < 50'}
DUTTA_WIN_TXT = {'1p3/2': '10 < E_m < 25', '1s1/2': '30 < E_m < 50'}
SHELLS = [('1p3/2', 'n_p', 'n_1p'), ('1s1/2', 'n_s', 'n_1s')]

def eep_corrected(name):
    """(e,e'p) momentum distributions of shell `name` with the final-state interactions removed, in MeV^-3:
    Dutta (|p_m|, both sides of q) divided by T, Mougey (absolute, (GeV/c)^-3) divided by eta."""
    key_d, key_m = {s[0]: s[1:] for s in SHELLS}[name]
    dd = dutta[key_d]; mm = mougey[key_m]
    return (np.abs(dd[:, 0]), dd[:, 1]/T_DUTTA), (mm[:, 0], mm[:, 1]*1e-9/ETA_MOUGEY[name])

def _count(p, n):
    """number of protons 4 pi int p^2 n dp from data points (the first point is extended to p = 0)."""
    o = np.argsort(p); p, n = p[o], n[o]
    return trapz(4*np.pi*np.r_[0, p]**2*np.r_[n[0], n], np.r_[0, p])

def eep_protons(name):
    """protons seen in shell `name` after removing the final-state interactions: Dutta (one side of q), Mougey."""
    key_d = {s[0]: s[1] for s in SHELLS}[name]
    (pm, nm) = eep_corrected(name)[1]
    return _dutta_protons(key_d)/T_DUTTA, _count(pm, nm)

def model_protons(orbs, name, pmax=300.0):
    """protons of the fully occupied shell below pmax (the range covered by the data)."""
    o = orbs[name]; p = np.linspace(0.5, pmax, 600)
    return trapz(4*np.pi*p**2*o['degeneracy']*momentum_distribution(o['r'], o['u'], o['l'], p), p)

def charge_radius(orbs):
    """rms charge radius [fm] from the 4 + 2 proton orbitals: point-proton radius with a centre-of-mass correction
    (harmonic-oscillator estimate, -3b^2/2A with b = 1.64 fm), plus the proton (0.707 fm^2) and neutron (-0.116 fm^2) charge radii."""
    r = orbs['1s1/2']['r']; rho = 4*orbs['1p3/2']['u']**2 + 2*orbs['1s1/2']['u']**2
    r2 = trapz(rho*r**2, r)/trapz(rho, r) - 3*1.64**2/(2*A_C)
    return np.sqrt(r2 + 0.707 - 0.116)

def panel_B(V0p, V0s, r0=1.32, a=0.55, Vso=9.0):
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.1, 1.1]))
    ax = axs[0]
    for k, (name, o) in enumerate(orbs.items()):
        Ee, dE, G = EXP_E[name]; x0 = 0.15 + 0.45*k
        ax.add_patch(plt.Rectangle((x0, Ee - G/2), 0.3, G, color=SHELLCOL[name], alpha=0.18, lw=0))
        ax.plot([x0, x0+0.3], [Ee, Ee], color=SHELLCOL[name], lw=1, ls='--')
        ax.plot([x0, x0+0.3], [-o['E']]*2, color=SHELLCOL[name], lw=3)
        ax.text(x0+0.15, -o['E'] - 2.0, f"{name}\nmodel {-o['E']:.1f}\nMougey 1976 {Ee:.1f}", ha='center', va='bottom', fontsize=9, color=SHELLCOL[name])
    ax.set(xlim=(0, 1), ylim=(60, 0), xticks=[], ylabel='removal energy [MeV]', title='energies: model (line) vs (e,e′p) (band)'); polish(ax)
    for k, (name, key, _) in enumerate(SHELLS):
        ax = axs[1+k]; o = orbs[name]
        n = o['degeneracy']*momentum_distribution(o['r'], o['u'], o['l'], pB)
        (pd, nd), (pm, nm) = eep_corrected(name)
        ax.plot(pB, n*1e8, color=SHELLCOL[name], lw=2.4, label=f'model: full shell ({o["degeneracy"]} protons)')
        ax.plot(pd, nd*1e8, 'o', ms=4.5, color=C['data'], label=f'Dutta et al. 2003 ÷ {T_DUTTA:.2f}\n({DUTTA_WIN_TXT[name]} MeV)')
        ax.plot(pm, nm*1e8, 's', ms=4.5, mfc='white', color='#777', label=f'Mougey et al. 1976 ÷ {ETA_MOUGEY[name]:.2f}\n({MOUGEY_WIN[name]} MeV)')
        top = max(n.max(), nd.max(), nm.max())*1e8
        ax.set(xlabel=r'$p_m$ [MeV/c]', ylabel=r'$n(p)$  [$10^{-8}$ MeV$^{-3}$]', xlim=(0, 320), ylim=(0, 1.55*top), title=f'{name} momentum distribution')
        ax.legend(fontsize=8.5, loc='upper right'); polish(ax)
    plt.show()

def show_B():
    """Panel B: two depths (both start at the same value), everything else fixed; energies and momentum distributions vs data."""
    show_panel(panel_B, W.VBox([w_V0p, w_V0s]), dict(V0p=w_V0p, V0s=w_V0s),
               static_calls=[dict(V0p=V0_DEFAULT, V0s=V0_DEFAULT), dict(V0p=55, V0s=66)])

w_Sp = W.FloatSlider(1.0, min=0.2, max=1.2, step=0.01, description='spectroscopic factor 1p', style=WIDE, readout_format='.2f', layout=W.Layout(width='400px'))
w_Ss = W.FloatSlider(1.0, min=0.2, max=1.2, step=0.01, description='spectroscopic factor 1s', style=WIDE, readout_format='.2f', layout=W.Layout(width='400px'))
w_Gs = W.FloatSlider(20, min=5, max=35, step=1, description='width of the 1s peak [MeV]', style=WIDE, readout_format='.0f', layout=W.Layout(width='400px'))
w_model = W.ToggleButtons(options=['orbitals (IPM)', 'local Fermi gas', 'Fermi gas'], description='model')
GP_1P = 7.0      # MeV, FWHM of the p-shell peak: the 1p strength is spread over several states of 11B (0, 2.1, 5.0 MeV) plus resolution

# ---- the model "as Dutta measured it": the spectral function integrated over the same E_m window and averaged over the same p_m bins
DUTTA_WIN = {'n_p': (10.0, 25.0), 'n_s': (30.0, 50.0)}; DUTTA_BIN = 40.0      # MeV, MeV/c
_pW = np.arange(1.0, 461.0, 2.0); _EW = np.arange(0.125, 80.0, 0.25)
def window_n(orbs, key, occ, Gs=20.0, Gp=None, p_out=None):
    """n(p) of the IPM in Dutta's E_m window `key` [MeV^-3, protons], i.e. int P(p,E) dE over the window (so the tail of the
    broad 1s peak that falls into the p-shell window is included), then averaged over a p_m bin of 40 MeV/c (volume weighted)."""
    P = ipm_spectral_function(orbs, _pW, _EW, widths={'1s1/2': Gs, '1p3/2': GP_1P if Gp is None else Gp}, occ=occ)
    lo, hi = DUTTA_WIN[key]; m = (_EW > lo) & (_EW < hi)
    nw = P[:, m].sum(1)*(_EW[1]-_EW[0])
    p_out = _pW if p_out is None else np.atleast_1d(p_out)
    out = []
    for c in p_out:
        a, b = max(c - DUTTA_BIN/2, 0), c + DUTTA_BIN/2; mm = (_pW >= a) & (_pW <= b)
        out.append(np.sum(nw[mm]*_pW[mm]**2)/np.sum(_pW[mm]**2))
    return p_out, np.array(out)

def fit_to_dutta(orbs, Gs=20.0):
    """S x T of the 1p and the 1s shell from the NUMBER of protons Dutta sees in the two E_m windows (Fig. 10 integrated over p_m),
    including the part of each shell that falls into the other window: a 2 x 2 linear system."""
    A = np.zeros((2, 2)); y = np.zeros(2)
    for i, key in enumerate(('n_p', 'n_s')):
        y[i] = _dutta_protons(key)
        for j, occ in enumerate(({'1p3/2': 1.0, '1s1/2': 0.0}, {'1p3/2': 0.0, '1s1/2': 1.0})):
            p, n = window_n(orbs, key, occ, Gs=Gs)
            A[i, j] = trapz(4*np.pi*p**2*n, p)
    st = np.linalg.solve(A, y)
    return {'1p3/2': st[0], '1s1/2': st[1]}

def _dutta_protons(key):
    d = dutta[key]; m = d[:, 0] > 0; p = np.r_[0, d[m, 0]]; v = np.r_[d[m, 1][0], d[m, 1]]; return trapz(4*np.pi*p**2*v, p)
DUTTA_TOTAL = _dutta_protons('n_p') + _dutta_protons('n_s')   # protons seen in Fig. 10 (distorted): ~2.45
def _fig9_scale():
    # Fig. 9 is published with its own normalisation: scale it so that its two windows (10-25 and 30-50 MeV) hold, together,
    # the protons that Fig. 10 shows in the same windows
    d = dutta['Em']; E, v = d[:, 0], d[:, 1]
    win = v[(E > 10) & (E < 25)].sum()*5 + v[(E > 30) & (E < 50)].sum()*5
    return DUTTA_TOTAL/win
FIG9_SCALE = _fig9_scale()
pC = np.linspace(1, 400, 200); EC = np.linspace(0, 80, 321)

def panel_C(Sp, Ss, Gs, model, V0p, V0s, r0, a, Vso):
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    occ = {'1s1/2': Ss, '1p3/2': Sp}
    if model == 'orbitals (IPM)': P = ipm_spectral_function(orbs, pC, EC, widths={'1s1/2': Gs, '1p3/2': GP_1P}, occ=occ)
    elif model == 'local Fermi gas': P = lfg_spectral_function(pC, EC, rho_c12_proton, S=16.0)
    else:
        P = np.zeros((len(pC), len(EC))); iE = np.argmin(np.abs(EC-25)); P[pC < 221, iE] = 6*3/(4*np.pi*221**3)/(EC[1]-EC[0])
    dp = pC[1]-pC[0]; dE = EC[1]-EC[0]
    fig = plt.figure(figsize=(14, 8.6)); gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
    ax = fig.add_subplot(gs[0, :2]); Z = 4*np.pi*pC[:, None]**2*P
    ax.pcolormesh(pC, EC, Z.T, cmap='Blues', shading='auto', vmax=np.percentile(Z, 99.5)); ax.invert_yaxis()
    ax.set(xlabel='p [MeV]', ylabel='E (removal) [MeV]', title=f'{model}: the spectral function  $4\\pi p^2 P(p,E)$'); ax.grid(False)
    # projection on E vs Dutta (final-state interactions removed)
    ax = fig.add_subplot(gs[0, 2]); projE = (4*np.pi*pC[:, None]**2*P).sum(0)*dp       # per MeV, total protons
    d = dutta['Em']; bars = d[:, 1]*FIG9_SCALE/T_DUTTA                                 # normalised to Fig. 10 in the same windows, then ÷ T
    ax.bar(d[:, 0], bars, width=4.6, color=C['data'], alpha=0.6, label=f'Dutta 2003 ÷ {T_DUTTA:.2f} (5 MeV bins)')
    edges = np.arange(0, 85, 5.0); binned = np.array([projE[(EC >= a_) & (EC < b_)].mean() for a_, b_ in zip(edges[:-1], edges[1:])])
    lab = 'model × S, same bins' if model == 'orbitals (IPM)' else f'{model}, same bins'
    ax.step(edges, np.r_[binned, binned[-1]], where='post', color=C['rfg'] if model != 'orbitals (IPM)' else C['L'], lw=2.2, label=lab)
    ax.set(xlabel=r'$E_m$ [MeV]', ylabel=r'$\int 4\pi p^2 P\,dp$  [MeV$^{-1}$]', xlim=(0, 80), title='projection on E'); ax.legend(fontsize=9, loc='center right'); polish(ax)
    # 4 pi p^2 n(p) per shell vs (e,e'p) with the final-state interactions removed: the area is the number of protons
    counts = {}
    for k, (name, key, _) in enumerate(SHELLS):
        ax = fig.add_subplot(gs[1, k]); col = SHELLCOL[name]
        (pd, nd), (pm, nm) = eep_corrected(name); Nd, Nm = eep_protons(name)
        if model == 'orbitals (IPM)':
            o = orbs[name]; n = o['degeneracy']*momentum_distribution(o['r'], o['u'], o['l'], pC)
            Nmod = occ[name]*model_protons(orbs, name)
            ax.plot(pC, 4*np.pi*pC**2*n*1e3, color=col, lw=1.2, ls=':', label=f'full shell: {model_protons(orbs, name):.1f} protons')
            ax.plot(pC, 4*np.pi*pC**2*n*occ[name]*1e3, color=col, lw=2.6, label=f'model × S = {occ[name]:.2f}: {Nmod:.1f} protons')
        else:
            lo, hi = DUTTA_WIN[key]
            n = Z[:, (EC >= lo) & (EC <= hi)].sum(1)*dE/(4*np.pi*pC**2)
            m3 = pC < 300; Nmod = trapz(4*np.pi*pC[m3]**2*n[m3], pC[m3])
            ax.plot(pC, 4*np.pi*pC**2*n*1e3, color=C['rfg'], lw=2.4, label=f'{model}, {DUTTA_WIN_TXT[name]} MeV: {Nmod:.1f} protons')
        counts[name] = (Nmod, Nd, Nm)
        ax.plot(pd, 4*np.pi*pd**2*nd*1e3, 'o', ms=4.5, color=C['data'], label=f'Dutta 2003 ÷ {T_DUTTA:.2f}: {Nd:.1f} protons')
        ax.plot(pm, 4*np.pi*pm**2*nm*1e3, 's', ms=4.5, mfc='white', color='#777', label=f'Mougey 1976 ÷ {ETA_MOUGEY[name]:.2f}: {Nm:.1f} protons')
        ax.set_ylim(0, 1.6*ax.get_ylim()[1])
        ax.set(xlabel=r'$p_m$ [MeV/c]', ylabel=r'$4\pi p^2 n(p)$  [$10^{-3}$ MeV$^{-1}$]', xlim=(0, 320), title=f'{name}: the area is the number of protons')
        ax.legend(fontsize=8.5, loc='upper right'); polish(ax)
    # strength bookkeeping
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    tot = (Z.sum()*dp*dE)
    if model == 'orbitals (IPM)':
        cp, cs = counts['1p3/2'], counts['1s1/2']
        txt = (f'protons below 300 MeV/c\n'
               f'          model  Dutta  Mougey\n'
               f'  1p      {cp[0]:4.1f}   {cp[1]:4.1f}   {cp[2]:4.1f}\n'
               f'  1s      {cs[0]:4.1f}   {cs[1]:4.1f}   {cs[2]:4.1f}\n\n'
               f'protons in the orbitals: {4*Sp + 2*Ss:.2f} of 6\n'
               f'missing: {6 - 4*Sp - 2*Ss:.2f} protons\n→ where are they?  (Panel A, Lecture 3)')
    else: txt = 'no shells, no spectroscopic factors:\nthe strength is a band'
    ax.text(0.0, 0.95, txt, va='top', fontsize=12.5, family='monospace', bbox=dict(boxstyle='round', fc='#F3F4F7', ec='none'))
    plt.show()

WELL = dict(V0p=w_V0p, V0s=w_V0s, r0=W.fixed(1.32), a=W.fixed(0.55), Vso=W.fixed(9.0))      # Panel B's sliders: C and D listen to them too
WELL0 = dict(V0p=V0_DEFAULT, V0s=V0_DEFAULT, r0=1.32, a=0.55, Vso=9)
def show_C():
    """Panel C: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_C, W.VBox([w_model, W.HBox([w_Sp, w_Ss]), w_Gs]), dict(Sp=w_Sp, Ss=w_Ss, Gs=w_Gs, model=w_model, **WELL),
               static_calls=[dict(Sp=1.0, Ss=1.0, Gs=20, model='orbitals (IPM)', **WELL0), dict(Sp=0.77, Ss=0.55, Gs=20, model='orbitals (IPM)', **WELL0), dict(Sp=1.0, Ss=1.0, Gs=20, model='local Fermi gas', **WELL0)])

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
    P = ipm_spectral_function(orbs, pD, ED, widths={'1s1/2': Gs, '1p3/2': GP_1P}, occ={'1s1/2': Ss, '1p3/2': Sp})/6.0
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
               static_calls=[dict(Sp=1.0, Ss=1.0, name=next(n for n in names if n.startswith('E = 0.961')), Gs=20, **WELL0), dict(Sp=0.77, Ss=0.55, name=next(n for n in names if n.startswith('E = 0.961')), Gs=20, **WELL0)])

# ---- Rosenbluth-separated world data (R_L, R_T in MeV^-1 at fixed q)
def _ld(fn, scale=1.0, cols=(0, 1, 2), mask_zero=False):
    d = np.loadtxt(os.path.join(DATA, 'LT', fn)); m = d[:, 1] > 0 if mask_zero else np.ones(len(d), bool)
    err = d[m, cols[2]] if len(cols) > 2 and d.shape[1] > cols[2] else 0.0*d[m, 1]
    return d[m, cols[0]], d[m, cols[1]]*scale, err*scale
LT = {}
for q in (300, 380, 570):
    LT[f'12C   q = {q} MeV   (Jourdan, world data)'] = dict(A=12, Z=6, N=6, q=q, L=_ld(f'c12_q{q}_L.dat'), T=_ld(f'c12_q{q}_T.dat'))
MONIZ = {12: (0.221, 0.025)}      # k_F, eps_B [GeV] from the Moniz fit
pE = np.linspace(1, 500, 170); EE = np.linspace(0, 100, 101)

def _peak_ratio(om_d, R_d, om_m, R_m, R_ref):
    """integral of data / integral of model over the quasielastic peak region, defined as the omega window where the orbital response R_ref exceeds 20 % of its maximum"""
    win = om_m[R_ref > 0.2*R_ref.max()]; sel = (om_d >= win.min()) & (om_d <= win.max())
    if sel.sum() < 3: return np.nan
    m = np.interp(om_d[sel], om_m, R_m); return trapz(R_d[sel], om_d[sel])/max(trapz(m, om_d[sel]), 1e-12)

w_LT = W.Dropdown(options=list(LT), value=list(LT)[1], description='momentum transfer', style=WIDE, layout=W.Layout(width='520px'))
w_rfgE = W.Checkbox(True, description='show Fermi gas')

def panel_E(name, show_rfg, Sp, Ss, Gs, V0p, V0s, r0, a, Vso):
    d = LT[name]; A, Z, N, q = d['A'], d['Z'], d['N'], d['q']
    orbs = c12_orbitals(V0p, V0s, r0, a, Vso)
    P = ipm_spectral_function(orbs, pE, EE, widths={'1s1/2': Gs, '1p3/2': GP_1P}, occ={'1s1/2': Ss, '1p3/2': Sp})/6.0; Slab = f'$S$(1p) = {Sp:.2f}, $S$(1s) = {Ss:.2f}'
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
               title=f'$R_{c}$ at $q$ = {q} MeV,  $^{{12}}$C\n' + f'peak region:  ∫data / ∫orbitals = {rs:.2f}' + (f',   ∫data / ∫Fermi gas = {rf:.2f}' if show_rfg else ''))
        ax.legend(fontsize=9, loc='upper right'); polish(ax); ax.title.set_fontsize(11)
    plt.show()

def show_E():
    """Panel E: display the controls and the live figure (or the static versions when USE_WIDGETS is False)."""
    show_panel(panel_E, W.VBox([w_LT, w_rfgE, W.HTML('<i>S(1p), S(1s), Γ(1s) and the carbon well are taken live from Panels B and C.</i>')]),
               dict(name=w_LT, show_rfg=w_rfgE, Sp=w_Sp, Ss=w_Ss, Gs=w_Gs, **WELL),
               static_calls=[dict(name=n, show_rfg=True, Sp=0.77, Ss=0.55, Gs=20, **WELL0) for n in LT])
