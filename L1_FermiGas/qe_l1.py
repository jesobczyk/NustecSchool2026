"""Physics engine and interactive panels for the Lecture 1 demo (NuSTEC School 2026).
Relativistic Fermi gas with single-nucleon form factors, superscaling, the GiBUU mean field, CCQE neutrino responses,
and the effect of a potential on the response.  The notebook calls one show_X() function per panel."""
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

raw = np.genfromtxt(os.path.join(DATA, '12C.dat'), usecols=(2,3,4,5,6), invalid_raise=False)
cite = np.genfromtxt(os.path.join(DATA, '12C.dat'), usecols=(-1), dtype=str, invalid_raise=False)
datasets = {}
for E, th in sorted({(round(r[0],4), round(r[1],3)) for r in raw}):
    m = (np.abs(raw[:,0]-E)<1e-4) & (np.abs(raw[:,1]-th)<1e-3)
    datasets[f'E = {E:.3f} GeV,  θ = {th:g}°   ({cite[m][0].split(":")[0]})'] = dict(E=E, th=np.radians(th), om=raw[m,2], sig=raw[m,3], err=raw[m,4])
def get(E, th):
    return next(v for v in datasets.values() if abs(v['E']-E)<1e-3 and abs(np.degrees(v['th'])-th)<0.05)

def show_A():
    """Panel A: anatomy of the quasielastic peak (static figure)."""
    d = get(0.961, 37.5); kF, eB = 0.221, 0.025
    om = np.linspace(0.02, 0.65, 400)
    tot, L, T = rfg_xsec(d['E'], d['th'], om, kF, eB)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.errorbar(d['om']*1e3, d['sig'], d['err'], fmt='o', ms=4, color=C['data'], ecolor=C['data_light'], elinewidth=1, capsize=0, label=r'$^{12}$C$(e,e^\prime)$  Sealock 1989', zorder=3)
    ax.fill_between(om*1e3, 0, L, color=C['L'], alpha=0.25, lw=0, label='longitudinal  $v_L R_L$')
    ax.fill_between(om*1e3, L, L+T, color=C['T'], alpha=0.25, lw=0, label='transverse  $v_T R_T$')
    ax.plot(om*1e3, tot, color=C['rfg'], label=f'RFG   $k_F$ = {kF*1e3:.0f} MeV,  $\\varepsilon_B$ = {eB*1e3:.0f} MeV')

    w0 = qe_peak_position(d['E'], d['th'], 0.0); wB = qe_peak_position(d['E'], d['th'], eB)
    ymax = tot.max()
    ax.annotate('', xy=(wB*1e3, ymax*1.08), xytext=(w0*1e3, ymax*1.08), arrowprops=dict(arrowstyle='->', color=C['accent'], lw=1.5))
    ax.plot([w0*1e3]*2, [0, ymax*1.08], ls=':', color=C['accent'], lw=1.2); ax.plot([wB*1e3]*2, [0, ymax*1.08], ls='--', color=C['accent'], lw=1.2)
    ax.text(w0*1e3-4, ymax*1.12, r'$\omega = Q^2/2M$' + '\n(free nucleon)', ha='right', color=C['accent'], fontsize=10)
    ax.text(wB*1e3+4, ymax*1.12, r'$+\,\varepsilon_B$', ha='left', color=C['accent'], fontsize=10)
    _, q0, _, _ = lepton_factors(d['E'], d['th'], wB); width = 2*q0*kF/M
    ax.annotate('', xy=((wB-width/2)*1e3, ymax*0.5), xytext=((wB+width/2)*1e3, ymax*0.5), arrowprops=dict(arrowstyle='<->', color=C['rfg'], lw=1.5))
    ax.text(wB*1e3, ymax*0.53, r'full width $\simeq 2qk_F/M$', ha='center', color=C['rfg'], fontsize=10)
    ax.annotate('Δ(1232) and beyond:\nnot in a Fermi gas', xy=(440, 4400), xytext=(470, 1500), fontsize=10, color='#555',
                arrowprops=dict(arrowstyle='-', color='#999', lw=1))
    ax.set(xlabel=r'energy transfer  $\omega$  [MeV]', ylabel=r'$d^2\sigma/d\Omega\,d\omega$   [nb sr$^{-1}$ GeV$^{-1}$]',
           title='Anatomy of the quasielastic peak', xlim=(0, 660), ylim=(0, ymax*1.32))
    ax.legend(loc='upper right'); polish(ax); plt.show()

names = list(datasets)
default = next(n for n in names if n.startswith('E = 0.961'))

def panel_B(name, kF, eB, showLT, pauli):
    d = datasets[name]; kF *= 1e-3; eB *= 1e-3
    om = np.linspace(max(d['om'].min()*0.6, 0.01), min(d['om'].max()*1.05, d['E']*0.97), 350)
    tot, L, T = (rfg_xsec if pauli else rfg_xsec_noblock)(d['E'], d['th'], om, kF, eB)
    fig = plt.figure(figsize=(9.5, 6.2)); gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharex=ax)
    ax.errorbar(d['om']*1e3, d['sig'], d['err'], fmt='o', ms=4, color=C['data'], ecolor=C['data_light'], elinewidth=1, capsize=0, zorder=3, label=name)
    if showLT:
        ax.fill_between(om*1e3, 0, L, color=C['L'], alpha=0.22, lw=0, label='longitudinal'); ax.fill_between(om*1e3, L, L+T, color=C['T'], alpha=0.22, lw=0, label='transverse')
    ax.plot(om*1e3, tot, color=C['rfg'], label='relativistic Fermi gas')
    wB = qe_peak_position(d['E'], d['th'], eB); ax.axvline(wB*1e3, color=C['accent'], ls='--', lw=1)
    _, q0, _, _ = lepton_factors(d['E'], d['th'], wB)
    ax.text(0.99, 0.97, f'$|\\mathbf{{q}}|$ at the peak = {q0*1e3:.0f} MeV\n$\\omega_{{\\rm QE}}$ = {wB*1e3:.0f} MeV\nFWHM$_{{\\rm RFG}}$ ≈ {2*q0*kF/M*1e3*0.7:.0f} MeV',
            transform=ax.transAxes, ha='right', va='top', fontsize=10, color='#444', bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd'))
    ymax = max(d['sig'].max(), tot.max())
    ax.set(ylim=(0, ymax*1.25), ylabel=r'$d^2\sigma/d\Omega\,d\omega$  [nb sr$^{-1}$ GeV$^{-1}$]', title='Relativistic Fermi gas vs. data')
    ax.legend(loc='upper left', fontsize=9.5); plt.setp(ax.get_xticklabels(), visible=False); polish(ax)
    # residual strip
    mod_at_data = np.interp(d['om'], om, tot)
    ratio = np.where(mod_at_data > 0, d['sig']/np.where(mod_at_data>0, mod_at_data, 1), np.nan)
    axr.axhline(1, color='#888', lw=1); axr.axhspan(0.9, 1.1, color=C['grid'], alpha=0.6, lw=0)
    axr.plot(d['om']*1e3, np.where(ratio<2.9, ratio, np.nan), 'o', ms=3.5, color=C['data'])
    off = ratio >= 2.9
    axr.plot(d['om'][off]*1e3, np.full(off.sum(), 2.85), '^', ms=4, color=C['warn'])
    axr.set(ylim=(0, 3), yticks=[0, 1, 2, 3], ylabel='data / RFG', xlabel=r'energy transfer  $\omega$  [MeV]', xlim=(om.min()*1e3, om.max()*1e3)); polish(axr)
    plt.show()

def rfg_xsec_noblock(E, th, omega, kF, eB, Z=6, N=6):
    """Same as rfg_xsec but the Pauli-blocking theta function is switched off (final nucleon may land inside the Fermi sea)."""
    Q2, q, vL, vT = lepton_factors(E, th, omega)
    omega = np.atleast_1d(omega).astype(float); w = (omega - eB)[:, None]; qq = q[:, None]
    p = np.linspace(1e-4, kF, 300)[None, :]; Ep = np.sqrt(p*p+M*M); Epp = Ep + w
    ok = Epp > M; pp2 = np.where(ok, Epp**2-M*M, 0.0); cosr = (pp2-p*p-qq*qq)/(2*p*qq); cos = np.clip(cosr, -1, 1)
    ok &= np.abs(cosr) <= 1
    Q2c = np.clip(qq*qq-w*w, 1e-6, None); pdotq = Ep*w - p*qq*cos; pt0 = Ep + pdotq*w/Q2c; pperp2 = p*p*(1-cos**2)
    jac = 2*np.pi*p*Epp/qq*(M*M/(Ep*Epp)); dp = p[0,1]-p[0,0]
    GEp, GMp, GEn, GMn = form_factors(Q2c); RL = 0; RT = 0
    for nN, GE, GM in ((Z, GEp, GMp), (N, GEn, GMn)):
        tau = Q2c/(4*M*M); W1 = tau*GM**2; W2 = (GE**2+tau*GM**2)/(1+tau)
        RL = RL + 3*nN/(4*np.pi*kF**3)*dp*np.sum(np.where(ok, jac*(-W1*qq*qq/Q2c + W2*pt0**2/(M*M)), 0), axis=1)
        RT = RT + 3*nN/(4*np.pi*kF**3)*dp*np.sum(np.where(ok, jac*(2*W1 + W2*pperp2/(M*M)), 0), axis=1)
    sM = mott(E, th)*HBARC2_NB
    return sM*(vL*RL+vT*RT), sM*vL*RL, sM*vT*RT

def panel_B_static(name=default):
    """Static version: what each knob does, as small multiples."""
    d = datasets[name]; om = np.linspace(max(d['om'].min()*0.6, 0.01), min(d['om'].max()*1.05, d['E']*0.97), 350)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (label, vals, fixed) in zip(axs, [(r'$k_F$', [180, 221, 260], ('eB', 25)), (r'$\varepsilon_B$', [0, 25, 50], ('kF', 221))]):
        ax.errorbar(d['om']*1e3, d['sig'], d['err'], fmt='o', ms=3.5, color=C['data'], ecolor=C['data_light'], elinewidth=1, capsize=0, zorder=3)
        for v, a, ls in zip(vals, [0.4, 1.0, 0.9], ['-', '-', '--']):
            kF, eB = (v, fixed[1]) if fixed[0] == 'eB' else (fixed[1], v)
            tot, _, _ = rfg_xsec(d['E'], d['th'], om, kF*1e-3, eB*1e-3)
            ax.plot(om*1e3, tot, color=C['rfg'], alpha=a, ls=ls, lw=2 if a == 1 else 1.6, label=f'{label} = {v} MeV')
        ax.set(xlabel=r'energy transfer  $\omega$  [MeV]', title=f'varying {label}   ({fixed[0].replace("eB", r"$\varepsilon_B$").replace("kF", r"$k_F$")} = {fixed[1]} MeV)')
        ax.legend(fontsize=9.5); polish(ax)
    axs[0].set_ylabel(r'$d^2\sigma/d\Omega\,d\omega$  [nb sr$^{-1}$ GeV$^{-1}$]'); fig.suptitle(name, y=1.01, fontsize=11, color='#444'); plt.show()

def show_B():
    """Panel B: fit the Fermi gas yourself."""
    w_data = W.Dropdown(options=names, value=default, description='dataset', layout=W.Layout(width='520px'))
    w_kF   = W.FloatSlider(221, min=150, max=300, step=1, description='$k_F$ [MeV]', continuous_update=True, readout_format='.0f', layout=W.Layout(width='420px'))
    w_eB   = W.FloatSlider(25, min=0, max=60, step=1, description='$\\varepsilon_B$ [MeV]', continuous_update=True, readout_format='.0f', layout=W.Layout(width='420px'))
    w_LT   = W.Checkbox(True, description='show L / T decomposition')
    w_pauli= W.Checkbox(True, description='Pauli blocking')
    show_panel(panel_B, W.VBox([w_data, W.HBox([w_kF, w_eB]), W.HBox([w_LT, w_pauli])]),
               dict(name=w_data, kF=w_kF, eB=w_eB, showLT=w_LT, pauli=w_pauli),
               static_calls=[])
    if not USE_WIDGETS:
        panel_B_static(default)
        panel_B(default, 221, 25, True, True)
        low_q = next(n for n in names if n.startswith('E = 0.400 GeV,  θ = 36'))
        panel_B(low_q, 221, 25, False, True); panel_B(low_q, 221, 25, False, False)

# spectra with a clean QE peak and q between ~300 and ~1000 MeV at the peak
def q_at_peak(d): return lepton_factors(d['E'], d['th'], qe_peak_position(d['E'], d['th'], 0.02))[1]
scaling_sets = [n for n, d in datasets.items() if 0.30 <= q_at_peak(d) <= 1.05 and d['E'] >= 0.4]

cmap = mpl.colormaps['Blues']   # sequential, single hue: colour encodes |q| (a magnitude)
def panel_C(kF, Es, show_rfg, qrange):
    kF *= 1e-3; Es *= 1e-3
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sets = [n for n in scaling_sets if qrange[0] <= q_at_peak(datasets[n]) <= qrange[1]]
    qs = np.array([q_at_peak(datasets[n]) for n in sets]); norm = mpl.colors.Normalize(0.25, 1.1)
    for n in sets:
        d = datasets[n]; psi, GL, GT, vL, vT, q = scaling_pieces(d['E'], d['th'], d['om'], kF, Es)
        f = d['sig']/HBARC2_NB/(mott(d['E'], d['th'])*(vL*GL + vT*GT))
        ax.plot(psi, f, 'o', ms=3.6, mec='none', color=cmap(norm(q_at_peak(d))), alpha=0.85)
    if show_rfg:
        ps = np.linspace(-2.5, 2.5, 500); ax.plot(ps, f_RFG(ps), color=C['warn'], lw=2, label=r'relativistic Fermi gas  $\frac{3}{4}(1-\psi^{\prime 2})$')
    ax.axvspan(1.0, 2.6, color=C['grid'], alpha=0.5, lw=0); ax.text(1.8, 0.86, 'Δ, MEC …\nno scaling expected', ha='center', fontsize=10, color='#555')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.045); cb.set_label(r'$|\mathbf{q}|$ at the QE peak  [GeV]')
    ax.set(xlim=(-2.1, 2.6), ylim=(0, 0.95), xlabel=r"scaling variable  $\psi'$", ylabel=r"superscaling function  $f(\psi')$",
           title=f'Superscaling of $^{{12}}$C$(e,e^\\prime)$ — {len(sets)} spectra, {sum(len(datasets[n]["om"]) for n in sets)} points')
    if show_rfg: ax.legend(loc='upper left')
    polish(ax); plt.show()

def show_C():
    """Panel C: superscaling."""
    w_kFs = W.FloatSlider(228, min=180, max=280, step=1, description='$k_F$ [MeV]', readout_format='.0f', layout=W.Layout(width='420px'))
    w_Es  = W.FloatSlider(20, min=0, max=50, step=1, description='$E_{\\rm shift}$ [MeV]', readout_format='.0f', layout=W.Layout(width='420px'))
    w_rfg = W.Checkbox(True, description='RFG parabola')
    w_cut = W.FloatRangeSlider(value=[0.3, 1.05], min=0.3, max=1.05, step=0.05, description='q range [GeV]', layout=W.Layout(width='420px'))
    show_panel(panel_C, W.VBox([W.HBox([w_kFs, w_Es]), W.HBox([w_cut, w_rfg])]),
               dict(kF=w_kFs, Es=w_Es, show_rfg=w_rfg, qrange=w_cut),
               static_calls=[dict(kF=228, Es=20, show_rfg=True, qrange=[0.3, 1.05]),
                             dict(kF=228, Es=0,  show_rfg=True, qrange=[0.3, 1.05])])


# ---------- scaling of the second kind: many nuclei ----------
# (Z, N, k_F [MeV], E_shift [MeV], from the table?)  k_F and E_shift: Maieron, Donnelly, Sick, PRC 65 (2002) 025502, Table 1.
# 4He and 16O are not in that table: the values below are our own starting guesses (marked with *).
# Any QES-archive file dropped into data/ (e.g. 56Fe.dat, 208Pb.dat) is picked up automatically.
NUCLEI = {'4He': (2, 2, 200, 20, False), '6Li': (3, 3, 165, 15, True), '12C': (6, 6, 228, 20, True), '16O': (8, 8, 230, 20, False),
          '27Al': (13, 14, 236, 18, True), '40Ca': (20, 20, 241, 28, True), '48Ca': (20, 28, 241, 28, False),
          '56Fe': (26, 30, 241, 23, True), '197Au': (79, 118, 245, 25, True), '208Pb': (82, 126, 248, 31, True)}
NUC_COL = {'4He': '#e34948', '6Li': '#e87ba4', '12C': '#2a78d6', '16O': '#1baf7a', '27Al': '#eda100',
           '40Ca': '#6250d6', '48Ca': '#9b8be0', '56Fe': '#eb6834', '197Au': '#8a6d00', '208Pb': '#2b2b2b'}

def load_nucleus(name):
    raw_ = np.genfromtxt(os.path.join(DATA, f'{name}.dat'), usecols=(2,3,4,5,6), invalid_raise=False)
    cite_ = np.genfromtxt(os.path.join(DATA, f'{name}.dat'), usecols=(-1), dtype=str, invalid_raise=False)
    out = {}
    for E, th in sorted({(round(r[0],4), round(r[1],3)) for r in raw_}):
        m = (np.abs(raw_[:,0]-E)<1e-4) & (np.abs(raw_[:,1]-th)<1e-3)
        out[(E, th)] = dict(E=E, th=np.radians(th), om=raw_[m,2], sig=raw_[m,3], err=raw_[m,4], ref=cite_[m][0].split(':')[0])
    return out

all_nuclei = {n: load_nucleus(n) for n in NUCLEI if os.path.exists(os.path.join(DATA, f'{n}.dat'))}

def scaling_sets_of(spectra, qlo=0.30, qhi=1.05):
    return [d for d in spectra.values() if qlo <= q_at_peak(d) <= qhi and d['E'] >= 0.4]

def panel_C2(chosen, same_kF, kF_all, Es_all, qrange):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(wspace=0.2))
    ps = np.linspace(-2.5, 2.5, 500)
    for n in chosen:
        Z, N, kF, Es, tab = NUCLEI[n]
        if same_kF: kF, Es = kF_all, Es_all
        sets = scaling_sets_of(all_nuclei[n], *qrange); npts = sum(len(d['om']) for d in sets)
        for i, d in enumerate(sets):
            psi, GL, GT, vL, vT, q = scaling_pieces(d['E'], d['th'], d['om'], kF*1e-3, Es*1e-3, Z, N)
            f = d['sig']/HBARC2_NB/(mott(d['E'], d['th'])*(vL*GL + vT*GT))
            lab = f'{n}{"" if tab or same_kF else "*"}   $k_F$ = {kF:.0f}, $E_s$ = {Es:.0f} MeV  ({len(sets)} spectra)' if i == 0 else None
            for ax in axs: ax.plot(psi, f, 'o', ms=3.2, mec='none', color=NUC_COL[n], alpha=0.75, label=lab if ax is axs[0] else None)
    for ax in axs:
        ax.plot(ps, f_RFG(ps), color='#888', lw=1.5, ls='--', label=r'Fermi gas  $\frac{3}{4}(1-\psi^{\prime 2})$' if ax is axs[0] else None)
        ax.axvspan(1.0, 2.6, color=C['grid'], alpha=0.5, lw=0); ax.set_xlabel(r"scaling variable  $\psi'$"); polish(ax)
    axs[0].set(xlim=(-2.1, 2.6), ylim=(0, 0.95), ylabel=r"$f(\psi')$", title='linear scale: the peak')
    axs[1].set(xlim=(-2.1, 1.0), ylim=(3e-3, 1.2), yscale='log', title=r"log scale: the low-$\omega$ side ($\psi'<0$)")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9.5)
    fig.suptitle('Scaling of the second kind: the same curve for different nuclei' + ('  — one $k_F$ for all' if same_kF else '  — each nucleus with its own $k_F$'),
                 y=1.02, fontsize=12.5, fontweight='bold')
    plt.show()

def show_C2():
    """Panel C2: superscaling across nuclei (scaling of the second kind)."""
    avail = list(all_nuclei)
    w_nuc = W.SelectMultiple(options=avail, value=[n for n in avail if n in ('6Li', '12C', '40Ca', '56Fe', '208Pb')] or avail[:2], description='nuclei', rows=min(6, len(avail)), layout=W.Layout(width='260px'))
    w_same = W.Checkbox(False, description='same k_F for all nuclei')
    w_kFa = W.FloatSlider(228, min=150, max=280, step=1, description='$k_F$ [MeV]', readout_format='.0f', continuous_update=False, layout=W.Layout(width='380px'))
    w_Esa = W.FloatSlider(20, min=0, max=50, step=1, description='$E_{\\rm shift}$ [MeV]', readout_format='.0f', continuous_update=False, layout=W.Layout(width='380px'))
    w_q = W.FloatRangeSlider(value=[0.3, 1.05], min=0.3, max=1.05, step=0.05, description='q range [GeV]', continuous_update=False, layout=W.Layout(width='380px'))
    show_panel(panel_C2, W.HBox([w_nuc, W.VBox([w_same, w_kFa, w_Esa, w_q])]),
               dict(chosen=w_nuc, same_kF=w_same, kF_all=w_kFa, Es_all=w_Esa, qrange=w_q),
               static_calls=[dict(chosen=avail, same_kF=False, kF_all=228, Es_all=20, qrange=[0.3, 1.05]),
                             dict(chosen=avail, same_kF=True, kF_all=228, Es_all=20, qrange=[0.3, 1.05])])

def panel_D(setname, x):
    par = EQS[setname]; rho = x*RHO0
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw=dict(wspace=0.32))
    # (1) V vs density at several momenta — reproduces Leitner Fig. 7.1 (left)
    ax = axs[0]; xs = np.linspace(0.02, 3, 120)
    for i, pm in enumerate([0.0, 0.5, 1.0, 1.5]):
        ax.plot(xs, [V_gibuu(pm, xx*RHO0, EQS[list(EQS)[1]])[0]*1e3 for xx in xs], color=C['eqs2'], alpha=1-0.22*i, label=f'EQS 2, |p| = {pm:.1f} GeV')
    ax.plot(xs, [V_gibuu(0.0, xx*RHO0, EQS[list(EQS)[0]])[0]*1e3 for xx in xs], color=C['eqs1'], label='EQS 1')
    ax.axvline(x, color='#999', ls=':', lw=1); ax.axhline(0, color='#999', lw=0.8)
    ax.set(xlabel=r'$\rho/\rho_0$', ylabel=r'$V_N$  [MeV]', title='density dependence', ylim=(-110, 320)); ax.legend(fontsize=8.5, loc='upper left'); polish(ax)
    # (2) V vs momentum at the chosen density, and the effective "binding" V(p')-V(0)
    ax = axs[1]; ps = np.linspace(0, 1.5, 150)
    V = V_gibuu(ps, rho, par)*1e3; ax.plot(ps, V, color=C['eqs2'] if par['C'] else C['eqs1'], label=setname.split('(')[0].strip())
    pF = HBARC*(1.5*np.pi**2*rho)**(1/3); ax.axvspan(0, pF, color=C['grid'], alpha=0.6, lw=0); ax.text(pF/2, V.min()+8, 'Fermi\nsea', ha='center', fontsize=9, color='#666')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set(xlabel=r'$|\mathbf{p}|$  [GeV]', ylabel=r'$V_N(p,\rho)$  [MeV]', title=f'momentum dependence at ρ = {x:.2f} ρ₀', ylim=(-90, 100)); polish(ax)
    shift = V - V[0]; ax.plot(ps, shift, color=C['accent'], ls='--', lw=1.6, label=r"$V_N(p')-V_N(0)$ : effective $\varepsilon_B$")
    ax.legend(fontsize=8.5, loc='lower right')
    # (3) LDA for 12C: density, local kF, local V(0)
    ax = axs[2]; r = np.linspace(0, 5, 200); rh = rho_C12(r)
    kFr = HBARC*(1.5*np.pi**2*rh)**(1/3)*1e3
    ax.plot(r, rh/RHO0, color=C['data'], label=r'$\rho(r)/\rho_0$'); ax.plot(r, kFr/300, color=C['rfg'], label=r'$k_F(r)/300$ MeV')
    Vr = np.array([V_gibuu(0.0, max(x_,1e-6), par)[0] for x_ in rh])*1e3
    ax.plot(r, -Vr/100, color=C['accent'], label=r'$-V_N(0,r)/100$ MeV')
    ax.axhline(x, color='#999', ls=':', lw=1); ax.text(4.9, x+0.03, 'ρ chosen in middle panel', ha='right', fontsize=8.5, color='#777')
    ax.set(xlabel='r  [fm]', title='local density approximation, $^{12}$C', ylim=(0, 1.6)); ax.legend(fontsize=9); polish(ax)
    fig.suptitle('The GiBUU nucleon mean field (Welke parametrisation, Teis fits)', y=1.02, fontsize=13, fontweight='bold')
    plt.show()

def show_D():
    """Panel D: the GiBUU mean field."""
    w_set = W.ToggleButtons(options=list(EQS), value=list(EQS)[1], description='parameter set', style={'description_width':'initial'})
    w_rho = W.FloatSlider(1.0, min=0.05, max=2.5, step=0.05, description='$\\rho/\\rho_0$', readout_format='.2f', layout=W.Layout(width='380px'))

    show_panel(panel_D, W.VBox([w_set, w_rho]), dict(setname=w_set, x=w_rho),
               static_calls=[dict(setname=list(EQS)[1], x=1.0), dict(setname=list(EQS)[0], x=1.0), dict(setname=list(EQS)[1], x=0.5)])

GF = 1.16638e-5; COSC = 0.9743; HBARC2_CM2 = 0.389379e-27; MMU = 0.10566; ME = 0.000511
METRIC = np.diag([1.,-1.,-1.,-1.])
EPS = np.zeros((4,4,4,4))
import itertools
for perm in itertools.permutations(range(4)):
    EPS[perm] = -np.linalg.det(np.eye(4)[list(perm)])          # eps^{0123} = -1

def form_factors_cc(Q2, MA=1.03):
    GD = 1/(1+Q2/0.71)**2; tau = Q2/(4*M*M)
    GEp, GMp, GEn, GMn = GD, MU_P*GD, -MU_N*tau*GD/(1+5.6*tau), MU_N*GD
    GEV, GMV = GEp-GEn, GMp-GMn
    F1 = (GEV+tau*GMV)/(1+tau); F2 = (GMV-GEV)/(1+tau)
    FA = -1.267/(1+Q2/MA**2)**2; FP = 2*M*M*FA/(0.1396**2+Q2)
    return F1, F2, FA, FP

def nucleon_tensor_cc(p4, pp4, q4, Q2, MA):
    # h^{mu nu} = 1/2 Tr[(p'+M) Gamma^mu (p+M) Gammabar^nu] in closed form (spinors normalised to 2M)
    F1, F2, FA, FP = form_factors_cc(Q2, MA); tau = Q2/(4*M*M); GM = F1+F2
    a1 = 4*M*M*(tau*GM**2 + (1+tau)*FA**2); a2 = F1**2 + tau*F2**2 + FA**2
    a3 = -2*GM*FA;  a4 = -GM**2 - FA**2 - 4*FA*FP + 4*tau*FP**2
    P = p4 + pp4; Pl = P @ METRIC; ql = q4 @ METRIC
    return (-METRIC*a1[...,None,None] + P[...,:,None]*P[...,None,:]*a2[...,None,None]
            + 1j*a3[...,None,None]*np.einsum('mnab,...a,...b->...mn', EPS, Pl, ql)
            + q4[...,:,None]*q4[...,None,:]*a4[...,None,None])

def lepton_tensor_cc(k4, kp4, nu=True):
    kdotkp = np.einsum('...m,mn,...n->...', k4, METRIC, kp4)
    eps_term = np.einsum('mnab,...a,...b->...mn', EPS, k4 @ METRIC, kp4 @ METRIC)
    sym = k4[...,:,None]*kp4[...,None,:] + k4[...,None,:]*kp4[...,:,None] - METRIC*kdotkp[...,None,None]
    return 8*(sym + (1 if nu else -1)*1j*eps_term)

def ccqe_rfg(Enu, theta, omega, kF, eB, MA=1.03, ml=MMU, nu=True, Z=6, N=6, npts=150):
    # Five RFG responses [GeV^-1], their weighted contributions and d2sigma/dOmega domega [cm^2/sr/GeV] for CC nu(nubar)-12C
    omega = np.atleast_1d(omega).astype(float); El = Enu - omega; kl = np.sqrt(np.clip(El**2-ml**2, 0, None))
    q = np.sqrt(np.clip(Enu**2 + kl**2 - 2*Enu*kl*np.cos(theta), 1e-12, None)); w = omega - eB
    nN = N if nu else Z
    p = (np.arange(npts)+0.5)/npts*kF; dp = kF/npts
    P, Wm = np.meshgrid(p, w, indexing='ij'); Q = np.broadcast_to(q, Wm.shape)
    Ep = np.sqrt(P*P+M*M); Epp = Ep + Wm; ok = Epp > M; pp2 = np.where(ok, Epp**2-M*M, 0)
    cosr = (pp2-P*P-Q*Q)/(2*P*Q); ok &= (np.abs(cosr) <= 1) & (pp2 > kF*kF); cos = np.clip(cosr, -1, 1); sin = np.sqrt(1-cos**2)
    Q2 = np.clip(Q*Q - Wm*Wm, 1e-6, None); q4 = np.stack([Wm, 0*Q, 0*Q, Q], axis=-1)
    h = 0
    for ph in (0, np.pi/2, np.pi, 3*np.pi/2):                     # azimuthal average, exact for a rank-2 tensor
        p4 = np.stack([Ep, P*sin*np.cos(ph), P*sin*np.sin(ph), P*cos], axis=-1); pp4 = p4 + q4; pp4[...,0] = Epp
        h = h + nucleon_tensor_cc(p4, pp4, q4, Q2, MA)
    h = h/4
    weight = np.where(ok, 2*np.pi*P*Epp/Q/(4*Ep*Epp), 0.0)*3*nN/(4*np.pi*kF**3)*dp
    Wn = np.einsum('pw,pwmn->wmn', weight, h)                      # nuclear tensor
    R = dict(CC=Wn[:,0,0].real, CL=-0.5*(Wn[:,0,3]+Wn[:,3,0]).real, LL=Wn[:,3,3].real,
             T=(Wn[:,1,1]+Wn[:,2,2]).real, Tp=(-0.5j*(Wn[:,1,2]-Wn[:,2,1])).real)
    cb = np.clip((Enu - kl*np.cos(theta))/q, -1, 1); sb = np.sqrt(1-cb**2)
    k4 = np.stack([Enu+0*q, Enu*sb, 0*q, Enu*cb], axis=-1); kp4 = k4 - q4[0]
    Ll = np.einsum('ma,wab,bn->wmn', METRIC, lepton_tensor_cc(k4, kp4, nu), METRIC)
    pref = GF**2*COSC**2/2 * kl/(2*(2*np.pi)**2)/(2*Enu) * HBARC2_CM2
    contrib = dict(CC=(Ll[:,0,0]*Wn[:,0,0]).real, CL=(Ll[:,0,3]*Wn[:,0,3]+Ll[:,3,0]*Wn[:,3,0]).real, LL=(Ll[:,3,3]*Wn[:,3,3]).real,
                   T=(Ll[:,1,1]*Wn[:,1,1]+Ll[:,2,2]*Wn[:,2,2]).real, Tp=(Ll[:,1,2]*Wn[:,1,2]+Ll[:,2,1]*Wn[:,2,1]).real)
    contrib = {k: pref*v for k, v in contrib.items()}
    return R, contrib, pref*np.einsum('wmn,wmn->w', Ll, Wn).real, q

RNAMES = {'CC': r'$R_{CC}$', 'CL': r'$R_{CL}$', 'LL': r'$R_{LL}$', 'T': r'$R_T$', 'Tp': r"$R_{T'}$"}
RCOL = {'CC': '#2a78d6', 'CL': '#eb6834', 'LL': '#1baf7a', 'T': '#eda100', 'Tp': '#e87ba4'}

def panel_E(Enu, thm, MA, show_nubar, ml, kF=0.221, eB=0.025):
    Enu *= 1e-3; th = np.radians(thm)
    om = np.linspace(0.005, Enu - ml - 0.002, 260)
    R, con, sig, q = ccqe_rfg(Enu, th, om, kF, eB, MA, ml, True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(wspace=0.25, width_ratios=[1.15, 1]))
    sc = 1e38
    axL.plot(om*1e3, sig*sc, color=C['data'], lw=2.4, label=r'$\nu_\ell$  total')
    for k in ('T', 'CC', 'CL', 'LL', 'Tp'):          # T first and thicker: at backward angles 2V_T'R_T' lies on top of it
        lab = {'CC': r'$V_{CC}R_{CC}$', 'CL': r'$2V_{CL}R_{CL}$', 'LL': r'$V_{LL}R_{LL}$', 'T': r'$V_TR_T$', 'Tp': r"$2V_{T'}R_{T'}$"}[k]
        axL.plot(om*1e3, con[k]*sc, color=RCOL[k], lw=3.2 if k == 'T' else 1.6, label=lab)
    if show_nubar:
        Rb, conb, sigb, _ = ccqe_rfg(Enu, th, om, kF, eB, MA, ml, False)
        axL.plot(om*1e3, sigb*sc, color=C['data'], lw=2.0, ls='--', label=r'$\bar\nu_\ell$  total')
        axL.plot(om*1e3, conb['Tp']*sc, color=RCOL['Tp'], lw=1.4, ls='--', label=r"$-2V_{T'}R_{T'}$  ($\bar\nu$)")
    axL.axhline(0, color='#999', lw=0.8)
    ipk = np.argmax(sig); Q2pk = q[ipk]**2 - om[ipk]**2
    axL.text(0.98, 0.97, f'at the peak:  $|\\mathbf{{q}}|$ = {q[ipk]*1e3:.0f} MeV,  $Q^2$ = {Q2pk:.2f} GeV$^2$', transform=axL.transAxes, ha='right', va='top', fontsize=10, color='#444',
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ddd'))
    axL.set(xlabel=r'energy transfer  $\omega$  [MeV]', ylabel=r'$d^2\sigma/d\Omega\,d\omega$   [$10^{-38}$ cm$^2$ sr$^{-1}$ GeV$^{-1}$ per $^{12}$C]',
            title=f'CCQE on the Fermi gas:  $E_\\nu$ = {Enu*1e3:.0f} MeV,  $\\theta_\\ell$ = {thm:.0f}°,  $M_A$ = {MA:.2f} GeV')
    nz = np.where(np.abs(sig) > 1e-4*np.abs(sig).max())[0]; whi = om[nz[-1]] if len(nz) else om[-1]
    xmax = min(om[-1], 1.3*whi)*1e3
    axL.set_xlim(0, xmax); axR.set_xlim(0, xmax)
    lo = min(0, min(v.min() for v in con.values())*sc, (conb['Tp'].min()*sc if show_nubar else 0)); axL.set_ylim(lo*1.15, sig.max()*sc*1.3)
    axL.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.13)); polish(axL)
    for k in ('CC', 'CL', 'LL', 'T', 'Tp'):
        axR.plot(om*1e3, R[k], color=RCOL[k], lw=1.8, label=RNAMES[k])
    axR.axhline(0, color='#999', lw=0.8)
    axR.set(xlabel=r'energy transfer  $\omega$  [MeV]', ylabel=r'nuclear response  [GeV$^{-1}$]', title='the five RFG response functions ($k_F$ = 221 MeV, $\\varepsilon_B$ = 25 MeV)')
    axR.legend(fontsize=10, loc='upper right'); polish(axR)
    plt.show()

def show_E():
    """Panel E: the same Fermi gas hit by a neutrino."""
    w_Enu = W.FloatSlider(600, min=200, max=1000, step=10, description='$E_\\nu$ [MeV]', readout_format='.0f', continuous_update=False, layout=W.Layout(width='420px'))
    w_thm = W.FloatSlider(30, min=2, max=180, step=1, description='$\\theta_\\mu$ [deg]', readout_format='.0f', continuous_update=False, layout=W.Layout(width='420px'))
    w_MA  = W.FloatSlider(1.03, min=0.9, max=1.5, step=0.01, description='$M_A$ [GeV]', readout_format='.2f', continuous_update=False, layout=W.Layout(width='420px'))
    w_nub = W.Checkbox(True, description='also antineutrino')
    w_lep = W.ToggleButtons(options=[('muon', MMU), ('electron', ME)], description='lepton')
    show_panel(panel_E, W.VBox([W.HBox([w_Enu, w_thm]), W.HBox([w_MA, w_lep]), w_nub]),
               dict(Enu=w_Enu, thm=w_thm, MA=w_MA, show_nubar=w_nub, ml=w_lep),
               static_calls=[dict(Enu=600, thm=30, MA=1.03, show_nubar=True, ml=MMU), dict(Enu=1000, thm=15, MA=1.03, show_nubar=True, ml=MMU),
                             dict(Enu=300, thm=60, MA=1.03, show_nubar=True, ml=MMU)])

class Potential:
    # kinds: 'none' | 'const' (V0 before and after) | 'moniz' (-eB on the initial nucleon only) | 'eqs1' | 'eqs2' (GiBUU at density rho)
    def __init__(self, kind='none', eB=0.025, V0=-0.05, rho=RHO0):
        self.kind = kind; self.eB = eB; self.pg = np.linspace(0, 3.0, 1201)
        par = {'eqs1': EQS[list(EQS)[0]], 'eqs2': EQS[list(EQS)[1]]}.get(kind)
        self.Vg = V_gibuu(self.pg, rho, par) if par else (V0 + 0*self.pg if kind == 'const' else 0*self.pg)
    def Vi(self, p): return -self.eB + 0*p if self.kind == 'moniz' else np.interp(p, self.pg, self.Vg)
    def Vf(self, p): return 0*p if self.kind == 'moniz' else np.interp(p, self.pg, self.Vg)
    def Ef(self, p): return np.sqrt(M*M + p*p) + self.Vf(p)
    def invert_Ef(self, E): return np.interp(E, self.Ef(self.pg), self.pg, left=np.nan, right=np.nan)

def ff_em_nucleon(Q2, proton=True):
    GEp, GMp, GEn, GMn = form_factors(Q2); tau = Q2/(4*M*M); GE, GM = (GEp, GMp) if proton else (GEn, GMn)
    return (GE+tau*GM)/(1+tau), (GM-GE)/(1+tau), 0*Q2, 0*Q2

def nucleon_tensor_general(p4, pp4, q4, Q2, F1, F2, FA, FP):
    tau = Q2/(4*M*M); GM = F1+F2
    a1 = 4*M*M*(tau*GM**2 + (1+tau)*FA**2); a2 = F1**2 + tau*F2**2 + FA**2; a3 = -2*GM*FA; a4 = -GM**2 - FA**2 - 4*FA*FP + 4*tau*FP**2
    P = p4 + pp4; Pl = P @ METRIC; ql = q4 @ METRIC
    return (-METRIC*a1[...,None,None] + P[...,:,None]*P[...,None,:]*a2[...,None,None]
            + 1j*a3[...,None,None]*np.einsum('mnab,...a,...b->...mn', EPS, Pl, ql) + q4[...,:,None]*q4[...,None,:]*a4[...,None,None])

def nuclear_tensor_pot(omega, q, kF, pot, ffun, nN, npts=150):
    omega = np.atleast_1d(omega).astype(float); q = np.broadcast_to(q, omega.shape).astype(float)
    p = (np.arange(npts)+0.5)/npts*kF; dp = kF/npts
    P, Wm = np.meshgrid(p, omega, indexing='ij'); Q = np.broadcast_to(q, Wm.shape); Ep = np.sqrt(P*P+M*M)
    pp = pot.invert_Ef(Ep + pot.Vi(P) + Wm); ok = np.isfinite(pp) & (pp > kF); pp = np.where(ok, pp, kF+1e-3); Epp = np.sqrt(M*M+pp*pp)
    cosr = (pp*pp - P*P - Q*Q)/(2*P*Q); ok &= np.abs(cosr) <= 1; cos = np.clip(cosr, -1, 1); sin = np.sqrt(1-cos**2)
    gprime = pp/Epp + np.interp(pp, pot.pg, np.gradient(pot.Vf(pot.pg), pot.pg))     # d/dp' [E_{p'} + V(p')]
    jac = 2*np.pi*P*P*pp/(gprime*P*Q)/(4*Ep*Epp)
    weff = Epp - Ep; Q2 = np.clip(Q*Q - weff*weff, 1e-6, None); q4 = np.stack([weff, 0*Q, 0*Q, Q], axis=-1); F = ffun(Q2); h = 0
    for ph in (0, np.pi/2, np.pi, 3*np.pi/2):
        p4 = np.stack([Ep, P*sin*np.cos(ph), P*sin*np.sin(ph), P*cos], axis=-1); h = h + nucleon_tensor_general(p4, p4+q4, q4, Q2, *F)
    return np.einsum('pw,pwmn->wmn', np.where(ok, jac, 0.0)*3*nN/(4*np.pi*kF**3)*dp, h/4)

def responses_pot(omega, q, kF, pot, probe='electron'):
    if probe == 'electron':
        Wn = nuclear_tensor_pot(omega, q, kF, pot, lambda Q2: ff_em_nucleon(Q2, True), 6) + nuclear_tensor_pot(omega, q, kF, pot, lambda Q2: ff_em_nucleon(Q2, False), 6)
        return {'L': Wn[:,0,0].real, 'T': (Wn[:,1,1]+Wn[:,2,2]).real}
    Wn = nuclear_tensor_pot(omega, q, kF, pot, lambda Q2: form_factors_cc(Q2), 6)
    return dict(CC=Wn[:,0,0].real, CL=-0.5*(Wn[:,0,3]+Wn[:,3,0]).real, LL=Wn[:,3,3].real, T=(Wn[:,1,1]+Wn[:,2,2]).real, Tp=(-0.5j*(Wn[:,1,2]-Wn[:,2,1])).real)

POTS = {'free nucleons': ('none', C['data'], '-'), 'constant well, before and after': ('const', C['accent'], '--'),
        'Moniz: ε_B on the initial nucleon only': ('moniz', C['rfg'], '-'), 'momentum-dependent V(p) at ρ₀': ('eqs2', C['L'], '-'), 'momentum-dependent V(p) at ρ₀/2': ('eqs2h', C['T'], '-')}
def make_pot(key): return Potential('eqs2', rho=RHO0/2) if key == 'eqs2h' else Potential(key, eB=0.025, V0=-0.05)

def panel_F(q, probe, chosen):
    q *= 1e-3; om = np.linspace(0.003, np.sqrt(M*M+(0.221+q)**2)-np.sqrt(M*M+0.221**2)+0.12, 400)
    if probe == 'electron':
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(wspace=0.22))
        for lab in chosen:
            key, col, ls = POTS[lab]; R = responses_pot(om, q, 0.221, make_pot(key), 'electron')
            axs[0].plot(om*1e3, R['L'], color=col, ls=ls, lw=2, label=lab); axs[1].plot(om*1e3, R['T'], color=col, ls=ls, lw=2, label=lab)
        axs[0].set(title='$R_L$', ylabel='response  [GeV$^{-1}$]'); axs[1].set(title='$R_T$'); axs[0].legend(fontsize=9.5)
        for ax in axs: ax.set_xlabel(r'energy transfer  $\omega$  [MeV]'); polish(ax)
    else:
        fig, axs = plt.subplots(1, len(chosen), figsize=(4.3*len(chosen), 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
        axs = np.atleast_1d(axs)
        for ax, lab in zip(axs, chosen):
            key, col, ls = POTS[lab]; R = responses_pot(om, q, 0.221, make_pot(key), 'neutrino')
            for k in ('T', 'CC', 'CL', 'LL', 'Tp'):
                y = R[k]/np.abs(R[k]).max()*np.sign(R[k][np.argmax(np.abs(R[k]))]); ax.plot(om*1e3, y, color=RCOL[k], lw=3 if k == 'T' else 1.6, label=RNAMES[k])
            i = np.argmax(R['T']); ax.axvline(om[i]*1e3, color='#999', ls=':', lw=1); ax.set(title=lab, xlabel=r'$\omega$  [MeV]', ylim=(0, 1.15)); ax.set_title(lab, fontsize=10.5); polish(ax)
        axs[0].set_ylabel('response / maximum'); axs[-1].legend(fontsize=9.5, loc='upper right')
    fig.suptitle(f'|q| = {q*1e3:.0f} MeV — the potential acts on the kinematics, so every response moves the same way', y=1.02, fontsize=12.5, fontweight='bold')
    plt.show()

def show_F():
    """Panel F: what the potential does to the response."""
    w_qF = W.FloatSlider(500, min=250, max=1100, step=25, description='$|\\mathbf{q}|$ [MeV]', readout_format='.0f', continuous_update=False, layout=W.Layout(width='420px'))
    w_probe = W.ToggleButtons(options=['electron', 'neutrino'], description='probe')
    w_pots = W.SelectMultiple(options=list(POTS), value=list(POTS)[:4], description='potentials', rows=5, layout=W.Layout(width='420px'))
    show_panel(panel_F, W.VBox([W.HBox([w_qF, w_probe]), w_pots]), dict(q=w_qF, probe=w_probe, chosen=w_pots),
               static_calls=[dict(q=500, probe='electron', chosen=list(POTS)[:4]), dict(q=500, probe='neutrino', chosen=[list(POTS)[0], list(POTS)[3]])])
