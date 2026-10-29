"""Interactive panels for the Lecture 3 demo: your own Skyrme Hartree-Fock solver.
The physics is in hf_core.py; this file only runs it (with a cache) and draws the figures.
One show_X() function per panel, as in the Lecture 2 notebook."""
import functools
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import ipywidgets as W
from IPython.display import display, Markdown
import hf_core as H

# ---------- visual language (same as the Lecture 2 notebook) ----------
C = dict(data='#2b2b2b', data_light='#9a9a9a', n='#2a78d6', p='#e34948', tot='#2b2b2b', ws='#1baf7a',
         accent='#6250d6', warn='#eb6834', grid='#e6e6e3')
mpl.rcParams.update({
    'figure.dpi': 110, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': 0.7,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False, 'lines.linewidth': 2.0,
    'mathtext.fontset': 'cm',
})
# If sliders show "Error displaying widget", your Jupyter front-end lacks the ipywidgets extension.
# Fix: `pip install -U ipywidgets widgetsnbextension`, restart the server.  Or set USE_WIDGETS = False.
USE_WIDGETS = True

def show_panel(fn, controls, widget_kwargs, static_calls):
    if USE_WIDGETS: display(controls, W.interactive_output(fn, widget_kwargs))
    else:
        for kw in static_calls: fn(**kw)

def polish(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='minor', length=2, color='#999'); ax.set_axisbelow(True)

SPECIES = {'n': 'neutrons', 'p': 'protons'}
MOUGEY = {'1s1/2': (38.1, 1.0), '1p3/2': (17.5, 0.4)}     # 12C proton removal energies, centroids [MeV]

@functools.lru_cache(maxsize=64)
def hf(nucleus, params='SLy4', t3_scale=1.0, W0_scale=1.0):
    """Cached Hartree-Fock run (the solver takes about a second)."""
    return H.run_hf(nucleus, params, t3_scale=round(t3_scale, 3), W0_scale=round(W0_scale, 3))

def summary(res):
    E = res['energy']['total']; Eexp, rexp = H.EXP.get(res['nucleus'], (np.nan, np.nan))
    return (f"**{res['nucleus']}, {res['status']}** after {res['iterations']} iterations  \n"
            f"binding energy {E:.1f} MeV (experiment {Eexp:.1f}),  E/A = {E/res['A']:.2f} MeV  \n"
            f"charge radius {res['radii']['charge']:.2f} fm (experiment {rexp:.2f}),  "
            f"effective mass in the centre m*/m = {res['mstar']['n'][0]:.2f}")

# ----------------------------------------------------------------------------- widgets shared by the panels
w_nuc = W.Dropdown(options=list(H.NUCLEI), value='16O', description='nucleus')
w_par = W.Dropdown(options=list(H.PARAMS), value='SLy4', description='Skyrme set')
w_t3 = W.FloatSlider(1.0, min=0.0, max=1.2, step=0.05, description='$t_3$ × ', readout_format='.2f', layout=W.Layout(width='320px'))
w_W0 = W.FloatSlider(1.0, min=0.0, max=1.5, step=0.05, description='$W_0$ × ', readout_format='.2f', layout=W.Layout(width='320px'))
CONTROLS = lambda: W.VBox([W.HBox([w_nuc, w_par]), W.HBox([w_t3, w_W0])])
KW = lambda: dict(nucleus=w_nuc, params=w_par, t3_scale=w_t3, W0_scale=w_W0)

# ----------------------------------------------------------------------------- Panel A: the loop
def panel_A(nucleus, params, t3_scale, W0_scale):
    res = hf(nucleus, params, t3_scale, W0_scale); h = res['history']
    it = [x['it'] for x in h]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]; ax.plot(it, [x['E'] for x in h], 'o-', ms=3, color=C['accent'])
    Eexp = H.EXP.get(nucleus, (np.nan,))[0]; ax.axhline(Eexp, color=C['data_light'], ls='--', lw=1.2); ax.text(it[-1], Eexp, ' experiment', va='bottom', ha='right', color='#777', fontsize=9)
    ax.set(xlabel='iteration', ylabel='total energy [MeV]', title='energy along the loop'); polish(ax)
    ax = axs[1]; ax.plot(it, [x['rms'] for x in h], 'o-', ms=3, color=C['tot'])
    ax.set(xlabel='iteration', ylabel='rms radius of the matter [fm]', title='size along the loop'); polish(ax)
    ax = axs[2]
    for q, key in (('n', 'e_n'), ('p', 'e_p')):
        arr = np.array([x[key] for x in h if len(x[key]) == len(h[-1][key])])
        if arr.size: ax.plot(np.arange(1, len(arr) + 1) + (len(h) - len(arr)), arr, color=C[q], lw=1.4)
    ax.plot([], [], color=C['n'], label='neutron orbitals'); ax.plot([], [], color=C['p'], label='proton orbitals')
    ax.set(xlabel='iteration', ylabel='orbital energy [MeV]', title='occupied orbital energies'); ax.legend(fontsize=9); polish(ax)
    plt.show(); display(Markdown(summary(res)))

def show_A():
    show_panel(panel_A, CONTROLS(), KW(), static_calls=[dict(nucleus='16O', params='SLy4', t3_scale=1.0, W0_scale=1.0), dict(nucleus='16O', params='SLy4', t3_scale=0.0, W0_scale=1.0)])

# ----------------------------------------------------------------------------- Panel B: densities and fields
def panel_B(nucleus, params, t3_scale, W0_scale):
    res = hf(nucleus, params, t3_scale, W0_scale); r = res['r']
    if res['status'] != 'converged': display(Markdown(f"**{res['status']}** — the figures show the last iteration."))
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    ax.plot(r, res['rho']['n'] + res['rho']['p'], color=C['tot'], label='all nucleons')
    ax.plot(r, res['rho']['n'], color=C['n'], lw=1.5, label='neutrons'); ax.plot(r, res['rho']['p'], color=C['p'], lw=1.5, label='protons')
    ax.axhline(0.16, color=C['data_light'], ls='--', lw=1); ax.text(r[-1]*0.7, 0.163, r'$\rho_0$ = 0.16 fm$^{-3}$', color='#777', fontsize=9)
    ax.set(xlabel='r [fm]', ylabel=r'density [fm$^{-3}$]', xlim=(0, 1.2*res['A']**(1/3) + 4), title='densities'); ax.legend(fontsize=9); polish(ax)
    ax = axs[1]
    for q in ('n', 'p'): ax.plot(r, res['U'][q], color=C[q], label=f'U ({SPECIES[q]})')
    ax.plot(r, H.woods_saxon(r, res['A']), color=C['ws'], ls='--', lw=1.5, label='Woods–Saxon guess (start)')
    ax.axhline(0, color='#999', lw=0.8)
    ax.set(xlabel='r [fm]', ylabel='U(r) [MeV]', xlim=(0, 1.2*res['A']**(1/3) + 4), ylim=(-110, 30), title='the mean field it makes'); ax.legend(fontsize=9); polish(ax)
    ax = axs[2]
    for q in ('n', 'p'): ax.plot(r, res['mstar'][q], color=C[q], label=SPECIES[q])
    ax.axhline(1, color='#999', lw=0.8)
    ax.set(xlabel='r [fm]', ylabel='m*/m', xlim=(0, 1.2*res['A']**(1/3) + 4), ylim=(0.4, 1.1), title='effective mass'); ax.legend(fontsize=9); polish(ax)
    plt.show(); display(Markdown(summary(res)))

def show_B():
    show_panel(panel_B, CONTROLS(), KW(), static_calls=[dict(nucleus='40Ca', params='SLy4', t3_scale=1.0, W0_scale=1.0)])

# ----------------------------------------------------------------------------- Panel C: levels
def panel_C(nucleus, params, t3_scale, W0_scale):
    res = hf(nucleus, params, t3_scale, W0_scale)
    if res['status'] != 'converged': display(Markdown(f"**{res['status']}** — the levels are from the last iteration."))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    emin = min(o['e'] for q in 'np' for o in res['levels'][q])
    for ax, q in zip(axs, ('n', 'p')):
        lv = [o for o in res['levels'][q] if o['e'] > emin - 1]
        count = 0; ylab = -1e9
        for o in lv:
            occ = o['occ'] > 0; deg = int(2*o['j'] + 1)
            ax.plot([0.1, 0.6], [o['e']]*2, color=C[q], lw=3 if occ else 1.5, ls='-' if occ else '--')
            yl = max(o['e'], ylab + 1.6); ylab = yl
            ax.text(0.63, yl, f"{o['label']}  ({deg})" + ('' if occ else '  empty'), va='center', fontsize=10, color='#222' if occ else '#888')
        if nucleus == '12C' and q == 'p':
            for name, (e, de) in MOUGEY.items():
                ax.errorbar([1.15], [-e], [de], fmt='o', ms=7, color=C['data'], capsize=0); ax.text(1.2, -e, f'(e,e′p) {name}: {e} MeV', va='center', fontsize=9)
        ax.set_title(SPECIES[q]); ax.set_xlim(0, 1.7); ax.set_xticks([]); ax.grid(False); ax.spines['bottom'].set_visible(False)
    axs[0].set_ylabel('orbital energy [MeV]'); axs[0].set_ylim(emin - 3, 2)
    plt.show(); display(Markdown(summary(res)))

def show_C():
    show_panel(panel_C, CONTROLS(), KW(), static_calls=[dict(nucleus='40Ca', params='SLy4', t3_scale=1.0, W0_scale=1.0), dict(nucleus='40Ca', params='SLy4', t3_scale=1.0, W0_scale=0.0)])

# ----------------------------------------------------------------------------- Panel D: momentum space and the spectral function
w_q = W.ToggleButtons(options=[('protons', 'p'), ('neutrons', 'n')], value='p', description='species')
w_G = W.FloatSlider(8, min=1, max=25, step=1, description='width Γ [MeV]', readout_format='.0f', layout=W.Layout(width='320px'))

def lfg_momentum(res, q, p):
    """Local Fermi gas from the same HF density: n(k) = (1/4 pi^3) int d^3r theta(k_F(r) - k)."""
    g = res['grid']; rf = np.linspace(1e-3, g.r[-1], 6000); kF = (3*np.pi**2*np.maximum(np.interp(rf, g.r, res['rho'][q]), 0))**(1/3); k = p/H.HBARC
    dr = rf[1] - rf[0]; nk = np.array([4*np.pi*np.sum(rf**2*(kF > kk))*dr for kk in k])/(4*np.pi**3)
    return nk/H.HBARC**3

def panel_D(nucleus, params, species, width):
    res = hf(nucleus, params, 1.0, 1.0); q = species
    p, ntot, per = H.momentum_distribution(res, q)
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1, 1.15]))
    ax = axs[0]
    for lab, nn in per.items(): ax.plot(p, 4*np.pi*p**2*nn*1e3, lw=1.6, label=lab)
    ax.set(xlabel='p [MeV]', ylabel=r'$4\pi p^2 n(p)$ [$10^{-3}$ MeV$^{-1}$]', xlim=(0, 500), title=f'orbitals in momentum space ({SPECIES[q]})'); ax.legend(fontsize=8.5); polish(ax)
    ax = axs[1]; nl = lfg_momentum(res, q, p)
    ax.plot(p, 4*np.pi*p**2*ntot*1e3, color=C[q], label='Hartree–Fock orbitals'); ax.plot(p, 4*np.pi*p**2*nl*1e3, color=C['ws'], ls='--', label='local Fermi gas, same density')
    ax.set(xlabel='p [MeV]', ylabel=r'$4\pi p^2 n(p)$ [$10^{-3}$ MeV$^{-1}$]', xlim=(0, 500), title='same density, two momentum distributions'); ax.legend(fontsize=9); polish(ax)
    ax = axs[2]; E = np.linspace(0, 80, 161); Pm = np.zeros((len(p), len(E)))
    occ = [o for o in res['levels'][q] if o['occ'] > 0]
    for o in occ:
        Lz = (width/2/np.pi)/((E + o['e'])**2 + (width/2)**2); Pm += np.outer(per[o['label']], Lz)
    Z = 4*np.pi*p[:, None]**2*Pm
    ax.pcolormesh(p, E, Z.T, cmap='Purples', shading='auto', norm=mpl.colors.PowerNorm(gamma=0.4, vmin=0, vmax=Z.max()))
    for o in occ: ax.text(480, -o['e'], o['label'], va='center', ha='right', fontsize=9, color=C['accent'])
    ax.invert_yaxis(); ax.grid(False)
    ax.set(xlabel='p [MeV]', ylabel='removal energy E [MeV]', xlim=(0, 500), title='spectral function: orbitals (stripes)')
    plt.show()

def show_D():
    show_panel(panel_D, W.VBox([W.HBox([w_nuc, w_par]), W.HBox([w_q, w_G])]), dict(nucleus=w_nuc, params=w_par, species=w_q, width=w_G),
               static_calls=[dict(nucleus='16O', params='SLy4', species='p', width=8)])

# ----------------------------------------------------------------------------- Panel E: a nucleon leaving through the same field
w_l = W.IntSlider(0, min=0, max=4, description='ℓ', layout=W.Layout(width='240px'))
w_T = W.FloatSlider(50, min=5, max=300, step=5, description='T [MeV]', readout_format='.0f', layout=W.Layout(width='320px'))

def panel_E(nucleus, params, l, T):
    res = hf(nucleus, params, 1.0, 1.0); q = 'n'; j = l + 0.5
    x, u, uf, delta = H.scattering_wave(res, q, l, j, T)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw=dict(wspace=0.25, width_ratios=[1, 1.5]))
    ax = axs[0]; r = res['r']
    ax.plot(r, res['U'][q], color=C['n'], label='U(r), neutrons'); ax.axhline(T, color=C['accent'], ls='--', lw=1.2, label=f'outgoing energy T = {T:.0f} MeV')
    ax.axhline(0, color='#999', lw=0.8); ax.set(xlabel='r [fm]', ylabel='MeV', xlim=(0, 12), title='the same field binds and distorts'); ax.legend(fontsize=9); polish(ax)
    ax = axs[1]
    ax.plot(x, uf, color=C['data_light'], lw=1.6, ls='--', label='plane wave (no potential)')
    ax.plot(x, u, color=C['n'], label='in the Hartree–Fock field')
    ax.axvspan(0, 1.2*res['A']**(1/3), color='#f3f1fb', lw=0); ax.axhline(0, color='#999', lw=0.8)
    ax.set(xlabel='r [fm]', ylabel='u(r), unit amplitude far away', xlim=(0, x[-1]), title=f'outgoing neutron, ℓ = {l}, j = {j:g}: phase shift {delta:.0f}°')
    ax.legend(fontsize=9, loc='lower right'); polish(ax)
    plt.show()

def show_E():
    show_panel(panel_E, W.VBox([W.HBox([w_nuc, w_par]), W.HBox([w_l, w_T])]), dict(nucleus=w_nuc, params=w_par, l=w_l, T=w_T),
               static_calls=[dict(nucleus='40Ca', params='SLy4', l=0, T=50), dict(nucleus='40Ca', params='SLy4', l=2, T=150)])
