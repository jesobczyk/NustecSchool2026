"""Spherical Skyrme Hartree-Fock solver (core physics, no plotting).

Nucleons in a spherical, self-consistent mean field built from a Skyrme energy density functional.
Protons and neutrons are treated separately; Coulomb (direct + Slater exchange) acts on protons.
Orbitals are labelled (n, l, j) and filled from the bottom (equal filling of the last open level).
Units: MeV and fm.

The single-particle equation for u(r) = r R(r), with B_q(r) = hbar^2 / 2 m*_q(r):
    -(B u')' + [ B l(l+1)/r^2 + B'/r + U_q + (W_q / r) (j(j+1) - l(l+1) - 3/4) ] u = e u
is solved on a radial grid as a symmetric tridiagonal eigenvalue problem.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn

HBARC = 197.327          # MeV fm
E2 = 1.43996             # e^2 in MeV fm
H2M = 20.73553           # hbar^2 / 2m [MeV fm^2] (average nucleon mass, as in SLy4)

# ----------------------------------------------------------------------------- parameter sets
PARAMS = {
    # E. Chabanat et al., Nucl. Phys. A 635, 231 (1998)
    'SLy4': dict(t0=-2488.91, t1=486.82, t2=-546.39, t3=13777.0, x0=0.834, x1=-0.344, x2=-1.0, x3=1.354, alpha=1/6, W0=123.0),
    # J. Bartel et al., Nucl. Phys. A 386, 79 (1982)
    'SkM*': dict(t0=-2645.0, t1=410.0, t2=-135.0, t3=15595.0, x0=0.09, x1=0.0, x2=0.0, x3=0.0, alpha=1/6, W0=130.0),
}

# ----------------------------------------------------------------------------- nuclei and data
NUCLEI = {'12C': (6, 6), '16O': (8, 8), '40Ca': (20, 20), '48Ca': (20, 28), '208Pb': (82, 126)}
# experimental binding energies [MeV] (atomic mass evaluation) and charge radii [fm] (Angeli & Marinova 2013)
EXP = {'12C': (-92.16, 2.470), '16O': (-127.62, 2.699), '40Ca': (-342.05, 3.478), '48Ca': (-415.99, 3.477), '208Pb': (-1636.43, 5.501)}

L_NAMES = 'spdfghijk'
def label(n, l, j): return f'{n}{L_NAMES[l]}{int(2*j)}/2'

# ----------------------------------------------------------------------------- grid helpers
class Grid:
    def __init__(self, rmax=15.0, h=0.1):
        self.h = h; self.r = h*np.arange(1, int(round(rmax/h)) + 1); self.n = len(self.r)
    def d(self, f): return np.gradient(f, self.h, edge_order=2)
    def lap(self, f): return self.d(self.d(f)) + 2*self.d(f)/self.r
    def integrate(self, f): return 4*np.pi*np.sum(f*self.r**2)*self.h       # volume integral of a radial function

# ----------------------------------------------------------------------------- one species: solve the radial equation
def solve_levels(g, B, U, W, lmax, nlev=6):
    """All bound levels (e < 0) for l = 0..lmax, both j. Returns a list of dicts sorted by energy."""
    r, h = g.r, g.h
    dB = g.d(B)
    Bh = 0.5*(B[1:] + B[:-1])                           # B at r_{i+1/2}
    Bm = np.r_[B[0], Bh]                                # B at r_{i-1/2} (B(h/2) ~ B(h))
    Bp = np.r_[Bh, B[-1]]
    out = []
    for l in range(lmax + 1):
        for j in ([0.5] if l == 0 else [l - 0.5, l + 0.5]):
            so = j*(j + 1) - l*(l + 1) - 0.75
            V = B*l*(l + 1)/r**2 + dB/r + U + W/r*so
            diag = (Bp + Bm)/h**2 + V
            off = -Bh/h**2
            e, vec = eigh_tridiagonal(diag, off, select='i', select_range=(0, nlev - 1))
            for k, (ek, uk) in enumerate(zip(e, vec.T)):
                if ek >= 0: break
                uk = uk/np.sqrt(np.sum(uk**2)*h)
                if uk[np.argmax(np.abs(uk[:40]))] < 0: uk = -uk           # positive near the origin
                out.append(dict(n=k + 1, l=l, j=j, e=ek, u=uk, label=label(k + 1, l, j)))
    return sorted(out, key=lambda o: o['e'])

def fill(levels, N):
    """Fill N nucleons from the bottom; the last level may be partially filled (equal filling)."""
    left = N
    for o in levels:
        deg = int(2*o['j'] + 1); o['occ'] = min(deg, left)/deg if left > 0 else 0.0; left -= min(deg, left)
    if left > 0: raise RuntimeError('not enough bound levels to place all nucleons')
    return levels

def densities(g, levels):
    """rho, tau, J (spin-orbit density) of one species from its occupied orbitals."""
    r = g.r; rho = np.zeros(g.n); tau = np.zeros(g.n); J = np.zeros(g.n)
    for o in levels:
        if o['occ'] <= 0: continue
        w = (2*o['j'] + 1)*o['occ']/(4*np.pi); u = o['u']; l, j = o['l'], o['j']
        R = u/r; dR = g.d(u)/r - u/r**2
        rho += w*R**2
        tau += w*(dR**2 + l*(l + 1)*R**2/r**2)
        J += w*(j*(j + 1) - l*(l + 1) - 0.75)*u**2/r**3
    return rho, tau, J

# ----------------------------------------------------------------------------- the functional
def coulomb_direct(g, rho_p):
    """Direct Coulomb potential of the proton density (cumulative trapezoid from the origin)."""
    from scipy.integrate import cumulative_trapezoid
    r = np.r_[0.0, g.r]; f = np.r_[rho_p[0], rho_p]
    inner = cumulative_trapezoid(f*r**2, r, initial=0)[1:]
    tot = cumulative_trapezoid(f*r, r, initial=0)
    outer = (tot[-1] - tot)[1:]
    return 4*np.pi*E2*(inner/g.r + outer)

def fields(g, P, rho, tau, J, A, coulomb=True):
    """Mean fields B_q, U_q, W_q for q = n (0), p (1)."""
    t0, t1, t2, t3, x0, x1, x2, x3, a, W0 = (P[k] for k in ('t0', 't1', 't2', 't3', 'x0', 'x1', 'x2', 'x3', 'alpha', 'W0'))
    rt, tt, Jt = rho[0] + rho[1], tau[0] + tau[1], J[0] + J[1]
    rts = np.maximum(rt, 1e-12)
    h2m = H2M*(1 - 1/A)                                 # one-body centre-of-mass correction
    c1 = (t1*(2 + x1) + t2*(2 + x2))/8; c2 = (t1*(2*x1 + 1) - t2*(2*x2 + 1))/8
    c3 = (3*t1*(2 + x1) - t2*(2 + x2))/16; c4 = (3*t1*(2*x1 + 1) + t2*(2*x2 + 1))/16
    divJt = g.d(g.r**2*Jt)/g.r**2
    B, U, Wf = [], [], []
    for q in (0, 1):
        divJq = g.d(g.r**2*J[q])/g.r**2
        Uq = (t0*((1 + x0/2)*rt - (x0 + 0.5)*rho[q])
              + t3/12*((1 + x3/2)*(2 + a)*rts**(a + 1) - (x3 + 0.5)*(2*rts**a*rho[q] + a*rts**(a - 1)*(rho[0]**2 + rho[1]**2)))
              + c1*tt - c2*tau[q]
              - c3*g.lap(rt) + c4*g.lap(rho[q])
              - W0/2*(divJt + divJq))
        if q == 1 and coulomb:
            Uq = Uq + coulomb_direct(g, rho[1]) - E2*(3/np.pi)**(1/3)*np.maximum(rho[1], 0)**(1/3)
        B.append(h2m + c1*rt - c2*rho[q]); U.append(Uq); Wf.append(W0/2*(g.d(rt) + g.d(rho[q])))
    return B, U, Wf

def energy(g, P, rho, tau, J, A, coulomb=True):
    """Total energy and its parts [MeV]."""
    t0, t1, t2, t3, x0, x1, x2, x3, a, W0 = (P[k] for k in ('t0', 't1', 't2', 't3', 'x0', 'x1', 'x2', 'x3', 'alpha', 'W0'))
    rt, tt, Jt = rho[0] + rho[1], tau[0] + tau[1], J[0] + J[1]; rts = np.maximum(rt, 1e-12)
    s2 = rho[0]**2 + rho[1]**2
    c1 = (t1*(2 + x1) + t2*(2 + x2))/8; c2 = (t1*(2*x1 + 1) - t2*(2*x2 + 1))/8
    d1 = (3*t1*(2 + x1) - t2*(2 + x2))/32; d2 = (3*t1*(2*x1 + 1) + t2*(2*x2 + 1))/32
    parts = dict(
        kinetic=H2M*(1 - 1/A)*tt,
        t0=t0/2*((1 + x0/2)*rt**2 - (x0 + 0.5)*s2),
        t3=t3/12*rts**a*((1 + x3/2)*rt**2 - (x3 + 0.5)*s2),
        effective_mass=c1*rt*tt - c2*(rho[0]*tau[0] + rho[1]*tau[1]),
        surface=d1*g.d(rt)**2 - d2*(g.d(rho[0])**2 + g.d(rho[1])**2),
        spin_orbit=W0/2*(Jt*g.d(rt) + J[0]*g.d(rho[0]) + J[1]*g.d(rho[1])),
    )
    if coulomb:
        parts['coulomb'] = 0.5*rho[1]*coulomb_direct(g, rho[1]) - 0.75*E2*(3/np.pi)**(1/3)*np.maximum(rho[1], 0)**(4/3)
    E = {k: g.integrate(v) for k, v in parts.items()}
    E['total'] = sum(E.values())
    return E

# ----------------------------------------------------------------------------- Woods-Saxon start (and comparison)
def woods_saxon(r, A, V0=52.0, r0=1.27, a=0.67):
    return -V0/(1 + np.exp((r - r0*A**(1/3))/a))

# ----------------------------------------------------------------------------- the self-consistent loop
def run_hf(nucleus='16O', params='SLy4', t3_scale=1.0, W0_scale=1.0, mix=0.4, maxiter=300, tol=1e-6,
           rmax=None, h=0.05, coulomb=True, verbose=False):
    """Hartree-Fock for one nucleus. Returns a dict with the levels, densities, fields, energies and the history."""
    Z, N = NUCLEI[nucleus] if isinstance(nucleus, str) else nucleus
    A = Z + N
    P = dict(PARAMS[params] if isinstance(params, str) else params); P['t3'] *= t3_scale; P['W0'] *= W0_scale
    rmax = rmax or max(12.0, 1.2*A**(1/3) + 9)
    g = Grid(rmax, h); lmax = 2 + int(1.2*A**(1/3)) if A > 40 else 4
    # start: Woods-Saxon orbitals, bare mass
    V = woods_saxon(g.r, A); B0 = np.full(g.n, H2M); W0f = np.zeros(g.n)
    lev = [fill(solve_levels(g, B0, V, W0f, lmax), N), fill(solve_levels(g, B0, V, W0f, lmax), Z)]
    dens = [densities(g, lev[0]), densities(g, lev[1])]
    rho = [dens[0][0], dens[1][0]]; tau = [dens[0][1], dens[1][1]]; J = [dens[0][2], dens[1][2]]
    hist = []; status = 'not converged'; Eold = 0.0
    for it in range(1, maxiter + 1):
        B, U, Wf = fields(g, P, rho, tau, J, A, coulomb)
        if min(B[0].min(), B[1].min()) <= 0:
            status = 'collapsed (effective mass became negative)'; break
        try:
            lev = [fill(solve_levels(g, B[0], U[0], Wf[0], lmax), N), fill(solve_levels(g, B[1], U[1], Wf[1], lmax), Z)]
        except RuntimeError as ex:
            status = f'stopped: {ex}'; break
        new = [densities(g, lev[0]), densities(g, lev[1])]
        rho = [(1 - mix)*rho[q] + mix*new[q][0] for q in (0, 1)]
        tau = [(1 - mix)*tau[q] + mix*new[q][1] for q in (0, 1)]
        J = [(1 - mix)*J[q] + mix*new[q][2] for q in (0, 1)]
        E = energy(g, P, rho, tau, J, A, coulomb)
        rms = np.sqrt(g.integrate((rho[0] + rho[1])*g.r**2)/A)
        hist.append(dict(it=it, E=E['total'], rms=rms, rho0=(rho[0] + rho[1])[:5].mean(),
                         e_n=[o['e'] for o in lev[0] if o['occ'] > 0], e_p=[o['e'] for o in lev[1] if o['occ'] > 0]))
        if verbose and it % 10 == 0: print(it, E['total'], rms)
        if (rho[0] + rho[1]).max() > 2.0 or rms < 0.4*1.2*A**(1/3)*np.sqrt(3/5):
            status = 'collapsed (the density keeps growing)'; break
        if abs(E['total'] - Eold) < tol and it > 5: status = 'converged'; break
        Eold = E['total']
    B, U, Wf = fields(g, P, rho, tau, J, A, coulomb)
    Ef = energy(g, P, rho, tau, J, A, coulomb)
    rp2 = g.integrate(rho[1]*g.r**2)/Z; rn2 = g.integrate(rho[0]*g.r**2)/N
    rch = np.sqrt(max(rp2 + 0.7071 - 0.1161*N/Z, 0))    # proton and neutron charge radii
    return dict(nucleus=nucleus, Z=Z, N=N, A=A, params=P, grid=g, r=g.r, levels=dict(n=lev[0], p=lev[1]),
                rho=dict(n=rho[0], p=rho[1]), tau=dict(n=tau[0], p=tau[1]), J=dict(n=J[0], p=J[1]),
                B=dict(n=B[0], p=B[1]), U=dict(n=U[0], p=U[1]), W=dict(n=Wf[0], p=Wf[1]),
                energy=Ef, history=hist, status=status, iterations=len(hist),
                radii=dict(proton=np.sqrt(rp2), neutron=np.sqrt(rn2), charge=rch),
                mstar=dict(n=H2M*(1 - 1/A)/B[0], p=H2M*(1 - 1/A)/B[1]))

# ----------------------------------------------------------------------------- after HF: momentum space and continuum
def momentum_distribution(res, q='p', p=None):
    """n(p) [MeV^-3] of species q from the occupied orbitals (normalised to the number of nucleons), and per orbital."""
    p = np.linspace(0, 600, 301) if p is None else p
    g = res['grid']; k = p/HBARC
    tot = np.zeros_like(p); per = {}
    for o in res['levels'][q]:
        if o['occ'] <= 0: continue
        phi = np.sqrt(2/np.pi)*np.array([np.sum(g.r*o['u']*spherical_jn(o['l'], kk*g.r))*g.h for kk in k])   # fm^{3/2}
        n_orb = (2*o['j'] + 1)*o['occ']*phi**2/(4*np.pi)/HBARC**3
        per[o['label']] = n_orb; tot += n_orb
    return p, tot, per

def scattering_wave(res, q='n', l=0, j=0.5, T=50.0, rmax=None):
    """Continuum solution at kinetic energy T [MeV] in the same HF field, and the free wave r j_l(kr).
    Both are normalised to unit asymptotic amplitude. Returns r, u_HF, u_free, phase shift [deg]."""
    g = res['grid']; r = g.r; rmax = rmax or r[-1]
    B, U, Wf = res['B'][q], res['U'][q], res['W'][q]
    so = j*(j + 1) - l*(l + 1) - 0.75; dB = g.d(B)
    Bf = lambda x: np.interp(x, r, B); Vf = lambda x: np.interp(x, r, B*l*(l + 1)/r**2 + dB/r + U + Wf/r*so)
    h2m = H2M*(1 - 1/res['A']); k = np.sqrt(T/h2m)
    def rhs(x, y): return [y[1]/Bf(x), (Vf(x) - T)*y[0]]
    x0 = r[0]; y0 = [x0**(l + 1), Bf(x0)*(l + 1)*x0**l]
    sol = solve_ivp(rhs, (x0, rmax), y0, t_eval=r[r <= rmax], rtol=1e-8, atol=1e-12)
    u = sol.y[0]; du = sol.y[1]/Bf(sol.t)
    # match at the edge to u = A [cos(d) k r j_l(kr) - sin(d) k r y_l(kr)] -> A sin(kr - l pi/2 + d)
    # (neutral-particle asymptotics; for protons the Coulomb tail is ignored)
    from scipy.special import spherical_yn
    xe = sol.t[-1]; z = k*xe
    fj = z*spherical_jn(l, z); fy = z*spherical_yn(l, z)
    dfj = k*(spherical_jn(l, z) + z*spherical_jn(l, z, derivative=True))
    dfy = k*(spherical_yn(l, z) + z*spherical_yn(l, z, derivative=True))
    a, b = np.linalg.solve(np.array([[fj, -fy], [dfj, -dfy]]), [u[-1], du[-1]])
    amp = np.hypot(a, b); delta = np.degrees(np.arctan2(b, a))
    u_free = k*sol.t*spherical_jn(l, k*sol.t)
    return sol.t, u/amp*np.sign(a if abs(a) > 1e-12 else 1), u_free, delta

