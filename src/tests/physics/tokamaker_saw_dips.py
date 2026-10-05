'''Investigation of the 1-D sawtooth q reset (saw_reset_1d) on synthetic q profiles with one or more dips.

Cylinder: rho = r/a, A = rho^2, I = A/q, c1 = 1 (dj in the units of I/A). Writes figures and a summary JSON.
Usage: python tokamaker_saw_dips.py [outdir]
'''
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from OpenFUSIONToolkit.TokaMaker.bootstrap import _saw_reset_1d

Q_S, DQ = 1.025, 0.03
rho = np.linspace(0.0, 1.0, 401)
A, c1 = rho**2, np.ones_like(rho)
g = lambda c, w: np.exp(-((rho - c)/w)**2)

CASES = {
    '1_single_axis_dip':   0.9 + 2.5*rho**2,
    '2_two_dips_hump_low': 1.035 + 0.12*rho**2 - 0.12*g(0.0, 0.08) - 0.05*g(0.4, 0.06),
    '3_two_dips_hump_high': 1.06 + 0.6*rho**2 - 0.12*g(0.0, 0.1) - 0.12*g(0.35, 0.06),
    '4_reversed_shear':    1.10 + 0.6*rho**2 - 0.17*g(0.35, 0.08),
}
RULES = {1: 'outermost', 2: 'innermost', 3: 'outermost deeper than ramp', 4: 'depth-weighted blend'}


def run(q, rule=1, ramp=0.01):
    q_new, dj, sc = _saw_reset_1d(rho, q, A/q, A, c1, Q_S, DQ, ramp=ramp, rule=rule)
    dI = (A/q)*(q/q_new - 1.0)
    return q_new, dj, dI, sc


def plot_case(name, q, outdir):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(rho, q, 'k', lw=2, label='q_base')
    out = {}
    for rule, col in zip(RULES, ('C0', 'C1', 'C2', 'C3')):
        q_new, dj, dI, sc = run(q, rule)
        out[rule] = {k: float(v) for k, v in sc.items()}
        out[rule]['q_new_axis'] = float(q_new[0])
        out[rule]['int_dj_dA'] = float(np.trapezoid(dj, A))
        out[rule]['max_abs_dj'] = float(np.max(np.abs(dj)))
        lab = f"rule {rule} ({RULES[rule]}): rho_m={sc['rho_m']:.3f}"
        ax[0].plot(rho, q_new, col, ls='--', label=lab)
        ax[1].plot(rho, dI, col, label=f'rule {rule}')
        ax[2].plot(rho, dj, col, label=f'rule {rule}')
        for a in ax:
            if sc['rho_m'] > 0:
                a.axvline(sc['rho_m'], color=col, ls=':', lw=0.8)
    for a, t in zip(ax, ('q', 'ΔI (enclosed)', 'Δj = dΔI/dA')):
        a.axhline(0, color='0.7', lw=0.5); a.set_xlabel('rho_tor_norm'); a.set_title(t); a.set_xlim(0, 0.8)
    ax[0].axhline(Q_S, color='r', lw=0.8, label='q_s'); ax[0].axhline(Q_S + DQ, color='r', ls=':', lw=0.8, label='q_s+dq')
    ax[0].set_ylim(0.85, 1.4); ax[0].legend(fontsize=7)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(os.path.join(outdir, f'case_{name}.png'), dpi=110)
    plt.close(fig)
    return out


SHAPES = {   # background + inner dip, and the outer dip (center, width); hump below / above q_s + dq
    'hump_low': (lambda: 1.035 + 0.12*rho**2 - 0.12*g(0.0, 0.08), 0.4, 0.06, np.linspace(0.02, 0.06, 81)),
    'hump_high': (lambda: 1.06 + 0.6*rho**2 - 0.12*g(0.0, 0.1), 0.35, 0.06, np.linspace(0.095, 0.15, 111)),
}


def sweep_marginal(outdir, shape='hump_low'):
    '''Outer dip depth swept through q_s, inner dip fixed'''
    bg, c, wd, depths = SHAPES[shape]
    res = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (rule, ramp), col in zip(((1, 0.0), (1, 0.01), (3, 0.01), (4, 0.01)), ('C0', 'C1', 'C2', 'C3')):
        rm, nj, qmin = [], [], []
        for d in depths:
            q = bg() - d*g(c, wd)
            q_new, dj, dI, sc = run(q, rule, ramp)
            rm.append(sc['rho_m']); nj.append(np.sqrt(np.trapezoid(dj**2, A))); qmin.append(q[rho > c - 2*wd].min())
        qmin = np.array(qmin)
        lab = f'rule {rule}, ramp {ramp}'
        ax[0].plot(Q_S - qmin, rm, col, marker='.', label=lab)
        ax[1].plot(Q_S - qmin, nj, col, marker='.', label=lab)
        res[lab] = {'outer_depth': (Q_S - qmin).tolist(), 'rho_m': rm, 'norm_dj': nj}
    for a, t in zip(ax[:2], ('rho_m', '||Δj|| (L2 over A)')):
        a.axvline(0, color='r', lw=0.8); a.set_xlabel('outer dip depth below q_s'); a.set_title(t); a.legend(fontsize=7)
    for d, ls in ((depths[len(depths)//2 - 8], '-'), (depths[len(depths)//2 + 8], '--')):
        q = bg() - d*g(c, wd)
        ax[2].plot(rho, q, 'k', ls=ls, label=f'q_base, outer depth {Q_S - q[rho > c - 2*wd].min():+.3f}')
        ax[2].plot(rho, run(q, 1, 0.01)[0], 'C1', ls=ls, label='q_new rule 1, ramp 0.01')
    ax[2].axhline(Q_S, color='r', lw=0.8); ax[2].set_xlim(0, 0.8); ax[2].legend(fontsize=7); ax[2].set_title('q, either side of the jump')
    fig.suptitle(f'Marginal outer dip, {shape} (case 5/6)'); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'sweep_marginal_outer_dip_{shape}.png'), dpi=110); plt.close(fig)
    return res


def sweep_trigger(outdir):
    '''Single dip depth swept through q_s: ramp 0 (hard trigger) vs 0.01'''
    q0s = np.linspace(Q_S - 0.03, Q_S + 0.01, 81)
    res = {}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for ramp, col in ((0.0, 'C0'), (0.01, 'C1')):
        nj = []
        for q0 in q0s:
            q = q0 + 2.5*rho**2
            nj.append(np.sqrt(np.trapezoid(run(q, 1, ramp)[1]**2, A)))
        ax.plot(Q_S - q0s, nj, col, marker='.', label=f'ramp {ramp}')
        res[f'ramp {ramp}'] = {'depth': (Q_S - q0s).tolist(), 'norm_dj': nj}
    ax.axvline(0, color='r', lw=0.8); ax.set_xlabel('q_s - q0'); ax.set_title('||Δj|| vs single-dip depth'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'sweep_trigger.png'), dpi=110); plt.close(fig)
    return res


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'saw_dips_out'
    os.makedirs(outdir, exist_ok=True)
    summary = {name: plot_case(name, q, outdir) for name, q in CASES.items()}
    summary['marginal_outer_dip'] = {sh: sweep_marginal(outdir, sh) for sh in SHAPES}
    summary['trigger'] = sweep_trigger(outdir)
    with open(os.path.join(outdir, 'summary.json'), 'w') as fid:
        json.dump(summary, fid, indent=1)
    for name in CASES:
        print(name)
        for rule, v in summary[name].items():
            print(f"  rule {rule}: n_dips {v['n_dips']:.0f} rho_s {v['rho_s']:.3f} rho_m {v['rho_m']:.3f} "
                  f"w {v['w']:.2f} q_new(0) {v['q_new_axis']:.3f} int(dj dA) {v['int_dj_dA']:+.2e} max|dj| {v['max_abs_dj']:.3e}")
