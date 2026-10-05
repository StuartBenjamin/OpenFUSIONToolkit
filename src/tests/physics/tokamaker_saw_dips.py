'''Investigation of the 1-D sawtooth q reset (saw_reset_1d): rule 2 (local) against rule 1 (fuse) on
synthetic q profiles, with a NumPy reference of the local rule (checked node by node against Fortran).

Cylinder: rho = r/a, A = rho^2, I = A/q, c1 = 1 (dj in the units of I/A). Writes figures, summary.json
and crosscheck/saw_local_crosscheck.{json,npz} (profiles and both rules' q_new, for the ida_fuse port).
Usage: python tokamaker_saw_dips.py [outdir]
'''
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from OpenFUSIONToolkit.TokaMaker.bootstrap import _saw_reset_1d

Q_S, DQ, RAMP = 1.025, 0.03, 0.01
QM = Q_S + DQ
rho = np.linspace(0.0, 1.0, 401)
g = lambda c, w, r=rho: np.exp(-((r - c)/w)**2)
RULES = {1: 'fuse', 2: 'local'}


def grad_nonuniform(x, y):
    '''saw_reset_1d's derivative (2nd-order interior, one-sided ends)'''
    d = np.empty_like(y)
    d[0] = (y[1] - y[0])/(x[1] - x[0]); d[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    h1, h2 = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    d[1:-1] = (h1**2*y[2:] - h2**2*y[:-2] + (h2**2 - h1**2)*y[1:-1])/(h1*h2*(h1 + h2))
    return d


def pshape(x, m):
    return (3.0 - 2.0*x)*x**2 + m*(x**3 - x**2) if m <= 3.0 else x**m


def local_ref(r, q, q_s=Q_S, dq=DQ, ramp=RAMP):
    '''NumPy reference of rule 2 (local, hump blend); returns q_new and the regions
    (rho_in, rho_c, rho_out, w, q_min, humps [(rho_h, h, s)])'''
    n, qm = len(q), q_s + dq
    gq = grad_nonuniform(r, q)
    weight = lambda a, b: min(max((q_s - q[a:b+1].min())/ramp, 0.0), 1.0) if ramp > 0 else 1.0
    def centre(a, b, lo, hi):
        ic = a + int(np.argmin(q[a:b+1]))
        if ic == 0: return 0.0
        d0, d2 = r[ic-1] - r[ic], r[ic+1] - r[ic]
        s0, s2 = (q[ic-1] - q[ic])/d0, (q[ic+1] - q[ic])/d2
        cc = (s2 - s0)/(d2 - d0)
        rc = r[ic] - 0.5*(s0 - cc*d0)/cc if cc > 0 else r[ic]
        return min(max(rc, lo), hi)
    def hump(i):
        d0, d2 = r[i-1] - r[i], r[i+1] - r[i]
        s0, s2 = (q[i-1] - q[i])/d0, (q[i+1] - q[i])/d2
        cc = (s2 - s0)/(d2 - d0)
        if cc >= 0: return r[i], q[i]
        bb = s0 - cc*d0
        t = min(max(-0.5*bb/cc, d0), d2)
        return r[i] + t, max(q[i] + bb*t + cc*t**2, q[i])
    def target(x, r_in, d_in, s_in, r_c, r_out, d_out, s_out):
        if x >= r_c:
            if d_out > 0 and r_out > r_c:
                return q_s + d_out*pshape((x - r_c)/(r_out - r_c), max((r_out - r_c)*s_out/d_out, 0.0))
            return q_s
        if d_in > 0:
            return q_s + d_in*pshape((r_c - x)/(r_c - r_in), max(-(r_c - r_in)*s_in/d_in, 0.0))
        return q_s
    q_new, regions, ia = q.copy(), [], None
    for k in range(n):
        if q[k] >= qm: continue
        ia = k if ia is None else ia
        if k == n - 1: break
        if q[k+1] < qm: continue
        ib = k
        if q[ia:ib+1].min() < q_s:
            r_out = r[ib] + (qm - q[ib])/(q[ib+1] - q[ib])*(r[ib+1] - r[ib])
            s_out = gq[ib] + (r_out - r[ib])/(r[ib+1] - r[ib])*(gq[ib+1] - gq[ib])
            if ia == 0: r_in, d_in, s_in = 0.0, max(q[0] - q_s, 0.0), 0.0
            else:
                r_in = r[ia-1] + (qm - q[ia-1])/(q[ia] - q[ia-1])*(r[ia] - r[ia-1])
                s_in = gq[ia-1] + (r_in - r[ia-1])/(r[ia] - r[ia-1])*(gq[ia] - gq[ia-1]); d_in = dq
            r_c, wk = centre(ia, ib, r_in, r_out), weight(ia, ib)
            low = np.flatnonzero(q[ia:ib+1] < q_s) + ia
            runs = np.split(low, np.flatnonzero(np.diff(low) > 1) + 1)
            # segments: ('free',) or ('run', j); humps between consecutive segments: (rho_h, h, s)
            segs, humps = [], []
            # left edge hump: interior local max between region start and run 0
            a0 = runs[0][0]
            if a0 > ia:
                lm = [i for i in range(max(ia, 1), a0) if q[i] >= q[i-1] and q[i] >= q[i+1]]
                if lm:
                    ih = max(lm, key=lambda i: q[i])
                    rh, h = hump(ih)
                    b = max(q_s, q[ia:ih+1].min())
                    s = min(max((h - b)/(qm - b), 0.0), 1.0) if qm > b else 0.0
                    segs.append(('free',)); humps.append((rh, h, s))
            for j in range(len(runs)):
                segs.append(('run', j))
                if j < len(runs) - 1:
                    i = runs[j][-1] + 1 + int(np.argmax(q[runs[j][-1]+1:runs[j+1][0]]))
                    rh, h = hump(i)
                    humps.append((rh, h, min(max((h - q_s)/dq, 0.0), 1.0)))
            ze = runs[-1][-1]
            if ze < ib:
                lm = [i for i in range(ze + 1, ib + 1) if q[i] >= q[i-1] and q[i] >= q[i+1]]
                if lm:
                    ih = max(lm, key=lambda i: q[i])
                    rh, h = hump(ih)
                    b = max(q_s, q[ih:ib+1].min())
                    s = min(max((h - b)/(qm - b), 0.0), 1.0) if qm > b else 0.0
                    segs.append(('free',)); humps.append((rh, h, s))
            ends = [r_in] + [h[0] for h in humps] + [r_out]
            hts = [None] + [max(h[1] - q_s, 0.0) for h in humps] + [None]
            for kk in range(ia, ib + 1):
                qt = target(r[kk], r_in, d_in, s_in, r_c, r_out, dq, s_out)
                if not humps:
                    q_new[kk] = q[kk] + wk*(qt - q[kk]); continue
                m = int(np.sum(np.array(ends[1:-1]) <= r[kk]))
                sk = float(np.interp(r[kk], [h[0] for h in humps], [h[2] for h in humps]))
                if segs[m][0] == 'free':
                    qsp, wj = q[kk], 0.0
                else:
                    j = segs[m][1]
                    dl = d_in if m == 0 else hts[m]; sl = s_in if m == 0 else 0.0
                    dr = dq if m == len(segs) - 1 else hts[m+1]; sr = s_out if m == len(segs) - 1 else 0.0
                    rcj = centre(runs[j][0], runs[j][-1], ends[m], ends[m+1])
                    qsp, wj = target(r[kk], ends[m], dl, sl, rcj, ends[m+1], dr, sr), weight(runs[j][0], runs[j][-1])
                q_new[kk] = q[kk] + (1 - sk)*wk*(qt - q[kk]) + sk*wj*(qsp - q[kk])
            regions.append(dict(rho_in=r_in, rho_c=r_c, rho_out=r_out, w=wk, q_min=float(q[ia:ib+1].min()),
                                humps=[dict(rho_h=float(h[0]), h=float(h[1]), s=float(h[2])) for h in humps]))
        ia = None
    return q_new, regions


REF_ERR = [0.0]   # max |q_new Fortran - reference| over every local call


def run(q, rule, r=rho):
    A = r**2
    q_new, dj, sc = _saw_reset_1d(r, q, A/q, A, np.ones_like(r), Q_S, DQ, ramp=RAMP, rule=rule)
    sc = dict(sc)
    if rule == 2:
        q_ref, regs = local_ref(r, q)
        REF_ERR[0] = max(REF_ERR[0], float(np.max(np.abs(q_new - q_ref))))
        sc['regions'] = regs
        sc['rho_c'] = regs[-1]['rho_c'] if regs else np.nan
    else:
        sc['rho_c'] = 0.0 if sc['rho_m'] > 0 else np.nan
    dI = (A/q)*(q/q_new - 1.0)
    return q_new, dj, dI, sc


CASES = {
    '1_single_axis_dip': 0.9 + 2.5*rho**2,
    '2_two_dips_hump_low': 1.035 + 0.12*rho**2 - 0.12*g(0.0, 0.08) - 0.05*g(0.4, 0.06),
    '3_two_dips_hump_high': 1.06 + 0.6*rho**2 - 0.12*g(0.0, 0.1) - 0.12*g(0.35, 0.06),
    '4_reversed_shear': 1.10 + 0.6*rho**2 - 0.17*g(0.35, 0.08),
    '5_axis_region_q0_between': 1.04 + 0.6*rho**2 - 0.1*g(0.2, 0.08),
    '6_axis_region_q0_below': 1.0 + 0.6*rho**2 - 0.1*g(0.2, 0.08),
    '7_three_dips_mixed': 1.0 + 0.4*rho**2 - 0.05*g(0.0, 0.06) + 0.07*g(0.15, 0.05) - 0.06*g(0.3, 0.05)
                          + 0.03*g(0.42, 0.04) - 0.10*g(0.55, 0.05),
}
RHO_NU = np.linspace(0.0, 1.0, 161)**1.5   # node-dense axis
CASES_NU = {'8_reversed_shear_nonuniform': 1.10 + 0.6*RHO_NU**2 - 0.17*g(0.35, 0.08, RHO_NU)}


def plot_case(name, q, outdir, r=rho):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(r, q, 'k', lw=2, label='q_base')
    out = {}
    for rule, col in ((1, 'C0'), (2, 'C3')):
        q_new, dj, dI, sc = run(q, rule, r)
        out[RULES[rule]] = dict(rho_s=sc['rho_s'], rho_m=sc['rho_m'], rho_out=sc['rho_out'], n_dips=sc['n_dips'],
                                w=sc['w'], q_new_axis=float(q_new[0]), int_dj_dA=float(np.trapezoid(dj, r**2)),
                                int_abs_dj_dA=float(np.trapezoid(np.abs(dj), r**2)), max_abs_dj=float(np.max(np.abs(dj))),
                                regions=sc.get('regions'))
        ax[0].plot(r, q_new, col, ls='--', label=f"{RULES[rule]}: rho_out={sc['rho_out']:.3f}, n={sc['n_dips']}")
        ax[1].plot(r, dI, col, label=RULES[rule])
        ax[2].plot(r, dj, col, label=RULES[rule])
        for reg in sc.get('regions', []):
            ax[0].axvline(reg['rho_c'], color=col, ls=':', lw=0.8)
    for a, t in zip(ax, ('q', 'ΔI (enclosed)', 'Δj = dΔI/dA')):
        a.axhline(0, color='0.7', lw=0.5); a.set_xlabel('rho_tor_norm'); a.set_title(t); a.set_xlim(0, 0.8)
    ax[0].axhline(Q_S, color='r', lw=0.8, label='q_s'); ax[0].axhline(QM, color='r', ls=':', lw=0.8, label='q_s+dq')
    ax[0].axvline(out['fuse']['rho_m'], color='k', ls=':', lw=0.8)
    ax[0].set_ylim(0.85, 1.35); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.suptitle(f'{name}  (red dotted: local rho_c; black dotted: rho_m)'); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'case_{name}.png'), dpi=110); plt.close(fig)
    return out


def _outer_min(q, lo):
    return q[rho > lo].min()


# Sweeps: name -> (q(p), parameter values, x(p, q) for the axis, x label, critical x)
SWEEPS = {
    'outer_dip_hump_high': (lambda d: 1.06 + 0.6*rho**2 - 0.12*g(0.0, 0.1) - d*g(0.35, 0.06),
                            np.linspace(0.095, 0.15, 111), lambda d, q: Q_S - _outer_min(q, 0.23),
                            'outer dip depth q_s - min q_outer', 0.0),
    'outer_dip_hump_low': (lambda d: 1.035 + 0.12*rho**2 - 0.12*g(0.0, 0.08) - d*g(0.4, 0.06),
                           np.linspace(0.02, 0.06, 81), lambda d, q: Q_S - _outer_min(q, 0.28),
                           'outer dip depth q_s - min q_outer', 0.0),
    'q0_through_qs': (lambda q0: q0 + 0.6*rho**2 - 0.12*(g(0.3, 0.2) + g(-0.3, 0.2)), np.linspace(1.0, 1.095, 191),
                      lambda q0, q: q[0] - Q_S, 'q0 - q_s  (reversed shear, q\'(0) = 0; q0 = q_s + dq at 0.03)', 0.0),
    'q0_through_qs_axis_slope': (lambda q0: q0 + 0.6*rho**2 - 0.12*g(0.3, 0.2), np.linspace(0.99, 1.085, 191),
                                 lambda q0, q: q[0] - Q_S, 'q0 - q_s  (reversed shear, q\'(0) = -0.19)', 0.0),
    'q0_through_qs_axis_hump': (lambda q0: q0 + 0.6*rho**2 - 0.12*g(0.3, 0.07), np.linspace(0.99, 1.085, 191),
                                lambda q0, q: q[0] - Q_S, 'q0 - q_s  (q rises off axis, then dips below q_s)', 0.0),
    'hump_through_qm_axis_deeper': (lambda h: 1.01 + 0.4*rho**2 - 0.04*g(0.0, 0.08) - 0.08*g(0.4, 0.06) + h*g(0.2, 0.06),
                                    np.linspace(-0.01, 0.06, 141), lambda h, q: q[(rho > 0.1) & (rho < 0.3)].max() - QM,
                                    'hump max - (q_s + dq)', 0.0),
    'hump_through_qm_outer_deeper': (lambda h: 1.01 + 0.4*rho**2 - 0.02*g(0.0, 0.08) - 0.12*g(0.4, 0.06) + h*g(0.2, 0.06),
                                     np.linspace(-0.01, 0.06, 141), lambda h, q: q[(rho > 0.1) & (rho < 0.3)].max() - QM,
                                     'hump max - (q_s + dq)', 0.0),
    'hump_through_qm_split': (lambda a: 0.995 + a*g(0.3, 0.06) + 0.6*rho**6, np.linspace(0.035, 0.075, 161),
                              lambda a, q: q[(rho > 0.2) & (rho < 0.4)].max() - QM, 'hump max - (q_s + dq)', 0.0),
    'equal_minima_swap': (lambda d2: 1.04 + 0.1*rho**2 - 0.06*g(0.15, 0.05) - d2*g(0.4, 0.05),
                          np.linspace(0.064, 0.084, 101), lambda d2, q: q[rho < 0.27].min() - q[rho > 0.27].min(),
                          'min q_inner - min q_outer', 0.0),
}


def sweep(name, outdir):
    qf, ps, xf, xlab, xc = SWEEPS[name]
    res = {'x': [], 'xlabel': xlab}
    prev = {}
    for rule in RULES.values():
        res[rule] = {k: [] for k in ('norm_dj', 'rho_c', 'rho_out', 'rho_m', 'max_dq', 'n_dips', 'step_q_new', 'step_dj')}
    step_q = []
    for p in ps:
        q = qf(p)
        res['x'].append(float(xf(p, q)))
        if 'q' in prev:
            step_q.append(float(np.max(np.abs(q - prev['q']))))
        prev['q'] = q
        for rule, rn in RULES.items():
            q_new, dj, dI, sc = run(q, rule)
            r = res[rn]
            r['norm_dj'].append(float(np.sqrt(np.trapezoid(dj**2, rho**2))))
            r['rho_c'].append(float(sc['rho_c'])); r['rho_out'].append(float(sc['rho_out'])); r['rho_m'].append(float(sc['rho_m']))
            r['max_dq'].append(float(np.max(np.abs(q_new - q)))); r['n_dips'].append(int(sc['n_dips']))
            if rn in prev:
                r['step_q_new'].append(float(np.max(np.abs(q_new - prev[rn][0]))))
                r['step_dj'].append(float(np.sqrt(np.trapezoid((dj - prev[rn][1])**2, rho**2))))
            prev[rn] = (q_new, dj)
    # Jump measure: largest step of q_new (sup norm) / largest step of q between neighbouring sweep points,
    # and the largest step of dj relative to its median step (a continuous map keeps both O(1))
    for rn in RULES.values():
        r = res[rn]
        r['q_new_step_ratio'] = max(r['step_q_new'])/max(step_q)
        r['dj_step_ratio'] = max(r['step_dj'])/max(np.median(r['step_dj']), 1e-300)
        r['max_step_q_new'] = max(r['step_q_new'])
        r['max_step_q_new_at'] = res['x'][1 + int(np.argmax(r['step_q_new']))]
    res['max_step_q'] = max(step_q)
    x = np.array(res['x'])
    fig, ax = plt.subplots(2, 4, figsize=(19, 8))
    for rn, col, ls in (('fuse', 'C0', '--'), ('local', 'C3', '-')):
        r = res[rn]
        for a, key in zip(ax[0], ('norm_dj', 'rho_c', 'rho_out', 'max_dq')):
            a.plot(x, r[key], col, ls=ls, marker='.', ms=3, label=rn)
        ax[1, 3].semilogy(0.5*(x[1:] + x[:-1]), r['step_q_new'], col, ls=ls, marker='.', ms=3, label=f'{rn}: |Δ q_new|∞')
    ax[1, 3].semilogy(0.5*(x[1:] + x[:-1]), step_q, 'k', lw=0.8, label='|Δ q|∞ (input step)')
    for a, t in zip(ax[0], ('||Δj|| (L2 over A)', 'rho_c (local: outermost region)', 'rho_out', 'max |q_new - q|')):
        a.axvline(xc, color='r', lw=0.8); a.set_xlabel(xlab, fontsize=8); a.set_title(t); a.legend(fontsize=7)
    ax[0, 2].plot(x, res['local']['rho_m'], 'k:', lw=1, label='rho_m (both)'); ax[0, 2].legend(fontsize=7)
    ax[1, 3].axvline(xc, color='r', lw=0.8); ax[1, 3].set_title('step between neighbouring sweep points')
    ax[1, 3].set_xlabel(xlab, fontsize=8); ax[1, 3].legend(fontsize=7)
    ic = int(np.argmin(np.abs(x - xc)))
    picks = sorted({max(ic - 6, 0), min(ic + 6, len(ps) - 1)})
    for i, ls in zip(picks, ('-', '--')):
        q = qf(ps[i])
        ax[1, 0].plot(rho, q, 'k', ls=ls, lw=1.5, label=f'q, x={x[i]:+.4f}')
        for rule, col in ((1, 'C0'), (2, 'C3')):
            q_new, dj, dI, sc = run(q, rule)
            ax[1, 0].plot(rho, q_new, col, ls=ls, label=f'{RULES[rule]}, x={x[i]:+.4f}')
            ax[1, 1].plot(rho, dj, col, ls=ls, label=f'{RULES[rule]}, x={x[i]:+.4f}')
            ax[1, 2].plot(rho, dI, col, ls=ls, label=f'{RULES[rule]}, x={x[i]:+.4f}')
    ax[1, 0].axhline(Q_S, color='r', lw=0.8); ax[1, 0].axhline(QM, color='r', ls=':', lw=0.8); ax[1, 0].set_ylim(0.9, 1.25)
    for a, t in zip(ax[1, :3], ('q / q_new either side of the critical x', 'Δj', 'ΔI')):
        a.set_xlim(0, 0.7); a.set_xlabel('rho_tor_norm'); a.set_title(t); a.legend(fontsize=6)
    fig.suptitle(f'sweep {name}'); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'sweep_{name}.png'), dpi=100); plt.close(fig)
    return res


def crosscheck(outdir):
    '''Profiles and both rules' outputs for the ida_fuse port (node-by-node comparison)'''
    cdir = os.path.join(outdir, 'crosscheck')
    os.makedirs(cdir, exist_ok=True)
    cases = [(k, rho, q) for k, q in CASES.items()] + [(k, RHO_NU, q) for k, q in CASES_NU.items()]
    for name, (qf, ps, xf, _, xc) in SWEEPS.items():   # sweep points either side of the critical value and
        x = np.array([xf(p, qf(p)) for p in ps])      # of local's largest q_new step
        ic = int(np.argmin(np.abs(x - xc)))
        qn = [_saw_reset_1d(rho, qf(p), rho**2/qf(p), rho**2, np.ones_like(rho), Q_S, DQ, ramp=RAMP, rule=2)[0] for p in ps]
        ij = int(np.argmax([np.max(np.abs(qn[i+1] - qn[i])) for i in range(len(ps) - 1)]))
        for i in sorted({max(ic - 1, 0), ic, min(ic + 1, len(ps) - 1), ij, ij + 1}):
            cases.append((f'sweep_{name}_{i}', rho, qf(ps[i])))
    out, npz = {'q_s': Q_S, 'dq': DQ, 'ramp': RAMP, 'note': 'rho axis first; A = rho^2, I = A/q, c1 = 1; '
                'dj = (d dI/dA)/c1 with dI = I*(q/q_new - 1); rule 1 = fuse, 2 = local', 'cases': {}}, {}
    for name, r, q in cases:
        A = r**2
        d = {'rho': r, 'q': q, 'I': A/q, 'A': A, 'c1': np.ones_like(r)}
        for rule, rn in RULES.items():
            q_new, dj, sc = _saw_reset_1d(r, q, A/q, A, np.ones_like(r), Q_S, DQ, ramp=RAMP, rule=rule)
            d[f'q_new_{rn}'], d[f'dj_{rn}'], d[f'wr_{rn}'] = q_new, dj, sc['wr']
            d[f'scal_{rn}'] = {k: float(sc[k]) for k in ('rho_s', 'rho_m', 'rho_out', 'n_dips', 'w')}
        d['regions_local'] = local_ref(r, q)[1]
        out['cases'][name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                npz[f'{name}__{k}'] = v
    with open(os.path.join(cdir, 'saw_local_crosscheck.json'), 'w') as fid:
        json.dump(out, fid)
    np.savez(os.path.join(cdir, 'saw_local_crosscheck.npz'), q_s=Q_S, dq=DQ, ramp=RAMP, **npz)
    return list(out['cases'])


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'saw_dips_out'
    os.makedirs(outdir, exist_ok=True)
    summary = {name: plot_case(name, q, outdir) for name, q in CASES.items()}
    summary.update({name: plot_case(name, q, outdir, RHO_NU) for name, q in CASES_NU.items()})
    summary['sweeps'] = {name: sweep(name, outdir) for name in SWEEPS}
    summary['crosscheck_cases'] = crosscheck(outdir)
    summary['max_abs_q_new_fortran_minus_reference'] = REF_ERR[0]
    with open(os.path.join(outdir, 'summary.json'), 'w') as fid:
        json.dump(summary, fid, indent=1)
    for name in list(CASES) + list(CASES_NU):
        print(name)
        for rn, v in summary[name].items():
            regs = ' '.join(f"[{r['rho_in']:.3f} {r['rho_c']:.3f} {r['rho_out']:.3f} w{r['w']:.2f}]" for r in (v['regions'] or []))
            print(f"  {rn:5s}: n_dips {v['n_dips']:.0f} rho_m {v['rho_m']:.3f} rho_out {v['rho_out']:.3f} q_new(0) {v['q_new_axis']:.4f} "
                  f"int(dj dA)/int|dj|dA {v['int_dj_dA']/max(v['int_abs_dj_dA'], 1e-300):+.1e} {regs}")
    for name, s in summary['sweeps'].items():
        print(f"sweep {name}: max input step {s['max_step_q']:.2e}")
        for rn in RULES.values():
            r = s[rn]
            print(f"  {rn:5s}: max step q_new {r['max_step_q_new']:.2e} at x={r['max_step_q_new_at']:+.4f} "
                  f"(ratio {r['q_new_step_ratio']:.1f}), dj step / median {r['dj_step_ratio']:.1f}")
    print(f"max |q_new Fortran - NumPy reference| (local) = {REF_ERR[0]:.2e}")
