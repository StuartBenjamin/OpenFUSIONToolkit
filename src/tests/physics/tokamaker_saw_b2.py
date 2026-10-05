'''Sawtooth reset in full ITER SWB solves: reversed-shear base, plus axis-peaked fixed current (W-shaped q).

Run from src/tests/physics (ITER_geom.json); writes figures and summary.json to outdir. Per-iteration
[saw] diagnostics go to stdout (diagnose_bs), so redirect it to a log.
Usage: python tokamaker_saw_b2.py outdir [q_s]
'''
import sys, os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_TokaMaker import _ITER_bootstrap_internal_setup

outdir = sys.argv[1] if len(sys.argv) > 1 else 'saw_b2_out'
os.makedirs(outdir, exist_ok=True)
setup = _ITER_bootstrap_internal_setup(1.0, 2)
mygs, x = setup['mygs'], setup['psi_sample']
common = dict(
    ffp_prof={'type': 'jphi-split-bootstrap', 'x': x, 'y': setup['inductive_jphi']},
    te_prof={'type': 'linterp', 'x': x, 'y': setup['Te'] / 1e3},
    ne_prof={'type': 'linterp', 'x': x, 'y': setup['ne']},
    ti_prof={'type': 'linterp', 'x': x, 'y': setup['Ti'] / 1e3},
    ni_prof={'type': 'linterp', 'x': x, 'y': setup['ni']},
    Zeff=setup['Zeff_val'], Ip_target=setup['Ip_target'], scale_jBS=1.0,
)


def solve(label, **kw):
    print(f'\n===== RUN {label} =====', flush=True)
    t0 = time.perf_counter()
    p = mygs.solve_bootstrap(**common, **kw)
    dt = time.perf_counter() - t0
    psi_q, q, _, _, _, _ = mygs.get_q(npsi=100)
    print(f'===== END {label}: {dt:.1f} s, q0 {q[0]:.4f}, min q {q.min():.4f}, '
          f'rho_m {p.get("saw_rho_m", 0):.3f}, n_dips {p.get("saw_n_dips", 0)}, Ip {mygs.get_stats()["Ip"]:.6e}', flush=True)
    return dict(label=label, t=dt, psi_q=psi_q.tolist(), q=q.tolist(), psi_n=p['psi_n'].tolist(),
                j_saw=p['j_saw'].tolist(), total=p['total_j_phi'].tolist(), rho_m=p.get('saw_rho_m', 0.0),
                n_dips=p.get('saw_n_dips', 0), Ip=float(mygs.get_stats()['Ip']))


runs = [solve('base', saw_q_s=0.0)]
q0, qmin = runs[0]['q'][0], min(runs[0]['q'])
# ITER base has reversed shear (off-axis q minimum below q0). q_s between them: an off-axis dip only;
# an axis-peaked extra current then pulls q0 below q_s too (W-shaped q_base)
q_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5*(q0 + qmin)
print(f'q0 {q0:.4f}, min q {qmin:.4f}, q_s {q_s:.4f}', flush=True)
for amp in (0.0, 1.5e5, 3.0e5):
    core = {'x': x, 'y': amp * np.exp(-(x / 0.12)**2)}
    lab = f'core{amp:.0e}'
    runs.append(solve(lab + '_off', jphi_fixed_prof=dict(core), saw_q_s=0.0))
    for rule in (1, 2, 4):
        runs.append(solve(f'{lab}_saw_rule{rule}', jphi_fixed_prof=dict(core), saw_q_s=q_s, saw_rule=rule,
                          diagnose_bs=True))

with open(os.path.join(outdir, 'summary.json'), 'w') as fid:
    json.dump({'q_s': q_s, 'runs': runs}, fid)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for r in runs:
    ls = '-' if 'saw' in r['label'] else '--'
    ax[0].plot(r['psi_q'], r['q'], ls=ls, label=r['label'])
    ax[1].plot(r['psi_n'], np.asarray(r['j_saw'])/1e6, ls=ls, label=r['label'])
ax[0].axhline(q_s, color='r', lw=0.8); ax[0].set_xlabel('psi_N'); ax[0].set_ylabel('q'); ax[0].set_ylim(0.8, 2.5); ax[0].set_xlim(0, 0.8)
ax[1].set_xlabel('psi_N'); ax[1].set_ylabel('j_saw [MA/m^2]')
for a in ax: a.legend(fontsize=6)
fig.suptitle(f'ITER SWB with saw reset, q_s = {q_s:.3f}'); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'b2_iter_core_current.png'), dpi=110)
print('done')
