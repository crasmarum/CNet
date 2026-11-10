#!/usr/bin/env python3
"""Phase-generalisation benchmark: does a network classify a complex signal correctly at carrier phases it never saw?

The task isolates the one thing complex arithmetic gives for free: equivariance to a global phase rotation
z -> e^{i theta} z. Each sample is one of K constant-modulus waveforms (unit-modulus entries with random phases, like
PSK/FM bursts) received with an unknown carrier phase theta, a random amplitude and complex Gaussian noise:

    z = a * e^{i theta} * s_k + sigma * w,      label k

Training phases are drawn from [0, pi/2) only; the test sweeps the full circle [0, 2 pi). The label does not depend
on theta, so a model that has the symmetry generalises to the unseen three quarters of the circle and one that has to
learn it from data does not.

Models (parameter counts matched by real degrees of freedom; a complex weight counts as two reals):
  complex       complex-linear layers without bias, modReLU activations, Born read-out p_k = |z_k|^2 / sum |z|^2.
                Invariant to the global phase by construction: W(e^{i t} z) = e^{i t} W z, modReLU keeps the phase, the
                Born rule removes it.
  tied_real     the same network written with real tensors only, every complex weight a tied 2x2 block [[a,-b],[b,a]].
                Control for the objection 'it is just a real net': it is, with the symmetry built into its structure.
  real_mlp      ReLU MLP on [Re z, Im z], trained on the same quarter circle. What a real network does by default.
  real_aug      the same MLP trained with phases augmented over the full circle. The symmetry is supplied through the
                data instead of the architecture: no out-of-range test, but it costs samples (see --sweep).
  real_modulus  ReLU MLP on |z_j|, a hand-made invariant feature. Invariant, but it discards the relative phases that
                carry the label for constant-modulus signals, so it sits at chance.
  matched       non-coherent matched filter argmax_k |<s_k, z>| with the true templates: the reference a model can
                approach, not a learned model.

usage:
  python3 phase_generalisation.py                       # one training-set size (4096), 3 seeds, ~20 s on a laptop CPU
  python3 phase_generalisation.py --sweep 256,1024,4096,16384   # sample efficiency of each model
  python3 phase_generalisation.py --snr_db -6 --n 32 --k 8 --hidden 48 --epochs 40 --out results_phase

Outputs in --out: results.csv (accuracy per model, seed, training size and test-phase bin), summary.md (tables),
accuracy_vs_phase.svg (no plotting library needed). Requires PyTorch only.
"""
import argparse, csv, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--n', type=int, default=32, help='samples per signal')
ap.add_argument('--k', type=int, default=8, help='number of classes (waveforms)')
ap.add_argument('--snr_db', type=float, default=-6.0, help='per-sample SNR in dB (matched-filter gain adds 10 log10 n)')
ap.add_argument('--hidden', type=int, default=48, help='complex hidden width; real widths are set to match parameters')
ap.add_argument('--train', type=int, default=4096, help='training-set size')
ap.add_argument('--sweep', default='', help='comma-separated training-set sizes (overrides --train)')
ap.add_argument('--train_phase', type=float, default=90.0, help='training phases are drawn from [0, this) degrees')
ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--bins', type=int, default=24, help='test-phase bins over the full circle')
ap.add_argument('--test_per_bin', type=int, default=500)
ap.add_argument('--models', default='complex,tied_real,real_mlp,real_aug,real_modulus,matched')
ap.add_argument('--out', default='results_phase')
a = ap.parse_args()
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


# ---------------------------------------------------------------- data

def make_templates(rng):
    """K constant-modulus waveforms: unit-modulus entries with random phases (fixed per seed)."""
    return np.exp(1j * rng.uniform(0, 2 * np.pi, (a.k, a.n)))

def sample(rng, S, m, phase_lo, phase_hi, labels=None):
    y = rng.integers(0, a.k, m) if labels is None else labels
    th = rng.uniform(phase_lo, phase_hi, m)
    amp = rng.uniform(0.5, 1.5, m)
    sigma = 10 ** (-a.snr_db / 20)                      # per complex sample; signal entries have modulus ~1
    w = (rng.standard_normal((m, a.n)) + 1j * rng.standard_normal((m, a.n))) / np.sqrt(2)
    z = (amp * np.exp(1j * th))[:, None] * S[y] + sigma * w
    return torch.tensor(z, dtype=torch.complex64), torch.tensor(y, dtype=torch.long)


# ---------------------------------------------------------------- models

def modrelu(z, b, eps=1e-6):
    r = z.abs()
    return F.relu(r + b) * z / (r + eps)

def born_logits(z, eps=1e-8):
    """Born rule: p_k = |z_k|^2 / sum_j |z_j|^2, returned as log|z_k|^2 so cross_entropy normalises it."""
    return torch.log(z.real ** 2 + z.imag ** 2 + eps)

class ComplexNet(nn.Module):
    """CLinear -> modReLU -> CLinear -> modReLU -> CLinear -> Born. No biases on the linear maps (a bias would break
    phase equivariance); modReLU's bias acts on the modulus only."""
    def __init__(self, n, h, k):
        super().__init__()
        def cw(i, o):
            return nn.Parameter(torch.randn(o, i, dtype=torch.complex64) / math.sqrt(i))
        self.w1, self.w2, self.w3 = cw(n, h), cw(h, h), cw(h, k)
        self.b1, self.b2 = nn.Parameter(torch.zeros(h)), nn.Parameter(torch.zeros(h))
    def forward(self, z):
        z = modrelu(z @ self.w1.T, self.b1)
        z = modrelu(z @ self.w2.T, self.b2)
        return born_logits(z @ self.w3.T)

class TiedRealNet(nn.Module):
    """The ComplexNet computed with real tensors only: every complex weight a + ib is the tied 2x2 block
    [[a, -b], [b, a]] acting on (Re, Im). Same function class, no complex dtype anywhere."""
    def __init__(self, n, h, k):
        super().__init__()
        def rw(i, o):
            return nn.Parameter(torch.randn(o, i) / math.sqrt(2 * i))
        self.a1, self.c1 = rw(n, h), rw(n, h)
        self.a2, self.c2 = rw(h, h), rw(h, h)
        self.a3, self.c3 = rw(h, k), rw(h, k)
        self.b1, self.b2 = nn.Parameter(torch.zeros(h)), nn.Parameter(torch.zeros(h))
    @staticmethod
    def lin(x, y, A, C):                       # (A + iC)(x + iy) = (Ax - Cy) + i(Cx + Ay)
        return x @ A.T - y @ C.T, x @ C.T + y @ A.T
    @staticmethod
    def act(x, y, b, eps=1e-6):
        r = torch.sqrt(x * x + y * y + eps * eps)
        g = F.relu(r + b) / (r + eps)
        return x * g, y * g
    def forward(self, z):
        x, y = z.real, z.imag
        x, y = self.act(*self.lin(x, y, self.a1, self.c1), self.b1)
        x, y = self.act(*self.lin(x, y, self.a2, self.c2), self.b2)
        x, y = self.lin(x, y, self.a3, self.c3)
        return torch.log(x * x + y * y + 1e-8)

class RealMLP(nn.Module):
    """ReLU MLP on [Re z, Im z] (or on |z| with modulus=True)."""
    def __init__(self, n_in, h, k, modulus=False):
        super().__init__()
        self.modulus = modulus
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, k))
    def forward(self, z):
        x = z.abs() if self.modulus else torch.cat([z.real, z.imag], -1)
        return self.net(x)

def n_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())

def matched_width(n_in, target):
    """Hidden width of a 2-hidden-layer real MLP with about `target` parameters."""
    best = min(range(4, 4 * target), key=lambda h: abs(n_in * h + h + h * h + h + h * a.k + a.k - target))
    return best

def build(name):
    ref = n_params(ComplexNet(a.n, a.hidden, a.k))
    if name == 'complex':
        return ComplexNet(a.n, a.hidden, a.k)
    if name == 'tied_real':
        return TiedRealNet(a.n, a.hidden, a.k)
    if name in ('real_mlp', 'real_aug'):
        return RealMLP(2 * a.n, matched_width(2 * a.n, ref), a.k)
    if name == 'real_modulus':
        return RealMLP(a.n, matched_width(a.n, ref), a.k, modulus=True)
    raise ValueError(name)


# ---------------------------------------------------------------- training and evaluation

def train(model, rng, S, m, augment):
    lo, hi = 0.0, (2 * np.pi if augment else np.radians(a.train_phase))
    z, y = sample(rng, S, m, lo, hi)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * math.ceil(m / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(int(rng.integers(1 << 30)))
    for _ in range(a.epochs):
        perm = torch.randperm(m, generator=g)
        for i in range(0, m, a.batch):
            idx = perm[i:i + a.batch]
            loss = F.cross_entropy(model(z[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

def test_sets(rng, S):
    """One fixed test set per phase bin, shared by every model of a seed."""
    edges = np.linspace(0, 2 * np.pi, a.bins + 1)
    return [(0.5 * (lo + hi), *sample(rng, S, a.test_per_bin, lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]

@torch.no_grad()
def evaluate(predict, tests):
    return [(c, (predict(z) == y).float().mean().item()) for c, z, y in tests]

@torch.no_grad()
def invariance_error(model, z):
    """max |p(z) - p(e^{it} z)| over random global phases t: zero up to float32 rounding for an invariant model."""
    p = F.softmax(model(z), -1)
    t = torch.rand(z.shape[0], 1) * 2 * math.pi
    p2 = F.softmax(model(z * torch.exp(1j * t).to(torch.complex64)), -1)
    return (p - p2).abs().max().item()


# ---------------------------------------------------------------- run

def main():
    os.makedirs(a.out, exist_ok=True)
    sizes = [int(s) for s in a.sweep.split(',')] if a.sweep else [a.train]
    names = [s for s in a.models.split(',') if s]
    in_range = lambda c: np.degrees(c) < a.train_phase
    rows, inv, counts = [], {}, {}
    t0 = time.time()
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed); torch.manual_seed(seed)
        S = make_templates(rng)
        tests = test_sets(rng, S)
        St = torch.tensor(S, dtype=torch.complex64)
        for m in sizes:
            for name in names:
                if name == 'matched':
                    acc = evaluate(lambda z: (z @ St.conj().T).abs().argmax(-1), tests)
                else:
                    torch.manual_seed(100 * seed + 7)
                    model = build(name); counts[name] = n_params(model)
                    train(model, np.random.default_rng(10_000 * seed + m), S, m, augment=(name == 'real_aug'))
                    acc = evaluate(lambda z: model(z).argmax(-1), tests)
                    inv.setdefault((name, m), []).append(invariance_error(model, tests[0][1]))
                for c, v in acc:
                    rows.append(dict(model=name, seed=seed, train=m, phase_deg=round(np.degrees(c), 2),
                                     in_train_range=int(in_range(c)), acc=round(v, 4)))
                ins = np.mean([v for c, v in acc if in_range(c)]); outs = np.mean([v for c, v in acc if not in_range(c)])
                print(f'seed {seed} train {m:6d} {name:13s} in-range {ins:.3f}  out-of-range {outs:.3f}  '
                      f'({time.time() - t0:.0f} s)', flush=True)

    with open(os.path.join(a.out, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    chance = 1 / a.k
    L = [f'# Phase-generalisation benchmark\n',
         f'K={a.k} constant-modulus waveforms of n={a.n} samples, per-sample SNR {a.snr_db:g} dB, random amplitude '
         f'0.5-1.5; training phases in [0, {a.train_phase:g}) deg (real_aug: full circle); test over the full circle in '
         f'{a.bins} bins x {a.test_per_bin}; {a.epochs} epochs, Adam {a.lr:g}, {a.seeds} seeds; chance {chance:.3f}.\n',
         'Parameters (real degrees of freedom): ' + ', '.join(f'{k} {v:,}' for k, v in counts.items()) + '.\n']
    for m in sizes:
        L += [f'\n## Training set {m:,}\n', '| model | in training range | outside it | worst bin | max change of p under a phase rotation |',
              '|---|---|---|---|---|']
        for name in names:
            R = [r for r in rows if r['model'] == name and r['train'] == m]
            per_seed = lambda flag: [np.mean([r['acc'] for r in R if r['seed'] == s and r['in_train_range'] == flag])
                                     for s in range(a.seeds)]
            fmt = lambda v: f'{np.mean(v):.3f} [{min(v):.3f}-{max(v):.3f}]'
            bins = {}
            for r in R: bins.setdefault(r['phase_deg'], []).append(r['acc'])
            worst = min(np.mean(v) for v in bins.values())
            ie = '-' if (name, m) not in inv else f'{max(inv[(name, m)]):.1e}'
            L.append(f'| {name} | {fmt(per_seed(1))} | {fmt(per_seed(0))} | {worst:.3f} | {ie} |')
    L.append('\nMean over seeds [min-max] of the accuracy averaged over the test-phase bins in / outside the training '
             'range; worst bin: lowest seed-mean bin accuracy. Last column: max |p_k(z) - p_k(e^{it} z)| over test '
             'signals and random t; below about 1e-4 means invariant up to float32 rounding.')
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    write_svg(rows, names, max(sizes), os.path.join(a.out, 'accuracy_vs_phase.svg'), chance)
    print('\n'.join(L))
    print(f'\nwrote {a.out}/results.csv, summary.md, accuracy_vs_phase.svg in {time.time() - t0:.0f} s')


def write_svg(rows, names, m, path, chance):
    """Accuracy against test phase for the largest training set, mean over seeds; dependency-free SVG."""
    W, H, l, r, t, b = 720, 400, 60, 150, 30, 50
    pw, ph = W - l - r, H - t - b
    x = lambda deg: l + pw * deg / 360
    y = lambda v: t + ph * (1 - v)
    colours = dict(complex='#1f6fb4', tied_real='#5aa9e6', real_mlp='#d1495b', real_aug='#edae49',
                   real_modulus='#8c8c8c', matched='#222222')
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<rect x="{x(0)}" y="{t}" width="{x(a.train_phase) - x(0)}" height="{ph}" fill="#eef4ea"/>',
         f'<text x="{x(a.train_phase / 2)}" y="{t + 14}" text-anchor="middle" fill="#557a46">training phases</text>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<line x1="{l}" x2="{l + pw}" y1="{y(v)}" y2="{y(v)}" stroke="#e5e5e5"/>'
                 f'<text x="{l - 8}" y="{y(v) + 4}" text-anchor="end">{v:.2f}</text>')
    for d in range(0, 361, 90):
        s.append(f'<text x="{x(d)}" y="{t + ph + 18}" text-anchor="middle">{d}</text>')
    s.append(f'<line x1="{l}" x2="{l + pw}" y1="{y(chance)}" y2="{y(chance)}" stroke="#999" stroke-dasharray="4 4"/>'
             f'<text x="{l + pw - 4}" y="{y(chance) - 4}" text-anchor="end" fill="#777">chance</text>')
    s.append(f'<text x="{l + pw / 2}" y="{H - 10}" text-anchor="middle">test carrier phase (degrees)</text>'
             f'<text x="16" y="{t + ph / 2}" text-anchor="middle" transform="rotate(-90 16 {t + ph / 2})">accuracy</text>')
    for i, name in enumerate(names):
        pts = {}
        for row in rows:
            if row['model'] == name and row['train'] == m:
                pts.setdefault(row['phase_deg'], []).append(row['acc'])
        path_d = ' '.join(f'{x(d):.1f},{y(np.mean(v)):.1f}' for d, v in sorted(pts.items()))
        c = colours.get(name, '#000')
        dash = ' stroke-dasharray="6 3"' if name in ('tied_real', 'matched') else ''
        s.append(f'<polyline points="{path_d}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        s.append(f'<line x1="{l + pw + 14}" x2="{l + pw + 34}" y1="{t + 10 + 18 * i}" y2="{t + 10 + 18 * i}" '
                 f'stroke="{c}" stroke-width="2"{dash}/><text x="{l + pw + 40}" y="{t + 14 + 18 * i}">{name}</text>')
    s.append(f'<text x="{l}" y="{t - 10}">training set {m:,}, mean over {a.seeds} seeds</text></svg>')
    open(path, 'w').write('\n'.join(s))


if __name__ == '__main__':
    main()
