"""Reproducible Part 1 figures: MNIST, storage budgets, restarts and 3D CSA.

Run from any directory; --output must be a NEW directory. Public data can be
downloaded explicitly with --download. No GPU, credentials or deep learning.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csa import CSA, PCA, accuracy_1nn
from csa_tensor import TensorCSA, mode_product

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'CSA': '#16677d', 'PCA': '#bc5135', 'Budget PCA': '#7b6398'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output, name):
    for ext in ('png', 'svg'):
        fig.savefig(output / (name + '.' + ext), dpi=200, bbox_inches='tight',
                    **({'metadata': {'Date': None}} if ext == 'svg' else {}))
        if ext == 'svg':
            path = output / (name + '.svg')
            path.write_bytes(b'\n'.join(line.rstrip() for line in path.read_bytes().splitlines()) + b'\n')
    plt.close(fig)


def panel_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)


def metrics(model, train, labels, test, target):
    return {
        'train_mse': float(np.square(train-model.reconstruct(train)).mean()),
        'test_mse': float(np.square(test-model.reconstruct(test)).mean()),
        'accuracy_1nn': accuracy_1nn(model.transform(train), labels,
                                     model.transform(test), target),
    }


def projection_diagram(output):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout='constrained')
    for ax, text in zip(axes, [
        'Input image X\n14 x 14\ncenter using training mean',
        'Core Y = U^T X V\n4 x 4\n16 coefficients / image',
        'Reconstruction U Y V^T\n14 x 14\nadd training mean',
        'Learn U and V together\nfix V -> eigenvectors of B_U\nfix new U -> eigenvectors of B_V',
    ]):
        ax.set_facecolor('#edf3f5')
        ax.text(.5, .5, text, ha='center', va='center', fontsize=11,
                transform=ax.transAxes, linespacing=1.8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#c0d3dc')
    fig.suptitle('2D CSA: two orthogonal projections, one compact core', fontsize=15)
    save(fig, output, '01-projection-pipeline')


def gallery(output, images, ids, models):
    fig, axes = plt.subplots(5, 10, figsize=(13, 7), layout='constrained')
    rows = [('Original', images)] + [(name, m.reconstruct(images)) for name, m in models.items()]
    for row, (name, data) in enumerate(rows):
        for col, image in enumerate(data):
            axes[row, col].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f'Digit {col}\nID {ids[col]}', fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=10)
    fig.suptitle('Held-out MNIST reconstructions: fixed first test ID per digit\n'
                 '30 training images/digit; subset seed 21; common grayscale [0,1]', fontsize=12)
    save(fig, output, '02-reconstruction-gallery')


def bases(output, csa, pca):
    fig, axes = plt.subplots(4, 8, figsize=(12, 6), layout='constrained')
    cb = np.kron(*csa.factors).T.reshape(16, 14, 14)
    pb = pca.basis.T.reshape(16, 14, 14)
    for j, bank in enumerate([cb, pb]):
        limit = np.abs(bank).max()
        for i, basis in enumerate(bank):
            ax = axes[i//4, j*4+i%4]
            ax.imshow(basis, cmap='RdBu_r', vmin=-limit, vmax=limit)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{"CSA" if j == 0 else "PCA"} {i+1}', fontsize=9)
    fig.suptitle('Sixteen basis images: separable CSA (left) vs unrestricted PCA (right)\n'
                 'Sign and order are arbitrary; color scale normalized within each method', fontsize=12)
    save(fig, output, '03-basis-images')


def summarize_grid(output, runs, plan):
    methods = ['CSA', 'PCA', 'Budget PCA']
    ranks = plan['ranks']
    agg = []
    for rank in ranks:
        group = [r for r in runs if r['rank'] == rank]
        record = {'rank': rank, 'latent': rank*rank, 'budget_pca_dimension': group[0]['budget_pca_dimension']}
        for method in methods:
            record[method] = {metric: {'mean': float(np.mean([r[method][metric] for r in group])),
                                        'sd': float(np.std([r[method][metric] for r in group], ddof=1))}
                              for metric in ['train_mse', 'test_mse', 'accuracy_1nn']}
        agg.append(record)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for method in ['CSA', 'PCA']:
        for ax, metric, scale in zip(axes, ['accuracy_1nn', 'test_mse'], [100, 1]):
            ax.errorbar([r*r for r in ranks], [g[method][metric]['mean']*scale for g in agg],
                        yerr=[g[method][metric]['sd']*scale for g in agg], marker='o',
                        capsize=3, label=method, color=COLORS[method])
    for ax in axes:
        panel_style(ax); ax.set_xlabel('Latent coefficients per image (r squared)'); ax.legend()
    axes[0].set_ylabel('Held-out 1-NN accuracy (%)'); axes[1].set_ylabel('Held-out pixel MSE')
    fig.suptitle('Equal latent size: mean +/- sample SD over three training subsets\n'
                 '30 images/digit; reused fixed test set; post-exploration illustration', fontsize=12)
    save(fig, output, '04-equal-latent-results')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for method in ['CSA', 'Budget PCA']:
        for ax, metric, scale in zip(axes, ['accuracy_1nn', 'test_mse'], [100, 1]):
            ax.errorbar(ranks, [g[method][metric]['mean']*scale for g in agg],
                        yerr=[g[method][metric]['sd']*scale for g in agg], marker='o',
                        capsize=3, label=method, color=COLORS[method])
    for ax in axes:
        panel_style(ax); ax.set_xticks(ranks); ax.set_xlabel('CSA rank r (core r x r)'); ax.legend()
    axes[0].set_ylabel('Held-out 1-NN accuracy (%)'); axes[1].set_ylabel('Held-out pixel MSE')
    fig.suptitle('Storage-constrained PCA: largest integer dimension within CSA budget\n'
                 'N=300 represented training images; bases + cores + common mean; PCA may leave slack', fontsize=12)
    save(fig, output, '05-storage-budget-results')
    d, n = 196, plan['train_per_class']*10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].plot(ranks, [28*r for r in ranks], 'o-', label='CSA basis', color=COLORS['CSA'])
    axes[0].plot(ranks, [d*r*r for r in ranks], 'o-', label='PCA basis (same latent)', color=COLORS['PCA'])
    axes[0].set_yscale('log'); axes[0].set_ylabel('Basis scalars (log scale; mean excluded)')
    axes[1].plot(ranks, [n*r*r+28*r+d for r in ranks], 'o-', label='CSA total', color=COLORS['CSA'])
    axes[1].plot(ranks, [(n+d)*r*r+d for r in ranks], 'o-', label='PCA total (same latent)', color=COLORS['PCA'])
    axes[1].set_ylabel('Total stored scalars: basis + 300 cores + mean')
    for ax in axes:
        panel_style(ax); ax.set_xlabel('CSA rank r; matched latent = r squared'); ax.legend()
    fig.suptitle('Projection storage and collection storage answer different questions', fontsize=13)
    save(fig, output, '06-storage-accounting')
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['seed', 'rank', 'latent', 'method', 'dimension', 'basis_scalars', 'total_scalars',
                  'train_mse', 'test_mse', 'accuracy_1nn']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for r in runs:
            for method in methods:
                dim = r['budget_pca_dimension'] if method == 'Budget PCA' else r['rank']**2
                basis = 28*r['rank'] if method == 'CSA' else d*dim
                writer.writerow(dict(seed=r['seed'], rank=r['rank'], latent=r['rank']**2,
                                     method=method, dimension=dim, basis_scalars=basis,
                                     total_scalars=basis+n*dim+d, **r[method]))
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'presentation-20261028/plan.json')
    parser.add_argument('--data', type=Path, default=ROOT.parent/'.cache/mnist.npz')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if args.output.exists():
        raise FileExistsError('choose a new --output directory; existing outputs are never replaced')
    if not args.data.exists() and args.download:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(plan['dataset_url'], timeout=60) as response:
            args.data.write_bytes(response.read())
    if sha(args.data) != plan['dataset_sha256']:
        raise ValueError('MNIST archive SHA-256 does not match plan')
    if plan['side'] != 14:
        raise ValueError('this teaching runner explicitly implements 28 -> 14 downsampling')
    args.output.mkdir(parents=True)
    output = args.output
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'comp9501-part1-20261028', 'axes.titlepad': 10})
    with np.load(args.data, allow_pickle=False) as raw:
        down = lambda a: a.astype(np.float64).reshape(-1, 14, 2, 14, 2).mean((2, 4))/255
        train, test = down(raw['x_train']), down(raw['x_test'])
        labels, target = raw['y_train'].copy(), raw['y_test'].copy()
    rng = np.random.default_rng(plan['test_seed'])
    test_ids = np.concatenate([rng.permutation(np.flatnonzero(target == c))[:plan['test_per_class']]
                               for c in range(10)])
    z, yt = test[test_ids], target[test_ids]
    runs, sample_ids, histories = [], {}, {}
    maxdim = max(plan['ranks'])**2
    archive = json.loads((ROOT/'artifacts/summary-2026-10-25.json').read_text())
    parity = []
    for seed in plan['seeds']:
        rng = np.random.default_rng(seed)
        pools = [rng.permutation(np.flatnonzero(labels == c)) for c in range(10)]
        ids = np.concatenate([p[:plan['train_per_class']] for p in pools])
        sample_ids[str(seed)] = ids.tolist()
        x, y = train[ids], labels[ids]
        fullpca = PCA(maxdim).fit(x)
        for rank in plan['ranks']:
            csa = TensorCSA((rank, rank), max_sweeps=plan['max_sweeps'], tol=plan['tol']).fit(x)
            latent = rank*rank
            budgetdim = (len(x)*latent+28*rank)//(len(x)+196)
            # d=0 is a mean-only reconstruction; 1-NN then uses a fixed first
            # training-label tie break. Report it rather than overspending.
            def truncated(dimension):
                p = PCA(dimension); p.mean = fullpca.mean; p.basis = fullpca.basis[:, :dimension]
                return p
            pca, bpca = truncated(latent), truncated(budgetdim)
            result = {'seed': seed, 'rank': rank, 'budget_pca_dimension': budgetdim,
                      'CSA': metrics(csa, x, y, z, yt), 'PCA': metrics(pca, x, y, z, yt),
                      'Budget PCA': metrics(bpca, x, y, z, yt),
                      'sweeps': csa.sweeps, 'converged': csa.converged, 'checks': csa.diagnostics}
            if result['PCA']['train_mse'] > result['CSA']['train_mse']+1e-10:
                raise ArithmeticError('PCA training reconstruction bound failed')
            runs.append(result)
            histories[f'{seed}-{rank}'] = csa.history
            for prior in archive['runs']:
                if prior['seed'] == seed and prior['n_per_class'] == plan['train_per_class'] and prior['rank'] == rank:
                    if prior['train_native_ids'] != ids.tolist() or archive['test_native_ids'] != test_ids.tolist():
                        raise ArithmeticError('archival sample IDs do not match')
                    error = max(abs(result[m][k]-prior['methods'][m][k])
                                for m in ['CSA', 'PCA'] for k in ['train_mse', 'test_mse', 'accuracy_1nn'])
                    if error > 1e-9:
                        raise ArithmeticError('archival metric parity failed')
                    parity.append({'seed': seed, 'rank': rank, 'max_metric_difference': error})
            if seed == 21 and rank == 4:
                show_ids = np.array([min(test_ids[yt == c]) for c in range(10)])
                models = {'CSA core 2x2': TensorCSA((2, 2)).fit(x), 'PCA latent 4': truncated(4),
                          'CSA core 4x4': csa, 'PCA latent 16': pca}
                gallery(output, test[show_ids], show_ids, models)
                bases(output, csa, pca)
            print(f'subset seed={seed}, rank={rank}, PCA budget dimension={budgetdim}', flush=True)
    aggregate = summarize_grid(output, runs, plan)
    lines = ['# Teaching extension: all ranks reported', '',
             'Three training-subset seeds; fixed reused 1,000-image test set; mean +/- sample SD.', '',
             '| CSA rank | Core scalars | Budget PCA dim | CSA acc (%) | Equal-latent PCA acc (%) | Budget PCA acc (%) | CSA test MSE | Equal-latent PCA test MSE | Budget PCA test MSE |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in aggregate:
        cells = [str(g['rank']), str(g['latent']), str(g['budget_pca_dimension'])]
        for metric, scale, precision in [('accuracy_1nn', 100, 2), ('test_mse', 1, 5)]:
            for method in ['CSA', 'PCA', 'Budget PCA']:
                stat = g[method][metric]
                cells.append(f"{stat['mean']*scale:.{precision}f} +/- {stat['sd']*scale:.{precision}f}")
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'For CSA r=1, the storage budget cannot fit one PCA component; d=0 is mean-only reconstruction.',
              'Its 1-NN prediction has all-distance ties, resolved to the first training label (10% here).',
              'Storage counts N=300 training-image cores, bases and the shared mean; test metrics do not enter this budget.',
              'This is not a same-byte compressed file comparison or a reproduction of original Fig. 5.']
    (output/'results-table.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    projection_diagram(output)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for seed in plan['seeds']:
        h = np.array(histories[f'{seed}-4'])
        ax.plot(np.arange(len(h)), h/h[-1], marker='.', label=f'Subset seed {seed}')
    ax.set_xlabel('Coordinate update (0=initialization; U then V)')
    ax.set_ylabel('Captured energy / final energy'); panel_style(ax); ax.legend()
    ax.set_title('Objective convergence, not proof of global optimality\n14x14 MNIST; r=4; 30 images/digit')
    save(fig, output, '07-objective-convergence')
    rp = plan['restarts']; rng = np.random.default_rng(rp['subset_seed'])
    ids = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:rp['train_per_class']] for c in range(10)])
    restart_results = []
    for name, seed in [('covariance', None)] + [('random', s) for s in rp['random_initializations']]:
        model = TensorCSA((rp['rank'], rp['rank']), init=name, random_state=seed).fit(train[ids])
        restart_results.append({'init': name, 'seed': seed, 'sweeps': model.sweeps,
                                'converged': model.converged, 'diagnostics': model.diagnostics,
                                **metrics(model, train[ids], labels[ids], z, yt)})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    names = ['Cov'] + [str(s) for s in rp['random_initializations']]
    for ax, metric, scale in zip(axes, ['train_mse', 'accuracy_1nn'], [1, 100]):
        ax.bar(names, [r[metric]*scale for r in restart_results], color=[COLORS['CSA']]+['#a6bbc4']*(len(names)-1))
        ax.set_xlabel('Covariance init / random init seed'); panel_style(ax)
    axes[0].set_ylabel('Final training pixel MSE'); axes[1].set_ylabel('Held-out 1-NN accuracy (%)')
    fig.suptitle('Initialization sensitivity: all starts shown; no test-based selection\n'
                 'One fixed subset (seed 21), 3 images/digit, r=4; descriptive only', fontsize=12)
    save(fig, output, '08-initialization-sensitivity')
    sp = plan['synthetic']; rng = np.random.default_rng(sp['seed'])
    factors = [np.linalg.qr(rng.normal(size=(m, r)))[0] for m, r in zip(sp['shape'], sp['ranks'])]
    offset = rng.normal(size=sp['shape'])
    def sample(n):
        tensor = rng.normal(size=(n, *sp['ranks']))
        for mode, factor in enumerate(factors):
            tensor = mode_product(tensor, factor, mode)
        return tensor+offset
    sx, st = sample(sp['n_train']), sample(sp['n_test'])
    sm = TensorCSA(sp['ranks']).fit(sx); reconstructed = sm.reconstruct(st)
    synthetic = {'plan': sp, 'test_mse': float(np.square(st-reconstructed).mean()),
                 'max_abs_error': float(np.abs(st-reconstructed).max()), 'diagnostics': sm.diagnostics,
                 'sweeps': sm.sweeps, 'converged': sm.converged}
    if synthetic['max_abs_error'] > 1e-10:
        raise ArithmeticError('synthetic exact-reconstruction check failed')
    fig, axes = plt.subplots(3, sp['shape'][2], figsize=(12, 6), layout='constrained')
    limit = np.abs(st[0]).max()
    for k in range(sp['shape'][2]):
        for row, image in enumerate([st[0, :, :, k], reconstructed[0, :, :, k], np.abs(st[0, :, :, k]-reconstructed[0, :, :, k])]):
            axes[row, k].imshow(image, cmap='magma' if row == 2 else 'RdBu_r',
                                vmin=0 if row == 2 else -limit, vmax=1e-12 if row == 2 else limit)
            axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
        axes[0, k].set_title(f'Slice {k}')
    for row, label in enumerate(['Original', 'CSA reconstruction', 'Absolute error\nscale 0 to 1e-12']):
        axes[row, 0].set_ylabel(label)
    fig.suptitle('3D mathematical check: 12x10x6 -> 3x2x2 -> 12x10x6\n'
                 'Noiseless synthetic separable tensors; not Gabor data or a recognition benchmark', fontsize=12)
    save(fig, output, '09-three-mode-reconstruction')
    # Values transcribed by visual inspection of original Table IV, PDF p9 / journal p44.
    original = {'source': 'Xu et al. 2008, Table IV, journal p44 / PDF p9',
                'doi': '10.1109/TCSVT.2007.903317',
                'protocol': '28x28x40 Gabor tensors; first half of native MNIST test set; author-reported best configurations',
                'rows': [{'setting': 'G30 / P470', 'PCA_Gabor': 88.5, 'CSA3': 92.8,
                          'pca_dim': '60', 'csa_core': '6 x 9 x 18'},
                         {'setting': 'G40 / P460', 'PCA_Gabor': 90.5, 'CSA3': 92.4,
                          'pca_dim': '50', 'csa_core': '15 x 5 x 13'}]}
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    t = np.arange(2)
    for delta, key, label, color in [(-.18, 'PCA_Gabor', 'PCA (Gabor)', COLORS['PCA']),
                                     (.18, 'CSA3', 'CSA-3', COLORS['CSA'])]:
        bars = ax.bar(t+delta, [r[key] for r in original['rows']], width=.34, label=label, color=color)
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set_xticks(t, [r['setting'] for r in original['rows']]); ax.set_ylim(0, 105)
    ax.set_ylabel('Reported top-1 accuracy (%)'); ax.legend(loc='lower right'); panel_style(ax)
    ax.set_title('Original paper Table IV (redrawn, not our experiment)\nXu et al., TCSVT 2008, p44; best configurations differ')
    save(fig, output, '10-original-paper-table-IV')
    result = {'state': 'passed', 'plan': plan, 'plan_sha256': sha(args.plan),
              'data_sha256': sha(args.data), 'environment': {'python': sys.version.split()[0],
                  'platform': platform.platform(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
              'sources': {p.name: sha(p) for p in (ROOT/'code').glob('*.py')},
              'train_native_ids': sample_ids, 'test_native_ids': test_ids.tolist(),
              'runs': runs, 'aggregate': aggregate, 'histories': histories,
              'archival_parity': parity, 'restarts': restart_results, 'synthetic': synthetic,
              'original_paper_table_IV': original}
    (output/'results.json').write_bytes((json.dumps(result, indent=2)+'\n').encode('utf-8'))
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'SHA256.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode('utf-8'))
    print('PART1_PRESENTATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
