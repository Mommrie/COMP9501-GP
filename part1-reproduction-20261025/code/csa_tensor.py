"""CSA for a batch of arbitrary-order real tensors (sample axis is never fitted).

Xu et al., TCSVT 2008, Eq. (5)-(6), Theorem 1 and Procedure 1.
This pedagogical implementation uses training centering, covariance initialization
and an objective-based stopping rule; it is not the authors' original software.
"""
import numbers
import numpy as np


def mode_product(x, matrix, mode):
    """Multiply sample mode `mode` (0-based) by matrix (new_size, old_size).

    x has shape (N, m0, ..., mk). Matrix acts on axis mode+1, preserving all
    other axes and their order. np.moveaxis undoes tensordot's appended axis.
    """
    axis = mode + 1
    if not 0 <= mode < x.ndim - 1 or matrix.ndim != 2:
        raise ValueError('invalid tensor mode or matrix')
    if matrix.shape[1] != x.shape[axis]:
        raise ValueError('matrix input dimension does not match tensor mode')
    return np.moveaxis(np.tensordot(x, matrix.T, axes=(axis, 0)), -1, axis)


def unfold_batch(x, mode):
    """Return (mode size, N * all other sizes); no new independent samples."""
    return np.moveaxis(x, mode + 1, 0).reshape(x.shape[mode + 1], -1)


def _leading(covariance, rank):
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, -rank:][:, ::-1]


class TensorCSA:
    """Orthogonal multilinear projection learned by sequential conditional PCA.

    ranks: one rank per sample mode; init: 'covariance' or seeded 'random'.
    history[0] is initial energy; each following entry is one mode update.
    """
    def __init__(self, ranks, max_sweeps=100, tol=1e-7,
                 init='covariance', random_state=0):
        self.ranks = tuple(ranks)
        if not self.ranks or any(isinstance(r, bool) or
                                 not isinstance(r, numbers.Integral) or r < 1
                                 for r in self.ranks):
            raise ValueError('ranks must be positive integers')
        if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, numbers.Integral) or max_sweeps < 1:
            raise ValueError('max_sweeps must be a positive integer')
        if not np.isfinite(tol) or tol < 0 or init not in ('covariance', 'random'):
            raise ValueError('invalid tolerance or initialization')
        self.max_sweeps, self.tol = max_sweeps, tol
        self.init, self.random_state = init, random_state

    def _input(self, x, fitting=False):
        raw = np.asarray(x)
        if np.iscomplexobj(raw):
            raise ValueError('only real tensors are supported')
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim != len(self.ranks) + 1 or any(s == 0 for s in x.shape) or not np.isfinite(x).all():
            raise ValueError('expected a nonempty finite batch with one axis per rank')
        if fitting:
            if any(r > m for r, m in zip(self.ranks, x.shape[1:])):
                raise ValueError('rank exceeds mode size')
        elif not hasattr(self, 'factors'):
            raise RuntimeError('call fit before transform/reconstruct')
        elif x.shape[1:] != self.mean.shape:
            raise ValueError('sample shape differs from training shape')
        return x

    def _project(self, z):
        for mode, factor in enumerate(self.factors):
            z = mode_product(z, factor.T, mode)
        return z

    def transform(self, x):
        return self._project(self._input(x) - self.mean)

    def inverse_transform(self, core):
        if not hasattr(self, 'factors'):
            raise RuntimeError('call fit before inverse_transform')
        z = np.asarray(core, dtype=np.float64)
        if z.ndim != len(self.ranks) + 1 or z.shape[1:] != self.ranks or not np.isfinite(z).all():
            raise ValueError('invalid core shape or values')
        for mode, factor in enumerate(self.factors):
            z = mode_product(z, factor, mode)
        return z + self.mean

    def reconstruct(self, x):
        return self.inverse_transform(self.transform(x))

    def fit(self, x):
        x = self._input(x, fitting=True)
        self.mean = x.mean(axis=0)
        z = x - self.mean
        rng = np.random.default_rng(self.random_state)
        self.factors = []
        for mode, (size, rank) in enumerate(zip(x.shape[1:], self.ranks)):
            if self.init == 'random':
                factor = np.linalg.qr(rng.normal(size=(size, rank)))[0]
            else:
                flat = unfold_batch(z, mode)
                factor = _leading(flat @ flat.T, rank)
            self.factors.append(factor)
        energy = lambda: float(np.square(self._project(z)).sum())
        self.history = [energy()]
        self.converged = False
        for sweep in range(self.max_sweeps):
            previous = self.history[-1]
            for mode, rank in enumerate(self.ranks):
                partial = z
                for other, factor in enumerate(self.factors):
                    if other != mode:
                        partial = mode_product(partial, factor.T, other)
                flat = unfold_batch(partial, mode)
                self.factors[mode] = _leading(flat @ flat.T, rank)
                self.history.append(energy())
            if self.history[-1] - previous <= self.tol * max(abs(previous), 1e-12):
                self.converged = True
                break
        self.sweeps = sweep + 1
        total = float(np.square(z).sum())
        self.diagnostics = {
            'total_centered_energy': total,
            'captured_energy': self.history[-1],
            'max_orthogonality_error': max(float(np.max(np.abs(f.T @ f - np.eye(r))))
                                         for f, r in zip(self.factors, self.ranks)),
            'min_coordinate_energy_change': float(np.diff(self.history).min()),
            'energy_identity_error': abs(total - self.history[-1] -
                                         float(np.square(x - self.reconstruct(x)).sum())),
        }
        if self.diagnostics['min_coordinate_energy_change'] < -1e-9 * max(total, 1):
            raise ArithmeticError('captured energy decreased beyond numerical tolerance')
        if self.diagnostics['max_orthogonality_error'] > 1e-9:
            raise ArithmeticError('projection columns are not orthonormal')
        if self.diagnostics['energy_identity_error'] > 1e-9 * max(total, 1):
            raise ArithmeticError('projection/reconstruction energy identity failed')
        return self
