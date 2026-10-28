"""Two-mode CSA: conditional eigenspace maximization of retained energy."""
import numpy as np


def leading(covariance, rank):
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argsort(values)[::-1][:rank]]


class CSA:
    def __init__(self, rank, max_sweeps=100, tol=1e-7):
        self.rank, self.max_sweeps, self.tol = rank, max_sweeps, tol

    def transform(self, x):
        return np.einsum('ha,nhw,wb->nab', self.u, x-self.mean, self.v)

    def reconstruct(self, x):
        return np.einsum('ha,nab,wb->nhw', self.u, self.transform(x), self.v)+self.mean

    def fit(self, x):
        self.mean = x.mean(axis=0)
        z = x-self.mean
        self.u = leading(np.einsum('nhw,nkw->hk', z, z), self.rank)
        self.v = leading(np.einsum('nhw,nhk->wk', z, z), self.rank)
        self.history = [float(np.square(self.transform(x)).sum())]
        self.converged = False
        for sweep in range(self.max_sweeps):
            previous = self.history[-1]
            zv = z @ self.v
            self.u = leading(np.einsum('nhb,nkb->hk', zv, zv), self.rank)
            self.history.append(float(np.square(self.transform(x)).sum()))
            uz = np.einsum('ha,nhw->naw', self.u, z)
            self.v = leading(np.einsum('naw,nak->wk', uz, uz), self.rank)
            self.history.append(float(np.square(self.transform(x)).sum()))
            if self.history[-1]-previous <= self.tol*max(abs(previous),1e-12):
                self.converged = True
                break
        self.sweeps = sweep+1
        assert np.diff(self.history).min() >= -1e-8*max(self.history[-1],1.)
        assert np.allclose(self.u.T @ self.u, np.eye(self.rank), atol=1e-10)
        assert np.allclose(self.v.T @ self.v, np.eye(self.rank), atol=1e-10)
        total = np.square(z).sum()
        assert np.isclose(total-self.history[-1], np.square(x-self.reconstruct(x)).sum(), atol=1e-8*max(total,1.))
        return self


class PCA:
    def __init__(self, rank):
        self.rank = rank

    def fit(self, x):
        self.mean = x.mean(axis=0)
        z = (x-self.mean).reshape(len(x),-1)
        assert self.rank <= min(len(x)-1,z.shape[1])
        _,_,v = np.linalg.svd(z,full_matrices=False)
        self.basis = v[:self.rank].T
        return self

    def transform(self, x):
        return (x-self.mean).reshape(len(x),-1) @ self.basis

    def reconstruct(self, x):
        return (self.transform(x) @ self.basis.T).reshape(x.shape)+self.mean


def accuracy_1nn(train, labels, test, target):
    a, b = train.reshape(len(train),-1), test.reshape(len(test),-1)
    distance = np.square(b).sum(1)[:,None]+np.square(a).sum(1)[None,:]-2*b @ a.T
    return float((labels[distance.argmin(axis=1)] == target).mean())
