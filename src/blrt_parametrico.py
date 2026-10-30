#!/usr/bin/env python3
"""BLRT parametrico (logica mclust) + nulo de copula gaussiana, em Python/scikit-learn.
Uso: python blrt_parametrico.py D22_scored.csv      (matriz n x 22 de escores brutos; e repadronizada)
Saida: blrt_resultados.json.  B = 1000 por padrao; n_init = 10; relata nao convergencia e falhas.
Inclui (i) autoteste do comportamento de GaussianMixture.sample() e (ii) validacao com K conhecido (1 e 2 grupos).
Ambiente de referencia: Python 3.13, numpy 2.3, scikit-learn 1.6.1, scipy 1.17.
"""
import sys, json, copy, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
warnings.simplefilter("ignore", ConvergenceWarning)
B, N_INIT, SEED = 1000, 10, 42

def fit(X, K, cov, seed):
    return GaussianMixture(K, covariance_type=cov, n_init=N_INIT, random_state=seed, max_iter=300, reg_covar=1e-6).fit(X)

def lr(X, K, cov, seed=SEED):
    a, c = fit(X, K, cov, seed), fit(X, K + 1, cov, seed)
    return float(2 * (c.score(X) - a.score(X)) * len(X)), a, bool(a.converged_ and c.converged_)

def rep_param(gK, n, K, cov, b):
    g = copy.deepcopy(gK); g.random_state = b   # ESSENCIAL: sample() reutiliza o random_state do objeto
    Xb, _ = g.sample(n); v, _, conv = lr(Xb, K, cov, b); return v, conv

def copula(D, L, b):
    rng = np.random.default_rng(b); U = stats.norm.cdf(rng.standard_normal(D.shape) @ L.T)
    X = np.column_stack([np.quantile(D[:, j], U[:, j]) for j in range(D.shape[1])])
    return StandardScaler().fit_transform(X)

def rep_cop(D, L, K, cov, b):
    v, _, conv = lr(copula(D, L, b), K, cov, b); return v, conv

def blrt(D, K, cov, nulo="param", B=B):
    Z = StandardScaler().fit_transform(D); LR, gK, conv = lr(Z, K, cov)
    if nulo == "param":
        out = Parallel(n_jobs=-1)(delayed(rep_param)(gK, len(Z), K, cov, b) for b in range(B))
    else:
        rho = stats.spearmanr(D).correlation
        L = np.linalg.cholesky(2 * np.sin(np.pi * rho / 6) + 1e-9 * np.eye(D.shape[1]))
        out = Parallel(n_jobs=-1)(delayed(rep_cop)(D, L, K, cov, b) for b in range(B))
    bt = np.array([v for v, _ in out if np.isfinite(v)])
    return dict(K=K, cov=cov, nulo=nulo, B=int(len(bt)), LR_obs=LR, p=float((1 + (bt >= LR).sum()) / (len(bt) + 1)),
                P50=float(np.median(bt)), P95=float(np.percentile(bt, 95)),
                n_nonconv=int(sum(not c for _, c in out)), n_fail=int(B - len(bt)), obs_converged=conv)

def autoteste():
    g = fit(np.random.default_rng(0).normal(size=(50, 3)), 1, "spherical", 0)
    assert np.allclose(g.sample(5)[0], g.sample(5)[0]), "sample() mudou de comportamento; revisar rep_param"
    rng = np.random.default_rng(SEED)
    sim1 = rng.normal(size=(238, 22))
    lab = rng.integers(0, 2, 238); sim2 = rng.normal(size=(238, 22)); sim2[:, :6] += 2.0 * lab[:, None]
    return [blrt(sim1, 1, "spherical"), blrt(sim2, 1, "spherical"), blrt(sim2, 2, "spherical")]

if __name__ == "__main__":
    D = pd.read_csv(sys.argv[1]).select_dtypes("number").values
    res = {"validacao_simulada": autoteste(), "dados_reais": []}
    for nulo in ("param", "copula"):
        for cov in ("spherical", "diag", "tied", "full"):
            for K in (1, 2, 3):
                r = blrt(D, K, cov, nulo); res["dados_reais"].append(r); print(r)
    json.dump(res, open("blrt_resultados.json", "w"), indent=1, default=float)
