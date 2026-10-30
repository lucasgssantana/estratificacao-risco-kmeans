#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE DO PAPER — "Gradiente ou categorias? Teste da estrutura de agrupamento
em indicadores de regulação emocional e esquemas emocionais em adultos brasileiros"

Estende pipeline_final.py (TCC) com os testes formais da hipótese de contínuo:
  0. Conferência de reprodução (K=2: 135/103, silhueta 0,212)  -> PARA se falhar
  1. Nulos da silhueta: normal multivariada e cópula gaussiana (B=1000)
  2. LPA/GMM K=1..5, 4 estruturas de covariância: BIC, aBIC, entropia, ARI vs K-Means
  3. BLRT K vs K+1 sob nulo paramétrico e nulo de cópula (ver blrt_parametrico.py)
  4. Estabilidade: bootstrap (100), split-half não circular (200), sementes K=3 (100)
  5. Sensibilidade à composição: α<0,50; 5 fatores Santana (2024); DERS+ERQ; sem Duração
  6. Tautologia DERS–DASS: perfil vs DERS dicotomizada (43,3%) vs DERS contínua vs 22 dims
  7. Normalidade multivariada (Mardia) — justificativa do nulo de cópula

Uso:  DATA_DIR=<csv> OUT_DIR=outputs/paper python src/pipeline_paper.py
Entrada alternativa (depósito): BASE_ANONIMIZADA_n238.csv (itens brutos; pula Etapa 1 do TCC)
Sementes: 42 (principal) e 2026 (reamostragens adicionais). Sem vazamento: toda padronização
e K-Means são reajustados dentro de cada partição/reamostra.
"""
import os, sys, json, copy, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from pipeline_final import (keyed, score22, scoredass, relabel, DIMS22, DERS_MAP, LESS_MAP,
                            EQR_MAP, DASS_MAP, DERS6, LESS_REV, DERS_REV)   # reutiliza escoragem do TCC
warnings.simplefilter("ignore", ConvergenceWarning)

SEED, SEED2 = 42, 2026
N_NULL, N_BOOT, N_SPLIT, N_SEEDS, B_BLRT = 1000, 100, 200, 100, 1000
DATA = os.environ.get("DATA_DIR", "data"); OUT = os.environ.get("OUT_DIR", "outputs/paper")
FIG, TAB = f"{OUT}/figuras", f"{OUT}/tabelas"
for d in (OUT, FIG, TAB): os.makedirs(d, exist_ok=True)
R = {}

def km(Z, k, seed=SEED, n_init=10):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(Z)

def zfit(D):                       # padronização SEMPRE dentro da partição
    return StandardScaler().fit_transform(D)

# ------------------------------------------------------------- 0. ENTRADA + CONFERÊNCIA
dep = os.path.join(DATA, "BASE_ANONIMIZADA_n238.csv")
if os.path.isfile(dep):                                      # depósito anonimizado (itens brutos)
    base = pd.read_csv(dep)
    items = base[[c for c in base.columns if c.split("_")[0] in ("ders", "eqr", "less", "dass")]]
    ref2, ref3 = base.get("perfil_k2_ref"), base.get("perfil_k3_ref")
else:                                                        # bases internas: reaproveita Etapa 1 do TCC
    from pipeline_final import X_items as items
    ref2 = ref3 = None
K95 = keyed(items); D22 = score22(K95); DASS = scoredass(K95); n = len(D22)
Z = zfit(D22.values)
L2 = relabel(km(Z, 2).labels_, pd.DataFrame(Z, columns=DIMS22), 2)
L3 = relabel(km(Z, 3).labels_, pd.DataFrame(Z, columns=DIMS22), 3)
sil2, sil3 = silhouette_score(Z, L2), silhouette_score(Z, L3)
R["conferencia"] = {"n": n, "K2_n": np.bincount(L2).tolist(), "K2_sil": round(sil2, 3),
                    "K3_n": np.bincount(L3).tolist(), "K3_sil": round(sil3, 3)}
assert np.bincount(L2).tolist() == [135, 103] and abs(sil2 - 0.212) < 0.002, \
    f"REPRODUÇÃO FALHOU: {R['conferencia']} — não prosseguir"
if ref2 is not None:
    R["conferencia"]["ARI_K2_vs_ref"] = adjusted_rand_score(L2, ref2)
    R["conferencia"]["ARI_K3_vs_ref"] = adjusted_rand_score(L3, ref3)   # ~0,989: 1 caso de fronteira (ver S-A)
# Regra: perfil_k3_ref (91/87/60) só para tabelas descritivas; nenhuma análise abaixo o usa como entrada.

# ------------------------------------------------------------- 7. NORMALIDADE (justifica cópula)
S = np.cov(Z.T); Si = np.linalg.inv(S); d2 = np.einsum("ij,jk,ik->i", Z, Si, Z); p = Z.shape[1]
R["normalidade"] = {"skew_abs_max": float(np.abs(stats.skew(Z, 0)).max()),
                    "n_dims_skew_gt_05": int((np.abs(stats.skew(Z, 0)) > .5).sum()),
                    "kurt_min": float(stats.kurtosis(Z, 0).min()),
                    "mardia_b2": float((d2 ** 2).mean()), "mardia_b2_expected": p * (p + 2),
                    "shapiro_p_lt_001": int(sum(stats.shapiro(D22.iloc[:, j]).pvalue < .001 for j in range(p)))}

# ------------------------------------------------------------- 1. NULOS DA SILHUETA
def copula_sample(D, L, rng):
    """Cópula gaussiana: dependência gaussiana (Cholesky da correlação) + marginais empíricas."""
    G = rng.standard_normal(D.shape) @ L.T; U = stats.norm.cdf(G)
    return np.column_stack([np.quantile(D[:, j], U[:, j]) for j in range(D.shape[1])])
rho = stats.spearmanr(D22.values).correlation; Lc = np.linalg.cholesky(2 * np.sin(np.pi * rho / 6) + 1e-9 * np.eye(p))
mu, cov = Z.mean(0), np.cov(Z.T); rng = np.random.default_rng(SEED)
nullG = {2: [], 3: []}; nullC = {2: [], 3: []}
for b in range(N_NULL):
    Xg = zfit(rng.multivariate_normal(mu, cov, n)); Xc = zfit(copula_sample(D22.values, Lc, rng))
    for k in (2, 3):
        nullG[k].append(silhouette_score(Xg, km(Xg, k).labels_)); nullC[k].append(silhouette_score(Xc, km(Xc, k).labels_))
R["nulos_silhueta"] = {}
for k, obs in ((2, sil2), (3, sil3)):
    g, c = np.array(nullG[k]), np.array(nullC[k])
    R["nulos_silhueta"][f"K{k}"] = {"obs": obs, "gauss_P95": float(np.percentile(g, 95)), "gauss_p": float((1 + (g >= obs).sum()) / (N_NULL + 1)),
                                    "copula_P95": float(np.percentile(c, 95)), "copula_p": float((1 + (c >= obs).sum()) / (N_NULL + 1))}
    np.save(f"{OUT}/null_gauss_K{k}.npy", g); np.save(f"{OUT}/null_copula_K{k}.npy", c)

# ------------------------------------------------------------- 2. LPA / GMM K=1..5
lpa = []
for cov_type, ninit in (("spherical", 10), ("diag", 10), ("tied", 5), ("full", 3)):
    for k in range(1, 6):
        g = GaussianMixture(k, covariance_type=cov_type, n_init=ninit, random_state=SEED, max_iter=300, reg_covar=1e-4).fit(Z)
        ll = g.score(Z) * n; d = g._n_parameters(); post = g.predict_proba(Z); pred = g.predict(Z)
        ent = -(post * np.log(np.clip(post, 1e-300, 1))).sum()
        lpa.append({"cov": cov_type, "K": k, "n_par": d, "BIC": -2 * ll + d * np.log(n), "aBIC": -2 * ll + d * np.log((n + 2) / 24),
                    "entropia_rel": 1 - ent / (n * np.log(k)) if k > 1 else 1.0, "menor_classe": int(np.bincount(pred, minlength=k).min()),
                    "ARI_vs_KMeans": adjusted_rand_score(L2 if k == 2 else L3, pred) if k in (2, 3) else np.nan})
LPA = pd.DataFrame(lpa); LPA.to_csv(f"{TAB}/lpa_gmm_K1a5.csv", index=False)
R["lpa_melhorK_BIC"] = {c: int(LPA[LPA.cov == c].loc[LPA[LPA.cov == c].BIC.idxmin(), "K"]) for c in LPA.cov.unique()}
# Fato 3 (sobreparametrização): plena = 275 par./componente; K=2 -> 551 com n=238 — declarar no texto.

# ------------------------------------------------------------- 3. BLRT (delegado)
from blrt_parametrico import blrt
R["blrt"] = [blrt(D22.values, K, cov_type, nulo, B=B_BLRT)
             for nulo in ("param", "copula") for cov_type in ("spherical", "diag", "tied", "full") for K in (1, 2, 3)]
pd.DataFrame(R["blrt"]).to_csv(f"{TAB}/blrt.csv", index=False)

# ------------------------------------------------------------- 4. ESTABILIDADE (sem vazamento)
rng = np.random.default_rng(SEED)
boot = {k: [] for k in (2, 3)}
for b in range(N_BOOT):
    idx = rng.integers(0, n, n); Zb = zfit(D22.values[idx])          # repadroniza na reamostra
    for k, lab in ((2, L2), (3, L3)): boot[k].append(adjusted_rand_score(lab[idx], km(Zb, k).labels_))
split = {k: [] for k in (2, 3)}
for r in range(N_SPLIT):
    perm = rng.permutation(n); A, B_ = perm[: n // 2], perm[n // 2:]
    for k in (2, 3):
        scA = StandardScaler().fit(D22.values[A]); scB = StandardScaler().fit(D22.values[B_])
        mA = km(scA.transform(D22.values[A]), k, seed=r, n_init=5); mB = km(scB.transform(D22.values[B_]), k, seed=r, n_init=5)
        split[k].append(adjusted_rand_score(mA.predict(scA.transform(D22.values[B_])), mB.labels_))   # ARI entre modelos independentes
seeds3 = [adjusted_rand_score(relabel(km(Z, 3, seed=s).labels_, pd.DataFrame(Z, columns=DIMS22), 3), L3) for s in range(N_SEEDS)]
R["estabilidade"] = {f"bootstrap_K{k}": {"media": float(np.mean(boot[k])), "dp": float(np.std(boot[k]))} for k in (2, 3)}
R["estabilidade"].update({f"splithalf_K{k}": {"mediana": float(np.median(split[k])), "P5": float(np.percentile(split[k], 5))} for k in (2, 3)})
R["estabilidade"]["K3_sementes"] = {"ARI_mediana": float(np.median(seeds3)), "ARI_min": float(np.min(seeds3)), "prop_lt_090": float(np.mean(np.array(seeds3) < .9))}
for k in (2, 3): np.save(f"{OUT}/splithalf_K{k}.npy", np.array(split[k]))

# ------------------------------------------------------------- 5. SENSIBILIDADE À COMPOSIÇÃO
def alpha2(i, j): r = np.corrcoef(K95[f"less_{i}"], K95[f"less_{j}"])[0, 1]; return 2 * r / (1 + r)
low = [nm for nm, it in LESS_MAP.items() if alpha2(*it) < .50]
SANTANA5 = {"F1": [3, 5, 7, 9, 10, 16, 17, 20, 22], "F2": [14, 21, 23, 26, 28], "F3": [11, 13, 27], "F4": [2, 18, 19], "F5": [1, 4, 6, 8, 12, 15, 24, 25]}
F5 = pd.DataFrame({f: K95[[f"less_{i}" for i in v]].mean(1) for f, v in SANTANA5.items()})
scen = {"C0_22dim": D22,
        "C6a_alpha>=0.50": D22.drop(columns=low),
        "C6b_5fatores": pd.concat([D22[["supressao", "reavaliacao"] + DERS6], F5], axis=1),
        "C6c_DERS+EQR": D22[["supressao", "reavaliacao"] + DERS6],
        "C6d_sem_duracao": D22.drop(columns=["duracao"])}
R["sensibilidade"] = {}
for nm, D in scen.items():
    Zs = zfit(D.values); l2 = km(Zs, 2).labels_; l3 = km(Zs, 3).labels_; s2 = silhouette_score(Zs, l2); s3 = silhouette_score(Zs, l3)
    rho_s = stats.spearmanr(D.values).correlation; Ls = np.linalg.cholesky(2 * np.sin(np.pi * rho_s / 6) + 1e-9 * np.eye(D.shape[1]))
    rng_s = np.random.default_rng(SEED); nc2 = []; nc3 = []
    for b in range(N_NULL):
        Xc = zfit(copula_sample(D.values, Ls, rng_s)); nc2.append(silhouette_score(Xc, km(Xc, 2).labels_)); nc3.append(silhouette_score(Xc, km(Xc, 3).labels_))
    bic = {c: int(np.argmin([GaussianMixture(k, covariance_type=c, n_init=5, random_state=SEED, reg_covar=1e-4).fit(Zs).bic(Zs) for k in range(1, 6)]) + 1) for c in ("spherical", "diag", "tied", "full")}
    R["sensibilidade"][nm] = {"p": D.shape[1], "sizes_K2": sorted(np.bincount(l2).tolist(), reverse=True), "sil_K2": round(s2, 3),
                              "pMC_K2": float((1 + (np.array(nc2) >= s2).sum()) / (N_NULL + 1)), "pMC_K3": float((1 + (np.array(nc3) >= s3).sum()) / (N_NULL + 1)),
                              "ARI_vs_main_K2": adjusted_rand_score(L2, l2), "ARI_vs_main_K3": adjusted_rand_score(L3, l3), "BIC_bestK": bic}

# ------------------------------------------------------------- 6. TAUTOLOGIA DERS–DASS
cut = {"depressao": 14, "ansiedade": 10, "estresse": 19}
anym = ((DASS["depressao"] >= 14) | (DASS["ansiedade"] >= 10) | (DASS["estresse"] >= 19)).astype(int).values
ders_tot = D22[DERS6].sum(1).values; thr = np.quantile(ders_tot, 1 - L2.mean()); Dd = (ders_tot >= thr).astype(int)
cv = StratifiedKFold(10, shuffle=True, random_state=SEED)
def cvauc(X): return roc_auc_score(anym, cross_val_predict(LogisticRegression(max_iter=5000, C=1e6), X, anym, cv=cv, method="predict_proba")[:, 1])
def OR(a): ct = pd.crosstab(a, anym).values; o = ct[1, 1] * ct[0, 0] / (ct[1, 0] * ct[0, 1]); se = np.sqrt((1 / ct).sum()); return [o, np.exp(np.log(o) - 1.96 * se), np.exp(np.log(o) + 1.96 * se)]
R["tautologia"] = {"perfil_K2": {"OR": OR(L2), "AUC": roc_auc_score(anym, L2), "prop_grupo_sup": float(L2.mean())},
                   "DERS_dicot_mesma_prop": {"OR": OR(Dd), "AUC": roc_auc_score(anym, Dd), "concordancia_com_perfil": float((Dd == L2).mean())},
                   "DERS_continua": {"AUC_cv": cvauc(ders_tot.reshape(-1, 1))}, "22_dimensoes": {"AUC_cv": cvauc(Z)}}

# ------------------------------------------------------------- FIGURAS E SAÍDA
def fig_nulos():
    f, ax = plt.subplots(1, 2, figsize=(9, 3.5), dpi=300)
    for a, k, obs in ((ax[0], 2, sil2), (ax[1], 3, sil3)):
        a.hist(nullG[k], bins=25, alpha=.55, label="nulo gaussiano"); a.hist(nullC[k], bins=25, alpha=.55, label="nulo cópula")
        a.axvline(obs, color="r", lw=2, label=f"observado {obs:.3f}"); a.set_title(f"K = {k}"); a.legend(fontsize=7)
    f.tight_layout(); f.savefig(f"{FIG}/F_nulos_silhueta.png"); plt.close(f)
def fig_bic():
    f, ax = plt.subplots(figsize=(6, 3.5), dpi=300)
    for c in LPA.cov.unique(): s = LPA[LPA.cov == c]; ax.plot(s.K, s.BIC, "o-", label=c)
    ax.set_xlabel("K"); ax.set_ylabel("BIC"); ax.legend(); f.tight_layout(); f.savefig(f"{FIG}/F_bic_lpa.png"); plt.close(f)
fig_nulos(); fig_bic()
json.dump(R, open(f"{OUT}/results_paper.json", "w"), indent=1, default=float)
print(json.dumps(R["conferencia"]), "\n", json.dumps(R["nulos_silhueta"]), "\n", R["lpa_melhorK_BIC"])
