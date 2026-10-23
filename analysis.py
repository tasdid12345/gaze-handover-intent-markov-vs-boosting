"""Class-balanced LightGBM on gaze n-gram features vs first-order Markov chain, on simulated gaze sequences."""
import json, os, sys
from itertools import groupby
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N, T, K = 3000, 60, 4  # sequences, steps per sequence, AOIs (0 object, 1 robot hand, 2 robot face, 3 elsewhere)
SEEDS = [42, 43, 44]
# shared base one-step transition matrix for both classes
P = np.array([[.78, .08, .08, .06], [.10, .75, .08, .07], [.10, .08, .75, .07], [.10, .08, .08, .74]])


def simulate(n, rng):
    """Same Markov base chain for both classes; handover has fewer but longer object<->robot back-and-forth bursts."""
    y = (rng.random(n) < 0.3).astype(int)
    S = np.zeros((n, T), dtype=int)
    for i in range(n):
        s = [rng.integers(K)]
        for _ in range(T - 1):
            s.append(rng.choice(K, p=P[s[-1]]))
        s = np.array(s)
        nb, lo, hi = (rng.integers(1, 3), 5, 11) if y[i] else (rng.integers(2, 4), 2, 6)
        for _ in range(nb):
            L = rng.integers(lo, hi + 1)
            st = rng.integers(0, T - L)
            r = rng.integers(1, 3)
            s[st:st + L] = [0 if j % 2 == 0 else r for j in range(L)]
        S[i] = s
    return S, y


def features(s):
    """Bigram/trigram counts, alternation stats, dwell stats, last-3 states."""
    bi = np.bincount(s[:-1] * 4 + s[1:], minlength=16)
    tri = np.bincount(s[:-2] * 16 + s[1:-1] * 4 + s[2:], minlength=64)
    runs = [(k, len(list(g))) for k, g in groupby(s)]
    st = [r[0] for r in runs]
    alt = lambda a, b: (a == 0) != (b == 0) and a != 3 and b != 3  # object <-> robot step
    n_alt, chain, best = 0, 1, 1
    for a, b in zip(st[:-1], st[1:]):
        if alt(a, b):
            n_alt += 1; chain += 1
        else:
            chain = 1
        best = max(best, chain)
    dw = []
    for k in range(K):
        d = [r[1] for r in runs if r[0] == k] or [0]
        dw += [np.mean(d), np.std(d), np.max(d)]
    return np.concatenate([bi, tri, [n_alt, best, len(runs)], dw, s[-3:]])


def best_thr(y, sc):
    """Threshold maximising F1 on given (out-of-fold) scores."""
    c = np.quantile(sc, np.linspace(0.02, 0.98, 97))
    return c[np.argmax([f1_score(y, sc >= t) for t in c])]


def evaluate(y, sc, thr):
    return {"f1": f1_score(y, sc >= thr), "auc": roc_auc_score(y, sc), "acc": accuracy_score(y, sc >= thr)}


def fit_markov(S, y):
    """Per-class transition matrices (Laplace smoothed); returns scorer = log-likelihood ratio."""
    lp, li = [], []
    for c in (0, 1):
        X = S[y == c]
        C = np.ones((K, K)); np.add.at(C, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1)
        I = np.bincount(X[:, 0], minlength=K) + 1.0
        lp.append(np.log(C / C.sum(1, keepdims=True))); li.append(np.log(I / I.sum()))
    Tm, I0 = lp[1] - lp[0], li[1] - li[0]
    return lambda A: I0[A[:, 0]] + Tm[A[:, :-1], A[:, 1:]].sum(1)


rng = np.random.default_rng(42)
print("Simulating sequences...")
S, y = simulate(N, rng)
idx = np.arange(N)
itr, ite = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)  # single split
F = np.array([features(s) for s in S])
Str, Ste, Ftr, Fte, ytr, yte = S[itr], S[ite], F[itr], F[ite], y[itr], y[ite]
print(f"train {len(itr)}, test {len(ite)}, handover rate {y.mean():.2f}, features {F.shape[1]}")

res = {"base": [], "imp": []}
for seed in SEEDS:
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    # baseline: out-of-fold LLR scores on train -> threshold; then refit on full train
    oof = np.zeros(len(ytr))
    for a, b in skf.split(Str, ytr):
        oof[b] = fit_markov(Str[a], ytr[a])(Str[b])
    thr = best_thr(ytr, oof)
    res["base"].append(evaluate(yte, fit_markov(Str, ytr)(Ste), thr))
    # improved: class-balanced LightGBM, small grid tuned by 3-fold CV on train; threshold from OOF probabilities
    gs = GridSearchCV(
        LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=seed, n_jobs=1, verbose=-1),
        {"num_leaves": [7, 15], "n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        scoring="f1", cv=skf, n_jobs=1).fit(Ftr, ytr)
    oofp = cross_val_predict(gs.best_estimator_, Ftr, ytr, cv=skf, method="predict_proba")[:, 1]
    res["imp"].append(evaluate(yte, gs.best_estimator_.predict_proba(Fte)[:, 1], best_thr(ytr, oofp)))
    print(f"seed {seed}: baseline F1 {res['base'][-1]['f1']:.3f} | improved F1 {res['imp'][-1]['f1']:.3f} | {gs.best_params_}")

m = {}
for tag, k in (("baseline", "base"), ("improved", "imp")):
    for met in ("f1", "auc", "acc"):
        v = [r[met] for r in res[k]]
        m[f"{tag}_{met}"] = float(np.mean(v)); m[f"{tag}_{met}_std"] = float(np.std(v))
    m[f"{tag}_score"], m[f"{tag}_score_std"] = m[f"{tag}_f1"], m[f"{tag}_f1_std"]
m["improvement"] = m["improved_score"] - m["baseline_score"]
if not all(np.isfinite(list(m.values()))):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
names = ["f1", "auc", "acc"]
w = 0.38
plt.bar(np.arange(3) - w / 2, [m[f"baseline_{n}"] for n in names], w, label="Markov chain")
plt.bar(np.arange(3) + w / 2, [m[f"improved_{n}"] for n in names], w, label="LightGBM")
plt.xticks(range(3), ["F1 (handover)", "ROC-AUC", "Accuracy"]); plt.legend(); plt.title("Test set, mean of 3 seeds")
plt.savefig("figures/comparison.png", dpi=120, bbox_inches="tight")

out = {
    "question": "Can a class-balanced LightGBM model on gaze n-gram and back-and-forth features predict handover intent better than a first-order Markov chain classifier on simulated gaze sequences?",
    "method": "Seeded simulated AOI gaze sequences (4 AOIs, 60 steps), stratified 70/30 split. Baseline: per-class first-order Markov chains with LLR and a CV-tuned threshold. Improved: class-balanced LightGBM tuned by 3-fold CV on n-gram, alternation, dwell and last-state features. 3 seeds.",
    "dataset": {"name": "Simulated gaze AOI sequences (handover vs no handover)", "source_url": "generated in script with numpy (no external data)",
                "n_rows": int(N), "n_train": int(len(itr)), "n_test": int(len(ite)), "same_as_paper": False,
                "baseline_is_papers_method": False},
    "metric_name": "Handover-class F1 on test set", "higher_is_better": True,
    "baseline": "First-order Markov chain per class, log-likelihood-ratio classifier, threshold chosen on out-of-fold train scores to maximise F1.",
    "improved": "Class-balanced LightGBM on bigram/trigram counts, object-robot alternation counts, dwell-time statistics and last-3 states; grid tuned by 3-fold CV; threshold chosen on out-of-fold train probabilities.",
    "metrics": m,
    "notes": "Simulated data built so handover differs mainly by longer object-robot back-and-forth bursts while one-step statistics are similar; the result reflects this design and is not a reproduction of the paper. Data simulated and split once (seed 42); seeds 42-44 vary CV folds and LightGBM randomness only, so the Markov baseline std comes only from threshold-fold variation. Baseline is a generic Markov chain, not the paper's method."}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Wrote results.json")
