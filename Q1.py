import numpy as np, pandas as pd, time, pickle
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans

SEED = 225
K = 15

# ---------- build X ----------
kdd = fetch_kddcup99(subset=None, as_frame=True)
df = kdd.data.copy()
cat = ['protocol_type', 'service', 'flag']
for c in cat:
    df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
num = [c for c in df.columns if c not in cat]
Xdf = pd.concat([df[num].astype(float),
                 pd.get_dummies(df[cat]).astype(float)], axis=1)
Xdf = Xdf.loc[:, Xdf.nunique() > 1]                 # drop constant columns
Xdf = (Xdf - Xdf.min()) / (Xdf.max() - Xdf.min())   # min-max to [0,1]
X = Xdf.to_numpy(dtype=np.float32)
n, d = X.shape
print("n =", n, "d =", d)


# ---------- helpers ----------
def cost(X, C, chunk=50000):
    """sum over rows of min squared distance to centres C (chunked)."""
    C = np.asarray(C, dtype=np.float64)
    c2 = (C**2).sum(1)
    tot = 0.0
    for i in range(0, len(X), chunk):
        B = X[i:i+chunk].astype(np.float64)
        D = (B**2).sum(1)[:, None] - 2 * B @ C.T + c2[None, :]
        tot += np.maximum(D.min(1), 0).sum()
    return tot

def centroids(X, labels, k=K):
    return np.vstack([X[labels == j].astype(np.float64).mean(0) for j in range(k) if (labels == j).any()])



def partition_cost(X, labels, k=K):
    tot = 0.0
    for j in range(k):
        if not (labels == j).any():
            continue
        B = X[labels == j].astype(np.float64)
        tot += ((B - B.mean(0))**2).sum()
    return tot


def run_kmeans(A, seed):
    km = KMeans(n_clusters=K, n_init=5, random_state=seed)
    km.fit(A)
    return km

# =====================  Q1  =====================
# ---- (a) ----
t0 = time.perf_counter()
km0 = run_kmeans(X, SEED)
t_full = time.perf_counter() - t0
P = km0.labels_
C = km0.cluster_centers_
cost_full = cost(X, C)
print("Q1a cost:", cost_full, "inertia:", km0.inertia_, "time:", t_full)

# ---- (b), (c), (d) ----
KS = [5, 10, 15, 20, 25]          
REPS = 5
res = []                          # rows: kprime, rep, ratio_hat, ratio_pinv, time
for kp in KS:
    for r in range(REPS):
        rng = np.random.default_rng([SEED, kp, r])
        Pi = rng.normal(0, 1/np.sqrt(kp), size=(kp, d)).astype(np.float32)
        t0 = time.perf_counter()
        Y = X @ Pi.T
        km = run_kmeans(Y, SEED + r)
        t = time.perf_counter() - t0

        Chat = centroids(X, km.labels_)
        if len(Chat) < K: print("WARNING: empty cluster", kp, r)
        ratio_hat = cost(X, Chat) / cost_full

        Cplus = km.cluster_centers_ @ np.linalg.pinv(Pi.astype(np.float64)).T  # (K, d)
        ratio_pinv = cost(X, Cplus) / cost_full

        res.append((kp, r, ratio_hat, ratio_pinv, t))
        print(kp, r, ratio_hat, ratio_pinv, t)
        pickle.dump(res, open("q1_res.pkl", "wb"))

R = pd.DataFrame(res, columns=["kp", "rep", "ratio_hat", "ratio_pinv", "time"])
summ = R.groupby("kp").agg(
    hat_mean=("ratio_hat", "mean"), hat_min=("ratio_hat", "min"), hat_max=("ratio_hat", "max"),
    pinv_mean=("ratio_pinv", "mean"), pinv_min=("ratio_pinv", "min"), pinv_max=("ratio_pinv", "max"),
    time_mean=("time", "mean"))
print(summ)

# (b) bar graph
fig, ax = plt.subplots()
ax.bar(np.arange(len(KS)), summ.hat_mean,
       yerr=[summ.hat_mean - summ.hat_min, summ.hat_max - summ.hat_mean], capsize=4)
ax.set_xticks(np.arange(len(KS))); ax.set_xticklabels(KS)
ax.set_xlabel("m"); ax.set_ylabel("cost(X, C_hat)/cost(X, C)")
ax.axhline(1, color='k', ls='--'); plt.savefig("q1b.png", dpi=150)

# (d) time plot
fig, ax = plt.subplots()
ax.plot(KS, summ.time_mean, 'o-', label="project + k-means")
ax.axhline(t_full, color='r', ls='--', label="k-means on X")
ax.set_xlabel("m"); ax.set_ylabel("seconds"); ax.legend()
plt.savefig("q1d.png", dpi=150)

# ---- (e) ----
rng = np.random.default_rng(SEED)
NP = 10000
I_list, J_list, have = [], [], 0
while have < NP:
    i = rng.integers(0, n, 30000)
    j = (i + rng.integers(1, n, 30000)) % n          # i != j
    ok = (X[i] != X[j]).any(axis=1)                  # rows actually differ
    I_list.append(i[ok]); J_list.append(j[ok]); have += ok.sum()
I = np.concatenate(I_list)[:NP]
J = np.concatenate(J_list)[:NP]
npairs = NP

def pair_ratios(Y, chunk=200000):
    out = []
    for s in range(0, npairs, chunk):
        i, j = I[s:s+chunk], J[s:s+chunk]
        dx = ((X[i].astype(np.float64) - X[j])**2).sum(1)
        dy = ((Y[i].astype(np.float64) - Y[j])**2).sum(1)
        m = dx > 0                                   # drop identical rows
        out.append(dy[m] / dx[m])
    return np.concatenate(out)

pair_data, cost_data = [], []
cost_X_P = partition_cost(X, P)
for kp in KS:
    rng = np.random.default_rng([SEED, 999, kp])
    Pi = rng.normal(0, 1/np.sqrt(kp), (kp, d)).astype(np.float32)
    pair_data.append(pair_ratios(X @ Pi.T))
    cr = []
    for r in range(20):
        Pi = rng.normal(0, 1/np.sqrt(kp), (kp, d)).astype(np.float32)
        cr.append(partition_cost(X @ Pi.T, P) / cost_X_P)
    cost_data.append(np.array(cr))
    print(kp, pair_data[-1].std(), np.std(cr))

fig, ax = plt.subplots(figsize=(10, 5))
pos = np.arange(len(KS))
b1 = ax.boxplot(pair_data, positions=pos - 0.18, widths=0.3, showfliers=False,
                patch_artist=True, boxprops=dict(facecolor='lightblue'))
b2 = ax.boxplot(cost_data, positions=pos + 0.18, widths=0.3, showfliers=False,
                patch_artist=True, boxprops=dict(facecolor='salmon'))
ax.set_xticks(pos); ax.set_xticklabels(KS); ax.axhline(1, color='k', ls='--')
ax.set_xlabel("m"); ax.legend([b1["boxes"][0], b2["boxes"][0]],
                              ["pairwise sq-distance ratio", "cost(Y,P)/cost(X,P)"])
plt.savefig("q1e.png", dpi=150)
