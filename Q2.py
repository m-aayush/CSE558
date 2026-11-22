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

y = (kdd.target != b'normal.').to_numpy(dtype=float)

# (run after the setup block above; uses X, y, n, d, SEED)
X64 = X.astype(np.float64)
rank = np.linalg.matrix_rank(X64)
print("rank(D) =",rank," d=",d)

def lossf(A, b, w):
    return float(((A @ w - b)**2).sum())

def countsketch(Z, m, rng):
    """Z: (n, p) array. Returns S Z (m, p) plus the hash arrays."""
    h = rng.integers(0, m, size=Z.shape[0])
    s = rng.choice([-1.0, 1.0], size=Z.shape[0])
    SZ = np.zeros((m, Z.shape[1]))
    np.add.at(SZ, h, s[:, None] * Z)
    return SZ, h, s

def solve(SX, Sy):
    return np.linalg.lstsq(SX, Sy, rcond=None)[0]

# ---- (a) ----
t0 = time.perf_counter()
w_star = np.linalg.lstsq(X64, y, rcond=None)[0]
t_a = time.perf_counter() - t0
loss_star = lossf(X64, y, w_star)
r_vec = y - X64 @ w_star
print("Q2a loss:", loss_star, "time:", t_a)

# ---- (b), (c) ----
MS = [50,100,200,500,1000,2000,5000]
REPS = 5
Z = np.hstack([X64, y[:, None]])          # sketch X and y together with the same S
rows = []
for m in MS:
    for r in range(REPS):
        rng = np.random.default_rng([SEED, 2, m, r])
        t0 = time.perf_counter()
        SZ, _, _ = countsketch(Z, m, rng)
        SX, Sy = SZ[:, :-1], SZ[:, -1]
        w_t = solve(SX, Sy)
        t = time.perf_counter() - t0
        L = lossf(X64, y, w_t)
        sk = float(((SX @ w_t - Sy)**2).sum())
        rows.append((m, r, L, L / loss_star, sk, t))
        print(m, r, L, L / loss_star, sk, t)
        pickle.dump(rows, open("q2_res.pkl", "wb"))

Rb = pd.DataFrame(rows, columns=["m", "rep", "loss", "ratio", "sketch_loss", "time"])
sb = Rb.groupby("m").agg(loss_mean=("loss","mean"),ratio_mean=("ratio", "mean"), ratio_min=("ratio", "min"),
                         ratio_max=("ratio", "max"), sk_mean=("sketch_loss", "mean"),
                         time_mean=("time", "mean"))
print(sb)
print("time of (a):", t_a)

fig, ax = plt.subplots()
ax.errorbar(sb.index, sb.ratio_mean,
            yerr=[sb.ratio_mean - sb.ratio_min, sb.ratio_max - sb.ratio_mean],
            fmt='o-', capsize=4)
ax.set_xscale('log'); ax.set_yscale('log')
print(np.linalg.matrix_rank(X64))
ax.axvline(np.linalg.matrix_rank(X64), color='gray', ls=':')
#ax.axvline(d, color='gray', ls=':')
ax.set_xlabel("m"); ax.set_ylabel("Loss(X,y,w~)/Loss(X,y,w*)")
plt.savefig("q2b.png", dpi=150)

# ---- (d) ----
MD = [50, 500, 5000]        
rn2 = float((r_vec**2).sum())
fig, axs = plt.subplots(1, len(MD), figsize=(12, 3.5))
for ax, m in zip(axs, MD):
    rng = np.random.default_rng([SEED, 4, m])
    vals = []
    for _ in range(100):
        h = rng.integers(0, m, n)
        s = rng.choice([-1.0, 1.0], n)
        Sr = np.bincount(h, weights=s * r_vec, minlength=m)
        vals.append((Sr**2).sum() / rn2)
    vals = np.array(vals)
    print(m, "mean", vals.mean(), "std", vals.std())
    ax.hist(vals, bins=20); ax.set_title(f"m={m}")
plt.savefig("q2d.png", dpi=150)

# ---- (e) ----
M_E = 500                   
rng = np.random.default_rng(SEED)
idx = rng.choice(n, 50000, replace=False)
Xs, ys = X64[idx], y[idx]
Zs = np.hstack([Xs, ys[:, None]])
w_sub = np.linalg.lstsq(Xs, ys, rcond=None)[0]
loss_sub = lossf(Xs, ys, w_sub)

def gaussian_sketch(Zs, m, rng):
    S = rng.normal(0, 1/np.sqrt(m), size=(m, Zs.shape[0]))
    return S @ Zs

out = {"count": [], "gauss": []}
for r in range(5):
    for name in ("count", "gauss"):
        rng = np.random.default_rng([SEED, 5, r, 0 if name == "count" else 1])
        t0 = time.perf_counter()
        if name == "count":
            SZ, _, _ = countsketch(Zs, M_E, rng)
        else:
            SZ = gaussian_sketch(Zs, M_E, rng)
        t_form = time.perf_counter() - t0
        w_t = solve(SZ[:, :-1], SZ[:, -1])
        out[name].append((t_form, lossf(Xs, ys, w_t) / loss_sub))
for name in out:
    a = np.array(out[name])
    print(name, "mean form time:", a[:, 0].mean(), "mean loss ratio:", a[:, 1].mean())

