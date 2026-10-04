#!/usr/bin/env python3
"""
CSE-307 Term Paper, Track 1: Page replacement with a small learned policy.
Student: Farhan Sadik Tahsin (202414072)

Pure Python 3.10+, no third-party packages.
Run from the repository root:  python3 src/run_experiment.py
"""
import csv, json, os, random, statistics

FRAMES, PAGES = 4, 10                 # 4 physical frames, 10 virtual pages
HALF = 400                            # references per phase (800 total)
EVAL_SEED, TRAIN_SEED = 307, 2026
WINDOW = 40                           # hybrid guard window
POLICIES = ["FIFO", "LRU", "Optimal", "Learned", "Hybrid"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# ------------------------------------------------------------------ workload
def locality_phase(rng, n):
    """Student is working on one assignment: pages 0-4 dominate, with repeats
    of recent pages and an occasional sequential run."""
    t = []
    while len(t) < n:
        r = rng.random()
        if t and r < 0.55:
            t.append(rng.choice(t[-3:]))                      # re-use a recent page
        elif r < 0.65:
            s = rng.randrange(0, 4); t += [s, s + 1, s + 2]    # short sequential run
        elif r < 0.92:
            t.append(rng.randrange(0, 5))                      # hot set 0-4
        else:
            t.append(rng.randrange(5, PAGES))                  # rare cold page
    return t[:n]

def shift_phase(rng, n):
    """Assignment submitted; student starts browsing: random pages over 0-9
    with short bursts on one page."""
    t = []
    while len(t) < n:
        p = rng.randrange(PAGES)
        t += [p] * (rng.randint(2, 3) if rng.random() < 0.2 else 1)
    return t[:n]

def make_trace(seed):
    rng = random.Random(seed)
    return locality_phase(rng, HALF) + shift_phase(rng, HALF)

# ------------------------------------------------------------------ helpers
def next_use_of(trace, i, p):
    for j in range(i + 1, len(trace)):
        if trace[j] == p: return j
    return 10**9

def candidate_features(i, frames, last, loaded, freq):
    """Features for every resident page at the moment of a fault. Only past
    information is used: age, frequency, recency rank, page id."""
    by_recency = sorted(frames, key=lambda p: last[p])
    return {p: [i - loaded[p], freq[p], by_recency.index(p), p] for p in frames}

# ------------------------------------------------------------------ tiny decision tree
def gini(pos, n):
    return 0.0 if n == 0 else 2 * (pos / n) * (1 - pos / n)

def build_tree(X, y, depth, max_depth=3, min_leaf=5):
    pos, n = sum(y), len(y)
    leaf = {"p": pos / n if n else 0.0}
    if depth >= max_depth or n < 2 * min_leaf or pos in (0, n):
        return leaf
    best = None
    for f in range(len(X[0])):
        vals = sorted(set(r[f] for r in X))
        for a, b in zip(vals, vals[1:]):
            thr = (a + b) / 2
            L = [k for k, r in enumerate(X) if r[f] <= thr]
            if len(L) < min_leaf or n - len(L) < min_leaf: continue
            lp = sum(y[k] for k in L)
            score = (len(L) * gini(lp, len(L)) + (n - len(L)) * gini(pos - lp, n - len(L))) / n
            if best is None or score < best[0]: best = (score, f, thr)
    if best is None: return leaf
    _, f, thr = best
    Li = [k for k, r in enumerate(X) if r[f] <= thr]; Ri = [k for k, r in enumerate(X) if r[f] > thr]
    return {"f": f, "t": thr, "p": leaf["p"],
            "L": build_tree([X[k] for k in Li], [y[k] for k in Li], depth + 1, max_depth, min_leaf),
            "R": build_tree([X[k] for k in Ri], [y[k] for k in Ri], depth + 1, max_depth, min_leaf)}

def tree_predict(node, x):
    while "f" in node: node = node["L"] if x[node["f"]] <= node["t"] else node["R"]
    return node["p"]

FEATURES = ["age", "frequency", "recency_rank", "page_id"]
def tree_text(node, ind=0):
    pad = "  " * ind
    if "f" not in node: return f"{pad}-> P(evict)={node['p']:.2f}\n"
    return (f"{pad}if {FEATURES[node['f']]} <= {node['t']:.1f}:\n" + tree_text(node["L"], ind + 1) +
            f"{pad}else:\n" + tree_text(node["R"], ind + 1))

def train_policy():
    """Train on a SEPARATE locality-only trace. Label = 1 if Belady would
    evict that candidate. Future knowledge is used only for labels here."""
    tr = locality_phase(random.Random(TRAIN_SEED), HALF)
    frames, last, loaded, freq = [], {}, {}, {}
    X, y = [], []
    for i, p in enumerate(tr):
        if p not in frames:
            if len(frames) == FRAMES:
                feats = candidate_features(i, frames, last, loaded, freq)
                victim = max(frames, key=lambda q: next_use_of(tr, i, q))
                for q in frames: X.append(feats[q]); y.append(int(q == victim))
                frames.remove(victim); freq.pop(victim)
            frames.append(p); loaded[p] = i; freq[p] = 0
        freq[p] += 1; last[p] = i
    return build_tree(X, y, 0), len(y)

# ------------------------------------------------------------------ simulator
def simulate(trace, policy, tree=None):
    """Return list of 1 (hit) / 0 (fault) per reference."""
    frames, last, loaded, freq, hits = [], {}, {}, {}, []
    sh_frames, sh_last, sh_hits = [], {}, []          # shadow LRU (hybrid guard)
    for i, p in enumerate(trace):
        if p in sh_frames: sh_hits.append(1)
        else:
            sh_hits.append(0)
            if len(sh_frames) == FRAMES: sh_frames.remove(min(sh_frames, key=lambda q: sh_last[q]))
            sh_frames.append(p)
        sh_last[p] = i
        if p in frames:
            hits.append(1)
        else:
            hits.append(0)
            if len(frames) == FRAMES:
                if policy == "FIFO":      v = min(frames, key=lambda q: loaded[q])
                elif policy == "LRU":     v = min(frames, key=lambda q: last[q])
                elif policy == "Optimal": v = max(frames, key=lambda q: next_use_of(trace, i, q))
                else:
                    feats = candidate_features(i, frames, last, loaded, freq)
                    # highest P(evict); ties -> least recently used
                    lv = max(frames, key=lambda q: (tree_predict(tree, feats[q]), -last[q]))
                    if policy == "Learned": v = lv
                    else:                   # Hybrid: LRU fallback while shadow LRU is winning
                        lo = max(0, i - WINDOW)
                        v = min(frames, key=lambda q: last[q]) if sum(sh_hits[lo:i]) > sum(hits[lo:i]) else lv
                frames.remove(v); freq.pop(v)
            frames.append(p); loaded[p] = i; freq[p] = 0
        freq[p] += 1; last[p] = i
    return hits

def run_all(trace, tree):
    return {pol: simulate(trace, pol, tree) for pol in POLICIES}

def phase_stats(h):
    a, b = h[:HALF], h[HALF:]
    return {"locality": (HALF - sum(a), sum(a) / HALF), "shift": (HALF - sum(b), sum(b) / HALF)}

# ------------------------------------------------------------------ outputs
def write_csv(path, header, rows):
    with open(os.path.join(OUT, path), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def svg_bars(res):
    W, H, L, B, T = 640, 330, 60, 50, 40
    cw, ch = W - L - 20, H - B - T
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15">Hit ratio before and after the workload shift</text>']
    for k in range(6):
        y = T + ch - ch * k / 5
        s.append(f'<line x1="{L}" x2="{W-20}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="{L-8}" y="{y+4}" text-anchor="end">{k/5:.1f}</text>')
    gw = cw / len(POLICIES)
    for gi, pol in enumerate(POLICIES):
        st = phase_stats(res[pol])
        for bi, (ph, col) in enumerate([("locality", "#2a6f97"), ("shift", "#e07a1f")]):
            v = st[ph][1]; x = L + gi * gw + gw * 0.15 + bi * gw * 0.35; h = ch * v
            s.append(f'<rect x="{x:.1f}" y="{T+ch-h:.1f}" width="{gw*0.35:.1f}" height="{h:.1f}" fill="{col}"/>')
            s.append(f'<text x="{x+gw*0.175:.1f}" y="{T+ch-h-4:.1f}" text-anchor="middle" font-size="10">{v:.2f}</text>')
        s.append(f'<text x="{L+gi*gw+gw/2:.1f}" y="{H-B+18}" text-anchor="middle">{pol}</text>')
    s.append(f'<rect x="{L+10}" y="{H-18}" width="10" height="10" fill="#2a6f97"/><text x="{L+25}" y="{H-9}">Locality phase</text>')
    s.append(f'<rect x="{L+140}" y="{H-18}" width="10" height="10" fill="#e07a1f"/><text x="{L+155}" y="{H-9}">Shift phase</text></svg>')
    return "\n".join(s)

def rolling(h, w=50):
    return [sum(h[max(0, i - w + 1):i + 1]) / min(w, i + 1) for i in range(len(h))]

def main():
    os.makedirs(OUT, exist_ok=True)
    tree, nrows = train_policy()
    trace = make_trace(EVAL_SEED)
    res = run_all(trace, tree)

    write_csv("access_trace.csv", ["t", "page", "phase"],
              [(i, p, "locality" if i < HALF else "shift") for i, p in enumerate(trace)])
    rows = []
    for pol in POLICIES:
        for ph, (f, hr) in phase_stats(res[pol]).items():
            rows.append((pol, ph, HALF, f, HALF - f, f"{hr:.4f}"))
    write_csv("policy_metrics.csv", ["policy", "phase", "references", "faults", "hits", "hit_ratio"], rows)

    roll = {p: rolling(res[p]) for p in POLICIES}
    write_csv("rolling_hit.csv", ["t"] + POLICIES, [[i] + [f"{roll[p][i]:.4f}" for p in POLICIES] for i in range(2 * HALF)])

    # multi-seed robustness (same trained tree, 20 unseen traces)
    per = {p: {"loc": [], "shift": []} for p in POLICIES}
    for s in range(1001, 1021):
        r = run_all(make_trace(s), tree)
        for p in POLICIES:
            st = phase_stats(r[p]); per[p]["loc"].append(st["locality"][1]); per[p]["shift"].append(st["shift"][1])
    ms = []
    for p in POLICIES:
        drops = [(a - b) * 100 for a, b in zip(per[p]["loc"], per[p]["shift"])]
        ms.append((p, *(f"{x:.4f}" for x in (statistics.mean(per[p]["loc"]), statistics.stdev(per[p]["loc"]),
                  statistics.mean(per[p]["shift"]), statistics.stdev(per[p]["shift"]))),
                  f"{statistics.mean(drops):.2f}", f"{statistics.stdev(drops):.2f}"))
    write_csv("multiseed_summary.csv", ["policy", "locality_hit_mean", "locality_hit_std", "shift_hit_mean",
                                        "shift_hit_std", "drop_pp_mean", "drop_pp_std"], ms)

    with open(os.path.join(OUT, "hit_ratio.svg"), "w") as f: f.write(svg_bars(res))
    with open(os.path.join(OUT, "learned_tree.txt"), "w") as f:
        f.write(f"Decision tree trained on {nrows} candidate rows (seed {TRAIN_SEED}, locality-only trace)\n")
        f.write("features: " + ", ".join(FEATURES) + "\n\n" + tree_text(tree))
    summary = {"seed": EVAL_SEED, "frames": FRAMES, "pages": PAGES, "refs": 2 * HALF,
               "policies": {p: {ph: {"faults": f, "hit_ratio": round(hr, 4)} for ph, (f, hr) in phase_stats(res[p]).items()} for p in POLICIES}}
    with open(os.path.join(OUT, "summary.json"), "w") as f: json.dump(summary, f, indent=2)

    print(f"{'Policy':8} {'Loc faults':>10} {'Loc hit':>8} {'Shift faults':>12} {'Shift hit':>9}")
    for p in POLICIES:
        st = phase_stats(res[p])
        print(f"{p:8} {st['locality'][0]:10d} {st['locality'][1]:8.1%} {st['shift'][0]:12d} {st['shift'][1]:9.1%}")
    print("\n20-trace summary (policy, loc mean/std, shift mean/std, drop pp mean/std):")
    for r in ms: print(" ", r)
    print("\n" + tree_text(tree))

if __name__ == "__main__":
    main()
