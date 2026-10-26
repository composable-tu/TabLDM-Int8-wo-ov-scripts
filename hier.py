"""numpy mirror of orig learning.py _grouping/_fit_node/_label_encoding/_predict_hierarchical + embedding.py mixed radix."""
import math, numpy as np
MAXC = 10
def mixed_radix_bases(C, maxc=MAXC):
    if C <= maxc: return [C]
    D = math.ceil(math.log(C) / math.log(maxc)); k = min(math.ceil(C ** (1.0 / D)), maxc)
    bases = [k] * D; prod = k ** D; i = 0
    while prod < C and i < D:
        if bases[i] < maxc: prod = prod // bases[i] * (bases[i] + 1); bases[i] += 1
        i += 1
    return bases
def label_views(y):
    """y: (n,) int labels 0..C-1 -> Yd (D, n) float32 for stage1 (C = y.max()+1, as in orig)."""
    y = y.astype(np.int64); C = int(y.max()) + 1
    if C <= MAXC: return y[None].astype(np.float32)
    b = mixed_radix_bases(C); out = []
    for d in range(len(b)):
        div = int(np.prod(b[d+1:])) if d+1 < len(b) else 1
        out.append((y // div) % b[d])
    return np.stack(out).astype(np.float32)
def grouping(C, maxc=MAXC):
    if C <= maxc: return np.zeros(C, int), 1
    G = min(math.ceil(C / maxc), maxc); ga = np.zeros(C, int); pos = 0; rc = C; rg = G
    for i in range(G):
        gs = math.ceil(rc / rg); ga[pos:pos+gs] = i; pos += gs; rc -= gs; rg -= 1
    return ga, G
def softmax(z, T=0.9):
    z = z / T; z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def fit_node(R, y):
    cls = np.unique(y).astype(int); node = {'classes': cls, 'R': R, 'y': y}
    if len(cls) <= MAXC: node['leaf'] = True; return node
    ga, G = grouping(len(cls)); cmap = dict(zip(cls.tolist(), ga.tolist()))
    gi = np.array([cmap[int(c)] for c in y]); node.update(leaf=False, gi=gi, children=[fit_node(R[gi == g], y[gi == g]) for g in range(G)])
    return node
def predict_standard(stage2, R, y):
    k = len(np.unique(y))
    return softmax(stage2(R[None], y[None].astype(np.float32))[0][:, :k])
def predict_hier(stage2, R_train, y, R_test, T=0.9):
    root = fit_node(R_train, y); C = len(root['classes']); nt = R_test.shape[0]
    def proc(node):
        RR = np.concatenate([node['R'], R_test])
        if node['leaf']:
            ly = np.searchsorted(np.unique(node['y']), node['y'])   # _label_encoding
            p = predict_standard(stage2, RR, ly); g = np.zeros((nt, C), np.float32)
            for li, gi in enumerate(node['classes']): g[:, gi] = p[:, li]
            return g
        gp = predict_standard(stage2, RR, node['gi']); out = np.zeros((nt, C), np.float32)
        for gidx, ch in enumerate(node['children']): out += proc(ch) * gp[:, gidx:gidx+1]
        return out
    return T * np.log(proc(root) + 1e-6)
def predict_logits(stage1, stage2, X, y):
    """X: (1, T, H) float32 (train rows first), y: (n,) int labels 0..C-1 -> logits (T-n, C)."""
    n = len(y); R = stage1(X, label_views(y))[0]
    C = len(np.unique(y))
    if C <= MAXC: return stage2(R[None], y[None].astype(np.float32))[0][:, :C]
    return predict_hier(stage2, R[:n], y.astype(int), R[n:])
