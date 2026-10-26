"""Export-only patches for Xiaomi-TabLDM (same math, no data-dependent control flow)."""
import torch, torch.nn.functional as F
from tabldm._model import layers, moe, inference, embedding_dual_stream as eds

# 1) SkippableLinear: `if skip_mask.any(): out[skip_mask]=v`  ->  where
def skl_forward(self, src):
    out = F.linear(src, self.weight, self.bias)
    m = (src == self.skip_value).all(dim=-1, keepdim=True)
    return torch.where(m, torch.full_like(out, self.skip_value), out)
layers.SkippableLinear.forward = skl_forward

# 2) InducedSelfAttentionBlock: mask/any/all branches -> compute all + where
def isab_forward(self, src, train_size=None):
    out = self.induced_attention(src, train_size)
    m = (src == self.skip_value).all(dim=(-2, -1))
    m = m.reshape(*m.shape, 1, 1)
    return torch.where(m, torch.full_like(out, self.skip_value), out)
layers.InducedSelfAttentionBlock.forward = isab_forward

# 3) Sparse MoE: nonzero/index_add_ routing -> dense experts weighted by top-k mask
def moe_forward(self, x):
    shp = x.shape; fx = x.reshape(-1, shp[-1])
    probs = torch.softmax(self.router(fx.float()), -1)
    if self.auxiliary_free:
        ti = torch.topk(probs + self.bias.to(probs.dtype), k=self.top_k, dim=-1).indices; tp = probs.gather(-1, ti)
    else:
        tp, ti = torch.topk(probs, k=self.top_k, dim=-1)
    tw = tp if self.router_weight_mode == "raw" else tp / tp.sum(-1, keepdim=True).clamp_min(1e-9)
    dense_w = torch.zeros_like(probs).scatter(-1, ti, tw).to(fx.dtype)
    out = torch.zeros_like(fx)
    for e, ex in enumerate(self.experts):
        out = out + ex(fx).to(fx.dtype) * dense_w[:, e:e+1]
    if self.shared_experts:
        out = out + sum(ex(fx).to(fx.dtype) for ex in self.shared_experts) / len(self.shared_experts)
    return out.reshape(shp)
moe.SparseMoEFeedForward.forward = moe_forward

# 4) InferenceManager: python batching/OOM/offload -> direct call
def mgr_call(self, forward_fn, inputs, auto_batch=True, output_repeat=1):
    return forward_fn(**inputs)
inference.InferenceManager.__call__ = mgr_call
inference.InferenceManager.configure = lambda self, *a, **k: None

# 5) Global-stream dilation shifts as tensors of H (valid for H >= 4; for H>=4
#    _ensure_distinct_mod_h is the identity: 1 < round(sqrt(m)) < m=min(span,H-1) < H)
def feature_grouping_global(self, X):
    B, T, H = X.shape
    assert self.feature_group in (True, "same") and self.global_dilation == "adaptive"
    m = torch.clamp(torch.ones(H, dtype=torch.float32, device=X.device).sum() - 1, max=float(self.global_max_span))
    shifts = [torch.ones((), dtype=torch.long), torch.round(torch.sqrt(m)).long(), m.long()]
    idxs = torch.arange(H, dtype=torch.long, device=X.device)
    Ht = torch.ones(H, dtype=torch.long, device=X.device).sum()  # tensor H (ONNX: aten.remainder.Scalar w/ SymInt unsupported)
    return torch.stack([X[:, :, torch.remainder(idxs + s, Ht)] for s in shifts], dim=-1)
eds.ColEmbeddingDualStream.feature_grouping_global = feature_grouping_global

# 6) _compute_embeddings: drop num_classes=int(y.max().item())+1 branch (<=10 classes only),
#    slice-assign -> cat
def compute_embeddings(self, features_local, features_global, train_size, y_train=None, embed_with_test=False):
    src = self.in_linear(features_local) + self.in_linear(features_global)
    src_projected = src
    if self.target_aware:
        y_emb = self.y_encoder(y_train.float()) if self.max_classes > 0 else self.y_encoder(y_train.unsqueeze(-1))
        src = torch.cat([src[..., :train_size, :] + y_emb, src[..., train_size:, :]], dim=-2)
    src = self.tf_col(src, train_size=None if embed_with_test else train_size)
    if self.affine:
        return src_projected * self.ln_w(self.out_w(src)) + self.ln_b(self.out_b(src))
    return src
eds.ColEmbeddingDualStream._compute_embeddings = compute_embeddings

# 7) ICLearning: drop len(torch.unique(y)) class counting/slicing; always return all
#    max_classes logits (TabICLv2-ONNX precedent). Caller slices [..., :n_classes].
from tabldm._model import learning
def icl_inference_forward(self, R, y_train, return_logits=True, softmax_temperature=0.9, mgr_config=None):
    out = self._icl_predictions(R, y_train)
    out = out[:, y_train.shape[1]:]
    if self.max_classes > 0 and not return_logits:
        out = torch.softmax(out / softmax_temperature, dim=-1)
    return out
learning.ICLearning._inference_forward = icl_inference_forward

# 8) SSMax: log(n) via math.log on a python int -> tensor op on a symbolic n
from tabldm._model import ssmax
def _logn(n, device, dtype):
    nt = torch.ones(n, dtype=torch.float64, device=device).sum() if not isinstance(n, int) else torch.tensor(float(n), dtype=torch.float64, device=device)
    return torch.log(torch.clamp(nt, min=1.0)).to(dtype)
ssmax._logn = _logn

# 9) Local stream grouping (H > size branch): `(idxs + 2**i) % H` with SymInt H -> tensor remainder
def feature_grouping_local(self, X):
    B, T, H = X.shape
    assert self.feature_group in (True, "same")
    idxs = torch.arange(H, dtype=torch.long, device=X.device)
    Ht = torch.ones(H, dtype=torch.long, device=X.device).sum()
    return torch.stack([X[:, :, torch.remainder(idxs + 2**i, Ht)] for i in range(self.feature_group_size)], dim=-1)
eds.ColEmbeddingDualStream.feature_grouping_local = feature_grouping_local

# 10) Feature grouping H<=3 vs H>3 (orig embedding_dual_stream.py:198-202 / 224-228 `if H <= size`).
#     The two branches give different numbers of column groups (1 vs H), so they are exported as two
#     subgraphs selected by torch.cond -> ONNX `If` (see stages.Stage1). The small-H subgraph uses a shallow copy of col_embedder (shared weights) with _group_small=True.
def _small_groups(X, size):
    B, T, H = X.shape
    Xp = torch.cat([X, X.new_zeros(B, T, size)], dim=-1)     # pad with zeros, keep first `size` cols
    return Xp[:, :, :size].reshape(B, T, 1, size)
_big_local, _big_global = feature_grouping_local, feature_grouping_global
def feature_grouping_local_all(self, X):
    return _small_groups(X, self.feature_group_size) if getattr(self, "_group_small", False) else _big_local(self, X)
def feature_grouping_global_all(self, X):
    return _small_groups(X, self.feature_group_size) if getattr(self, "_group_small", False) else _big_global(self, X)
eds.ColEmbeddingDualStream.feature_grouping_local = feature_grouping_local_all
eds.ColEmbeddingDualStream.feature_grouping_global = feature_grouping_global_all

import copy
def make_small_twin(m):
    """Shallow twin of the TabLDM model sharing all parameters, whose col_embedder uses the H<=3 grouping."""
    ce2 = copy.copy(m.col_embedder); ce2._group_small = True
    m2 = copy.copy(m); m2._modules = dict(m._modules); m2._modules["col_embedder"] = ce2
    return m2

# 11) attention.sdpa_with_flattened_batch: drop global call counters / prints / flash dispatch
#     (orig attention.py:234-289; side effects are illegal inside torch.cond subgraphs). Same math (SDPA path).
from tabldm._model import attention as _att
def sdpa_pure(q, k, v, attn_mask=None, dropout_p=0.0, ssmax_layer=None):
    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:]); k = k.reshape(-1, *k.shape[-3:]); v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    if ssmax_layer is not None:
        q = ssmax_layer(q, k.size(-2))
    return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p).view(q_shape)
_att.sdpa_with_flattened_batch = sdpa_pure
