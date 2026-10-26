"""Split TabLDM into stage1 (X, Yd) -> R and stage2 (R, y_node) -> 10 logits / quantiles."""
import torch, dynpatch
from tabldm._model import embedding_dual_stream as eds

# 12) target-aware col embedding with D label "views" stacked on dim 0 (Yd: (D, n_train)).
#     D=1, Yd=y      -> identical to the <=10-class path.
#     D>1, Yd=digits -> identical to the original mixed-radix ensemble (orig embedding_dual_stream.py:313-331):
#                       mean_d tf_col(src + y_encoder(digit_d) on train rows), affine applied after averaging.
def compute_embeddings_md(self, features_local, features_global, train_size, y_train=None, embed_with_test=False):
    src = self.in_linear(features_local) + self.in_linear(features_global)          # (1, G, T, E)
    src_projected = src
    y_emb = self.y_encoder(y_train.float()) if self.max_classes > 0 else self.y_encoder(y_train.unsqueeze(-1))  # (D, G, n, E)
    D = y_emb.shape[0]
    tr = src[..., :train_size, :] + y_emb
    te = src[..., train_size:, :].expand(D, -1, -1, -1)
    src = self.tf_col(torch.cat([tr, te], dim=-2), train_size=train_size).mean(0, keepdim=True)
    if self.affine:
        return src_projected * self.ln_w(self.out_w(src)) + self.ln_b(self.out_b(src))
    return src
eds.ColEmbeddingDualStream._compute_embeddings = compute_embeddings_md

class Stage1(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
        object.__setattr__(s, "m_small", dynpatch.make_small_twin(m))
    def run(s, mm, X, Yd):
        R = mm.row_interactor(mm.col_embedder(X, y_train=Yd))
        return R.new_empty(R.shape).copy_(R)
    def forward(s, X, Yd):
        return torch.cond(X.shape[2] <= 3, lambda X, Yd: s.run(s.m_small, X, Yd), lambda X, Yd: s.run(s.m, X, Yd), (X, Yd))

class Stage2(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.icl = m.icl_predictor
    def forward(s, R, y):
        icl = s.icl; n = y.shape[1]
        Ry = icl.y_encoder(y.float()) if icl.max_classes > 0 else icl.y_encoder(y.unsqueeze(-1))
        R = torch.cat([R[:, :n] + Ry, R[:, n:]], dim=1)
        src = icl.tf_icl(R, train_size=n)
        if icl.norm_first: src = icl.ln(src)
        return icl.decoder(src)[:, n:]
