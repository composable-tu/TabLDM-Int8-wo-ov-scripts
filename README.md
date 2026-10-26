# Xiaomi-TabLDM INT8 weight-only OpenVINO IR — export scripts

Scripts that produce [Composable/Xiaomi-TabLDM-Int8-wo-ov](https://modelscope.cn/models/Composable/Xiaomi-TabLDM-Int8-wo-ov)
from upstream [xiaomi-research/xiaomi-tabldm](https://github.com/xiaomi-research/xiaomi-tabldm) (commit `e83da2b`)
and the checkpoints in [occams/Xiaomi-TabLDM](https://huggingface.co/occams/Xiaomi-TabLDM). Upstream code and weights
are Apache-2.0; this is a modified version under the same license (see `LICENSE`).

## Changes relative to upstream

- `patches/attention_headmerge.patch`: the attention head merge is rewritten as matmul + sum
  (mathematically equivalent), avoiding an OpenVINO CPU crash on Reshape(Transpose) with dynamic dims.
- `dynpatch.py` (export-time monkey patches, same math): data-dependent Python control flow is replaced by
  tensor ops — sparse MoE computes all experts weighted by a top-k mask, skip paths use `where`, the class-count
  branch is removed (head fixed at 10 logits), and the H<=3 feature grouping becomes a `torch.cond` (ONNX `If`).
- `stages.py`: the network is split at `R = row_interactor(col_embedder(X, Yd))` into stage1 `(X, Yd) -> R`
  and stage2 `(R, y) -> logits / 999 quantiles`, so >10-class hierarchical prediction (`hier.py`) can reuse R.
  Mixed-radix label views `Yd` are averaged inside stage1.
- `export_split.py`: torch.export (dynamic rows/train/features, batch 1) -> ONNX -> OpenVINO IR ->
  NNCF `compress_weights(INT8_SYM)` (per-channel, weight-only, no calibration data, activations not quantized).

## Reproduce

```bash
pip install -r requirements.txt            # Python 3.13, CPU
git clone https://github.com/xiaomi-research/xiaomi-tabldm upstream && git -C upstream checkout e83da2b
cp -r upstream upstream-orig               # unpatched copy, used only as the reference in evaluate.py
git -C upstream apply "$PWD/patches/attention_headmerge.patch"
hf download occams/Xiaomi-TabLDM checkpoints/clf_default.ckpt checkpoints/reg_default.ckpt \
   --revision 712e5635be9fd8728084cd5ae105ddecb31dc03c --local-dir .

python export_split.py clf --src upstream --ckpt checkpoints/clf_default.ckpt --out-dir out
python export_split.py reg --src upstream --ckpt checkpoints/reg_default.ckpt --out-dir out
python evaluate.py clf --src upstream-orig --ckpt checkpoints/clf_default.ckpt --ir-dir out
python evaluate.py reg --src upstream-orig --ckpt checkpoints/reg_default.ckpt --ir-dir out
```

With the pinned versions, the eight `out/tabldm_*_stage*_int8wo.{xml,bin}` files are byte-identical to the
published ones (`logs/sha256_check.log`).

## Accuracy (from `logs/evaluate_*.log`)

INT8 IR (OpenVINO CPU, f32 inference hint) vs the original PyTorch network, 30% test split, `random_state=0`:

| Dataset | Classes | Argmax agreement | Acc IR | Acc PyTorch |
|---|---|---|---|---|
| breast_cancer | 2 | 100.00% | 95.32% | 95.32% |
| iris | 3 | 100.00% | 100.00% | 100.00% |
| wine | 3 | 100.00% | 100.00% | 100.00% |
| synthetic, 12 classes | 12 | 99.33% | 94.67% | 94.67% |
| synthetic, 60 classes | 60 | 98.44% | 82.44% | 82.67% |

Regression, diabetes: RMSE 55.810 (IR) vs 55.745 (PyTorch); largest per-sample difference in the point
prediction is 14.760. Numbers are for a single network forward pass, not the upstream sklearn wrapper
(no ensembling or preprocessing). Stage2 regression quantiles are unsorted; sort along the last axis before use.
