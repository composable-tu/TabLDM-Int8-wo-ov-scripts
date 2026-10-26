"""Compare the two-stage INT8 weight-only IR with the original (unpatched) PyTorch TabLDM network.

Reference: raw network forward of upstream code (clf: model(X, y, return_logits=True), which uses the
upstream mixed-radix + hierarchical path for >10 classes; reg: model(X, y) quantiles).
IR: stage1 + stage2 driven by hier.py, OpenVINO CPU with INFERENCE_PRECISION_HINT=f32.
Data: sklearn breast_cancer / iris / wine / diabetes, plus synthetic 12- and 60-class sets;
30% test split, random_state=0, z-scored with train statistics.

--src must be an UNPATCHED xiaomi-tabldm checkout at e83da2b (dynpatch is never imported here).
"""
import argparse, os, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('task', choices=['clf', 'reg'])
ap.add_argument('--src', required=True, help='unpatched xiaomi-tabldm checkout')
ap.add_argument('--ckpt', required=True, help='clf_default.ckpt / reg_default.ckpt')
ap.add_argument('--ir-dir', required=True, help='directory with tabldm_<task>_stage{1,2}_int8wo.xml')
ap.add_argument('--threads', type=int, default=0, help='torch threads (0 = default)')
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common; common.use_upstream(a.src)
import numpy as np, torch, openvino as ov
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes, make_classification
from sklearn.model_selection import train_test_split
import hier
if a.threads: torch.set_num_threads(a.threads)

SYNTH = {'synthetic_12cls': dict(n_samples=1500, n_features=20, n_informative=12, n_classes=12, n_clusters_per_class=1, class_sep=1.5, random_state=0),
         'synthetic_60cls': dict(n_samples=3000, n_features=30, n_informative=16, n_classes=60, n_clusters_per_class=1, class_sep=2.0, random_state=1)}

def prep(X, y, clf):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y if clf else None)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xa = ((np.concatenate([Xtr, Xte]) - mu) / sd).astype(np.float32)[None]
    return Xa, ytr, yte

core = ov.Core(); cfg = {'INFERENCE_PRECISION_HINT': 'f32'}
p = lambda st: os.path.join(a.ir_dir, f'tabldm_{a.task}_{st}_int8wo.xml')
c1, c2 = core.compile_model(p('stage1'), 'CPU', cfg), core.compile_model(p('stage2'), 'CPU', cfg)
f1 = lambda X, Yd: c1([X, Yd])[0]
f2 = lambda R, y: c2([np.ascontiguousarray(R, np.float32), y])[0]
m = common.load(a.task, a.ckpt)

if a.task == 'clf':
    sets = {'breast_cancer': load_breast_cancer, 'iris': load_iris, 'wine': load_wine}
    data = {k: prep(*f(return_X_y=True), True) for k, f in sets.items()}
    data.update({k: prep(*make_classification(**kw), True) for k, kw in SYNTH.items()})
    for k, (X, ytr, yte) in data.items():
        with torch.no_grad():
            r = m(torch.from_numpy(X), torch.from_numpy(ytr.astype(np.float32)[None]), return_logits=True).numpy()[0]
        o = hier.predict_logits(f1, f2, X, ytr.astype(int))
        agree = (o.argmax(-1) == r.argmax(-1)).mean() * 100
        print(f'{k}: classes={len(np.unique(ytr))} n_test={len(yte)} argmax_agree_vs_pytorch={agree:.2f}% '
              f'acc_ir={(o.argmax(-1) == yte).mean()*100:.2f}% acc_pytorch={(r.argmax(-1) == yte).mean()*100:.2f}%', flush=True)
else:
    X, ytr, yte = prep(*load_diabetes(return_X_y=True), False)
    ym, ys = ytr.mean(), ytr.std(); yt = ((ytr - ym) / ys).astype(np.float32)[None]
    with torch.no_grad():
        r = m(torch.from_numpy(X), torch.from_numpy(yt)).numpy()[0]
    q = f2(f1(X, yt), yt)[0]                      # IR quantiles are unsorted: sort before use
    pr = np.sort(r, -1).mean(-1) * ys + ym; po = np.sort(q, -1).mean(-1) * ys + ym
    rmse = lambda z: float(np.sqrt(((z - yte) ** 2).mean()))
    print(f'diabetes: n_test={len(yte)} rmse_ir={rmse(po):.3f} rmse_pytorch={rmse(pr):.3f} '
          f'pred_maxabs_diff={np.abs(po - pr).max():.3f}', flush=True)
