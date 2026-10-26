"""Export Xiaomi-TabLDM to a two-stage OpenVINO IR with INT8 weight-only compression.

stage1: (X, Yd) -> R        R = row_interactor(col_embedder(X, Yd))
stage2: (R, y)  -> 10 logits (clf) / 999 unsorted quantiles (reg)
Pipeline: torch.export (dynamic rows/train/feat) -> ONNX -> OpenVINO IR (fp32) -> NNCF INT8_SYM weight-only.

--src must be a xiaomi-tabldm checkout at e83da2b WITH patches/attention_headmerge.patch applied.
"""
import argparse, os, sys, time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('task', choices=['clf', 'reg'])
ap.add_argument('--src', required=True, help='patched xiaomi-tabldm checkout')
ap.add_argument('--ckpt', required=True, help='clf_default.ckpt / reg_default.ckpt')
ap.add_argument('--out-dir', default='out', help='output directory (default: ./out)')
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common; common.use_upstream(a.src)
import numpy as np, torch, openvino as ov, nncf
from torch.export import Dim
import stages  # imports dynpatch (export-only monkey patches)

os.makedirs(a.out_dir, exist_ok=True); od = lambda f: os.path.join(a.out_dir, f)
task = a.task
m = common.load(task, a.ckpt)
s1, s2 = stages.Stage1(m).eval(), stages.Stage2(m).eval()
g = np.random.default_rng(0); ntr, nte, H = 200, 100, 8
X = torch.from_numpy(g.standard_normal((1, ntr + nte, H)).astype(np.float32))
y = torch.from_numpy((g.integers(0, 3, (1, ntr)) if task == 'clf' else g.standard_normal((1, ntr))).astype(np.float32))
Yd = torch.cat([y, y]) if task == 'clf' else y          # clf: D label views (mixed radix), example D=2
T = Dim('rows', min=8, max=20000); N = Dim('train', min=4, max=16000); F = Dim('feat', min=1, max=500)
ds1 = {'X': {1: T, 2: F}, 'Yd': ({0: Dim('views', min=1, max=8), 1: N} if task == 'clf' else {1: N})}
with torch.no_grad():
    ep1 = torch.export.export(s1, (X, Yd), dynamic_shapes=ds1, strict=False); print('EXPORT1 OK', flush=True)
    R = s1(X, y)
    ep2 = torch.export.export(s2, (R, y), dynamic_shapes={'R': {1: T}, 'y': {1: N}}, strict=False); print('EXPORT2 OK', flush=True)
md = nncf.CompressWeightsMode.INT8_SYM
for name, ep, inn, outn in [('stage1', ep1, ['X', 'Yd'], ['R']), ('stage2', ep2, ['R', 'y'], ['out'])]:
    t0 = time.time(); f = od(f'{task}_{name}.onnx')
    torch.onnx.export(ep, f=f, dynamo=True, input_names=inn, output_names=outn, external_data=True, optimize=True)
    om = ov.convert_model(f); ov.save_model(om, od(f'{task}_{name}_fp32.xml'), compress_to_fp16=False)
    n_if = 0
    for op in om.get_ops():          # NNCF does not descend into If bodies (stage1 torch.cond): compress each body
        if op.get_type_name() == 'If':
            for i in range(2): op.set_function(i, nncf.compress_weights(op.get_function(i), mode=md)); n_if += 1
    if n_if == 0: om = nncf.compress_weights(om, mode=md)
    ov.save_model(om, od(f'tabldm_{task}_{name}_int8wo.xml'), compress_to_fp16=False)
    print(name, 'ONNX+OV+NNCF OK', 'If bodies', n_if, round(time.time() - t0, 1), 's', flush=True)
