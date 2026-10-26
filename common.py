"""Shared helpers: put an upstream xiaomi-tabldm checkout on sys.path and load the raw TabLDM network."""
import os, sys


def use_upstream(src):
    """src: path to a xiaomi-tabldm checkout (commit e83da2b). Must be called before importing tabldm."""
    src = os.path.abspath(src)
    if not os.path.isdir(os.path.join(src, 'tabldm')):
        sys.exit(f'not a xiaomi-tabldm checkout: {src}')
    sys.path.insert(0, src)


def load(task, ckpt):
    """task: clf|reg; ckpt: checkpoints/clf_default.ckpt or checkpoints/reg_default.ckpt from occams/Xiaomi-TabLDM."""
    if task == 'clf':
        from tabldm import TabLDMClassifier as C
    else:
        from tabldm import TabLDMRegressor as C
    c = C(model_path=ckpt, device='cpu'); c._load_model()
    return c.model_.eval()
