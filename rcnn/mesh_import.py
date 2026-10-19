"""Brings the Colab Mesh R-CNN output back into the repo's usual format and rescores.

Reads mesh_rcnn_colab/output/ (mesh_pred.npz, config.json, train.log, meshes/) and writes
rcnn_out/mesh_rcnn/: predictions.pkl (one outline per slice, cut from the predicted mesh),
config.json, train.log and the meshes, then runs rcnn.evaluate and make_results_table.py.

Usage (from the repo root):  python3 -m rcnn.mesh_import [path/to/output]
"""

import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, load_split, mask_to_box


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "mesh_rcnn_colab" / "output"
    out = ROOT / "rcnn_out" / "mesh_rcnn"
    out.mkdir(parents=True, exist_ok=True)
    pred = np.load(src / "mesh_pred.npz")
    vols = {int(n): (v, float(s)) for n, v, s in zip(pred["rats"], pred["volumes"], pred["scores"])}
    test = load_split("test")
    preds = []
    for f, n, z in zip(test["files"], test["subjects"], test["zs"]):
        p = {"file": str(f), "subject": int(n), "z": int(z), "boxes": np.zeros((0, 4), np.float32),
             "scores": np.zeros((0,), np.float32), "masks": None}
        if int(n) in vols:
            vol, score = vols[int(n)]
            m = vol[int(z)]
            box = mask_to_box(m)
            if box is not None:
                p.update(boxes=box[None], scores=np.array([score], np.float32),
                         masks=(m[None].astype(np.uint8) * 255))
        preds.append(p)
    with open(out / "predictions.pkl", "wb") as fh:
        pickle.dump(preds, fh)
    for name in ["config.json", "train.log"]:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    if (src / "meshes").exists():
        shutil.copytree(src / "meshes", out / "meshes", dirs_exist_ok=True)
    print(f"wrote {out}/predictions.pkl ({sum(len(p['scores']) for p in preds)} slices with an outline)")
    subprocess.run([sys.executable, "-m", "rcnn.evaluate"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "make_results_table.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
