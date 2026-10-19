"""Packs everything the Colab Mesh R-CNN run needs into one folder to upload to Google Drive.

mesh_rcnn_colab/
  mesh_data.npz     training slices + atlas masks (53 rats), test slices (9 rats) WITHOUT
                    their hand traces, and the one stage-1 slice + box picked per test rat
  mesh_rcnn.py      the training/prediction script (copied from rcnn/)
  mesh_rcnn.ipynb   the notebook to open in Colab

The hand traces never leave the Mac: scoring happens locally, in rcnn/mesh_import.py.
The folder is gitignored, since it holds scan images.

Stage-1 pick per test rat: of the slices where Faster R-CNN is confident (score >= 0.5),
the one with the largest box. The largest slice box covers the structure's whole in-plane
extent (checked on the 53 training rats: the 3D extent is at most 1.05x the largest box).

Usage (from the repo root):  python3 -m rcnn.mesh_pack
"""

import json
import pickle
import shutil

import numpy as np

from .common import ROOT, load_split

OUT = ROOT / "mesh_rcnn_colab"
STAGE1 = "faster_rcnn"
THRESHOLD = 0.5


def pick_slices(test):
    with open(ROOT / "rcnn_out" / STAGE1 / "predictions.pkl", "rb") as f:
        stage1 = {p["file"]: p for p in pickle.load(f)}
    picks = {}
    for f, n, z in zip(test["files"], test["subjects"], test["zs"]):
        p = stage1[str(f)]
        if not len(p["scores"]) or p["scores"].max() < THRESHOLD:
            continue
        k = int(np.argmax(p["scores"]))
        box = p["boxes"][k]
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        if int(n) not in picks or area > picks[int(n)][3]:
            picks[int(n)] = (int(z), box.astype(np.float32), float(p["scores"][k]), area)
    return picks


def main():
    train, test = load_split("train"), load_split("test")
    picks = pick_slices(test)
    OUT.mkdir(exist_ok=True)
    rats = sorted(picks)
    np.savez_compressed(
        OUT / "mesh_data.npz",
        train_images=train["images"], train_masks=train["masks"], train_subjects=train["subjects"],
        train_zs=train["zs"],
        test_images=test["images"], test_subjects=test["subjects"], test_zs=test["zs"], test_files=test["files"],
        pick_rats=np.array(rats), pick_z=np.array([picks[n][0] for n in rats]),
        pick_boxes=np.stack([picks[n][1] for n in rats]), pick_scores=np.array([picks[n][2] for n in rats]))
    shutil.copy(ROOT / "rcnn" / "mesh_rcnn.py", OUT / "mesh_rcnn.py")
    write_notebook(OUT / "mesh_rcnn.ipynb")
    size = (OUT / "mesh_data.npz").stat().st_size / 1e6
    print(f"wrote {OUT}/ (mesh_data.npz {size:.1f} MB, mesh_rcnn.py, mesh_rcnn.ipynb)")
    for n in rats:
        print(f"  rat {n}: stage-1 slice {picks[n][0]}, box {np.round(picks[n][1]).astype(int).tolist()}, "
              f"score {picks[n][2]:.2f}")
    missing = sorted(set(test["subjects"].tolist()) - set(rats))
    if missing:
        print(f"  no confident stage-1 box for rats {missing}: they get an empty prediction")


def write_notebook(path):
    md = lambda text: {"cell_type": "markdown", "metadata": {}, "source": text}
    code = lambda text: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}
    cells = [
        md("# Mesh R-CNN: hypothalamus 3D shape from one slice\n\n"
           "1. **Runtime > Change runtime type > T4 GPU**, then run the cells top to bottom.\n"
           "2. The folder `mesh_rcnn_colab` must be in **My Drive** (upload the whole folder).\n"
           "3. Output goes to `mesh_rcnn_colab/output` on Drive, so a dropped session loses nothing: "
           "re-run the cells and training resumes from the last finished epoch.\n\n"
           "No extra installs: the script needs only PyTorch, torchvision and NumPy, which Colab already has."),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code("from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             "FOLDER = '/content/drive/MyDrive/mesh_rcnn_colab'\n"
             "!ls -la \"$FOLDER\""),
        md("**Smoke test** (about a minute): a few training steps and a full prediction pass, "
           "written to a scratch folder. If this fails, stop and send the error."),
        code("!python \"$FOLDER/mesh_rcnn.py\" --data \"$FOLDER/mesh_data.npz\" --out /content/smoke --smoke"),
        md("**Full run** (50 epochs, roughly 30-60 minutes on a T4; it took about 1 minute per epoch on the M1). The log prints one line per epoch."),
        code("!python \"$FOLDER/mesh_rcnn.py\" --data \"$FOLDER/mesh_data.npz\" --out \"$FOLDER/output\""),
        md("**Done.** Download `mesh_rcnn_colab/output` from Drive into the repo as "
           "`mesh_rcnn_colab/output/`, then on the Mac:\n\n"
           "```\npython3 -m rcnn.mesh_import\n```\n\n"
           "That scores it against the hand traces and adds the Mesh R-CNN row to RESULTS.md."),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    path.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    main()
