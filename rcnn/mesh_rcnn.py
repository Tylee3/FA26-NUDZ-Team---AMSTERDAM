"""Mesh R-CNN (Gkioxari et al. 2019) for the hypothalamus: from ONE 2D slice, predict the
whole 3D shape as a triangle mesh, then cut the mesh back into slices for scoring.

Standalone on purpose: needs only torch, torchvision and numpy, so the same file runs on
Google Colab (CUDA) or on the Mac (MPS). Mesh R-CNN's 3D operations (cubify, graph
convolution, vertex alignment, point sampling, chamfer/normal/edge losses) are written
here in plain PyTorch instead of installing PyTorch3D, which does not build on macOS and
is a slow, fragile build on Colab.

How it is set up:
  2D part   Mesh R-CNN normally adds its 3D branch to a Mask R-CNN. Here the box comes from
            our trained Faster R-CNN (stage 1, picked in rcnn/mesh_pack.py), and the 3D
            branch works on a zoomed window around it: twice the box, scaled to 256x256,
            as in rcnn/zoom.py. Backbone = the COCO-pretrained Mask R-CNN ResNet-50-FPN.
  voxel     a G x G x D occupancy grid over the window: G=32 in-plane, D=24 in depth
            covering +-6 slices around the input slice (0.5 slice per cell). Depth is
            predicted as channels per in-plane cell, like Mesh R-CNN's voxel head.
  mesh      cubify the grid (cells above 0.2 become cubes), then 3 refinement stages, each
            sampling image features at the vertices and moving them with graph convolutions.
  losses    voxel BCE 1.0, chamfer 1.0, normal 0.1, edge 1.0 (Mesh R-CNN's Pix3D weights).
  training  every training slice whose box is at least 0.6x the rat's largest, labels =
            the atlas outline in 3D (like every other model); windows jittered 1.6-2.4x the
            box and +-25% shift, random left-right flip. Adam 1e-4, x0.1 at 70%, 50 epochs.
  test      one slice per rat (stage 1's largest confident box); the final mesh is
            voxelized with the generalized winding number at every pixel centre of every
            slice, giving a 12-slice mask scored like the other models.

Usage:
  python3 mesh_rcnn.py --data mesh_data.npz --out output [--smoke] [--epochs N]
Writes to --out: mesh_pred.npz (predicted 12x256x256 masks per test rat), meshes/*.obj
(the predicted 3D shapes, in mm, open in any 3D viewer), model.pt, train.log, config.json.
A checkpoint is saved every epoch and picked up again if the run is restarted.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 256
CONTEXT = 2.0
MIN_SIDE = 48
G = 32
G_HI = 64
D = 24
Z_HALF = 6.0
CUBIFY_THRESH = 0.2
STAGES = 3
HIDDEN = 128
N_POINTS = 3000
MAX_FACES = 30000
W_VOX, W_CHAMFER, W_NORMAL, W_EDGE = 1.0, 1.0, 0.1, 1.0
EPOCHS = 50
BATCH = 8
LR = 1e-4
MIN_SHARE = 0.6
SPACING_MM = (0.1, 0.1, 1.0)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------- geometry helpers (window, sampling the label volume into the window) ----------

def mask_to_box(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def window(box, scale=CONTEXT, shift=(0.0, 0.0)):
    x0, y0, x1, y1 = box
    side = int(round(max(MIN_SIDE, scale * max(x1 - x0, y1 - y0))))
    cx, cy = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def cut(gray, win):
    """Square window out of a slice (zero-padded at the edges), scaled to CROP x CROP."""
    x, y, side = win
    padded = np.pad(gray, side)
    patch = padded[y + side:y + 2 * side, x + side:x + 2 * side].astype(np.float32)
    t = torch.from_numpy(patch)[None, None]
    return F.interpolate(t, size=(CROP, CROP), mode="bilinear", align_corners=False)[0, 0]


def target_grid(volume, win, z0, gh):
    """The 3D label resampled onto the window's grid: (D, gh, gh) bool. Cell (b, i, j) covers
    in-plane pixels x + j*side/gh ... and depth (z0 + 0.5) - Z_HALF + b*2*Z_HALF/D ...,
    the same layout cubify uses, so the grid spans [-1, 1] on every axis."""
    x, y, side = win
    nz, h, w = volume.shape
    cols = np.floor(x + (np.arange(gh) + 0.5) * side / gh).astype(int)
    rows = np.floor(y + (np.arange(gh) + 0.5) * side / gh).astype(int)
    deps = np.floor(z0 + 0.5 - Z_HALF + (np.arange(D) + 0.5) * 2 * Z_HALF / D).astype(int)
    ok = (deps[:, None, None] >= 0) & (deps[:, None, None] < nz) & (rows[None, :, None] >= 0) & \
         (rows[None, :, None] < h) & (cols[None, None, :] >= 0) & (cols[None, None, :] < w)
    vals = volume[np.clip(deps, 0, nz - 1)][:, np.clip(rows, 0, h - 1)][:, :, np.clip(cols, 0, w - 1)]
    return vals & ok


def image_input(crop):
    t = (crop / 255.0).clamp(0, 1).unsqueeze(0).repeat(3, 1, 1)
    return (t - MEAN) / STD


# ---------- Mesh R-CNN's 3D operations in plain PyTorch ----------

def cubify(prob, thresh):
    """Occupancy grids (N, D, H, W) -> list of (verts (V, 3), faces (F, 3)) per grid: one cube
    per cell above thresh, only the faces between an occupied and an empty cell, shared
    corners merged, every triangle wound so its normal points out of the shape. Vertex
    coordinates are (x, y, z) in [-1, 1], x along W, y along H, z along D; the outer cell
    corners sit at -1 and 1 (PyTorch3D's align="corner")."""
    n, dd, hh, ww = prob.shape
    occ = F.pad((prob > thresh).float(), (1, 1, 1, 1, 1, 1)).bool()
    scale = torch.tensor([2.0 / ww, 2.0 / hh, 2.0 / dd], device=prob.device)
    out = []
    for k in range(n):
        o = occ[k]
        core = o[1:-1, 1:-1, 1:-1]
        corners_all = []
        for axis in range(3):
            for s in (-1, 1):
                sl = [slice(1, -1)] * 3
                sl[axis] = slice(2, None) if s > 0 else slice(0, -2)
                idx = torch.nonzero(core & ~o[tuple(sl)])
                if len(idx) == 0:
                    continue
                base = idx.clone()
                base[:, axis] += 1 if s > 0 else 0
                u, v = [a for a in range(3) if a != axis]
                eu = torch.zeros(3, dtype=torch.long, device=prob.device)
                ev = torch.zeros(3, dtype=torch.long, device=prob.device)
                eu[u], ev[v] = 1, 1
                c0, c1, c2, c3 = base, base + eu, base + eu + ev, base + ev
                tris = torch.cat([torch.stack([c0, c1, c2], 1), torch.stack([c0, c2, c3], 1)])
                # (d, h, w) -> (x, y, z) ordering for the winding check
                p = tris[0, :, [2, 1, 0]].float()
                normal = torch.cross(p[1] - p[0], p[2] - p[0], dim=0)
                want = torch.zeros(3, device=prob.device)
                want[2 - axis] = s
                if float((normal * want).sum()) < 0:
                    tris = tris[:, [0, 2, 1]]
                corners_all.append(tris)
        if not corners_all:
            out.append((torch.zeros((0, 3), device=prob.device), torch.zeros((0, 3), dtype=torch.long, device=prob.device)))
            continue
        tris = torch.cat(corners_all)
        ids = (tris[..., 0] * (hh + 1) + tris[..., 1]) * (ww + 1) + tris[..., 2]
        uniq, inv = torch.unique(ids.reshape(-1), return_inverse=True)
        d_, rem = uniq // ((hh + 1) * (ww + 1)), uniq % ((hh + 1) * (ww + 1))
        h_, w_ = rem // (ww + 1), rem % (ww + 1)
        verts = torch.stack([w_, h_, d_], 1).float() * scale - 1
        out.append((verts, inv.view(-1, 3)))
    return out


def mesh_edges(faces):
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)


class GraphConv(nn.Module):
    """PyTorch3D's GraphConv: each vertex = W0 * itself + W1 * sum of its neighbours."""

    def __init__(self, cin, cout):
        super().__init__()
        self.w0, self.w1 = nn.Linear(cin, cout), nn.Linear(cin, cout)

    def forward(self, x, edges):
        y = self.w1(x)
        agg = torch.zeros_like(y)
        agg.index_add_(0, edges[:, 0], y[edges[:, 1]])
        agg.index_add_(0, edges[:, 1], y[edges[:, 0]])
        return self.w0(x) + agg


def vert_align(feats, verts):
    """Bilinear sample of image features (C, H, W) at each vertex's (x, y)."""
    grid = verts[:, :2].view(1, 1, -1, 2)
    return F.grid_sample(feats[None], grid, mode="bilinear", align_corners=False)[0, :, 0].t()


def face_normals_areas(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    norm = cross.norm(dim=1, keepdim=True)
    return cross / norm.clamp_min(1e-12), norm[:, 0] / 2


def sample_points(verts, faces, n):
    """Area-weighted random points on the surface, with the surface normal at each point
    (PyTorch3D's sample_points_from_meshes). Differentiable in the vertex positions."""
    normals, areas = face_normals_areas(verts, faces)
    pick = torch.multinomial(areas.detach() + 1e-12, n, replacement=True)
    r1, r2 = torch.rand(n, 1, device=verts.device), torch.rand(n, 1, device=verts.device)
    s = r1.sqrt()
    f = faces[pick]
    pts = (1 - s) * verts[f[:, 0]] + s * (1 - r2) * verts[f[:, 1]] + s * r2 * verts[f[:, 2]]
    return pts, normals[pick]


def chamfer(p, pn, q, qn):
    """Chamfer distance (squared, both directions) and normal loss (1 - |cos| to the nearest
    point), as in PyTorch3D's chamfer_distance."""
    d = torch.cdist(p, q) ** 2
    dp, ip = d.min(1)
    dq, iq = d.min(0)
    cham = dp.mean() + dq.mean()
    norm = (1 - F.cosine_similarity(pn, qn[ip], dim=1).abs()).mean() + \
           (1 - F.cosine_similarity(qn, pn[iq], dim=1).abs()).mean()
    return cham, norm


def edge_loss(verts, edges):
    return ((verts[edges[:, 0]] - verts[edges[:, 1]]) ** 2).sum(1).mean()


def winding_inside(points, verts, faces, chunk=2048):
    """Generalized winding number of a closed mesh at each point (Jacobson et al. 2013):
    about 1 inside, 0 outside, robust to small holes. Returns a bool per point."""
    tri = verts[faces]
    inside = []
    for k in range(0, len(points), chunk):
        p = points[k:k + chunk]
        a, b, c = (tri[None, :, i] - p[:, None] for i in range(3))
        la, lb, lc = a.norm(dim=2), b.norm(dim=2), c.norm(dim=2)
        det = (a * torch.cross(b, c, dim=2)).sum(2)
        den = la * lb * lc + (a * b).sum(2) * lc + (b * c).sum(2) * la + (c * a).sum(2) * lb
        w = (2 * torch.atan2(det, den)).sum(1) / (4 * math.pi)
        inside.append(w.abs() > 0.5)
    return torch.cat(inside) if inside else torch.zeros(0, dtype=torch.bool, device=points.device)


# ---------- network ----------

class RefineStage(nn.Module):
    def __init__(self, feat_dim, first):
        super().__init__()
        self.bottleneck = nn.Linear(feat_dim, HIDDEN)
        cin = HIDDEN + 3 + (0 if first else HIDDEN)
        self.convs = nn.ModuleList([GraphConv(cin, HIDDEN), GraphConv(HIDDEN, HIDDEN), GraphConv(HIDDEN, HIDDEN)])
        self.offset = nn.Linear(HIDDEN + 3, 3)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, feats, verts, edges, prev):
        x = F.relu(self.bottleneck(torch.cat([vert_align(f, verts) for f in feats], 1)))
        x = torch.cat([x, verts] + ([] if prev is None else [prev]), 1)
        for conv in self.convs:
            x = F.relu(conv(x, edges))
        verts = verts + torch.tanh(self.offset(torch.cat([x, verts], 1)))
        return verts, x


class MeshRCNN3D(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        det = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
                                    weights_backbone=None)
        self.backbone = det.backbone
        self.voxel = nn.Sequential(*[m for _ in range(4) for m in (nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())],
                                   nn.Conv2d(256, D, 1))
        self.stages = nn.ModuleList([RefineStage(3 * 256, i == 0) for i in range(STAGES)])

    def forward(self, images, thresh=CUBIFY_THRESH):
        fp = self.backbone(images)
        vox_logits = self.voxel(fp["1"])
        feats = [fp["0"], fp["1"], fp["2"]]
        meshes = cubify(torch.sigmoid(vox_logits.detach()), thresh)
        results = []
        for k, (verts, faces) in enumerate(meshes):
            if len(faces) == 0 or len(faces) > MAX_FACES:
                results.append(None)
                continue
            edges = mesh_edges(faces)
            stages, prev = [], None
            for stage in self.stages:
                verts, prev = stage([f[k] for f in feats], verts, edges, prev)
                stages.append(verts)
            results.append({"stages": stages, "faces": faces, "edges": edges})
        return vox_logits, results


# ---------- data ----------

def by_rat(subjects, zs):
    rats = {}
    for i, (n, z) in enumerate(zip(subjects.tolist(), zs.tolist())):
        rats.setdefault(n, {})[z] = i
    return rats


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data, limit=None):
        self.images, masks = data["train_images"], data["train_masks"]
        self.volumes, self.items = {}, []
        for n, zmap in by_rat(data["train_subjects"], data["train_zs"]).items():
            order = [zmap[z] for z in sorted(zmap)]
            self.volumes[n] = np.stack([masks[i] for i in order])
            boxes = {z: mask_to_box(masks[zmap[z]]) for z in sorted(zmap)}
            sides = {z: max(b[2] - b[0], b[3] - b[1]) for z, b in boxes.items() if b is not None}
            if not sides:
                continue
            top = max(sides.values())
            self.items += [(n, z, zmap[z], boxes[z]) for z, s in sides.items() if s >= MIN_SHARE * top]
        if limit:
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        n, z, i, box = self.items[k]
        size = max(box[2] - box[0], box[3] - box[1])
        scale = float(torch.empty(()).uniform_(1.6, 2.4))
        shift = tuple(float(torch.empty(()).uniform_(-0.25, 0.25)) * size for _ in range(2))
        win = window(box, scale, shift)
        img = image_input(cut(self.images[i], win))
        lo = torch.from_numpy(target_grid(self.volumes[n], win, z, G)).float()
        hi = torch.from_numpy(target_grid(self.volumes[n], win, z, G_HI)).float()
        if torch.rand(()) < 0.5:
            img, lo, hi = img.flip(-1), lo.flip(-1), hi.flip(-1)
        return img, lo, hi


# ---------- train / predict ----------

def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def losses(model, img, lo, hi):
    vox_logits, results = model(img)
    loss_vox = F.binary_cross_entropy_with_logits(vox_logits, lo)
    gt = cubify(hi, 0.5)
    cham = norm = edge = torch.zeros((), device=img.device)
    used = 0
    for res, (gv, gf) in zip(results, gt):
        if res is None or len(gf) == 0:
            continue
        with torch.no_grad():
            q, qn = sample_points(gv, gf, N_POINTS)
        for verts in res["stages"]:
            p, pn = sample_points(verts, res["faces"], N_POINTS)
            c, nl = chamfer(p, pn, q, qn)
            cham, norm, edge = cham + c, norm + nl, edge + edge_loss(verts, res["edges"])
        used += 1
    if used:
        cham, norm, edge = cham / used, norm / used, edge / used
    total = W_VOX * loss_vox + W_CHAMFER * cham + W_NORMAL * norm + W_EDGE * edge
    return total, {"vox": loss_vox.item(), "chamfer": cham.item(), "normal": norm.item(), "edge": edge.item(),
                   "meshes": used / len(results)}


@torch.no_grad()
def predict(model, data, device, out_dir):
    model.eval()
    images, rats = data["test_images"], by_rat(data["test_subjects"], data["test_zs"])
    choice = {int(n): (int(z), b, float(s)) for n, z, b, s in
              zip(data["pick_rats"], data["pick_z"], data["pick_boxes"], data["pick_scores"])}
    (out_dir / "meshes").mkdir(exist_ok=True)
    rat_ids, vols, scores = [], [], []
    for n in sorted(rats):
        zmap = rats[n]
        nz = max(zmap) + 1
        h, w = images[0].shape
        vol = np.zeros((nz, h, w), dtype=bool)
        score = 0.0
        if n in choice:
            z0, box, score = choice[n]
            win = window(box)
            img = image_input(cut(images[zmap[z0]], win))[None].to(device)
            _, results = model(img)
            res = results[0]
            if res is not None:
                verts, faces = res["stages"][-1], res["faces"]
                x, y, side = win
                xs = np.arange(max(0, x), min(w, x + side))
                ys = np.arange(max(0, y), min(h, y + side))
                for zk in range(nz):
                    zn = (zk - z0) / Z_HALF
                    if abs(zn) > 1:
                        continue
                    gy, gx = np.meshgrid(ys, xs, indexing="ij")
                    pts = np.stack([((gx + 0.5) - x) / side * 2 - 1, ((gy + 0.5) - y) / side * 2 - 1,
                                    np.full(gx.shape, zn)], -1).reshape(-1, 3)
                    inside = winding_inside(torch.from_numpy(pts).float().to(device), verts, faces).cpu().numpy()
                    vol[zk, gy.reshape(-1)[inside], gx.reshape(-1)[inside]] = True
                write_obj(out_dir / "meshes" / f"rat_{n:03d}.obj", verts.cpu().numpy(), faces.cpu().numpy(), win, z0)
        rat_ids.append(n)
        vols.append(vol)
        scores.append(score)
        print(f"rat {n}: {int(vol.sum())} voxels predicted", flush=True)
    np.savez_compressed(out_dir / "mesh_pred.npz", rats=np.array(rat_ids), volumes=np.stack(vols),
                        scores=np.array(scores, np.float32))


def write_obj(path, verts, faces, win, z0):
    """Mesh in millimetres, in the scan's voxel frame (x, y in-plane, z along the slices)."""
    x, y, side = win
    px = (verts[:, 0] + 1) / 2 * side + x
    py = (verts[:, 1] + 1) / 2 * side + y
    pz = verts[:, 2] * Z_HALF + z0 + 0.5
    mm = np.stack([px * SPACING_MM[0], py * SPACING_MM[1], pz * SPACING_MM[2]], 1)
    with open(path, "w") as f:
        f.writelines(f"v {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in mm)
        f.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="a few iterations, to check the setup works")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.data))
    device = pick_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    epochs, limit, max_iters = (1, 16, 3) if args.smoke else (args.epochs, None, None)
    config = {"model": "mesh_rcnn", "stage1": "faster_rcnn", "crop": CROP, "context": CONTEXT, "grid": [D, G, G],
              "z_half_slices": Z_HALF, "cubify_thresh": CUBIFY_THRESH, "stages": STAGES, "epochs": epochs,
              "batch": BATCH, "lr": LR, "loss_weights": [W_VOX, W_CHAMFER, W_NORMAL, W_EDGE]}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    log_file = open(out_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    train = TrainSet(data, limit)
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=False)
    model = MeshRCNN3D().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7 * epochs)], 0.1)
    ckpt, start = out_dir / "checkpoint.pt", 0
    if ckpt.exists() and not args.smoke:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start = state["epoch"] + 1
        log(f"resuming after epoch {start}")
    log(f"device {device}, {len(train)} training windows, {epochs} epochs")
    t0 = time.time()
    for epoch in range(start, epochs):
        model.train()
        sums, count = {}, 0
        for it, (img, lo, hi) in enumerate(loader):
            if max_iters and it >= max_iters:
                break
            total, parts = losses(model, img.to(device), lo.to(device), hi.to(device))
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
        sched.step()
        log(f"epoch {epoch + 1}/{epochs}  " + "  ".join(f"{k} {v / max(count, 1):.4f}" for k, v in sums.items())
            + f"  {(time.time() - t0) / 60:.1f} min")
        if not args.smoke:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch}, ckpt)
    torch.save(model.state_dict(), out_dir / "model.pt")
    predict(model, data, device, out_dir)
    log(f"done: {(time.time() - t0) / 60:.1f} min total")


if __name__ == "__main__":
    main()
