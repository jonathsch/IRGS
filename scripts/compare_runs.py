"""Compare metrics between two IRGS training runs (e.g. baseline vs --use_gt_supervision).

Usage:
    python scripts/compare_runs.py <baseline_model_path> <gt_model_path>
    python scripts/compare_runs.py outputs/bedroom/refgs_baseline outputs/bedroom/refgs_gt

Reads test PSNR from each run's eval/ours_<iter>/psnr.txt (written by training).
Optionally recomputes SSIM + LPIPS over the eval PNGs against the test-set GT
when --recompute is passed.
"""
import argparse, os, re, sys, glob

import torch
import numpy as np
from PIL import Image


def _parse_psnr_file(path):
    """Read 'tensor(12.85, device=...)' or '12.85' style strings into a float."""
    with open(path) as f:
        s = f.read().strip()
    m = re.search(r"-?\d+\.\d+", s)
    return float(m.group(0)) if m else float("nan")


def _latest_eval_dir(model_path):
    """Return the eval/ours_<iter>/ dir with the highest iter count."""
    cands = glob.glob(os.path.join(model_path, "eval", "ours_*"))
    if not cands:
        return None
    return max(cands, key=lambda p: int(re.search(r"ours_(\d+)", p).group(1)))


def _load_eval_renders(eval_dir):
    """Return list of (filename, tensor [3,H,W] in [0,1] on CPU)."""
    out = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "*.png"))):
        im = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        out.append((os.path.basename(p), torch.from_numpy(im).permute(2, 0, 1)))
    return out


def _gt_test_images(model_path):
    """Recover test-set GT tensors by re-loading the scene from cfg_args."""
    # The training process writes cfg_args (a Namespace) into model_path; we
    # reconstruct the source path and reload the test cameras.
    import ast
    with open(os.path.join(model_path, "cfg_args")) as f:
        cfg_src = f.read().strip()
    # cfg_args is a repr of Namespace(...) — eval needs Namespace in scope.
    from argparse import Namespace  # noqa: F401
    cfg = eval(cfg_src)
    source_path = cfg.source_path

    # Build a minimal Scene for the test cameras only.
    sys.path.insert(0, os.getcwd())
    from arguments.refgs import ModelParams
    from argparse import ArgumentParser
    parser = ArgumentParser()
    lp = ModelParams(parser)
    args = parser.parse_args([
        "-s", source_path, "-m", model_path, "--eval",
        "--resolution", str(getattr(cfg, "resolution", 1)),
    ])
    # Avoid loading the full gaussian model / point cloud — we just want cameras.
    # Use the same scene_info-only path the training code uses for image loading.
    from scene.dataset_readers import sceneLoadTypeCallbacks
    if os.path.exists(os.path.join(source_path, "transforms.json")) \
            and not os.path.exists(os.path.join(source_path, "transforms_train.json")):
        si = sceneLoadTypeCallbacks["NerfTransformsPly"](source_path, cfg.white_background, True)
    else:
        raise SystemExit(
            f"compare_runs.py only handles the NerfTransformsPly format right now "
            f"(source={source_path})."
        )
    from utils.camera_utils import cameraList_from_camInfos
    test_cams = cameraList_from_camInfos(sorted(si.test_cameras, key=lambda c: c.image_name),
                                          1.0, lp.extract(args))
    return [(c.image_name, c.original_image.detach().cpu()) for c in test_cams]


def _ssim_lpips(renders, gts):
    """Compute mean SSIM and LPIPS over paired (render, gt) tensors."""
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    ssims, lpipss = [], []
    for (_, r), (_, g) in zip(renders, gts):
        r = r.unsqueeze(0).cuda()
        g = g.unsqueeze(0).cuda()
        ssims.append(ssim(r, g).item())
        lpipss.append(lpips(r, g, net_type="vgg").item())
    return float(np.mean(ssims)), float(np.mean(lpipss))


def _summarize(model_path, recompute):
    eval_dir = _latest_eval_dir(model_path)
    if eval_dir is None:
        return None
    out = {"eval_dir": eval_dir, "iter": int(re.search(r"ours_(\d+)", eval_dir).group(1))}
    psnr_path = os.path.join(eval_dir, "psnr.txt")
    out["psnr"] = _parse_psnr_file(psnr_path) if os.path.exists(psnr_path) else float("nan")
    if recompute:
        renders = _load_eval_renders(eval_dir)
        gts = _gt_test_images(model_path)
        # match on basename order — both lists are pre-sorted.
        out["ssim"], out["lpips"] = _ssim_lpips(renders, gts)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("gt")
    p.add_argument("--recompute", action="store_true",
                   help="Re-run SSIM/LPIPS over the eval PNGs (slower).")
    args = p.parse_args()

    base = _summarize(args.baseline, args.recompute)
    gt   = _summarize(args.gt,       args.recompute)
    if base is None or gt is None:
        sys.exit(f"Missing eval dir(s):\n  baseline: {base}\n  gt: {gt}")

    print(f"\nBaseline: {args.baseline}  (iter {base['iter']})")
    print(f"GT:       {args.gt}        (iter {gt['iter']})\n")

    rows = [("PSNR  (test)", "psnr", "+", 2)]
    if args.recompute:
        rows += [("SSIM  (test)", "ssim", "+", 3),
                 ("LPIPS (test)", "lpips", "-", 3)]
    print(f"{'metric':<14} {'Baseline':>10} {'+GT':>10} {'Δ':>10}")
    print("-" * 47)
    for label, key, sign_hint, prec in rows:
        b, g = base[key], gt[key]
        delta = g - b
        marker = ""
        if sign_hint == "+":
            marker = "  ✓ better" if delta > 0 else ("  ✗ worse" if delta < 0 else "")
        elif sign_hint == "-":
            marker = "  ✓ better" if delta < 0 else ("  ✗ worse" if delta > 0 else "")
        fmt = f"{{:.{prec}f}}"
        print(f"{label:<14} {fmt.format(b):>10} {fmt.format(g):>10} {fmt.format(delta):>+10}{marker}")


if __name__ == "__main__":
    main()
