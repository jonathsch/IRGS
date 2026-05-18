# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IRGS (CVPR 2025) — Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing. Joint geometry + inverse-rendering pipeline that decomposes a scene into base color, roughness, metallic, an environment map, and indirect lighting, then supports relighting. Built on 2D Gaussian Splatting (2DGS) with a custom CUDA/OptiX ray tracer over the surfels.

## Environment & build

The repo was upgraded from the original Python 3.8 / PyTorch 2.0 stack (see `environment.yml`) to Python 3.12 / PyTorch 2.9 / CUDA 12.8 (see `requirements.txt`). Prefer the new stack unless reproducing the published results; `environment.yml` is kept for reference.

Submodules build separately and are *not* pip-pinned from `requirements.txt`:

```bash
pip install submodules/diff-surfel-rasterization submodules/simple-knn submodules/raytracing
# 2D Gaussian Ray Tracer (CMake + pip)
cd submodules/surfel_tracer && rm -rf build && mkdir build && cd build && cmake .. && make && cd ../../..
pip install submodules/surfel_tracer
```

`submodules/raytracing/setup.py` downloads and vendors Eigen 3.4.0 into `submodules/raytracing/eigen-3.4.0/` on first build (the path is `.gitignore`d).

## Two-stage training pipeline

Stage 1 (`train_refgaussian.py`) — geometry reconstruction using Ref-Gaussian (`scene/ref_gaussian_model.py`, `gaussian_renderer/ref_gaussian.py`, args in `arguments/refgs.py`). Produces `chkpnt50000.pth`.

Stage 2 (`train.py`) — material decomposition. Loads the stage-1 checkpoint via `--start_checkpoint_refgs` and trains `GaussianModel` (`scene/gaussian_model.py`) with the inter-reflection renderer `render_ir` (`gaussian_renderer/__init__.py`). The `--train_ray` flag switches the loss from full-image rendering to a sub-sampled ray budget (`opt.trace_num_rays`).

`run_syn4relight.sh` and `run_tensoir.sh` are the canonical recipes for the two paper datasets and show the full per-scene hyperparameters.

### Common commands

```bash
# Stage 1
python train_refgaussian.py -s data/Synthetic4Relight/jugs -m outputs/.../refgs --eval -w --lambda_mask_entropy 0.05

# Stage 2
python train.py -s data/Synthetic4Relight/jugs --iterations 20000 \
  --start_checkpoint_refgs outputs/.../refgs/chkpnt50000.pth \
  --envmap_resolution 128 --diffuse_sample_num 256 \
  -m outputs/.../irgs --train_ray

# Render / evaluate
python render.py -m outputs/.../irgs --eval --diffuse_sample_num 512
python compute_albedo_scale_{syn4,tensoir}.py -m outputs/.../irgs
python eval_material_{syn4,tensoir}.py    -m outputs/.../irgs --albedo_rescale 2
python eval_relighting_{syn4,tensoir}.py  -m outputs/.../irgs --diffuse_sample_num 512 --light_sample_num 256 -e light
```

No test suite, no linter. Validation is by running `render.py` + the eval scripts against the checkpoint and comparing PSNR/SSIM/LPIPS metrics.

## Architecture notes

- **Renderer.** `gaussian_renderer/__init__.py:render_ir` is the main entry. It rasterizes surfels via `diff_surfel_rasterization` to get base_color / roughness / normal / alpha maps, then calls `rendering_equation` (or chunked variant `rendering_equation_chunk`) per surface pixel. The rendering equation samples incident rays (Fibonacci sphere + optional light-importance sampling against the envmap), traces each ray against the BVH via `pc.trace(...)`, and combines direct envmap radiance with traced indirect radiance through a GGX BRDF. Pipeline flags `wo_indirect`, `detach_indirect`, `wo_indirect_relight` toggle the indirect path.
- **Ray tracing.** Two backends coexist: `submodules/raytracing` (used in stage 1) and `submodules/surfel_tracer` (the 2D Gaussian Ray Tracer used in stage 2 via `surfel_tracer.GaussianTracer` inside `GaussianModel`). The BVH is (re)built by `GaussianModel.build_bvh()` / `update_bvh()` — the training loop calls one of these every step (build after densification, update otherwise).
- **Environment map.** `scene/light.py:EnvLight` is a learned cubemap; `pc.get_envmap` provides `pure_env`, `diffuse`, and `specular` queries plus light-importance sampling. For Synthetic4Relight / TensoIR a fixed coordinate-system rotation is applied (see `train.py` and `render.py`).
- **Dataset loaders.** `scene/dataset_readers.py:sceneLoadTypeCallbacks` dispatches on directory layout: `sparse/` → COLMAP, `transforms.json` (no `_train` variant) → `NerfTransformsPly`, `transforms_train.json` → Blender / Synthetic4Relight / StanfordORB (chosen by path keyword). `_try_load_gt_maps` opportunistically loads sibling `albedo/`, `normal/`, `roughness/`, `metallic/` folders for the optional GT material supervision (`OptimizationParams.use_gt_supervision`).
- **Argument layering.** `arguments/__init__.py` is the stage-2 schema; `arguments/refgs.py` is the stage-1 schema. They are independent `ParamGroup` classes. `get_combined_args` merges `cfg_args` (written at training time into `model_path/cfg_args`) with the current CLI overrides — render/eval scripts rely on this to pick up training-time params.
- **Model checkpoints.** `*.pth` files are tuples of `(model_params, iteration)`. `GaussianModel.restore_from_refgs` is the bridge that loads a stage-1 Ref-Gaussian checkpoint into the stage-2 `GaussianModel`.

## Conventions

- All training/eval scripts assume CUDA. The `data_device` arg defaults to `"cuda"`.
- sRGB ↔ linear conversions live in `utils.graphics_utils` (`rgb_to_srgb`, `srgb_to_rgb`); base color is stored linear and converted at the output boundary.
- Outputs land in `outputs/<dataset>/<scene>/{refgs,irgs}/`. Training visualizations write to `<model_path>/visualize/`.
