# Co-Student RF-DETR training

## Detection (default)

BBox Co-Student training with pseudo-label merging across weak/strong views:

```bash
uv run python train_costudent.py \
  --task detection \
  --train-image-dir datasets/squidle_coco/squidle_urchin_full_train_sparse/images \
  --train-ann-file datasets/squidle_coco/squidle_urchin_full_train_sparse/annotations/instances_train.json \
  --val-image-dir datasets/squidle_coco/squidle_urchin_2011/test2023 \
  --val-ann-file datasets/squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json \
  --model small \
  --freeze-encoder
```


```bash
uv run python train_costudent.py \
  --task detection \
  --train-image-dir datasets/exemplarsegmentation_outputs/coco_19616/images \
  --train-ann-file datasets/exemplarsegmentation_outputs/coco_19616/annotations/annotations_coco.json \
  --val-image-dir datasets/exemplarsegmentation_outputs/coco_13393/images \
  --val-ann-file datasets/exemplarsegmentation_outputs/coco_13393/annotations/annotations_coco.json \
  --model small \
  --wandb --wandb-project co-student-rf-detr \
  --freeze-encoder 
```

Pseudo-label merging is **on by default** for detection (`--pseudo-labels` is implicit). Disable with `--no-pseudo-labels` to train on sparse GT boxes only.

## Segmentation

Train an RF-DETR **Seg** variant (`RFDETRSegNano` / `Small` / `Medium` / `Large`) with COCO polygon mask supervision:

```bash
uv run python train_costudent.py \
  --task segmentation \
  --train-image-dir .../images \
  --train-ann-file .../instances_train.json \
  --val-image-dir .../test/images \
  --val-ann-file .../instances_test.json \
  --model small \
  --freeze-encoder
```

Requirements for segmentation:

- COCO annotations must include `segmentation` polygons (not bbox-only exports).
- Co-Student pseudo labels are **on by default** (same as detection); disable with `--no-pseudo-labels`.
- Checkpoints have `segmentation_head: true`; load with `RFDETRSeg*` at inference time.

## Co-Student pseudo labels and segmentation

The Co-Student method augments each image into **raw**, **weak**, and **strong** views. Student predictions on one view are warped into another and merged with sparse ground truth when they do not overlap existing labels (IoU + class match). A teacher (EMA) also denoises student predictions via `revision_pred`.

**Mask-aware pseudo labels (implemented):**

| Step | Behaviour |
|------|-----------|
| `result_to_detections` | Keeps post-NMS boolean masks from RF-DETR Seg postprocess |
| `cvt_detections` | Warps mask rasters with the same view transform as boxes (`cv2.warpPerspective`) |
| `revision_pred` | When teacher wins an overlap, copies teacher **box + mask** onto the student detection |
| `merge_ground_truth` | Appends unmatched pseudo **boxes and masks** to sparse GT |
| `triple_augment` | GT masks transformed on resize / flip / affine (same as images) |

**Matching still uses box IoU** (same as the original FCOS Co-Student port), not mask IoU. That is usually sufficient when boxes are tight; mask IoU matching could be added later for finer control.

Disable pseudo-label expansion with `--no-pseudo-labels` to train on sparse GT only (weak/strong branches still run).

## Parameter notes

| Flag | Notes |
|------|--------|
| `--task detection\|segmentation` | Model family and mask loading |
| `--pseudo-labels` / `--no-pseudo-labels` | Default: on for both tasks; use `--no-pseudo-labels` for sparse GT only |
| `--freeze-encoder` | Recommended for small/sparse datasets |
| `weight-decay` | Often raised (e.g. `3e-4`) for small datasets |

See existing README sections below for lr/warmup/freeze-encoder rationale.

```bash
uv run python train_costudent.py \
--train-image-dir datasets/squidle_coco/squidle_urchin_full_train_sparse/images \
--train-ann-file datasets/squidle_coco/squidle_urchin_full_train_sparse/annotations/instances_train.json \
--val-image-dir datasets/squidle_coco/squidle_urchin_2011/test2023 \
--val-ann-file datasets/squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json \
--wandb --wandb-project co-student-rf-detr \
--weight-decay 3e-4 \
--freeze-encoder
  ```

```bash
uv run python train_costudent.py \
--train-image-dir datasets/squidle_coco/squidle_urchin_full_train_sparse/images \
--train-ann-file datasets/squidle_coco/squidle_urchin_full_train_sparse/annotations/instances_train.json \
--val-image-dir datasets/squidle_coco/squidle_urchin_2011/test2023 \
--val-ann-file datasets/squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json \
--wandb --wandb-project co-student-rf-detr \
--weight-decay 5e-4 \
--freeze-encoder \
--model small \
--lr-drop-epochs 40,50 --lr-drop-gamma 0.1 
```

```bash
uv run python train_costudent.py \
  --task detection \
  --train-image-dir datasets/exemplarsegmentation_outputs/coco_19627/images \
  --train-ann-file datasets/exemplarsegmentation_outputs/coco_19627/annotations/annotations_coco.json \
  --val-image-dir datasets/exemplarsegmentation_outputs/coco_13393/images \
  --val-ann-file datasets/exemplarsegmentation_outputs/coco_13393/annotations/annotations_coco.json \
--wandb --wandb-project co-student-rf-detr \
--weight-decay 5e-4 \
--freeze-encoder \
--model small \
--mean-teacher \
--lr-drop-epochs 40,50 --lr-drop-gamma 0.1 \
--output-dir outputs/costudent_19627_gtonly \
--no-pseudo-labels


# Restart
uv run python train_costudent.py \
  --task detection \
  --train-image-dir datasets/exemplarsegmentation_outputs/coco_19616/images \
  --train-ann-file datasets/exemplarsegmentation_outputs/coco_19616/annotations/annotations_coco.json \
  --val-image-dir datasets/exemplarsegmentation_outputs/coco_13393/images \
  --val-ann-file datasets/exemplarsegmentation_outputs/coco_13393/annotations/annotations_coco.json \
--wandb --wandb-project co-student-rf-detr \
--weight-decay 5e-4 \
--model small \
--mean-teacher \
--lr-drop-epochs 50 --lr-drop-gamma 0.1 \
  --lr 1e-5 \
  --lr-encoder 1e-6 \
  --warmup-epochs 0 \
  --epochs 50 \
  --resume outputs/costudent/checkpoint_39.ckpt   --resume-weights-only

```

```bash
uv run python train_costudent.py \
  --task detection \
  --train-image-dir datasets/exemplarsegmentation_outputs/coco_19617/images \
  --train-ann-file datasets/exemplarsegmentation_outputs/coco_19617/annotations/annotations_coco.json \
  --val-image-dir datasets/exemplarsegmentation_outputs/coco_13393/images \
  --val-ann-file datasets/exemplarsegmentation_outputs/coco_13393/annotations/annotations_coco.json \
--wandb --wandb-project co-student-rf-detr \
--weight-decay 5e-4 \
--freeze-encoder \
--model small \
--mean-teacher \
--lr-drop-epochs 40,50 --lr-drop-gamma 0.1 \
--batch-size 16 --grad-accum-steps 1 --lr 1e-4 \
--output-dir outputs/costudent_19617_lr1e_4
```

### Visualise
```bash
uv run python viz_predictions.py \
  --checkpoint outputs/costudent/checkpoint_last_regular.pth \
  --image-dir datasets/exemplarsegmentation_outputs/coco_13393/images \
  --ann-file datasets/exemplarsegmentation_outputs/coco_13393/annotations/annotations_coco.json \
  --show-gt \
  --max-images 40 \
  --output-dir outputs/costudent/viz_outputs
  ```



## Parameter changes
weight-decay - set higher to clamp down large weight changes due to small dataset 3e-4
lr, lr_encoder - reduce by factor of 10
warmup epochs - 3 
--freeze-encoder

#### Freeze-encoder
Primary Academic Citation:

Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2025). RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. arXiv preprint arXiv:2511.09554. (Accepted at ICLR 2026).

The Context: The authors outline how their Weight-Sharing Neural Architecture Search (NAS) operates over the transformer layers. The paper addresses handling out-of-distribution transfer learning, explicitly documenting that while massive datasets can benefit from joint training (unfrozen backbones with a geometric layer-decay multiplier), data pools defined by small sizes or highly sparse target annotations require locking down the high-capacity DINOv2/DINOv3 spatial representations to prevent catastrophic representation collapse.

# notes on where changes were influenced from
Defaults for lr, lr-encoder, warmup_epochs based on.

1. RF-DETR Core Suggestions & Configuration API
The recommendations regarding RF-DETR Nano (its ~30.5M parameter count, its DINOv2 Vision Transformer backbone layer-decay, the specific 1e-4 default learning rate, and the small dataset adjustment rules) are sourced directly from:

The Official GitHub Repository: roboflow/rf-detr and the corresponding extension repository [roboflow/rf-detr_plus].

The Academic Citation: > Robinson, I., Robicheaux, P., Popov, M., Ramanan, D., & Peri, N. (2025). RF-DETR: Neural Architecture Search for Real-Time Detection Transformers. arXiv preprint arXiv:2511.09554. (Accepted at ICLR 2026).

Official Documentation: The rfdetr.roboflow.com core manuals covering the PyTorch Lightning training configurations, skip_best_epochs handling, and dataset-scale tuning guides.
