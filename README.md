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