#!/bin/bash
# Train FMLM (cross-entropy distillation) on OpenWebText.
# Requires a pretrained FLM teacher checkpoint at TEACHER_PATH.

TEACHER_PATH="/path/to/your/teacher/checkpoint.ckpt"
DATA_CACHE_DIR="/path/to/your/data"

python -u -m main \
  loader.global_batch_size=128 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  data.cache_dir=${DATA_CACHE_DIR} \
  model=small \
  model.length=1024 \
  algo=fmlm \
  algo.double_temb=True \
  algo.learnable_loss_weighting=False \
  algo.distillation_method=PSD \
  algo.use_mse_loss_psd=False \
  algo.diagonal_fraction=0.5 \
  algo.add_boundary=fixed \
  algo.boundary_prob=32 \
  algo.offdiagonal_sampling=uniform_diff \
  algo.use_simulated_entropy=False \
  algo.use_ema_for_psd_target=False \
  algo.teacher_path=${TEACHER_PATH} \
  algo.initialize_student_from_teacher=True \
  algo.use_teacher_for_D_t_lsd=False \
  algo.entmax_temp_lsd=2.0 \
  algo.backprop_entmax_temp_lsd=False \
  sampling.noise_removal=uniform_alpha \
  sampling.steps=[1,2,4,8,16,32,64,128] \
  trainer.max_steps=1000000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10000 \
  trainer.limit_val_batches=10 \
  optim.lr=3e-4 \
  optim.beta2=0.95 \
  wandb.project=owt_full \
  wandb.name=owt_fmlm_PSD
