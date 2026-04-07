#!/bin/bash
# Train FMLM (cross-entropy distillation) on LM1B.
# Requires a pretrained FLM teacher checkpoint at TEACHER_PATH.

TEACHER_PATH="YOUR_FLM_CHECKPOINT_PATH"
DATA_CACHE_DIR="YOUR_DATA_DIR"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  data=lm1b-wrap \
  data.cache_dir=${DATA_CACHE_DIR} \
  model=small \
  model.length=128 \
  algo=fmlm \
  algo.double_temb=True \
  algo.use_discrete_schedule=True \
  algo.time_condition=alpha_t \
  algo.loss_scale=none \
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
  wandb.project=lm1b_full \
  wandb.name=lm1b_fmlm_PSD
