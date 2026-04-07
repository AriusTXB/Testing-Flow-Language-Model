#!/bin/bash
CKPT_PATH="/home/david3684/projects/discrete-mean-flow/text/outputs/lm1b/2026.03.04/063736/checkpoints/last.ckpt"
STEPS=1

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      algo=fmlm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=8 \
      sampling.steps=$STEPS \
      algo.double_temb=True \
      eval.disable_ema=True \
      algo.learnable_loss_weighting=False \
      sampling.gamma=0.8 \
      +wandb.offline=true \
