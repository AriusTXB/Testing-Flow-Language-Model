#!/bin/bash
CKPT_PATH="path/to/your/checkpoint.ckpt"
STEPS=32

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
      eval.disable_ema=False \
      algo.learnable_loss_weighting=False \
      sampling.gamma=0.8 \
      +wandb.offline=true \
