CKPT_PATH="/home/david3684/checkpoints/ours_flow_ce_1m.ckpt"
STEPS=1024

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      algo=flm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=32 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      algo.double_temb=False \
      eval.disable_ema=False \
      +wandb.offline=true \