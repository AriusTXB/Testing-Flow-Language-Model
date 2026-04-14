import torch
import torch.nn.functional as F
import transformers
from models.dit import DIT
import utils
import time
import os
from omegaconf import OmegaConf

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration (DIT-Small)
    config = OmegaConf.create({
        "model": {"hidden_size": 768, "cond_dim": 128, "length": 128, "n_blocks": 12, "n_heads": 12, 
                  "scale_by_sigma": True, "dropout": 0.0, "tie_word_embeddings": False, "vocab_lookup": True},
        "algo": {"backbone": "dit", "double_temb": False, "learnable_loss_weighting": False, "causal_attention": False}
    })

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size 
    model = DIT(config, vocab_size=vocab_size).to(device).to(torch.float16)
    
    ckpt_path = "pretrained_weights/checkpoints/lm1b_flm.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("backbone.")}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    L = config.model.length
    V = vocab_size
    
    # --- Experiment Settings ---
    batch_size = 32
    num_batches = 4
    total_samples = batch_size * num_batches
    
    print(f"\n🚀 Starting RL Rollout Throughput Benchmark...")
    print(f"Total Samples to generate: {total_samples} (Batch Size: {batch_size})")

    # 1. Flow-Based (FMLM) One-Step Throughput (Simulated One-Step)
    print("\n--- Measuring One-Step Flow Map (FMLM) Generation ---")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_batches):
            z = torch.randn((batch_size, L, V), device=device, dtype=torch.float16)
            tau = torch.tensor([1.0], device=device, dtype=torch.float16).expand(batch_size) # Jump to end
            
            # Simulated one-step jump
            logits = model(z, tau)
            tokens = logits.argmax(dim=-1)
    
    end_time = time.time()
    fmlm_time = end_time - start_time
    fmlm_tps = (total_samples * L) / fmlm_time
    print(f"FMLM Total Time: {fmlm_time:.4f}s")
    print(f"FMLM Throughput: {fmlm_tps:.2f} tokens/sec")

    # 2. Autoregressive (AR) Baseline (Simulated Baseline)
    # Autoregressive models require L forward passes per sample.
    # To be fair, we simulate the time it would take to do 128 sequential forward passes.
    print("\n--- Measuring Autoregressive (AR) Sequential Generation ---")
    start_time = time.time()
    
    with torch.no_grad():
        # AR needs L steps
        # To avoid actual memory OOM, we do a smaller batch for AR measurement if needed, 
        # but here we keep it equal for direct math.
        for step in range(L // 16): # Sampling 1/16th to estimate full L faster
            dummy_x = torch.randn((batch_size, step+1, V), device=device, dtype=torch.float16)
            dummy_tau = torch.zeros(batch_size, device=device, dtype=torch.float16)
            # Just one block of sequence to simulate AR context window
            _ = model(torch.randn((batch_size, L, V), device=device), dummy_tau)
            
    end_time = time.time()
    ar_estimated_time = (end_time - start_time) * 16 # Scale back up to full L
    ar_tps = (total_samples * L) / ar_estimated_time
    print(f"AR Estimated Time: {ar_estimated_time:.4f}s")
    print(f"AR Throughput: {ar_tps:.2f} tokens/sec")

    print("\n--- Summary ---")
    speedup = fmlm_tps / ar_tps
    print(f"Flow Model is approx. {speedup:.2f}x faster for rollouts than AR.")

if __name__ == "__main__":
    main()
