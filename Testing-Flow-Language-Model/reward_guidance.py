import torch
import torch.nn.functional as F
import transformers
from models.dit import DIT
import utils
import numpy as np
import os
import json
from omegaconf import OmegaConf

def calculate_keyword_reward(logits, target_tokens, device):
    """
    Differentiable reward function.
    Higher values if probabilities of target_tokens are high.
    logits: (1, L, V)
    target_tokens: list of token IDs
    """
    probs = F.softmax(logits, dim=-1) # (1, L, V)
    
    # Measure total probability of target tokens across all positions
    # We want to maximize this.
    reward = probs[:, :, target_tokens].sum()
    return reward

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration (DIT-Small)
    config = OmegaConf.create({
        "model": {
            "hidden_size": 768, "cond_dim": 128, "length": 128,
            "n_blocks": 12, "n_heads": 12, "scale_by_sigma": True,
            "dropout": 0.0, "tie_word_embeddings": False, "vocab_lookup": True
        },
        "algo": {
            "backbone": "dit", "double_temb": False, "learnable_loss_weighting": False,
            "causal_attention": False
        }
    })

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size 
    
    model = DIT(config, vocab_size=vocab_size).to(device).to(torch.float16)
    ckpt_path = "pretrained_weights/checkpoints/lm1b_flm.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    lut_a2g, _ = utils.build_luts(K=vocab_size)

    # Keywords to steer toward: "Technology", "Computer", "Software", "Internet"
    keyword_list = ["technology", "computer", "software", "internet", "programming", "data", "system"]
    target_tokens = []
    for kw in keyword_list:
        target_tokens.extend(tokenizer.encode(kw, add_special_tokens=False))
    target_tokens = list(set(target_tokens)) # Unique
    print(f"Targeting tokens: {target_tokens}")

    num_steps = 50 # Fewer steps for faster gradient iteration
    L = config.model.length
    V = vocab_size
    guidance_scale = 5.0 # lambda in steering equation

    print(f"\nStarting Reward-Guided Generation (FMTG)...")
    
    # Initialize from noise
    z = torch.randn((1, L, V), device=device, dtype=torch.float16, requires_grad=True)
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        tau_t_curr = tau_vals[i]
        tau_t_next = tau_vals[i + 1]
        t_curr = torch.tensor(utils.alpha_to_gamma(tau_t_curr.cpu().numpy(), lut_a2g), device=device, dtype=torch.float16)
        t_next = torch.tensor(utils.alpha_to_gamma(tau_t_next.cpu().numpy(), lut_a2g), device=device, dtype=torch.float16)
        dt = t_next - t_curr
        
        # 1. Base Model Forward Pass (with grad enabled for z)
        # Note: We wrap the model pass to get gradients w.r.t the input z
        z_in = z.detach().requires_grad_(True)
        with torch.set_grad_enabled(True):
            logits = model(z_in, tau_t_curr.expand(1).to(torch.float16))
            cap_value = 30.0
            logits = cap_value * torch.tanh(logits / cap_value)
            
            # 2. Reward Calculation on lookahead (x_1 prediction)
            # We treat normalized logits as a proxy for clean x_1 probabilities
            target_prob = calculate_keyword_reward(logits, target_tokens, device)
            
            # 3. Calculate Reward Gradient w.r.t input z
            grad_z = torch.autograd.grad(target_prob, z_in)[0]

        # 4. Integrate Base Velocity + Guided Steering
        with torch.no_grad():
            x_1_pred_probs = F.softmax(logits, dim=-1)
            v_base = (x_1_pred_probs - z_in) / (1.0 - t_curr + 1e-5)
            
            # Steered velocity: v = v_base + scale * grad(Reward)
            v_guided = v_base + guidance_scale * grad_z
            
            # Euler update
            z = z_in + dt * v_guided

    # Decode Result
    res_token_ids = z.argmax(dim=-1)[0].cpu().numpy()
    decoded_text = tokenizer.decode(res_token_ids, skip_special_tokens=True)
    
    print("\n--- Guided Generated Text ---")
    print(decoded_text)
    print("------------------------------\n")

if __name__ == "__main__":
    main()
