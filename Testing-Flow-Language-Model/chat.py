import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), "outputs/14_04_2026_Version_2/src"))
import torch
import transformers
from omegaconf import OmegaConf
from algo import FLM
import os
import sys

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Configuration (Matching the SFT setup)
    base_ckpt = "pretrained_weights/checkpoints/owt_flm.ckpt"
    sft_ckpt_dir = "outputs/14_04_2026_Version_2/checkpoints"
    
    config = OmegaConf.create({
        "model": {
            "hidden_size": 768,
            "cond_dim": 128,
            "length": 256,
            "n_blocks": 12,
            "n_heads": 12,
            "scale_by_sigma": True,
            "dropout": 0.0,
            "tie_word_embeddings": False,
            "vocab_lookup": True
        },
        "data": {
            "cache_dir": "./cache",
            "vocab_size": 50258
        },
        "algo": {
            "name": "flm",
            "backbone": "dit",
            "parameterization": "mean",
            "time_conditioning": True,
            "T": 0,
            "subs_masking": False,
            "double_temb": False,
            "learnable_loss_weighting": False,
            "causal_attention": False,
            "ignore_bos": False,
            "t_min": 1e-5,
            "t_max": 1.0
        },
        "sampling": {
            "steps": 100,
            "predictor": "euler"
        },
        "training": { "ema": 0.999, "sampling_eps": 1e-5 },
        "optim": { "lr": 1e-4 },
        "eval": { "checkpoint_path": base_ckpt }
    })

    # 2. Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 3. Model Setup
    print("Initializing Model...")
    model = FLM(config, tokenizer).to(device).to(torch.float16)
    
    # Try to load the latest SFT checkpoint if it exists, else load base
    ckpt_path = config.eval.checkpoint_path # Default to base
    
    if os.path.exists(sft_ckpt_dir):
        files = [os.path.join(sft_ckpt_dir, f) for f in os.listdir(sft_ckpt_dir) if f.endswith(".ckpt")]
        if files:
            ckpt_path = max(files, key=os.path.getctime)
            print(f"Loading SFT checkpoint: {ckpt_path}")
    else:
        print(f"SFT checkpoint not found. Loading base model: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    print("\n" + "="*50)
    print(" FLM Chat Interface (Alpaca Template)")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Format Prompt
            prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_tensor = torch.tensor(prompt_ids, device=device)
            
            # Generate
            print("FLM is thinking...", end="\r")
            with torch.no_grad():
                generated_ids = model.generate_with_prefix(
                    prompt_ids=prompt_tensor,
                    num_steps=config.sampling.steps
                )
            
            # Decode Suffix only (after the prompt)
            full_ids = generated_ids[0].tolist()
            # Extract only the newly generated tokens
            # prompt_ids might be shorter than prefix_len used in model due to padding/truncation
            # But generate_with_prefix uses the real prompt_ids length
            suffix_ids = full_ids[len(prompt_ids):]
            
            response = tokenizer.decode(suffix_ids, skip_special_tokens=True)
            
            print(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
