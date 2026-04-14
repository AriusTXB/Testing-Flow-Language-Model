# Master Experiment Log - Flow Language Model (FLM)

This document tracks all Supervised Fine-Tuning (SFT) and Reasoning experiments conducted on the Flow Language Model.

## Chronological Index

| Date | Folder | Base Model | Goal | Result / Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **13/04/2026** | `outputs/13_04_2026_Version_1` | **LM1B** (139M) | Initial Chat SFT | Successfully aligned to instructions. Prompt clamping verified. Passed simple math test. |
| **14/04/2026** | `outputs/14_04_2026_Version_1` | **LM1B** (139M) | Reasoning Benchmark | Baseline assessment for Zero-Shot/Few-Shot/CoT. Limited by model scale. |
| **14/04/2026** | `outputs/14_04_2026_Version_2` | **OWT** (50k Vocab) | High-Vocab Alignment | Successfully resolved UniCode collapse using Logit Masks. Achieved fluent English. |
| **14/04/2026** | `outputs/14_04_2026_Version_3` | **OWT-Fidelity** | Tiny Benchmark | Verified coherence but identified "Article Bias" (rambling). Logic EM: 0%. |
| **14/04/2026** | `outputs/14_04_2026_Version_4` | **OWT + Reasoning** | Logic Augmentation | Training on GSM8K + Alpaca mix to break generative priors. Encountered catastrophic forgetting/gradient spike. |
| **14/04/2026** | `outputs/14_04_2026_Version_5` | **OWT + SFT Stabilized** | Smooth Alignment | 20 Epochs, clipped grads, low LR (3e-5). Model converges but ignores logic conditioning without CFG. |
| **Active** | `outputs/14_04_2026_Version_6` | **OWT + CFG Anchor** | CFG Null Training | Rewrote Dataloader for 15% prefix dropout. Model learns unconditional anchor. Generative capacity limit verified. |

---

## Technical History (Summaries)

### 1. Version 1 (LM1B)
- **Tokenization**: `bert-base-uncased` (30.5k tokens).
- **Setup**: SFT on cleaned Alpaca dataset. 
- **Discovery**: The model showed "Arithmetic Discovery"—it could solve $5+3=8$ after alignment, even though it was originally trained on random news text.

### 2. Version 2 (OWT)
- **Tokenization**: `gpt2` + added `[PAD]` (50,258 tokens).
- **Discovery**: Successfully resolved "Unicode Collapse" using Logit Masks.
- **Current Status**: Baseline for instruction following.

### 3. Version 3 (The Benchmark)
- **Status**: Completed.
- **Discovery**: OWT models generate clean text but treat reasoning prompts as "writing topics."
- **Failure Mode**: When asked $1+1$, the model writes a paragraph about why math is important rather than saying $2$.

### 4. Version 4 (Targeted Reasoner - Unstable)
- **Paradigim**: Simplex Flow Matching on One-Hot Encodings.
- **Focus**: Mixing GSM8K (7.5k math) into the instruction set.
- **Outcome**: Divergence due to aggressive learning rate ($1 \times 10^{-4}$) clashing with OpenWebText base-vector field.

### 5. Version 5 (Stabilized Reasoning SFT & CFG)
- **Modifications**: Extended to 20 epochs, reduced LR ($3 \times 10^{-5}$), batch accumulation to 8 (eff 64), added linear warmup, and gradient clipping at 1.0.
- **Status**: Completed SFT convergence ($Loss \approx 4.17$). Model successfully avoids Unicode collapse and produces highly fluent English words.
- **Discovery**: Testing via Tiny Benchmark revealed the 139M Flow Language Model ignores instruction-prompt conditions completely, favoring the unconditional "Article Mode" prior.
- **CFG Experiment**: Implemented Classifier-Free Guidance ($v = v_{uncond} + 2.0 \cdot (v_{cond} - v_{uncond})$) using `[PAD]` sequence substitution during inference.
- **CFG Outcome**: Failed structurally. The unconditional vector field produced out-of-distribution noise because the model pre-training lacked a native "null condition" dropout. However, we proved SFT modified the vector field significantly as the model outputted disjointed math vocabulary (`5<<<< 16 because they 50 each them number`) showing generative shift.
- **Next Step Requirement**: Rewrite the data loaders to introduce a 10-15% probability of randomly dropping the instruction prefix during training, cementing an explicit mathematical anchor for CFG generation.

### 6. Version 6 (CFG Dropout Anchor)
- **Modifications**: Modified `reasoning_dataloader.py` to drop the instruction prefix (`[PAD]` replacement) in 15% of samples while retaining `attention_mask = 1.0`. Evaluated using CFG 2.0.
- **Status**: Completed ($Loss \approx 1.04$). The generative SFT objective is fully solved.
- **Final Discovery**: The 139M Simplex Flow Language Model still exhibits generative collapse during logic logic generation even with a properly trained CFG anchor.
- **Architectural Conclusion**: Flow Matching is geometrically complex compared to Autoregressive models. 139M parameters lack the representational depth to follow zero-shot mathematical/structural syntax securely over a 256 sequence context. Scaling to $\geq 1$ Billion parameters is functionally mandatory to resolve logic alignment.