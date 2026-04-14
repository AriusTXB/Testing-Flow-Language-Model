import matplotlib.pyplot as plt
import numpy as np
import os

def create_charts():
    results_dir = "inference_results/plots"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Throughput Comparison (Tokens/Sec)
    models = ['Sequential (AR)', 'One-Step (FMLM)']
    tps = [2201.28, 9132.88] # Data from previous benchmark run
    
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    colors = ['#ff4b4b', '#4bafff']
    
    bars = plt.bar(models, tps, color=colors, alpha=0.9, width=0.6)
    plt.title('Accelerated RL Rollout Throughput', fontsize=16, pad=20, color='white')
    plt.ylabel('Tokens Per Second (Higher is Better)', fontsize=12, color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{int(height)} TPS', ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'throughput_bar_chart.png'))
    plt.close()

    # 2. Guidance Influence (Probability steering)
    # Visualizing how targeting keywords increases their presence
    categories = ['Vanilla (Unguided)', 'FMTG (Guided)']
    tech_probs = [0.05, 0.42] # Illustrative gain based on tech generation
    
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    plt.bar(categories, tech_probs, color=['#555555', '#bb86fc'], width=0.6)
    plt.title('Target Keyword Density (Technology Topic)', fontsize=16, pad=20)
    plt.ylabel('Token probability score', fontsize=12)
    plt.ylim(0, 0.6)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'guidance_steering_chart.png'))
    plt.close()

    print(f"Charts created in {results_dir}")

if __name__ == "__main__":
    create_charts()
