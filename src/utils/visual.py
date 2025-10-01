import matplotlib.pyplot as plt
import numpy as np
import os
def visual_lys_2_func(video_name, scores, labels):

    fig = plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, :])
    
    x = np.arange(len(scores))
    ax3.plot(x, scores, color="#4e79a7", linewidth=1)
    ymin, ymax = 0, 1
    xmin, xmax = 0, len(scores)
    ax3.set_xlim([xmin, xmax])
    ax3.set_ylim([ymin, ymax])
    title = video_name
    ax3.text(0.02, 0.90, title, fontsize=16, transform=ax3.transAxes)
    for y_value in [0.25, 0.5, 0.75]:
        ax3.axhline(y=y_value, color="grey", linestyle="--", linewidth=0.8)
    ax3.set_yticks([0.25, 0.5, 0.75])
    ax3.tick_params(axis="y", labelsize=16)
    ax3.set_ylabel("Anomaly score", fontsize=18)
    ax3.set_xlabel("Frame number", fontsize=18)
    
    start_idx = None
    for frame_idx, label in enumerate(labels):
        if label!= 0 and start_idx is None:
            start_idx = frame_idx
        elif label == 0 and start_idx is not None:
            rect = plt.Rectangle((start_idx, ymin), frame_idx - start_idx, ymax - ymin, color="#e15759", alpha=0.5)
            ax3.add_patch(rect)
            start_idx = None
    
    if start_idx is not None:
        rect = plt.Rectangle((start_idx, ymin), len(labels) - start_idx, ymax - ymin, color="#e15759", alpha=0.5)
        ax3.add_patch(rect)
    
    ax1.set_title("Video frame", fontsize=18)
    ax2.set_title("Temporal summary", fontsize=18)
    
    plt.tight_layout()
    plt.show()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_image_path = os.path.join(current_directory, f"{video_name}.png")
    fig.savefig(output_image_path)
