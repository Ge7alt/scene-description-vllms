from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from PIL import Image

def visualize_and_save(
    image: Path, 
    captions: List[str], 
    output_path: Path
):
    """Displays an image and its captions, then saves the figure."""
    # image = Image.open(image_path)

    plt.figure(figsize=(8, 10))
    plt.imshow(image)
    plt.axis('off')
    
    caption_text = "Generated Captions:\n" + "\n".join(
        [f"- {cap}" for cap in captions]
    )
    
    plt.title(caption_text, wrap=True, ha='center', fontsize=10, pad=-10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 