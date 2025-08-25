from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


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
    
    # caption_text = "Generated Captions:\n" + "\n".join(
    #     [f"{cap}" for cap in captions]
    # )
    caption_text = captions[0]
    wrapped_caption = "\n".join(textwrap.wrap(caption_text, width=100))
    
    plt.title(wrapped_caption, wrap=True, ha='center', fontsize=10, pad=-10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 

# from pathlib import Path
# from typing import List, Union
# import matplotlib.pyplot as plt
# from PIL import Image
# import textwrap

# def visualize_and_save(
#     image: Image.Image, 
#     captions: List[str], 
#     output_path: Path
# ):
#     """Displays an image and its captions, then saves the figure."""
#     # Create a figure and axes
#     fig, ax = plt.subplots(figsize=(8, 10))
    
#     # Display the image
#     ax.imshow(image)
#     ax.axis('off')  # Hide the axes
    
#     caption_to_display = captions[0]

# # Create a figure and axes
#     fig, ax = plt.subplots(figsize=(8, 10))
    
#     # Display the image
#     ax.imshow(image)
#     ax.axis('off')
    
#     # Prepare the single caption text
#     wrapped_caption = "\n".join(textwrap.wrap(caption_to_display, width=100))
#     caption_text = f"Generated Caption:\n\n{wrapped_caption}"
    
#     # Add the text below the image
#     fig.text(0.5, 0.01, caption_text, 
#              ha="center",
#              va="bottom",
#              wrap=True, 
#              fontsize=9)
    
#     # Adjust layout to make space for the text
#     plt.subplots_adjust(bottom=0.2)
    
#     plt.savefig(output_path, bbox_inches='tight')
#     plt.close(fig)