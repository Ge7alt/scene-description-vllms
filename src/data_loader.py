from pathlib import Path
from typing import List
from PIL import Image

def get_image_paths(folder_path: str) -> List[Path]:
    """Gets all JPEG and PNG image paths from a folder."""
    image_extensions = ['.jpg', '.jpeg', '.png']
    return [p for p in Path(folder_path).glob('*') if p.suffix.lower() in image_extensions]

def load_image(image_path: Path) -> Image.Image:
    """Loads an image and converts it to RGB."""
    img = Image.open(image_path).convert("RGB")
    return img