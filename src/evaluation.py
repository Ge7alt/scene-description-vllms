import json
import torch
import evaluate
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from utils import setup_logger
from PIL import Image
import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore
from datasets import load_dataset
from tqdm import tqdm

from typing import Union

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

logger = setup_logger()

def load_generated_captions(results_dir: str) -> Dict[str, List[str]]:
    """Loads the generated captions from the results JSON file"""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        logger.error(f"Results file not found at: {results_file}")
        raise FileNotFoundError(f"Results file not found at: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Standardize to a dict mapping image filenames to list of captions ({"image1.jpg": ["caption1", "caption2", ...], ...})
    captions_map = {}
    for img_id, entry in data["per_image_metrics"].items():
        captions_map[img_id] = entry["captions"]
    logger.info(f"Loaded generated captions for {len(captions_map)} images.") 
    return captions_map

def load_nocaps_references(split: str) -> Dict[str, List[str]]:
    """
    Loads NoCaps reference captions from the Hugging Face dataset.
    
    Args:
        split (str): The dataset split to load (e.g., "validation", "test").

    Returns:
        A dictionary mapping image file names to a list of reference captions.
    """
    logger.info(f"Loading NoCaps '{split}' split from HuggingFace...")
    try:
        dataset = load_dataset("HuggingFaceM4/NoCaps", split=split)
    except Exception as e:
        logger.error(f"Failed to load NoCaps dataset from HuggingFace: {e}")
        return {}
    
    filename_to_captions = {}
    filename_to_image = {}
    for item in dataset:
        filename = item['image_file_name']
        filename_to_image[filename] = item['image']
        
        # Using the corrected key based on your dataset schema
        captions = item['annotations_captions']
        if filename not in filename_to_captions:
            filename_to_captions[filename] = []
        filename_to_captions[filename].extend(captions)

    logger.info(f"Loaded {len(filename_to_captions)} reference captions and {len(filename_to_image)} images from NoCaps '{split}' split.")
    return filename_to_captions, filename_to_image



def load_coco_references(reference_path: str) -> Dict[str, List[str]]:
    """
    Parses a COCO captions annotation file into a usable dictionary.

    The COCO format is complex. This function maps image file names to a list
    of their corresponding ground truth captions.

    Args:
        reference_path (str): Path to the COCO captions JSON file.

    Returns:
        A dictionary mapping image file names (e.g., "00000012345.jpg") to a
        list of reference caption strings.
    """
    if not Path(reference_path).exists():
        logger.error(f"Reference file not found at: {reference_path}")
        return {}

    with open(reference_path, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from image ID to image file name
    image_id_to_filename = {
        image['id']: image['file_name'] for image in coco_data['images']
    }

    # Create a mapping from image file name to a list of captions
    filename_to_captions = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        filename = image_id_to_filename.get(image_id)
        if filename:
            if filename not in filename_to_captions:
                filename_to_captions[filename] = []
            filename_to_captions[filename].append(caption)
            
    return filename_to_captions

def calculate_bertscore(
    generated_captions_map: Dict[str, List[str]],
    reference_captions_map: Dict[str, List[str]]
) -> Optional[Dict[str, float]]:
    """Calculates BERTScore."""
    logger.info("Calculating BERTScore...")
    predictions_flat, references_flat = [], []
    for img_name, gen_caps in generated_captions_map.items():
        if img_name in reference_captions_map:
            predictions_flat.append(gen_caps[0])
            references_flat.append(reference_captions_map[img_name])

    if not predictions_flat:
        logger.warning("No matching images found for BERTScore calculation.")
        return {"bertscore_error": "No matching images."}

    try:
        bertscore = evaluate.load("bertscore")
        bert_score_results = bertscore.compute(predictions=predictions_flat, references=references_flat, lang="en")
        return {
            "bertscore_precision": round(np.mean(bert_score_results["precision"]), 4),
            "bertscore_recall": round(np.mean(bert_score_results["recall"]), 4),
            "bertscore_f1": round(np.mean(bert_score_results["f1"]), 4),
        }
    except Exception as e:
        logger.error(f"BERTScore calculation failed: {e}")
        return {"bertscore_error": str(e)}

def calculate_clip_score(
    generated_captions_map: Dict[str, List[str]],
    images_source: Union[str, Dict[str, Image.Image]],
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Calculates CLIPScore for images and captions using batch updates.
    Automatically handles long captions with LongCLIP if needed.
    """
    logger.info("Calculating CLIPScore...")
    results = {}

    # Prepare images and captions based on the source type
    images_data, captions = [], []
    is_folder = isinstance(images_source, (str, Path))
    
    for img_name, gen_caps in generated_captions_map.items():
        if is_folder:
            image_path = Path(images_source) / img_name
            if image_path.exists():
                images_data.append(image_path)
                captions.append(gen_caps[0])
        else: # Assumes dict of PIL images
            if img_name in images_source:
                images_data.append(images_source[img_name])
                captions.append(gen_caps[0])

    if not images_data:
        logger.warning("No matching images found for CLIPScore calculation.")
        return {"clip_score_error": "No matching images found."}

    # Calculate CLIPScore 
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Choose model based on caption length using real token count
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        max_tokens = max([tokenizer(c, return_tensors="pt").input_ids.shape[1] for c in captions])
        if max_tokens > 77:
            logger.info("Caption exceeds 77 tokens. Using LongCLIP model.")
            model_name = "zer0int/LongCLIP-L-Diffusers"
        else:
            model_name = "openai/clip-vit-base-patch16"

        metric = CLIPScore(model_name_or_path=model_name).to(device)

        num_batches = (len(images_data) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="CLIPScore batches"):
            batch_paths = images_data[i*batch_size : (i+1)*batch_size]
            batch_caps = captions[i*batch_size : (i+1)*batch_size]

            if is_folder:
                pil_images = [Image.open(p).convert("RGB").resize((224, 224)) for p in batch_paths]
            else:
                pil_images = [img.convert("RGB").resize((224, 224)) for img in batch_paths]

            img_tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 for img in pil_images]
            metric.update(img_tensors, batch_caps)
            logger.info(f"Processed CLIPScore batch {i+1}/{num_batches}")

        results["clip_score"] = round(metric.compute().item(), 4)


    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("CUDA out of memory. Reduce batch_size or resize images.")
        results["clip_score_error"] = str(e)
    except Exception as e:
        logger.error(f"Error during CLIPScore calculation: {e}")
        results["clip_score_error"] = str(e)

    return results

def calculate_coco_metrics(
    generated_captions_map: Dict[str, List[str]],
    reference_captions_map: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Calculates BLEU, ROUGE, CIDEr, and METEOR using pycocoevalcap.
    """
    logger.info("Calculating COCO captioning metrics (BLEU, ROUGE, CIDEr, METEOR)...")
    if not generated_captions_map or not reference_captions_map:
        return {"coco_metrics_error": "Empty generated or reference captions."}

    # Convert to COCOEvalCap style dicts
    gts, res = {}, {}
    for i, (img_name, gen_caps) in enumerate(generated_captions_map.items()):
        if img_name in reference_captions_map:
            gts[i] = [{"caption": cap} for cap in reference_captions_map[img_name]]
            res[i] = [{"caption": gen_caps[0]}]  # only one genearated caption per image

    # Tokenize captions (COCO-style preprocessing)
    tokenizer = PTBTokenizer()
    gts_tok = tokenizer.tokenize(gts)
    res_tok = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["bleu1", "bleu2", "bleu3", "bleu4"]),
        (Meteor(), "meteor"),
        (Rouge(), "rouge_l"),
        (Spice(), "spice"),
        (Cider(), "cider")
    ]

    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts_tok, res_tok)
        if isinstance(method, list):  # BLEU returns 4 numbers
            for m, s in zip(method, score):
                final_scores[m] = round(s, 4)
        else:
            final_scores[method] = round(score, 4)

    return final_scores

    
def main():
    """Main function to run standalone evaluation."""
    parser = argparse.ArgumentParser(description="Run standalone evaluation for image captioning results.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the output directory containing the 'results.json' file.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["coco", "nocaps"], help="The type of dataset to evaluate against ('coco' or 'nocaps').")
    parser.add_argument("--reference_path", type=str, required=True, help="Path to the COCO reference JSON file, OR the split name for NoCaps (e.g., 'validation').")
    parser.add_argument("--image_dir", type=str, help="Path to the directory with original images (required for CLIPScore).")
    parser.add_argument("--output_dir", type=str,help="Directory to save the evaluation summary. Defaults to --results_dir.")
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir) if args.output_dir else results_path
    output_path.mkdir(exist_ok=True)
    
    # Load generated captions from the specified results directory
    generated_captions = load_generated_captions(results_path)
    
    # Dictionary to hold all evaluation scores
    final_eval_metrics = {}
    # Load the appropriate reference captions
    references = None
    if args.dataset_type == "coco":
        references = load_coco_references(args.reference_path)
        if args.image_dir:
            clip_scores = calculate_clip_score(generated_captions, args.image_dir, batch_size=16)
            final_eval_metrics.update(clip_scores)
        else:
            logger.warning("No --image_dir provided for COCO. Skipping CLIPScore calculation.")

    elif args.dataset_type == "nocaps":
        # For NoCaps, load both references and images from Hugging Face
        references, image_map = load_nocaps_references(args.reference_path)
        # Calculate CLIPScore using the in-memory images
        clip_scores = calculate_clip_score(generated_captions, image_map, batch_size=16)
        final_eval_metrics.update(clip_scores)



    # Calculate reference-based metrics
    if references:
        ref_metrics = calculate_coco_metrics(generated_captions, references)
        final_eval_metrics.update(ref_metrics)

        # BERTScore
        bert_metrics = calculate_bertscore(generated_captions, references)
        final_eval_metrics.update(bert_metrics)

    else:
        logger.warning("Could not load references, skipping reference-based metrics.")

    
    # Save the combined evaluation summary
    if final_eval_metrics:
        eval_output_file = output_path / "evaluation_summary.json"
        with open(eval_output_file, 'w') as f:
            json.dump(final_eval_metrics, f, indent=4)
        logger.info(f"Evaluation summary saved to {eval_output_file}")
        logger.info(f"Final Evaluation Metrics: {final_eval_metrics}")
    else:
        logger.warning("No evaluation metrics were calculated.")

if __name__ == "__main__":
    main()
