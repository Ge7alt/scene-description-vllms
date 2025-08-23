import json
import torch
import evaluate
from pathlib import Path
from typing import Dict, List, Optional
from utils import setup_logger
from PIL import Image
import numpy as np
from torchmetrics.multimodal.clip_score import CLIPScore

logger = setup_logger()

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

def calculate_clip_score(
    generated_captions_map: Dict[str, List[str]],
    image_folder: str
) -> Dict[str, float]:
    """
    Calculates the CLIPScore for a set of images and generated captions.
    This is a reference-free metric.
    """
    logger.info("Calculating CLIPScore...")
    results = {}
    
    # Prepare data for CLIPScore ---
    image_paths = []
    predictions_flat = []
    for img_name, gen_caps in generated_captions_map.items():
        image_path = Path(image_folder) / img_name
        if image_path.exists():
            image_paths.append(str(image_path))
            # Use the first generated caption for a standard comparison
            predictions_flat.append(gen_caps[0])

    if not image_paths:
        logger.warning("No images found for CLIPScore calculation.")
        return {"clip_score_error": "No matching images found."}

    # Calculate CLIPScore 
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        
        images = [Image.open(path).convert("RGB") for path in image_paths]
        img_tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images]

        metric.update(img_tensors, predictions_flat)
        clip_score = metric.compute()
        results["clip_score"] = round(clip_score.item(), 4)
    except Exception as e:
        logger.error(f"An error occurred during CLIPScore calculation: {e}")
        results["clip_score_error"] = str(e)
        
    return results

def calculate_reference_based_metrics(
    generated_captions_map: Dict[str, List[str]],
    reference_captions_path: Optional[str]
) -> Optional[Dict[str, float]]:
    """
    Calculates BLEU, ROUGE, and BERTScore metrics against COCO references.
    """
    if not reference_captions_path:
        logger.warning("No reference captions path provided. Skipping evaluation.")
        return None
        
    logger.info("Loading and parsing COCO reference captions...")
    references = load_coco_references(reference_captions_path)
    if not references:
        logger.error("Failed to load reference captions. Aborting evaluation.")
        return {"error": "Failed to load or parse reference captions file."}

    # Align predictions and references based on image filenames
    predictions_flat = []
    references_flat = []
    image_paths_for_clip = []
    for img_name, gen_caps in generated_captions_map.items():
        if img_name in references:
            # We use the first generated caption for a standard comparison
            predictions_flat.append(gen_caps[0])
            references_flat.append(references[img_name])

    if not predictions_flat:
        return {"error": "No matching images found between generated and reference sets."}

    logger.info(f"Evaluating metrics for {len(predictions_flat)} matched images.")
    try:
        bleu = evaluate.load("sacrebleu")
        rouge = evaluate.load("rouge")
        bertscore = evaluate.load("bertscore")
        
        results = {}
        # Compute all scores
        bleu_score = bleu.compute(predictions=predictions_flat, references=references_flat)
        rouge_score = rouge.compute(predictions=predictions_flat, references=references_flat)
        bert_score = bertscore.compute(predictions=predictions_flat, references=references_flat, lang="en")

        results = {
            "bleu": {
                "score": bleu_score["score"],
                "precisions": bleu_score["precisions"],
                "brevity_penalty": bleu_score["bp"],
                "system_length": bleu_score["sys_len"],
                "reference_length": bleu_score["ref_len"]
            },
            "rouge": {
                "rouge1": rouge_score["rouge1"],
                "rouge2": rouge_score["rouge2"],
                "rougeL": rouge_score["rougeL"],
                "rougeLsum": rouge_score["rougeLsum"]
            },
            "bertscore": {
                # Average scores for a single summary number
                "precision": sum(bert_score["precision"]) / len(bert_score["precision"]),
                "recall": sum(bert_score["recall"]) / len(bert_score["recall"]),
                "f1": sum(bert_score["f1"]) / len(bert_score["f1"])
            }
        }

        return results
    except Exception as e:
        logger.error(f"An error occurred during metric calculation: {e}")
        return {"error": str(e)}