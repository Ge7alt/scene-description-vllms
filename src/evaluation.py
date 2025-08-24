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
    for item in dataset:
        filename = item['image_file_name']
        # The 'annotations_nocaps' field contains the list of reference captions
        captions = [ann['caption'] for ann in item['annotations_nocaps']]
        if filename not in filename_to_captions:
            filename_to_captions[filename] = []
        filename_to_captions[filename].extend(captions)

    logger.info(f"Loaded {len(filename_to_captions)} reference image captions from NoCaps '{split}' split.")
    return filename_to_captions


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
    
def main():
    """Main function to run standalone evaluation."""
    parser = argparse.ArgumentParser(description="Run standalone evaluation for image captioning results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the output directory containing the 'results.json' file."
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["coco", "nocaps"],
        help="The type of dataset to evaluate against ('coco' or 'nocaps')."
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to the COCO reference JSON file, OR the split name for NoCaps (e.g., 'validation')."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to the directory with original images (required for CLIPScore)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the evaluation summary. Defaults to --results_dir."
    )
    args = parser.parse_args()
    
    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir) if args.output_dir else results_path
    output_path.mkdir(exist_ok=True)
    
    # Load generated captions from the specified results directory
    generated_captions = load_generated_captions(results_path)
    
    # Load the appropriate reference captions
    references = None
    if args.dataset_type == "coco":
        references = load_coco_references(args.reference_path)
    elif args.dataset_type == "nocaps":
        # For NoCaps, reference_path is the split name
        references = load_nocaps_references(args.reference_path)

    # Dictionary to hold all evaluation scores
    final_eval_metrics = {}

    # Calculate reference-based metrics
    if references:
        ref_metrics = calculate_reference_based_metrics(generated_captions, references)
        final_eval_metrics.update(ref_metrics)
    else:
        logger.warning("Could not load references, skipping reference-based metrics.")

    # Calculate CLIPScore (reference-free) if image directory is provided
    if args.image_dir:
        clip_scores = calculate_clip_score(generated_captions, args.image_dir)
        final_eval_metrics.update(clip_scores)
    else:
        logger.warning("No --image_dir provided. Skipping CLIPScore calculation.")
    
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
