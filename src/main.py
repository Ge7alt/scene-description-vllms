import json
from pathlib import Path
from datetime import datetime
import numpy as np
import argparse
from omegaconf import OmegaConf

from utils import setup_logger, load_config
from data_loader import get_image_paths, load_image
from model_handler import load_model, generate_captions
from performance import PerformanceTracker
from visualization import visualize_and_save
# from evaluation import calculate_clip_score, calculate_reference_based_metrics
import torch
import random
from datasets import load_dataset


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    """Main function to run the image captioning pipeline."""
    parser = argparse.ArgumentParser(description="Run the image captioning pipeline.")
    parser.add_argument("--image_folder", type=str, default=None, help="Path to a folder of images for caption generation (overrides dataset options).")
    parser.add_argument("--config", type=str, required=True,help="Path to the YAML model configuration file.")
    parser.add_argument("--dataset", type=str, choices=["coco", "nocaps"], help="Dataset to use: 'coco' or 'nocaps'.")
    parser.add_argument("--nocaps_split", type=str, default="validation", choices=["validation", "test"], help="Split for NoCaps dataset if selected.")
    parser.add_argument("--coco_path", type=str, default=None, help="Path to COCO images (required if dataset==coco).")
    args = parser.parse_args()
    # breakpoint()

    # ========= Setup =========
    config = load_config(args.config)
    logger = setup_logger()
    
    # Create a unique output directory based on the model name from the config
    model_name_path = config.model.id.replace("/", "_")
    # dataset_name = Path(args.dataset).name

    if args.image_folder:
        run_name = f"custom_{Path(args.image_folder).name}"
    elif args.dataset:
        run_name = Path(args.dataset).name
    else:
        raise ValueError("You must provide either --image_folder or --dataset.")
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # output_path = Path(config.paths.output_dir) / f"{model_name_path}_{dataset_name}_{run_timestamp}"
    output_path = Path(config.paths.output_dir) / f"{model_name_path}_{run_name}_{run_timestamp}"
    viz_path = output_path / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path.mkdir(exist_ok=True)
    
    # ========= Load Model  =========
    logger.info(f"Loading model specified in: {args.config}")
    logger.info(f'Model Type: {config.model.type}')
    logger.info(f"Model ID: {config.model.id}")
    processor, model = load_model(
        config.model.id, 
        config.model.type,
        config.model.device, 
        config.model.torch_dtype
    )
    # ========= Load Data =========

    image_items = []
    run_mode = ""
    if args.image_folder:
        logger.info(f"Loading images from custom folder: {args.image_folder}")
        image_items = get_image_paths(args.image_folder)
        run_mode = "folder"
        # For a custom folder, process all images
        images_to_process = image_items[:1]
        logger.info(f"Found {len(images_to_process)} images to process.")

    elif args.dataset:
        if args.dataset == "coco":
            if not args.coco_path:
                raise ValueError("For COCO dataset, --coco_path must be provided.")
            logger.info(f"Loading COCO images from {args.coco_path}")
            image_items = get_image_paths(args.coco_path)
            run_mode = "coco"
        elif args.dataset == "nocaps":
            logger.info(f"Loading NoCaps dataset ({args.nocaps_split} split) from HuggingFace")
            nocaps_ds = load_dataset("HuggingFaceM4/NoCaps", split=args.nocaps_split)
            image_items = [(item["image"], item["image_file_name"]) for item in nocaps_ds]
            run_mode = "nocaps"
        
        # For datasets, take a random sample for evaluation
        num_to_sample = 200
        images_to_process = random.sample(image_items, num_to_sample) if len(image_items) > num_to_sample else image_items
        logger.info(f"Randomly selected {len(images_to_process)} images for evaluation.")


    # ========= Main Generation Loop =========
    results = {"per_image_metrics": {}}
    results["model_name"] = config.model.id
    with PerformanceTracker(device=config.model.device) as tracker:
        for i, img_item in enumerate(images_to_process):
            logger.info(f"Processing image {i+1}/{len(image_items)}: {image_path.name}")
            # image = load_image(image_path)
            # Load image and assign an ID or name for both datasets
            if run_mode in  ["coco", "folder"]:
                image_path = img_item
                image = load_image(image_path)
                image_id = image_path.name
                logger.info(f"Processing image {i+1}/{len(image_items)}: {image_id}")
            elif run_mode == "nocaps":
                image, image_id = img_item  # img_item is (PIL.Image, filename)
                logger.info(f"Processing NoCaps image {i+1}/{len(image_items)}: {image_id}")

            
            tracker.start()
            captions, num_new_tokens = generate_captions(
                image, model, config.model.type, processor, config.generation, config.model.device
            )
            perf_metrics = tracker.stop(num_new_tokens=num_new_tokens)
            
            results["per_image_metrics"][image_id] = {
                "captions": captions,
                "performance": perf_metrics
            }
            
            # Visualization
            if i < 5:  # Limit to first 5 images for visualization
                viz_output_file = viz_path / f"{Path(image_id).stem}.png"
                visualize_and_save(image, captions, viz_output_file)
    
    all_latencies = [v["performance"]["latency_ms"] for v in results["per_image_metrics"].values()]
    all_gpu_usages = [v["performance"]["peak_gpu_mem_used_mb"] for v in results["per_image_metrics"].values()]
    all_throughputs = [v["performance"]["tokens_per_second"] for v in results["per_image_metrics"].values()]

    summary_stats = {
        "average_latency_ms": np.mean(all_latencies),
        "average_peak_gpu_mem_used_mb": np.mean(all_gpu_usages),
        "average_tokens_per_second": np.mean(all_throughputs),
        "total_images_processed": len(all_latencies)
    }
    
    results["summary_statistics"] = summary_stats
    logger.info(f"Summary Stats: {summary_stats}")

    # =================== Save Outputs ===================
    output_json_path = output_path / "results.json"
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_json_path}")
            
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()