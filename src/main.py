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
from evaluation import calculate_clip_score, calculate_reference_based_metrics

def main():
    """Main function to run the image captioning pipeline."""
    parser = argparse.ArgumentParser(description="Run the image captioning pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file for the experiment."
    )
    args = parser.parse_args()
    # breakpoint()

    # ========= Setup =========
    config = load_config(args.config)
    logger = setup_logger()
    
    # Create a unique output directory based on the model name from the config
    model_name_path = config.model.id.replace("/", "_")
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(config.paths.output_dir) / f"{model_name_path}_{run_timestamp}"
    viz_path = output_path / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    viz_path.mkdir(exist_ok=True)
    
    # ========= Load Model and Data =========
    logger.info(f"Loading model specified in: {args.config}")
    logger.info(f'Model Type: {config.model.type}')
    logger.info(f"Model ID: {config.model.id}")
    processor, model = load_model(
        config.model.id, 
        config.model.type,
        config.model.device, 
        config.model.torch_dtype
    )
    
    logger.info(f"Loading images from: {config.paths.image_folder}")
    image_paths = get_image_paths(config.paths.image_folder)
    
    # ========= Main Generation Loop =========
    results = {"per_image_metrics": {}}
    results["model_name"] = config.model.id
    with PerformanceTracker(device=config.model.device) as tracker:
        for i, image_path in enumerate(image_paths[:5]):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
            image = load_image(image_path)
            
            tracker.start()
            captions = generate_captions(
                image, model, config.model.type, processor, config.generation, config.model.device
            )
            perf_metrics = tracker.stop()
            
            results["per_image_metrics"][image_path.name] = {
                "captions": captions,
                "performance": perf_metrics
            }
            
            viz_output_file = viz_path / f"{image_path.stem}.png"
            visualize_and_save(image_path, captions, viz_output_file)

    
    all_latencies = [
        v["performance"]["latency_ms"] 
        for v in results["per_image_metrics"].values()
    ]
    all_gpu_usages = [
        v["performance"]["gpu_mem_used_mb"] 
        for v in results["per_image_metrics"].values()
    ]

    summary_stats = {
        "average_latency_ms": np.mean(all_latencies),
        "average_gpu_mem_used_mb": np.mean(all_gpu_usages),
        "total_images_processed": len(all_latencies)
    }
    
    results["summary_statistics"] = summary_stats
    logger.info(f"Summary Stats: {summary_stats}")

    # =================== Save Outputs ===================
    output_json_path = output_path / "results.json"
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_json_path}")

    # =================== Final Evaluation ===================
    logger.info("Starting final evaluation...")
    
    # Prepare generated captions in the required format
    generated_captions_map = {
        k: v["captions"] for k, v in results["per_image_metrics"].items()
    }
    
    # Dictionary to hold all evaluation scores
    final_eval_metrics = {}

    # Calculate CLIPScore (reference-free)
    clip_scores = calculate_clip_score(
        generated_captions_map=generated_captions_map,
        image_folder=config.paths.image_folder
    )
    final_eval_metrics.update(clip_scores)

    # Calculate reference-based metrics
    if config.evaluation.get("reference_captions_path"):
        ref_metrics = calculate_reference_based_metrics(
            generated_captions_map=generated_captions_map,
            reference_captions_path=config.evaluation.reference_captions_path
        )
        final_eval_metrics.update(ref_metrics)
    else:
        logger.warning("No reference_captions_path in config. Skipping reference-based metrics.")

    # Save the combined evaluation summary
    if final_eval_metrics:
        eval_output_path = output_path / "evaluation_summary.json"
        with open(eval_output_path, 'w') as f:
            json.dump(final_eval_metrics, f, indent=4)
        logger.info(f"Evaluation summary saved to {eval_output_path}")
        logger.info(f"Combined Evaluation Metrics: {final_eval_metrics}")
            
    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()