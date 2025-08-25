import subprocess
import time
from pathlib import Path
import sys
import logging

# =================== CONFIGURATION ===================
# --- List the models you want to run ---
MODELS_TO_RUN = [
    "blip2-opt",
    "smolvlm",
    "qwen-vl-instruct",
    "llava-hf",
]

# --- List the datasets to run on ---
DATASETS_TO_RUN = ["coco"]

# --- Paths ---
COCO_IMAGES_PATH = Path("datasets/coco/val2017") # IMPORTANT: Update this path
MAIN_SCRIPT_PATH = Path("src/main.py")
CONFIGS_DIR = Path("configs")
# =====================================================

def setup_logger():
    """Sets up a logger that outputs to both console and a file."""
    # Create logger
    logger = logging.getLogger("ExperimentRunner")
    logger.setLevel(logging.INFO)

    # Prevent logs from propagating to the root logger
    logger.propagate = False

    # Create console handler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler("runner_log.log", mode='w')
        fh.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        
    return logger

def main():
    """Main function to orchestrate and run the experiments."""
    logger = setup_logger()

    jobs = [
        {"model": model, "dataset": dataset}
        for model in MODELS_TO_RUN
        for dataset in DATASETS_TO_RUN
    ]
    total_jobs = len(jobs)
    
    if not MAIN_SCRIPT_PATH.exists():
        logger.error(f"Main script not found at '{MAIN_SCRIPT_PATH}'. Exiting.")
        return

    logger.info(f"Found {total_jobs} jobs to run. Starting experiments...")
    
    successful_jobs = 0
    failed_jobs = 0
    total_start_time = time.time()

    for i, job in enumerate(jobs):
        model_name = job['model']
        dataset_name = job['dataset']
        job_num = i + 1
        
        config_path = CONFIGS_DIR / f"{model_name}.yaml"
        if not config_path.exists():
            logger.warning(f"Config file not found for model '{model_name}' at '{config_path}'. Skipping job.")
            failed_jobs += 1
            continue

        logger.info("=" * 80)
        logger.info(f"Starting job {job_num}/{total_jobs}: Model='{model_name}', Dataset='{dataset_name}'")
        logger.info("=" * 80)

        command = [
            sys.executable, str(MAIN_SCRIPT_PATH),
            "--config", str(config_path),
            "--dataset", dataset_name
        ]

        if dataset_name == "coco":
            if not COCO_IMAGES_PATH or not COCO_IMAGES_PATH.exists():
                logger.error(f"COCO path '{COCO_IMAGES_PATH}' is not valid. Skipping COCO jobs.")
                failed_jobs += 1
                continue
            command.extend(["--coco_path", str(COCO_IMAGES_PATH)])

        job_start_time = time.time()
        
        try:
            # Using Popen to capture and log output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Log the output from the subprocess
            logger.info(f"--- Output from main.py for job {job_num} ---")
            for line in iter(process.stdout.readline, ''):
                # We log each line from the child process
                logging.getLogger("ExperimentRunner").info(line.strip())
            process.stdout.close()
            logger.info(f"--- End of output for job {job_num} ---")

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)

            job_end_time = time.time()
            duration = job_end_time - job_start_time
            logger.info(f"SUCCESS: Job {job_num} finished in {duration:.2f} seconds.")
            successful_jobs += 1

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            job_end_time = time.time()
            duration = job_end_time - job_start_time
            logger.error(f"FAILED: Job {job_num} failed after {duration:.2f} seconds. Error: {e}")
            failed_jobs += 1

        elapsed_time = time.time() - total_start_time
        avg_time_per_job = elapsed_time / job_num
        remaining_jobs = total_jobs - job_num
        estimated_remaining_time = remaining_jobs * avg_time_per_job
        
        est_h = int(estimated_remaining_time / 3600)
        est_m = int((estimated_remaining_time % 3600) / 60)
        
        logger.info("-" * 80)
        logger.info(f"Progress: {job_num}/{total_jobs} complete.")
        if remaining_jobs > 0:
            logger.info(f"Estimated time remaining: ~{est_h}h {est_m}m")
        logger.info("-" * 80)

    total_duration = time.time() - total_start_time
    total_h = int(total_duration / 3600)
    total_m = int((total_duration % 3600) / 60)
    total_s = int(total_duration % 60)
    
    logger.info("=" * 80)
    logger.info("All experiments finished!")
    logger.info(f"Total time taken: {total_h:02d}:{total_m:02d}:{total_s:02d}")
    logger.info(f"Successful jobs: {successful_jobs}/{total_jobs}")
    logger.info(f"Failed jobs: {failed_jobs}/{total_jobs}")
    logger.info(f"Full log saved to runner_log.log")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()