import logging
import sys
from omegaconf import OmegaConf, DictConfig 

def setup_logger() -> logging.Logger:
    """Sets up a logger that outputs to stdout."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def load_config(config_path: str) -> DictConfig:
    """Loads a YAML configuration file using OmegaConf."""
    config = OmegaConf.load(config_path)
    return config
