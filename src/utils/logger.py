"""Logging configuration utilities."""
import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


def setup_logging(
    config_path: Optional[str] = None, 
    level: str = "INFO",
    log_file: Optional[str] = None
):
    """Setup logging configuration."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Default configuration
        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        }
        
        if log_file:
            handlers['file'] = {
                'class': 'logging.FileHandler',
                'level': level,
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'a'
            }
        
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with given name."""
    return logging.getLogger(name)