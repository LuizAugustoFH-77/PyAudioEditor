import logging
import sys

def setup_logger():
    logger = logging.getLogger("PyAudacity")
    logger.setLevel(logging.DEBUG)
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
        
    return logger

logger = setup_logger()
