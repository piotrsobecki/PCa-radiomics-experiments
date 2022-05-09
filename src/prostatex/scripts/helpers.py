import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(logger_name, base_dir=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s.%(funcName)s(): %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if base_dir is not None:
        fh = RotatingFileHandler(os.path.join(base_dir, 'debug.log'), maxBytes=10 ^ 6 * 1024)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
