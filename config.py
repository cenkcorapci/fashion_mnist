# -*- coding: utf-8 -*-
import logging
import pathlib
from os.path import expanduser

# Logs
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Experiments
RANDOM_STATE = 41

# Local files and folders
HOME = expanduser("~")
DL_MODELS_PATH = HOME + 'Research/pre-trained-models/cv/fashion_mnist'
TB_LOGS_PATH = HOME + 'Research/tb-logs/cv/fashion_mnist'

# create directories
logging.info("Checking/creating directories...")
pathlib.Path(DL_MODELS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(TB_LOGS_PATH).mkdir(parents=True, exist_ok=True)
logging.info("Directories are set.")
