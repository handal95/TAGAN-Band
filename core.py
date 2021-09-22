import os
import json
import pandas as pd
import numpy as np
import torch
from utils.seed import seeding
from models.TAGANBand import TAGANBand
from utils.logger import Logger

logger = Logger(__file__)

torch.set_printoptions(precision=2, sci_mode=False)
pd.options.display.float_format = '{:.1f}'.format
np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)


def use_default_config(path: os.path = "config/config.json"):
    """
    User Default Configuration settings
    """
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
    return config


if __name__ == "__main__":
    logger.info("*** TAGAN-BAND ***")
    logger.info("- System setting -")
    config = use_default_config("config/config.json")
    seeding(31)

    logger.info("- Model Setting -")
    model = TAGANBand(config=config)

    logger.info("- Model Running -")
    try:
        model.train()
    except KeyboardInterrupt:
        print("Abort!")

    # model.run()

    # get OUTPUT Option
    # output file path : ./output_{data_title}.csv
    # model.get_labels(text=True)
