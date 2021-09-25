import os
import json
import torch
import random
import numpy as np
import pandas as pd
from TAGAN.core import TAGANCore
from utils.logger import Logger

logger = Logger(__file__)

torch.set_printoptions(precision=3, sci_mode=False)
pd.options.display.float_format = "{:.3f}".format
np.set_printoptions(linewidth=np.inf, precision=3, suppress=True)


def use_default_config(path: os.path = "config/config.json"):
    """
    User Default Configuration settings
    """
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    return config


def seeding(seed=31):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"  Seed   : {seed}")


if __name__ == "__main__":
    logger.info("*** TAGAN-BAND ***")
    logger.info("- System setting -")
    config = use_default_config("config/config.json")
    seeding(31)

    logger.info("- Model Setting -")
    model = TAGANCore(config=config)

    logger.info("- Model Running -")
    try:
        model.train()
    except KeyboardInterrupt:
        print("Abort!")

    model.run()

    # get OUTPUT Option
    # output file path : ./output_{data_title}.csv
    # model.get_labels(text=True)
