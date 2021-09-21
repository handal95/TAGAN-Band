import os
import json
import numpy as np
import torch
from utils.seed import seeding
from models.TAGANBand import TAGANBand

torch.set_printoptions(precision=4)
np.set_printoptions(linewidth=np.inf, precision=6, suppress=True)


def use_default_config(path:os.path="config/config.json"):
    """
    User Default Configuration settings
    """
    if os.path.exists(path):
        with open(path) as f:
            config = json.load(f)
    return config


if __name__ == "__main__":
    # Argument options - JSON
    config = None

    # Setting the JSON configuration for parameter
    # If use the RESTFul API, Skip this section and using JSON params.
    config = use_default_config("config/config.json")
    seeding(31)

    # Input the parameters when creating the model
    # By using 'config.json' or json format parameter
    model = TAGANBand(config=config)
    model.train()
    # model.run()

    # get OUTPUT Option
    # output file path : ./output_{data_title}.csv
    # model.get_labels(text=True)
