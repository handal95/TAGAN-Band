import os
import json
import torch
import numpy as np

from models.TAGANBand import TAGANBand

torch.manual_seed(31)
np.set_printoptions(linewidth=np.inf)


if __name__ == "__main__":
    # Argument options - JSON
    config = None

    json_config_path = "config/config.json"
    if os.path.exists(json_config_path):
        with open(json_config_path) as f:
            config = json.load(f)

    # Input the parameters when creating the model
    # By using 'config.json' or json format parameter
    model = TAGANBand(config=config)
    model.run()
