import torch
import random
import numpy as np
from utils.logger import Logger

logger = Logger(__file__)


def seeding(seed=31):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"  Seed   : {seed}")
