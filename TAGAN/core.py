from utils.args import load_yaml
from utils.logger import Logger
from utils.device import init_device

from TAGAN.dashboard import Dashboard_v2
from TAGAN.trainer import TAGANTrainer
from TAGAN.dataset import TAGANDataset
from TAGAN.metric import TAGANMetric
from TAGAN.model import TAGANModel

logger = Logger(__file__)


class TAGANCore:
    """
    TAGANBand : Timeseries Analysis using GAN Band

    The Model for Detecting anomalies / Imputating missing value in timeseries data

    """

    def __init__(self, config: dict = None) -> None:
        # Set device
        self.device = init_device()

        # Set Config
        self.set_config(config=config)

        # Dataset option
        self.dataset = TAGANDataset(self.dataset_cfg, self.device)

        # Model option
        self.models = TAGANModel(self.models_cfg, self.device)
        self.netD, self.netG = self.models.init_model(self.dataset.dims)

        # Metric option
        self.metric = TAGANMetric(self.metric_cfg, self.device)

        # self.dashboard = Dashboard_v2(self.dataset)

    def set_config(self, config: dict = None) -> None:
        """
        Setting configuration

        If config/config.json is not exists,
        Use default config 'config.yml'
        """
        if config is None:
            logger.info("  Config : Default configs")
            config = load_yaml()
        else:
            logger.info("  Config : JSON configs")

        core = config["core"]

        # Core configs
        self.workers = core["workers"]
        self.batch_size = core["batch_size"]
        self.seed = core["seed"]  # UNUSED

        # Configuration Categories
        self.dataset_cfg = config["dataset"]
        self.models_cfg = config["models"]
        self.metric_cfg = config["metric"]
        self.trainer_cfg = config["trainer"]

    def run(self):
        logger.info("Inference the future")

    def train(self) -> None:
        logger.info("Train the model")
        
        trainer = TAGANTrainer(
            self.trainer_cfg,
            self.dataset,
            self.models,
            self.metric,
            self.device,
        )

        trainset = self.dataset.loader(self.batch_size, self.workers, train=True)
        validset = self.dataset.loader(self.batch_size, self.workers)
        netD, netG = trainer.run(
            self.netD,
            self.netG,
            trainset,
            validset
        )
        
        return netD, netG
