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
        
        # Trainer option
        self.trainer = TAGANTrainer(
            self.trainer_cfg,
            self.dataset,
            self.models,
            self.metric,
            self.device,
        )

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

    def run(self, netG=None):
        predsset = self.dataset.loader(self.batch_size, self.workers, preds=True)
        if netG == None:
            _, netG = self.models.load('0.1989')
        
        output = self.trainer.inference(netG, predsset)
        
        output.to_csv("./미나리.csv", index=False)
        
        # encoder_input = encoder_input.to(self.device)
        # decoder_input = torch.zeros([1, future_size+1, target_n], dtype=torch.float32).to(device)
        # output = model(encoder_input, decoder_input, False)
        # return output.cpu()

    def train(self) -> None:
        trainset = self.dataset.loader(self.batch_size, self.workers, train=True)
        validset = self.dataset.loader(self.batch_size, self.workers)
        netD, netG = self.trainer.run(
            self.netD,
            self.netG,
            trainset,
            validset
        )
        
        return netD, netG
