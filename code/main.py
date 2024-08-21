import logging
import hydra
import submitit
import os
import time

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from code.utils.utils import get_output_dir
from code.trainers.hugginface_based_trainer import Hugginface_Trainer
LOG = logging.getLogger(__name__)


@record
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.trainer.type == "hugginface":
        trainer = Hugginface_Trainer(cfg)
    else:
        raise NotImplementedError

    with open_dict(cfg):
        cfg.trainer.logdir = str(get_output_dir(cfg, cfg.trainer.sync_key))
    if cfg.trainer.platform == "local":
        LOG.info(f"Output directory {cfg.trainer.logdir}/{cfg.trainer.sync_key}")
        trainer.setup_platform()
        trainer.setup_trainer()
        if cfg.trainer.qualitative_experiment:
            trainer.run_qualitative_experiments()
        else:
            raise NotImplementedError
        return 0
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
