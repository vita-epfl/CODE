import logging
import hydra
import submitit

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from ddpm.trainers.audio_trainer import AudioTrainer 
from ddpm.trainers.cityscape_trainer import Cityscape_Trainer
from ddpm.utils.utils import get_output_dir
from ddpm.trainers.ddib_based_trainer import DDIB_Trainer
from ddpm.trainers.hugginface_based_trainer import Hugginface_Trainer
LOG = logging.getLogger(__name__)


@record
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.trainer.type == "audio" :
        trainer = AudioTrainer(cfg)
    elif cfg.trainer.type == 'ddib':
        trainer = DDIB_Trainer(cfg)
    elif cfg.trainer.type == "cityscape" :
        trainer = DDIB_Trainer(cfg)
    elif cfg.trainer.type == "hugginface":
        trainer = Hugginface_Trainer(cfg)
        # trainer = Cityscape_Trainer(cfg)
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
            trainer.run_experiments()
        return 0
    else:
        raise NotImplementedError
    # # Mode SLURM
    # executor = submitit.AutoExecutor(folder=cfg.trainer.logdir, slurm_max_num_timeout=30)
    # executor.update_parameters(
    #     mem_gb=cfg.trainer.slurm.mem,
    #     gpus_per_node=cfg.trainer.slurm.gpus_per_node,
    #     tasks_per_node=cfg.trainer.slurm.gpus_per_node,  # one task per GPU
    #     cpus_per_task=cfg.trainer.slurm.cpus_per_task,
    #     nodes=cfg.trainer.slurm.nodes,
    #     timeout_min=int(cfg.trainer.slurm.timeout) * 60,  # max is 60 * 72
    #     # Below are cluster dependent parameters
    #     slurm_partition=cfg.trainer.slurm.partition,
    #     slurm_qos=cfg.trainer.slurm.qos,
    #     slurm_gres=f"gpu:{cfg.trainer.slurm.gpus_per_node}"
    #     # slurm_signal_delay_s=120,
    #     # **kwargs
    # )

    # executor.update_parameters(name=cfg.trainer.name)

    # slurm_additional_parameters = {
    #     'requeue': True
    # }

    # if cfg.trainer.slurm.account:
    #     slurm_additional_parameters['account'] = cfg.trainer.slurm.account
    # if cfg.trainer.slurm.reservation:
    #     slurm_additional_parameters['reservation'] = cfg.trainer.slurm.reservation

    # executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)

    # job = executor.submit(trainer)
    # LOG.info(f"Submitted job_id: {job.job_id}")
    # return job


if __name__ == "__main__":
    main()
