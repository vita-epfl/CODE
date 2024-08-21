import logging
import hydra
import submitit

from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record
from code.utils.utils import get_output_dir
from code.trainers.hugginface_based_trainer import Hugginface_Trainer

LOG = logging.getLogger(__name__)


@record
@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> int:
    if cfg.trainer.type == 'hugginface':
        trainer = Hugginface_Trainer(cfg)
    else:
        raise NotImplementedError

    with open_dict(cfg):
        cfg.trainer.logdir = str(get_output_dir(cfg, cfg.trainer.sync_key))

    if cfg.trainer.platform == "local":
        try:
            LOG.info(f"Output directory {cfg.trainer.logdir}/{cfg.trainer.sync_key}")
            trainer.setup_platform()
            trainer.setup_trainer()
            trainer.run()
        except Exception as e:
            LOG.info(f"Error: {e}")
            print(e)
        return 0

    # Mode SLURM
    executor = submitit.AutoExecutor(folder=cfg.trainer.logdir, slurm_max_num_timeout=30)
    executor.update_parameters(
        mem_gb=cfg.trainer.slurm.mem,
        gpus_per_node=cfg.trainer.slurm.gpus_per_node,
        tasks_per_node=cfg.trainer.slurm.gpus_per_node,  
        cpus_per_task=cfg.trainer.slurm.cpus_per_task,
        nodes=cfg.trainer.slurm.nodes,
        timeout_min=int(cfg.trainer.slurm.timeout) * 60,  
        # Below are cluster dependent parameters
        slurm_partition=cfg.trainer.slurm.partition,
        slurm_qos=cfg.trainer.slurm.qos,
        slurm_gres=f"gpu:{cfg.trainer.slurm.gpus_per_node}"
        # slurm_signal_delay_s=120,
        # **kwargs
    )

    executor.update_parameters(name=cfg.trainer.name)

    slurm_additional_parameters = {
        'requeue': True
    }

    if cfg.trainer.slurm.account:
        slurm_additional_parameters['account'] = cfg.trainer.slurm.account
    if cfg.trainer.slurm.reservation:
        slurm_additional_parameters['reservation'] = cfg.trainer.slurm.reservation

    executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)

    job = executor.submit(trainer)
    LOG.info(f"Submitted job_id: {job.job_id}")
    return job


if __name__ == "__main__":
    main()
