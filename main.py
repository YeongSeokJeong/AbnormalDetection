import hydra.utils
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.loggers import WandbLogger
import dotenv
from typing import List
from src import utils

dotenv.load_dotenv(override=True)
OmegaConf.register_new_resolver('eval', lambda x: eval(x))


def train(config: DictConfig):
    datasets = hydra.utils.instantiate(config['dataloader']['datasets'])
    train_loader, valid_loader = datasets.get_train_valid_dataloaders()

    # if "seed" in config:
    #     pl.seed_everything(config['seed'])

    model = hydra.utils.instantiate(config['framework'])
    ## callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))
        if any([isinstance(l, WandbLogger) for l in logger]):
            utils.wandb_login(key=config.wandb_api_key)

    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )


    utils.log_hyperparameters(
        config=config,
        model=model,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger
    )

    for l in [l for l in logger if isinstance(l, WandbLogger)]:
        l.watch(model=model, log='all', log_freq=25)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    utils.finish(
        logger=logger
    )

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@hydra.main('configs/','train.yaml')
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return train(config)

if __name__ == '__main__':
    main()
