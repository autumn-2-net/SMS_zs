import pathlib

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from PL_callbacks.save_checkpoint import ModelCheckpoints
from utils.config_loader import get_config
# from utils_model.ssvc import ssvc
from utils_model.ssvc_v2_waug_v2 import ssvc
import lightning as pl


config=get_config('configs/a_v2_aug.yaml')
# config=get_config('configs/a1.yaml')
config.update({'infer':False})

models_ssvc=ssvc(config=config)
models_ssvc.build_losses_and_metrics()


# work_dir = pathlib.Path(config['base_work_dir'])/'testckpt'
# work_dir = pathlib.Path(config['base_work_dir'])/'largex_v4x'
work_dir = pathlib.Path(config['base_work_dir'])/'aug_2encoder_1_ff0_ne'
if __name__ == '__main__':
    trainer = pl.Trainer(
        accelerator=config['pl_trainer_accelerator'],
        devices=config['pl_trainer_devices'],
        num_nodes=config['pl_trainer_num_nodes'],
        strategy='auto',
        precision=config['pl_trainer_precision'],
        callbacks=[
            ModelCheckpoints(
                dirpath=work_dir,
                filename='model_ckpt_steps_{step}',
                auto_insert_metric_name=False,
                monitor='step',
                mode='max',
                save_last=False,
                # every_n_train_steps=hparams['val_check_interval'],
                save_top_k=config['num_ckpt_keep'],
                permanent_ckpt_start=config['permanent_ckpt_start'],
                permanent_ckpt_interval=config['permanent_ckpt_interval'],
                verbose=True
            ),
            # LearningRateMonitor(logging_interval='step'),
            # DsTQDMProgressBar(),
        ],
        logger=TensorBoardLogger(
            save_dir=str(work_dir),
            name='lightning_logs',
            version='lastest'
        ),
        gradient_clip_val=config['clip_grad_norm'],
        val_check_interval=config['val_check_interval'] * config['accumulate_grad_batches'],
        # so this is global_steps
        check_val_every_n_epoch=None,
        log_every_n_steps=1,
        max_steps=config['max_updates'],
        use_distributed_sampler=False,
        num_sanity_val_steps=config['num_sanity_val_steps'],
        accumulate_grad_batches=config['accumulate_grad_batches']
    )
    # models_ssvc.load_state_dict(torch.load(r'D:\propj\sum_a\ckpt\aug_v1_llr\model_ckpt_steps_53999.ckpt')['state_dict'])

    trainer.fit(models_ssvc,ckpt_path=r'D:\propj\sum_a\ckpt\aug_2encoder_1_ff0_ne\model_ckpt_steps_343999.ckpt' #ckpt_path=get_latest_checkpoint_path(work_dir)
                )