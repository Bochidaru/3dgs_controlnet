from share import *

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class LROverrideCallback(pl.Callback):
    def __init__(self, new_lr: dict):
        self.new_lr = new_lr

    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]
        for group in opt.param_groups:
            name = group.get("name", "")
            if name in self.new_lr:
                old = group["lr"]
                group["lr"]         = self.new_lr[name]
                group["initial_lr"] = self.new_lr[name]
                print(f"[LROverride] group '{name}': {old:.2e} → {self.new_lr[name]:.2e}")


# Configs
resume_ckpt_path = ""                             ## For example: "./models/weights-epoch=30-step=2000.ckpt"
pretrain_path = './models/control_sd15_ini.ckpt'
pl.seed_everything(42, workers=True)
use_cache_latent = True
learning_rate_for_controlnet = 1e-5
learning_rate_for_new_module = 2e-5
learning_rate_for_unet_out = 2e-6
sd_locked = False
only_mid_control = False
num_val_batches = 1
image_logger_freq = 500


lr_override_values = {
    "controlnet": learning_rate_for_controlnet,
    "new": learning_rate_for_new_module,
    "unet_out": learning_rate_for_unet_out,
}


accumulate_grad_batches = 1
# DataLoader Config
batch_size = 128
num_workers = 8
prefetch_factor = 4 if num_workers > 0 else None
pin_memory = num_workers > 0
persistent_workers = num_workers > 0


checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="control_sd15_3dgs_{epoch:02d}_{step}",
    save_top_k=-1,
    every_n_train_steps=1000,
    save_weights_only=False
)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/3dgs_cldm_v15.yaml').cpu()
model.learning_rate_for_controlnet = learning_rate_for_controlnet
model.learning_rate_for_new_module = learning_rate_for_new_module
model.learning_rate_for_unet_out = learning_rate_for_unet_out
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(isTest=False, use_cached_latent=use_cache_latent)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

val_dataset = MyDataset(isTest=True, use_cached_latent=True)
val_batches = []
val_bs = 25
g = torch.Generator()
g.manual_seed(42)
val_dataloader = DataLoader(val_dataset, batch_size=val_bs, shuffle=True, generator=g)
val_iter = iter(val_dataloader)
for _ in range(num_val_batches):
    val_batches.append(next(val_iter))

log_images_kwargs = {
    "sample": True,
    "unconditional_guidance_scale": 1.0,
    "N": val_bs
}
logger = ImageLogger(max_images=val_bs, batch_frequency=image_logger_freq,
                     log_images_kwargs=log_images_kwargs, val_batch_cache=val_batches)

callbacks = [logger, checkpoint_callback, LROverrideCallback(lr_override_values)]

trainer = pl.Trainer(accelerator="gpu",
                     precision="bf16-mixed",
                     callbacks=callbacks,
                     accumulate_grad_batches=accumulate_grad_batches,
                     max_steps=100000)

# Train!
if resume_ckpt_path and os.path.exists(resume_ckpt_path):
    print(f"!!! CONTINUE FROM CHECKPOINT {resume_ckpt_path}!!!")
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_ckpt_path)
else:
    print("!!! INIT TRAIN !!!")
    model.load_state_dict(load_state_dict(pretrain_path))
    trainer.fit(model, train_dataloaders=dataloader)
