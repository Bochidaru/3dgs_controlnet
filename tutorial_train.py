from share import *

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from temp_tutorial_dataset import MyDataset   ## !!!!!!!!!!!!!!! RM THIS WHEN DONE NEW DS
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configs
resume_ckpt_path = ""                             ## For example: "./models/weights-epoch=30-step=2000.ckpt"
pretrain_path = './models/control_sd15_ini.ckpt'
pl.seed_everything(42, workers=True)
use_cache_latent = True
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
num_val_batches = 1
image_logger_freq = 250

accumulate_grad_batches = 4             ## With 80gb vram, use bs=24, accu=4

# DataLoader Config
batch_size = 24

num_workers = 6
prefetch_factor = 2 if num_workers > 0 else None
pin_memory = num_workers > 0
persistent_workers = num_workers > 0


checkpoint_callback = ModelCheckpoint(
    dirpath="./weights",
    filename="weights-{epoch:02d}-{step}",
    save_top_k=-1,
    every_n_train_steps=1000,
    save_weights_only=False
)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/3dgs_cldm_v15.yaml').cpu()
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(isTest=False, use_cached_latent=use_cache_latent)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=num_workers, pin_memory=pin_memory, 
                        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

val_dataset = MyDataset(isTest=True, use_cached_latent=True)
val_batches = []
g = torch.Generator()
g.manual_seed(42)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=g)
val_iter = iter(val_dataloader)
for _ in range(num_val_batches):
    val_batches.append(next(val_iter))

log_images_kwargs = {
    "sample": True,             ### Tắt CFG, vì text luôn empty
    "unconditional_guidance_scale": 1.0   
}
logger = ImageLogger(batch_frequency=image_logger_freq, log_images_kwargs=log_images_kwargs, val_batch_cache=val_batches)
trainer = pl.Trainer(accelerator="gpu", 
                     precision="bf16-mixed", 
                     callbacks=[logger, checkpoint_callback],
                     accumulate_grad_batches=accumulate_grad_batches, 
                     max_steps=30000)


# Train!
if resume_ckpt_path and os.path.exists(resume_ckpt_path):
    print(f"!!! CONTINUE FROM CHECKPOINT {resume_ckpt_path}!!!")
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume_ckpt_path)
else:
    print("!!! INIT TRAIN !!!")
    model.load_state_dict(load_state_dict(pretrain_path))
    trainer.fit(model, train_dataloaders=dataloader)
