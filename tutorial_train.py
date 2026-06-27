from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 8
logger_freq = 250
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
accumulate_grad_batches = 8


checkpoint_callback = ModelCheckpoint(
    dirpath="./weights",
    filename="weights-{epoch:02d}-{step}",
    save_top_k=-1,
    every_n_train_steps=1000,
    save_weights_only=True
)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/3dgs_cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

log_images_kwargs = {
    "sample": True,             ### Tắt CFG, vì text luôn empty
    "unconditional_guidance_scale": 1.0   
}
logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=log_images_kwargs)
trainer = pl.Trainer(gpus=1, 
                     precision="bf16", 
                     callbacks=[logger, checkpoint_callback], 
                     accumulate_grad_batches=accumulate_grad_batches, 
                     max_steps=8000)  ## 1 step = 8 batch (batchsize x accu) -> 8000 steps = 64000 batch -> with 8k dataset = 64 epochs (1000 batch = 1 epoch)


# Train!
trainer.fit(model, dataloader)
