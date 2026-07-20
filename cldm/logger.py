import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import json


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, val_batch_cache=[]):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_mode_toggle = False

        # For global step
        self.train_last_log_step = -1

        self.val_batch_cache = val_batch_cache

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, use_artifact_decode):
        sampler_name = "t200" if use_artifact_decode else "full"
        root = os.path.join(save_dir, "image_log", split, f"step_{global_step:06}_{sampler_name}")
        for k in images:
            if k == "scene_name":
                scene_names = images[k]
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.json".format(k, global_step, current_epoch, batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"scene_names": list(scene_names)}, f, ensure_ascii=False, indent=2)
                continue

            n = images[k].shape[0]
            nrow = int(n ** 0.5)  # sqrt → layout vuông nhất có thể
            grid = torchvision.utils.make_grid(images[k], nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train", use_artifact_decode=False):

        if ( # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, use_artifact_decode=use_artifact_decode, **self.log_images_kwargs)

            for k in images:
                if not isinstance(images[k], torch.Tensor):
                    continue
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, use_artifact_decode)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if not self.disabled:
            check_idx = pl_module.global_step
            if check_idx == self.train_last_log_step:
                return
            if self.check_frequency(check_idx):
                self.train_last_log_step = check_idx

                self.log_mode_toggle = not self.log_mode_toggle

                self.log_img(pl_module, batch, batch_idx, split="train", use_artifact_decode=self.log_mode_toggle)
                for idx, val_batch in enumerate(self.val_batch_cache):
                    self.log_img(pl_module, val_batch, idx, split="val", use_artifact_decode=self.log_mode_toggle)

            