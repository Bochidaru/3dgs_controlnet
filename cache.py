import os
import json
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.encoders.modules import FrozenDINOv2ImageEmbedder


def resize_and_pad_to_square(img, target_size):
    h, w, c = img.shape
    target_w, target_h = target_size

    # Bước 1: resize theo cạnh lớn
    if h > w:
        new_h = target_h
        new_w = int(w * target_h / h)
    else:
        new_w = target_w
        new_h = int(h * target_w / w)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Bước 2: pad để thành vuông
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # màu đen
    )

    pad_info = (pad_top, pad_bottom, pad_left, pad_right)

    return img_padded, pad_info


@torch.no_grad()
def precompute_latents(root_path, new_root_path, model, model_dinov2, 
                       target_size=(512,512), device='cuda'):
    def preprocess_img(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        target_w, target_h = target_size
        if h > w:
            new_h = target_h
            new_w = int(w * target_h / h)
        else:
            new_w = target_w
            new_h = int(h * target_w / w)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        img_padded = img_padded.astype(np.float32) / 255.0
        return torch.from_numpy(img_padded).permute(2, 0, 1)

    def process_folder(folder, run_dino=False):
        folder_path = os.path.join(root_path, folder)
        for dataset in tqdm(os.listdir(folder_path), desc=f"Scene {folder}"):
            dataset_dir = os.path.join(folder_path, dataset)
            for run in os.listdir(dataset_dir):
                run_dir = os.path.join(dataset_dir, run)
                fnames = sorted(os.listdir(run_dir))

                imgs, valid_fnames = [], []
                for fname in fnames:
                    img_path = os.path.join(run_dir, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        print("Warning: cannot read", img_path)
                        continue
                    imgs.append(preprocess_img(img))
                    valid_fnames.append(fname)

                if not imgs:
                    continue

                batch_tensor = torch.stack(imgs).to(device)  # [B,C,H,W]

                # VAE encode theo mini-batch
                vae_latents = []
                for chunk in torch.split(batch_tensor * 2.0 - 1.0, 4):
                    posterior = model.encode_first_stage(chunk)
                    vae_latents.append(model.get_first_stage_encoding(posterior))
                vae_latents = torch.cat(vae_latents, dim=0)  # [B,4,64,64]

                # DINOv2 encode
                dino_feats = None
                if run_dino:
                    dino_feats = model_dinov2(batch_tensor)  # [B,dim]

                # Lưu từng ảnh
                for i, fname in enumerate(valid_fnames):
                    base = fname.replace(".png", ".pt")
                    out_dir = os.path.join(new_root_path, folder, dataset, run)
                    os.makedirs(out_dir, exist_ok=True)

                    torch.save(vae_latents[i].cpu(), os.path.join(out_dir, f"vae_{base}"))

                    if run_dino and dino_feats is not None:
                        torch.save(dino_feats[i].cpu(), os.path.join(out_dir, f"dinov2_{base}"))

    for f in ["artifact_image", "gt_image", "ref_image"]:
        process_folder(f, run_dino=False)
    process_folder("trained_image", run_dino=True)


if __name__ == '__main__':
    # Load model trước
    config = OmegaConf.load('models/3dgs_cldm_v15.yaml')
    model = instantiate_from_config(config).to('cuda')
    ckpt = torch.load('models/control_sd15_ini.ckpt', map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    model_dinov2 = FrozenDINOv2ImageEmbedder()
    model_dinov2.eval()

    precompute_latents(
        root_path='./cldm_dataset',
        new_root_path='./cache_cldm_dataset',
        model=model,
        model_dinov2=model_dinov2,
        target_size=(512, 512),
        device='cuda',
    )