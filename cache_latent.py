import os
import json
import shutil
import cv2
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


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


def get_cache_key(img_path, dataset_tag):
    filename = os.path.basename(img_path)
    # ref1_frame000002.png -> frame000002.png
    # ref2_frame000002.png -> frame000002.png
    # frame000002.png      -> frame000002.png  (source không có prefix)
    parts = filename.split('_', 1)
    frame_name = parts[1] if parts[0] in ('ref1', 'ref2') else filename
    return f"{dataset_tag}/{frame_name}"


def get_cache_key(img_path, dataset_tag):
    filename = os.path.basename(img_path)
    parts = filename.split('_', 1)
    frame_name = parts[1] if parts[0] in ('ref1', 'ref2') else filename
    return f"{dataset_tag}/{frame_name}"


def get_save_path(img_path):
    dirname = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    parts = filename.split('_', 1)
    clean_name = parts[1] if parts[0] in ('ref1', 'ref2') else filename
    return os.path.join(dirname, os.path.splitext(clean_name)[0] + '.pt')


@torch.no_grad()
def precompute_latents(config_path, ckpt_path, root_path, new_root_path, target_size=(512, 512), device='cuda'):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device).eval()

    with open(f'{root_path}/dataset.jsonl', 'rt') as f:
        lines = f.readlines()

    encoded_cache = {}  # cache_key -> save_path

    def encode_and_save(img_path, save_path, dataset_tag, is_source=False):
        if is_source:
            cache_key = get_cache_key(img_path, dataset_tag)
            
            if cache_key in encoded_cache:
                src = encoded_cache[cache_key]
                if src != save_path and not os.path.exists(save_path):
                    shutil.copy(src, save_path)
                return

            if os.path.exists(save_path):
                encoded_cache[cache_key] = save_path
                return

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}, skipping")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = resize_and_pad_to_square(img, target_size)
        img = img.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        img_vae = img_tensor * 2.0 - 1.0
        posterior = model.encode_first_stage(img_vae)
        vae_latent = model.get_first_stage_encoding(posterior)  # 1, 4, 64, 64

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(vae_latent.squeeze(0).cpu(), save_path)  # 4, 64, 64

        if is_source:
            encoded_cache[cache_key] = save_path

    for line in tqdm(lines):
        item = json.loads(line)
        dataset_tag = item['scene_tag']

        ref1_img_path   = root_path + '/' + item['ref']['ref1']['path']
        ref2_img_path   = root_path + '/' + item['ref']['ref2']['path']
        source_img_path = root_path + '/' + item['source']

        ref1_save_path   = new_root_path + '/' + os.path.normpath(get_save_path(ref1_img_path))
        ref2_save_path   = new_root_path + '/' + os.path.normpath(get_save_path(ref2_img_path))
        source_save_path = new_root_path + '/' + os.path.normpath(get_save_path(source_img_path))

        encode_and_save(ref1_img_path,   ref1_save_path,   dataset_tag)
        encode_and_save(ref2_img_path,   ref2_save_path,   dataset_tag)
        encode_and_save(source_img_path, source_save_path, dataset_tag, is_source=True)

if __name__ == '__main__':
    precompute_latents(
        config_path='models/3dgs_cldm_v15.yaml',
        ckpt_path='models/control_sd15_ini.ckpt',
        root_path='./cldm_dataset',
        new_root_path='./cache_latent',
        target_size=(512, 512),
        device='cuda',
    )