import os
import sys

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import share  # áp dụng các thiết lập runtime của project
from cldm.model import create_model, load_state_dict
from inference_dataset import InferDataset


# ========================= Settings =========================
CONFIG_PATH = "./models/3dgs_cldm_v15.yaml"
CHECKPOINT_PATH = "./checkpoints/control_sd15_3dgs.ckpt"
DATASET_ROOT = "./cldm_dataset"
POSE_STATS_PATH = "./models/pose_stats.npy"
OUTPUT_DIR = "./inference/inference_outputs"

BATCH_SIZE = 4
NUM_WORKERS = 4
DDIM_STEPS = 50
START_TIMESTEP = 300
TARGET_SIZE = (512, 512)


def encode_condition(model, batch, device):
    artifact = batch["artifact"].permute(0, 3, 1, 2).contiguous().to(device)
    ref1 = batch["ref1"].permute(0, 3, 1, 2).contiguous().to(device)
    ref2 = batch["ref2"].permute(0, 3, 1, 2).contiguous().to(device)

    # Encode chung để giảm số lần gọi VAE encoder.
    images = torch.cat([artifact, ref1, ref2], dim=0)
    latents = model.get_first_stage_encoding(
        model.encode_first_stage(images)
    ).detach()
    z_artifact, z_ref1, z_ref2 = latents.chunk(3, dim=0)

    batch_size = z_artifact.shape[0]
    condition = {
        "c_concat": [z_artifact],
        "c_crossattn": [model.get_empty_clip(batch_size)],
        "c_ref1": [z_ref1],
        "c_ref2": [z_ref2],
        "c_ref1_pose": [batch["ref1_pose"].to(device)],
        "c_ref2_pose": [batch["ref2_pose"].to(device)],
    }
    return condition


def to_uint8_rgb(image):
    image = image.clamp(-1, 1)
    image = ((image + 1) * 127.5).round().byte()
    return image.permute(1, 2, 0).cpu().numpy()


def remove_padding(image, pad_info):
    top, bottom, left, right = [int(v) for v in pad_info]
    height, width = image.shape[:2]
    return image[
        top:height - bottom if bottom > 0 else height,
        left:width - right if right > 0 else width,
    ]


def main():
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = InferDataset(
        root_path=DATASET_ROOT,
        target_size=TARGET_SIZE,
        pose_stats_path=POSE_STATS_PATH,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = create_model(CONFIG_PATH).cpu()
    model.load_state_dict(load_state_dict(CHECKPOINT_PATH, location="cpu"))

    # LPIPS chỉ phục vụ training loss, không dùng khi inference.
    if hasattr(model, "lpips_loss"):
        model.lpips_loss = None
    if hasattr(model, "cond_stage_model"):
        model.cond_stage_model = None

    model.requires_grad_(False).eval().to(device)

    with torch.inference_mode():
        for batch in tqdm(dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                condition = encode_condition(model, batch, device)

                samples, _ = model.sample_log(
                    cond=condition,
                    batch_size=len(batch["scene_name"]),
                    ddim=True,
                    ddim_steps=DDIM_STEPS,
                    timestep=START_TIMESTEP,
                )
                predictions = model.decode_first_stage(samples)

            for prediction, pad_info, scene_name in zip(
                predictions, batch["pad_info"], batch["scene_name"]
            ):
                prediction = remove_padding(
                    to_uint8_rgb(prediction), pad_info.tolist()
                )

                output_path = os.path.join(OUTPUT_DIR, scene_name + ".png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR),
                )


if __name__ == "__main__":
    main()
