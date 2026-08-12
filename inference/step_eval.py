import csv
import os
import sys
from collections import defaultdict

import cv2
import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import share
from cldm.model import create_model, load_state_dict
from tutorial_dataset import MyDataset
from inference.inference import encode_condition, to_uint8_rgb


# ========================= Settings =========================
CONFIG_PATH = "./models/3dgs_cldm_v15.yaml"
DATASET_ROOT = "./cldm_dataset"
OUTPUT_DIR = "./step_eval_outputs"

CHECKPOINTS = {
    "step_10000": "./checkpoints/control_sd15_3dgs_step=10000.ckpt",
    "step_15000": "./checkpoints/control_sd15_3dgs_step=15000.ckpt",
    "step_20000": "./checkpoints/control_sd15_3dgs_step=20000.ckpt",
}

BATCH_SIZE = 4
NUM_WORKERS = 4
DDIM_STEPS = 50
START_TIMESTEP = 300
TARGET_SIZE = (512, 512)
LPIPS_BACKBONE = "vgg"
SAVE_IMAGES = False


def get_pad_infos(batch):
    """Đưa pad_info do DataLoader tạo về shape [B, 4]."""
    pad_info = batch["pad_info"]
    if isinstance(pad_info, (list, tuple)):
        return torch.stack(pad_info, dim=1)
    return pad_info


def crop_tensor(image, pad_info):
    top, bottom, left, right = [int(value) for value in pad_info]
    height, width = image.shape[-2:]

    return image[
        :,
        top:height - bottom if bottom > 0 else height,
        left:width - right if right > 0 else width,
    ]


def get_scene_id(scene_name):
    """Lấy `dataset/scene_tag` từ chuỗi tên đầy đủ của một sample."""
    parts = scene_name.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid scene_name: {scene_name}")
    return "/".join(parts[:2])


def aggregate_by_scene(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)

    scene_rows = []
    for scene_id, scene_samples in sorted(groups.items()):
        scene_rows.append(
            {
                "scene_id": scene_id,
                "num_images": len(scene_samples),
                "psnr": float(np.mean([row["psnr"] for row in scene_samples])),
                "ssim": float(np.mean([row["ssim"] for row in scene_samples])),
                "lpips": float(np.mean([row["lpips"] for row in scene_samples])),
            }
        )
    return scene_rows


@torch.no_grad()
def compute_metrics(prediction, target, pad_info, lpips_metric):
    # Không tính metric trên vùng padding màu đen.
    prediction = crop_tensor(prediction.float().clamp(-1, 1), pad_info)
    target = crop_tensor(target.float().clamp(-1, 1), pad_info)

    prediction_01 = (prediction + 1.0) / 2.0
    target_01 = (target + 1.0) / 2.0

    mse = torch.mean((prediction_01 - target_01) ** 2)
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))

    ssim = structural_similarity_index_measure(
        prediction_01.unsqueeze(0),
        target_01.unsqueeze(0),
        data_range=1.0,
    )

    lpips_value = lpips_metric(
        prediction.unsqueeze(0),
        target.unsqueeze(0),
    ).mean()

    return psnr.item(), ssim.item(), lpips_value.item()


@torch.no_grad()
def evaluate_checkpoint(label, checkpoint_path, dataloader, lpips_metric, device):
    model = create_model(CONFIG_PATH).cpu()
    model.load_state_dict(load_state_dict(checkpoint_path, location="cpu"))

    # Hai module này không được dùng trong đường sampling hiện tại.
    if hasattr(model, "lpips_loss"):
        model.lpips_loss = None
    if hasattr(model, "cond_stage_model"):
        model.cond_stage_model = None

    model.requires_grad_(False).eval().to(device)

    checkpoint_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows = []

    for batch in tqdm(dataloader, desc=label):
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

        targets = batch["groundtruth"].permute(0, 3, 1, 2).to(device)
        pad_infos = get_pad_infos(batch)

        for prediction, target, pad_info, scene_name in zip(
            predictions, targets, pad_infos, batch["scene_name"]
        ):
            pad_info = pad_info.tolist()
            psnr, ssim, lpips_value = compute_metrics(
                prediction, target, pad_info, lpips_metric
            )

            rows.append(
                {
                    "scene_id": get_scene_id(scene_name),
                    "scene_name": scene_name,
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips_value,
                }
            )

            if SAVE_IMAGES:
                image = to_uint8_rgb(crop_tensor(prediction, pad_info))
                output_path = os.path.join(checkpoint_dir, scene_name + ".png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )

    if not rows:
        raise ValueError("MyDataset(isTest=True) returned no validation samples.")

    per_image_path = os.path.join(checkpoint_dir, "metrics_per_image.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Trung bình các ảnh thuộc cùng `dataset/scene_tag`.
    scene_rows = aggregate_by_scene(rows)
    per_scene_path = os.path.join(checkpoint_dir, "metrics_per_scene.csv")
    with open(per_scene_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=scene_rows[0].keys())
        writer.writeheader()
        writer.writerows(scene_rows)

    # Mean của checkpoint được tính trên các scene, để mỗi scene có trọng số bằng nhau.
    result = {
        "checkpoint": label,
        "num_images": len(rows),
        "num_scenes": len(scene_rows),
        "psnr": float(np.mean([row["psnr"] for row in scene_rows])),
        "ssim": float(np.mean([row["ssim"] for row in scene_rows])),
        "lpips": float(np.mean([row["lpips"] for row in scene_rows])),
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # isTest=True chính là tập validation cố định dùng để chọn checkpoint.
    dataset = MyDataset(
        root_path=DATASET_ROOT,
        target_size=TARGET_SIZE,
        isTest=True,
        use_cached_latent=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    lpips_metric = lpips.LPIPS(net=LPIPS_BACKBONE).eval().to(device)
    lpips_metric.requires_grad_(False)

    summary = []
    for label, checkpoint_path in CHECKPOINTS.items():
        result = evaluate_checkpoint(
            label,
            checkpoint_path,
            dataloader,
            lpips_metric,
            device,
        )
        summary.append(result)
        print(result)

    summary_path = os.path.join(OUTPUT_DIR, "checkpoint_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved checkpoint summary to: {summary_path}")


if __name__ == "__main__":
    main()
