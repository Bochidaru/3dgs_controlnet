import json
import cv2
import numpy as np
import os
import torch
import pickle
import random

from torch.utils.data import Dataset


def resize_and_pad_to_square(img, target_size=(512, 512)):
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


def get_save_path(img_path):
    dirname = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    parts = filename.split('_', 1)
    clean_name = parts[1] if parts[0] in ('ref1', 'ref2') else filename
    return os.path.join(dirname, os.path.splitext(clean_name)[0] + '.pt')


class MyDataset(Dataset):
    def __init__(self, root_path="./cldm_dataset/" ,target_size=(512,512), 
                 isTest=False, use_cached_latent=False):
        self.use_cached_latent = use_cached_latent
        self.cache_latent_root_path = "./cache_cldm_dataset/" if self.use_cached_latent else None
        self.data = []
        self.root_path = root_path
        self.target_size = target_size
        self.isTest = isTest

        poses_xyz_alldataset = []
        with open(f'{self.root_path}/dataset.jsonl', 'rt') as f:
            for line in f:
                json_data = json.loads(line)
                
                ref_pose_bank_path = os.path.join(self.root_path, json_data["ref"]["ref_bank_path"])
                with open(ref_pose_bank_path, "rb") as f:
                    ref_bank = pickle.load(f)
                
                for pose in ref_bank.values():
                    poses_xyz_alldataset.append(pose[:3])
                
                if json_data["is_test"] != isTest:  # isTest = False -> Train dataset; isTest = True -> Test dataset
                    continue

                self.data.append(json_data)
        
        poses_xyz_alldataset = np.stack(poses_xyz_alldataset)
        self.mean_xyz = poses_xyz_alldataset.mean(axis=0).astype(np.float32)
        self.std_xyz = poses_xyz_alldataset.std(axis=0).astype(np.float32)
        stats_xyz = {
            "mean": self.mean_xyz,
            "std": self.std_xyz
        }
        np.save(f"./models/pose_stats.npy", stats_xyz)  # Useful for inference
    

    def __len__(self):
        return len(self.data)
    

    def normalize_pose(self, pose_9d: np.ndarray) -> np.ndarray:
        """
            Input: pose 9D (numpy array), normalize 3 first elements (x,y,z) by self.mean_xyz và self.std_xyz
            Output: return xyz normalized pose 9D
        """
        pose_norm = pose_9d.copy().astype(np.float32)
        pose_norm[:3] = (pose_norm[:3] - self.mean_xyz) / self.std_xyz
        return pose_norm


    def __getitem__(self, idx):
        item = self.data[idx]

        source_path = os.path.join(self.root_path, item['source'])
        target_path = os.path.join(self.root_path, item['target'])

        ref1_name = item["ref"]["ref1_name"]
        ref2_name = item["ref"]["ref2_name"]
        scene_name = (item["dataset"] + "/" + item["scene_tag"] + "/" 
                      + item["image_name"] + "_" 
                      + f"ref1_{ref1_name}_ref2_{ref2_name}")

        # Load ref bank
        ref_pose_bank_path = os.path.join(self.root_path, item["ref"]["ref_bank_path"])
        with open(ref_pose_bank_path, "rb") as f:
            ref_pose_bank = pickle.load(f)

        trained_folder_path = os.path.join(
            self.root_path, "trained_image", item["dataset"], item["scene_tag"]
        )

        # Pose
        ref1_pose = self.normalize_pose(ref_pose_bank[ref1_name])  # [9]
        ref2_pose = self.normalize_pose(ref_pose_bank[ref2_name])  # [9]

        result = dict(
            txt        = "",
            ref1_pose  = ref1_pose,
            ref2_pose  = ref2_pose,
            scene_name = scene_name,
        )

        if self.use_cached_latent:
            def img_to_cache_path(abs_img_path, prefix):
                rel        = os.path.relpath(abs_img_path, self.root_path)
                dirname, fname = os.path.split(rel)
                cache_fname = f"{prefix}_{os.path.splitext(fname)[0]}.pt"
                return os.path.join(self.cache_latent_root_path, dirname, cache_fname)

            ref1_path = os.path.join(trained_folder_path, ref1_name)
            ref2_path = os.path.join(trained_folder_path, ref2_name)

            result["use_cache"]   = True
            result["z_artifact"]  = torch.load(img_to_cache_path(source_path, "vae"), map_location="cpu")
            result["z_target"]    = torch.load(img_to_cache_path(target_path, "vae"), map_location="cpu")
            result["z_ref1"]      = torch.load(img_to_cache_path(ref1_path,   "vae"), map_location="cpu")
            result["z_ref2"]      = torch.load(img_to_cache_path(ref2_path,   "vae"), map_location="cpu")

        else:
            ref1_path = os.path.join(trained_folder_path, ref1_name)
            ref2_path = os.path.join(trained_folder_path, ref2_name)

            source = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
            # target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
            ref1   = cv2.cvtColor(cv2.imread(ref1_path),   cv2.COLOR_BGR2RGB)
            ref2   = cv2.cvtColor(cv2.imread(ref2_path),   cv2.COLOR_BGR2RGB)

            source, source_pad_info = resize_and_pad_to_square(source, self.target_size)
            # target, _               = resize_and_pad_to_square(target, self.target_size)
            ref1,   _               = resize_and_pad_to_square(ref1,   self.target_size)
            ref2,   _               = resize_and_pad_to_square(ref2,   self.target_size)

            result["use_cache"]   = False
            # result["groundtruth"] = (target.astype(np.float32) / 127.5) - 1.0   # [-1, 1]
            result["artifact"]    = (source.astype(np.float32) / 127.5) - 1.0
            result["ref1"]        = (ref1.astype(np.float32) / 127.5) - 1.0
            result["ref2"]        = (ref2.astype(np.float32) / 127.5) - 1.0
            result["pad_info"]    = source_pad_info

        target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
        target, _ = resize_and_pad_to_square(target, self.target_size)
        result["groundtruth"] = (target.astype(np.float32) / 127.5) - 1.0

        return result

