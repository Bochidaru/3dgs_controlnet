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
    def __init__(self, max_ref, root_path="./cldm_dataset/" ,target_size=(512,512), 
                 isTest=False, use_cached_latent=False, max_ref_vram_test=False):
        assert isinstance(max_ref, int) and max_ref >= 2, \
            f"max_ref must be int >= 2, but got {max_ref} ({type(max_ref)})"

        self.use_cached_latent = use_cached_latent
        self.cache_latent_root_path = "./cache_cldm_dataset/" if self.use_cached_latent else None
        self.data = []
        self.root_path = root_path
        self.target_size = target_size
        self.max_ref = max_ref
        self.max_ref_vram_test = max_ref_vram_test
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

                if max_ref_vram_test:
                    if len(ref_bank) >= max_ref:
                        self.data.append(json_data)     ## test vram, chỉ thêm những scene nào có trained image nhiều hơn max_ref
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
        # print("Mean x,y,z:", self.mean_xyz)
        # print("Std x,y,z:", self.std_xyz)
    

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

        source_path = os.path.join(self.root_path, item['source'])  # Artifact image
        target_path = os.path.join(self.root_path, item['target'])  # Groundtruth image
        prompt      = ""
        scene_name  = item["dataset"] + "/" + item["scene_tag"]

        # Load
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize
        source, source_pad_info = resize_and_pad_to_square(source, self.target_size)
        target, target_pad_info = resize_and_pad_to_square(target, self.target_size)

        # Normalize source and ref images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # Load ref bank
        ref_pose_bank_path = os.path.join(self.root_path, item["ref"]["ref_bank_path"])
        with open(ref_pose_bank_path, "rb") as f:
            ref_pose_bank = pickle.load(f)

        trained_folder_path = os.path.join(self.root_path, "trained_image", item["dataset"], item["scene_tag"])
        all_ref_names = sorted(os.listdir(trained_folder_path))

        best_refs = [item["ref"]["ref1_name"] ,item["ref"]["ref2_name"]]

        remaining_refs = [
            x for x in all_ref_names
            if x not in best_refs
        ]

        if self.isTest:
            n_extra = min(self.max_ref - 2, len(remaining_refs))

            if n_extra > 0:
                indices = np.round(
                    np.linspace(0, len(remaining_refs) - 1, n_extra)
                ).astype(int)
                sampled_refs = [remaining_refs[i] for i in indices]
            else:
                sampled_refs = []

            selected_refs = list(best_refs) + sampled_refs

        else:
            r = random.random()
            chosen_size = (
                self.max_ref if r < 0.7
                else max(2, self.max_ref - 2) if r < 0.9
                else max(2, self.max_ref - 4)
            )
            n_extra = min(self.max_ref - 2, len(remaining_refs))
            n_extra = min(n_extra, chosen_size - 2)

            selected_refs = (
                list(best_refs)
                + random.sample(remaining_refs, n_extra)
            )

        refs = []
        ref_poses = []

        for trained_image in selected_refs:
            ref_path = os.path.join(trained_folder_path, trained_image)
            ref_pose = self.normalize_pose(ref_pose_bank[trained_image])
            ref      = cv2.imread(ref_path)
            ref      = cv2.cvtColor(ref,   cv2.COLOR_BGR2RGB)
            ref, _   = resize_and_pad_to_square(ref, self.target_size)
            ref      = ref.astype(np.float32) / 255.0
            refs.append(ref)
            ref_poses.append(ref_pose)

        while len(refs) < self.max_ref:
            refs.append(np.zeros_like(refs[0]))
            ref_poses.append(np.zeros_like(ref_poses[0]))
        
        refs = np.stack(refs)
        ref_poses = np.stack(ref_poses)
        
        ref_masks = np.zeros(self.max_ref, dtype=bool)
        ref_masks[:len(selected_refs)] = True

        result = dict(groundtruth=target, txt=prompt, artifact=source,
                      ref=refs, ref_pose=ref_poses, ref_mask=ref_masks,
                      pad_info=source_pad_info, scene_name=scene_name)
        
        if self.use_cached_latent:
            def img_to_cache_path(abs_img_path, prefix):
                rel = os.path.relpath(abs_img_path, self.root_path)
                dirname, fname = os.path.split(rel)
                cache_fname = f"{prefix}_{os.path.splitext(fname)[0]}.pt"
                return os.path.join(self.cache_latent_root_path, dirname, cache_fname)

            # Source + Target latent
            result["z_artifact"] = torch.load(img_to_cache_path(source_path, "vae"), map_location="cpu")
            result["z_target"]   = torch.load(img_to_cache_path(target_path, "vae"), map_location="cpu")

            # Chỉ load VAE cho ref1 và ref2 (best_refs[0], best_refs[1])
            ref_vaes = []
            for ref_name in best_refs:  # luôn đúng 2 phần tử
                ref_path = os.path.join(trained_folder_path, ref_name)
                ref_vaes.append(torch.load(img_to_cache_path(ref_path, "vae"), map_location="cpu"))
            result["ref_vae"] = torch.stack(ref_vaes)  # [2, 4, 64, 64]

            # DINOv2 cho tất cả selected_refs, pad đến max_ref
            ref_dinos = []
            for trained_image in selected_refs:
                ref_path = os.path.join(trained_folder_path, trained_image)
                ref_dinos.append(torch.load(img_to_cache_path(ref_path, "dinov2"), map_location="cpu"))

            while len(ref_dinos) < self.max_ref:
                ref_dinos.append(torch.zeros_like(ref_dinos[0]))

            result["use_cache"] = True
            result["ref_dino"]  = torch.stack(ref_dinos)  # [max_ref, dim]

        return result

