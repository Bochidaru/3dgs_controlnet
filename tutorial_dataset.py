import json
import cv2
import numpy as np

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


class MyDataset(Dataset):
    def __init__(self, target_size=(512,512), isTest=False):
        self.data = []
        self.root_path = "./cldm_dataset/"
        self.target_size = target_size

        poses_xyz_alldataset = []
        with open(f'{self.root_path}/dataset.jsonl', 'rt') as f:
            for line in f:
                json_data = json.loads(line)
                
                for ref_key in ["ref1", "ref2"]:
                    pose = np.array(json_data["ref"][ref_key]["pose_rel"], dtype=np.float32)
                    json_data["ref"][ref_key]["pose_rel"] = pose  # reassign to np.array

                    poses_xyz_alldataset.append(pose[:3])

                if json_data["is_test"] != isTest:  # isTest = False -> Train dataset; isTest = True -> Test dataset
                    continue
                self.data.append(json_data)
        
        poses_xyz_alldataset = np.stack(poses_xyz_alldataset)
        self.mean_xyz = poses_xyz_alldataset.mean(axis=0)
        self.std_xyz = poses_xyz_alldataset.std(axis=0)
        stats_xyz = {
            "mean": self.mean_xyz,
            "std": self.std_xyz
        }
        np.save(f"{self.root_path}/pose_stats.npy", stats_xyz)  # Useful for inference
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

        source_path = self.root_path + item['source']  # Artifact image
        target_path = self.root_path + item['target']  # Groundtruth image
        ref1_path   = self.root_path + item["ref"]["ref1"]["path"]
        ref2_path   = self.root_path + item["ref"]["ref2"]["path"]
        
        prompt      = ""

        ref1_pose   = self.normalize_pose(item["ref"]["ref1"]["pose_rel"])
        ref2_pose   = self.normalize_pose(item["ref"]["ref2"]["pose_rel"])

        # Load
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)
        ref1   = cv2.imread(ref1_path)
        ref2   = cv2.imread(ref2_path)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        ref1   = cv2.cvtColor(ref1,   cv2.COLOR_BGR2RGB)
        ref2   = cv2.cvtColor(ref2,   cv2.COLOR_BGR2RGB)

        # Resize
        source, source_pad_info = resize_and_pad_to_square(source, self.target_size)
        target, target_pad_info = resize_and_pad_to_square(target, self.target_size)
        ref1, ref1_pad_info     = resize_and_pad_to_square(ref1,   self.target_size)
        ref2, ref2_pad_info     = resize_and_pad_to_square(ref2,   self.target_size)

        # Normalize source and ref images to [0, 1].
        source = source.astype(np.float32) / 255.0
        ref1   = ref1.astype(np.float32) / 255.0
        ref2   = ref2.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, 
                    ref1=ref1, ref1_pose=ref1_pose, 
                    ref2=ref2, ref2_pose=ref2_pose,
                    pad_info=source_pad_info)  # các ảnh đều như nhau về shape nên pad giống nhau?

