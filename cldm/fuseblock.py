import torch
import torch.nn as nn

from ldm.modules.attention import MemoryEfficientCrossAttentionSDPA


class FuseBlock(nn.Module):
    def __init__(self, channels, num_heads=8, pose_dim=9, pose_hidden_dim=128):
        super().__init__()

        assert channels % num_heads == 0

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.SiLU(),
            nn.Linear(pose_hidden_dim, channels),
        )

        self.cross_attn = MemoryEfficientCrossAttentionSDPA(
            query_dim=channels,
            context_dim=channels,
            heads=num_heads,
            dim_head=channels // num_heads,
        )

    def forward(self, art, ref1, ref2, ref1_pose, ref2_pose):
        b, c, h, w = art.shape

        ref1_pose = ref1_pose.to(device=art.device, dtype=art.dtype)
        ref2_pose = ref2_pose.to(device=art.device, dtype=art.dtype)

        pose1 = self.pose_encoder(ref1_pose)
        pose2 = self.pose_encoder(ref2_pose)

        art_tokens = art.flatten(2).transpose(1, 2)

        ref1_tokens = ref1.flatten(2).transpose(1, 2)
        ref2_tokens = ref2.flatten(2).transpose(1, 2)

        ref1_tokens = ref1_tokens + pose1.unsqueeze(1)
        ref2_tokens = ref2_tokens + pose2.unsqueeze(1)

        ref_tokens = torch.cat(
            [
                ref1_tokens,
                ref2_tokens,
            ],
            dim=1,
        )

        r_attn_tokens = self.cross_attn(
            x=art_tokens,
            context=ref_tokens,
        )

        r_attn = r_attn_tokens.transpose(1, 2).reshape(b, c, h, w)

        return art + r_attn


class TinyFuseBlock(nn.Module):
    def __init__(self, channels, pose_dim=9, pose_hidden_dim=128):
        super().__init__()

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.SiLU(),
            nn.Linear(pose_hidden_dim, channels * 2),
        )

        # Bắt đầu từ phép biến đổi identity:
        # scale = 0 và shift = 0.
        nn.init.zeros_(self.pose_encoder[-1].weight)
        nn.init.zeros_(self.pose_encoder[-1].bias)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.SiLU(),
        )

    def condition_ref(self, ref, pose):
        pose_feat = self.pose_encoder(pose)
        scale, shift = torch.chunk(pose_feat, 2, dim=-1)

        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        return ref * (1.0 + scale) + shift

    def forward(self, art, ref1, ref2, ref1_pose=None, ref2_pose=None):
        if ref1_pose is not None:
            ref1 = self.condition_ref(ref1, ref1_pose)

        if ref2_pose is not None:
            ref2 = self.condition_ref(ref2, ref2_pose)

        return self.fuse(torch.cat([art, ref1, ref2], dim=1))