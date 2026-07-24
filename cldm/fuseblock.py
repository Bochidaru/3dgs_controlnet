import torch
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.attention import MemoryEfficientCrossAttentionSDPA


class FuseBlock(nn.Module):
    """A minimal three-way artifact/reference fusion block.

    The two gates implement:
      1. art vs. reference;
      2. local reference correspondence vs. global reference semantics.
    """

    def __init__(self, channels, num_heads=8, pose_dim=9, pose_hidden_dim=128):
        super().__init__()
        assert channels % num_heads == 0

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.SiLU(),
            nn.Linear(pose_hidden_dim, channels),
        )

        # One normalization before attention is enough for this first version.
        self.art_norm = nn.LayerNorm(channels)
        self.ref_norm = nn.LayerNorm(channels)

        self.local_attn = MemoryEfficientCrossAttentionSDPA(
            query_dim=channels,
            context_dim=channels,
            heads=num_heads,
            dim_head=channels // num_heads,
        )

        # Turns pooled ref information into a global semantic feature.
        self.global_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        # Chooses local correspondence or global semantic information.
        # It sees the artifact feature and both poses, hence can learn whether
        # the references are close enough for local matching.
        self.route_gate = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.SiLU(),
            nn.Linear(channels, 2),
        )

        # Chooses artifact or the selected reference feature.
        self.art_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )
        # Start with a mild preference for preserving art (sigmoid(1) ~ 0.73).
        nn.init.constant_(self.art_gate[-1].bias, 1.0)

    def _condition_ref(self, ref_tokens, pose):
        """Apply a small pose-dependent FiLM modulation to one reference."""
        pose_feat = self.pose_encoder(pose)  # B,C
        ref_tokens = self.ref_norm(ref_tokens)

        # Unlike only adding a shared pose bias, this changes token content
        # through a pose-dependent scale as well.
        return (
            ref_tokens * (1.0 + 0.1 * torch.tanh(pose_feat).unsqueeze(1))
            + pose_feat.unsqueeze(1)
        )

    def forward(self, art, ref1, ref2, ref1_pose, ref2_pose):
        batch, channels, height, width = art.shape

        ref1_pose = ref1_pose.to(device=art.device, dtype=art.dtype)
        ref2_pose = ref2_pose.to(device=art.device, dtype=art.dtype)

        art_tokens = art.flatten(2).transpose(1, 2)    # B,HW,C
        ref1_tokens = ref1.flatten(2).transpose(1, 2) # B,HW,C
        ref2_tokens = ref2.flatten(2).transpose(1, 2) # B,HW,C

        art_tokens = self.art_norm(art_tokens)
        ref1_tokens = self._condition_ref(ref1_tokens, ref1_pose)
        ref2_tokens = self._condition_ref(ref2_tokens, ref2_pose)

        # Local branch: use when a reference is sufficiently close.
        local_tokens = self.local_attn(
            x=art_tokens,
            context=torch.cat([ref1_tokens, ref2_tokens], dim=1),
        )
        local_feat = local_tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        # Global branch: pool semantic information after pose conditioning.
        # This deliberately does not average the two camera feature maps.
        global_desc = 0.5 * (
            ref1_tokens.mean(dim=1) + ref2_tokens.mean(dim=1)
        )
        global_desc = self.global_proj(global_desc)
        global_feat = global_desc[:, :, None, None].expand(
            -1, -1, height, width
        )

        art_desc = art_tokens.mean(dim=1)
        pose1_desc = self.pose_encoder(ref1_pose)
        pose2_desc = self.pose_encoder(ref2_pose)

        route = F.softmax(
            self.route_gate(
                torch.cat([art_desc, pose1_desc, pose2_desc], dim=-1)
            ),
            dim=-1,
        )
        ref_feat = (
            route[:, 0:1, None, None] * local_feat
            + route[:, 1:2, None, None] * global_feat
        )

        # If art and the selected ref agree, keep art. If they disagree in a
        # way correlated with an artifact, the learned gate can trust ref.
        diff_desc = torch.abs(art - ref_feat).mean(dim=(2, 3))
        art_weight = torch.sigmoid(
            self.art_gate(torch.cat([art_desc, diff_desc], dim=-1))
        )

        fused = (
            art_weight[:, :, None, None] * art
            + (1.0 - art_weight[:, :, None, None]) * ref_feat
        )
        return fused
