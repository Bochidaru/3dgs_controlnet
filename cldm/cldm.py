import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
import lpips

torch.set_float32_matmul_precision('high')

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.fuseblock import FuseBlock, TinyFuseBlock


# FUSE_IDX = [ 2, 5, 8, 11 ]  # start from 0 to 12 (12 is middleblock), middle block is always being added to unet diff
#                                     # 2: 64; 5: 32; 8: 16; 11: 8; middle: 8 


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            pose_emb_dim=64,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.pose_emb_dim = pose_emb_dim
                                        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        '''
        # self.input_hint_block = TimestepEmbedSequential(
        #     conv_nd(dims, hint_channels, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 32, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 96, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )
        '''


        self.ref_time_emb = nn.Parameter(torch.zeros(1, time_embed_dim))
        self.input_art_ref_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, in_channels, model_channels, 3, padding=1)),
        )
        self.fuse_blocks = nn.ModuleList([
            TinyFuseBlock(320),   # 0

            TinyFuseBlock(320),   # 1
            TinyFuseBlock(320),   # 2
            TinyFuseBlock(320),   # 3

            TinyFuseBlock(640),   # 4
            TinyFuseBlock(640),   # 5
            TinyFuseBlock(640),   # 6

            TinyFuseBlock(1280),  # 7
            TinyFuseBlock(1280),  # 8
            TinyFuseBlock(1280),  # 9

            TinyFuseBlock(1280),  # 10
            TinyFuseBlock(1280),  # 11

            TinyFuseBlock(1280),  # middle
        ])



        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, use_simple_tf_block=True    # tắt 1 lần attn với cond là text
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, use_simple_tf_block=True   # tắt 1 lần attn với cond là text
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, ref1, ref2, ref1_pose, ref2_pose, context=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        art_emb = self.time_embed(t_emb)  # B, 1280
        batch_size = art_emb.shape[0]
        ref_emb = self.ref_time_emb.expand(batch_size, -1)  # B, 1280

        ref1, ref2, ref1_pose, ref2_pose = ref1[0], ref2[0], ref1_pose[0], ref2_pose[0]

        # [artifact, ref1, ref2]
        emb = torch.cat(
            [art_emb, ref_emb, ref_emb],
            dim=0,
        )  # 3B, 1280

        art_ref_cat = torch.cat([hint, ref1, ref2], dim=0)       # 3B,4,64,64
        art_ref_cat = self.input_art_ref_block(art_ref_cat, emb) # 3B,320,64,64

        outs = []

        h = x.type(self.dtype)   # B,4,64,64   THIS IS X_T !!!

        # count from 0 to 11, 12 is middle block, 0 is input block
        for idx, (module, zero_conv) in enumerate(zip(self.input_blocks, self.zero_convs)):
            if art_ref_cat is not None:
                h = module(h, emb)    # B,320,64,64    x_t đi vào input_blocks[0] để nâng chiều
                h = torch.cat(
                    [
                        h,
                        torch.zeros_like(h),
                        torch.zeros_like(h)
                    ],
                    dim=0
                )      # 3B,4,64,64   not allow adding between ref and x_t
                h += art_ref_cat
                art_ref_cat = None
            else:
                h = module(h, emb)
                
            art, ref1, ref2 = torch.chunk(h, 3, dim=0)
            fused = self.fuse_blocks[idx](art, ref1, ref2, ref1_pose, ref2_pose)
            outs.append(zero_conv(fused, emb[:batch_size]))

        h = self.middle_block(h, emb)
        art, ref1, ref2 = torch.chunk(h, 3, dim=0)
        fused = self.fuse_blocks[-1](art, ref1, ref2, ref1_pose, ref2_pose)
        outs.append(self.middle_block_out(fused, emb[:batch_size]))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.register_buffer("empty_clip", torch.load("./cldm/empty_clip.pt"))
        self.instantiate_lpips()

    def instantiate_lpips(self):
        self.lpips_loss = lpips.LPIPS(net="vgg")
        self.lpips_loss.requires_grad_(False)
        self.lpips_loss.eval()
        self.lpips_loss.train = disabled_train
        self.lpips_weight = 0.5
        self.rgb_recon_weight = 0.5
        self.rgb_recon_loss_type = "l1"

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        ref1_pose = batch["ref1_pose"].to(self.device)  # B, 9
        ref2_pose = batch["ref2_pose"].to(self.device)  # B, 9
        target    = batch["groundtruth"].to(self.device) # B, 512, 512, 3
        target = einops.rearrange(target, 'b h w c -> b c h w').to(self.device)   # B, 3, 512, 512
        target = target.to(memory_format=torch.contiguous_format).float()  

        if bs is not None:
            target    = target[:bs]
            ref1_pose = ref1_pose[:bs]
            ref2_pose = ref2_pose[:bs]

        if "use_cache" in batch:
            x         = batch["z_target"].to(self.device)    # B, 4, 64, 64
            z_control = batch["z_artifact"].to(self.device)  # B, 4, 64, 64
            z_ref1    = batch["z_ref1"].to(self.device)      # B, 4, 64, 64
            z_ref2    = batch["z_ref2"].to(self.device)      # B, 4, 64, 64
            B         = x.shape[0]
            c         = self.get_empty_clip(batch_size=B)

            if bs is not None:
                x         = x[:bs]
                z_control = z_control[:bs]
                z_ref1    = z_ref1[:bs]
                z_ref2    = z_ref2[:bs]
                c         = c[:bs]

        else:
            # x = super().get_input()

            x = batch[self.first_stage_key]     # B, 512, 512, 3
            control = batch[self.control_key]   # B, 512, 512, 3
            ref1    = batch["ref1"]             # B, 512, 512, 3
            ref2    = batch["ref2"]             # B, 512, 512, 3

            B         = x.shape[0]
            c         = self.get_empty_clip(batch_size=B)

            if bs is not None:
                x       = x[:bs]
                control = control[:bs]
                ref1    = ref1[:bs]
                ref2    = ref2[:bs]
            
            x = einops.rearrange(x, 'b h w c -> b c h w').to(self.device)
            x = x.to(memory_format=torch.contiguous_format).float()            

            control = einops.rearrange(control, 'b h w c -> b c h w').to(self.device)
            control = control.to(memory_format=torch.contiguous_format).float()

            ref1 = einops.rearrange(ref1, 'b h w c -> b c h w').to(self.device)
            ref1 = ref1.to(memory_format=torch.contiguous_format).float()

            ref2 = einops.rearrange(ref2, 'b h w c -> b c h w').to(self.device)
            ref2 = ref2.to(memory_format=torch.contiguous_format).float()

            # x         = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            # z_control = self.get_first_stage_encoding(self.encode_first_stage(control)).detach()
            # z_ref1    = self.get_first_stage_encoding(self.encode_first_stage(ref1)).detach()
            # z_ref2    = self.get_first_stage_encoding(self.encode_first_stage(ref2)).detach()

            # Stack thành 1 batch lớn: [4B, C, H, W]
            all_imgs = torch.cat([x, control, ref1, ref2], dim=0)
            all_latents = self.get_first_stage_encoding(self.encode_first_stage(all_imgs)).detach()
            x, z_control, z_ref1, z_ref2 = all_latents.chunk(4, dim=0)

        return x, dict(c_crossattn = [c], c_concat    = [z_control],
                    c_ref1      = [z_ref1], c_ref2      = [z_ref2],
                    c_ref1_pose = [ref1_pose], c_ref2_pose = [ref2_pose],
                    rgb_x = [target])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,
                                         ref1=cond['c_ref1'], ref2=cond['c_ref2'],
                                         ref1_pose=cond['c_ref1_pose'], ref2_pose=cond['c_ref2_pose'])
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def get_empty_clip(self, batch_size=1):
        return self.empty_clip.repeat(batch_size, 1, 1)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True, use_artifact_decode=False,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c_orig = self.get_input(batch, self.first_stage_key, bs=N)

        c_cat, c = c_orig["c_concat"][0][:N], c_orig["c_crossattn"][0][:N]  ## This c var is text emb

        c_ref1 = c_orig["c_ref1"][0][:N]
        c_ref2 = c_orig["c_ref2"][0][:N]
        c_ref1_pose = c_orig["c_ref1_pose"][0][:N]
        c_ref2_pose = c_orig["c_ref2_pose"][0][:N]

        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["ground_truth"] = self.decode_first_stage(z)
        log["control"] = self.decode_first_stage(c_cat)   # = decode(z_artifact); ref1/ref2 đã nằm trong scene_name
        log["scene_name"] = batch["scene_name"][:N]

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        # ts_list = [200, 300, 500]
        ts_list = [300]
        if sample:
            if use_artifact_decode:
                for ts in ts_list:
                    samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],
                                                        "c_ref1": [c_ref1], "c_ref2": [c_ref2],
                                                        "c_ref1_pose": [c_ref1_pose], "c_ref2_pose": [c_ref2_pose],},
                                                    batch_size=N, ddim=use_ddim, timestep=ts,
                                                    ddim_steps=ddim_steps, eta=ddim_eta)
                    x_samples = self.decode_first_stage(samples)
                    log[f"samplesT{ts}"] = x_samples
            else:
                samples, z_denoise_row = self.sample_log_full(cond={"c_concat": [c_cat], "c_crossattn": [c],
                                                            "c_ref1": [c_ref1], "c_ref2": [c_ref2],
                                                            "c_ref1_pose": [c_ref1_pose], "c_ref2_pose": [c_ref2_pose],},
                                                        batch_size=N, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples = self.decode_first_stage(samples)
                log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        # if unconditional_guidance_scale > 1.0:
        #     uc_cross = self.get_unconditional_conditioning(N)
        #     uc_cat = c_cat  # torch.zeros_like(c_cat)
        #     uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        #     samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],
        #                                             "c_ref1": [c_ref1], "c_ref2": [c_ref2],
        #                                             "c_ref1_pose": [c_ref1_pose], "c_ref2_pose": [c_ref2_pose],},
        #                                      batch_size=N, ddim=use_ddim,
        #                                      ddim_steps=ddim_steps, eta=ddim_eta,
        #                                      unconditional_guidance_scale=unconditional_guidance_scale,
        #                                      unconditional_conditioning=uc_full,
        #                                      )
        #     x_samples_cfg = self.decode_first_stage(samples_cfg)
        #     log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log_full(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)         ## nếu không đưa control vào latent: shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, timestep, **kwargs):
        artifact_latent = cond["c_concat"][0]
        ddpm_t = timestep
        ddim_sampler = DDIMSampler(self)

        ddim_sampler.make_schedule(
            ddim_num_steps=ddim_steps,
            ddim_eta=0,
            verbose=False
        )

        ddim_idx = np.argmin(
            np.abs(ddim_sampler.ddim_timesteps - ddpm_t)
        )

        t = torch.full((batch_size,), ddim_sampler.ddim_timesteps[ddim_idx], device=self.device, dtype=torch.long)

        g = torch.Generator(device=self.device)
        g.manual_seed(42)
        noise = torch.randn_like(artifact_latent, generator=g, device=self.device)
        x_t = self.q_sample(x_start=artifact_latent, t=t, noise=noise)

        samples = ddim_sampler.decode(x_t, cond, t_start=ddim_idx + 1)

        return samples, None


    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = list(self.control_model.parameters())
    #     if not self.sd_locked:
    #         params += list(self.model.diffusion_model.output_blocks.parameters())
    #         params += list(self.model.diffusion_model.out.parameters())
    #     opt = torch.optim.AdamW(params, lr=lr)
    #     return opt

    def configure_optimizers(self):
        lr_base = self.learning_rate                    # ControlNet pretrained (1e-5)
        lr_new = self.learning_rate_for_new_module      # New modules (2e-5)
        lr_unet = self.learning_rate_for_unet_out       # SD UNet output blocks

        new_module_names = {
            "input_art_ref_block",
            "fuse_blocks",
            "ref_time_emb",
        }

        base_params = []
        new_params = []
        unet_params = []

        for name, param in self.control_model.named_parameters():
            if not param.requires_grad:
                continue

            top_module = name.split(".")[0]
            if top_module in new_module_names:
                new_params.append(param)
            else:
                base_params.append(param)

        total_assigned = len(base_params) + len(new_params)
        total_trainable = sum(1 for p in self.control_model.parameters() if p.requires_grad)

        assert total_assigned == total_trainable, \
            f"Bỏ sót param: assigned={total_assigned}, trainable={total_trainable}"

        if not self.sd_locked:
            unet_params = (
                list(self.model.diffusion_model.output_blocks.parameters())
                + list(self.model.diffusion_model.out.parameters())
            )

        param_groups = [
            {"params": base_params, "lr": lr_base, "name": "controlnet"},
            {"params": new_params, "lr": lr_new, "name": "new"},
        ]

        if len(unet_params) > 0:
            param_groups.append(
                {"params": unet_params, "lr": lr_unet, "name": "unet_out"}
            )

        opt = torch.optim.AdamW(param_groups)

        warmup_steps = 1000

        def base_lr_lambda(step):
            return 1.0

        def new_lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        def unet_lr_lambda(step):
            return 1.0

        lr_lambdas = [
            base_lr_lambda,
            new_lr_lambda,
        ]

        if len(unet_params) > 0:
            lr_lambdas.append(unet_lr_lambda)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lr_lambdas
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
