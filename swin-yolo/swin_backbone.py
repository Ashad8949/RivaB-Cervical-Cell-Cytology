"""
Swin Transformer Backbone for YOLO Detection.

Replaces YOLO's CNN backbone with a pretrained Swin Transformer (timm),
keeping the FPN/PANet neck and YOLO detection head.

Usage:
    from swin_backbone import SwinTrainer

    SwinTrainer.swin_variant = 'swin_base_patch4_window12_384_in22k'
    trainer = SwinTrainer(overrides={'model': 'yolo11x.yaml', 'data': 'data.yaml', ...})
    trainer.train()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")

from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer


class SwinDetectionModel(DetectionModel):
    """DetectionModel with Swin Transformer backbone.

    Builds the standard YOLO architecture, then:
    - Adds a pretrained Swin backbone (timm) for feature extraction
    - Adds 1x1 conv channel adapters to match YOLO neck expectations (512ch)
    - Freezes unused CNN backbone layers 0-8
    - Overrides forward pass to route Swin features through YOLO neck/head

    The YOLO head references backbone indices 4 (P3), 6 (P4), 10 (P5).
    Swin stages 1,2,3 produce features at 1/8, 1/16, 1/32 resolution.
    SPPF (layer 9) and C2PSA (layer 10) are kept and applied to P5.

    The Swin model is created with img_size matching the YOLO training
    resolution so PatchEmbed and attention masks are correct for that size.
    We avoid features_only=True to guarantee img_size propagation across
    all timm versions.
    """

    def __init__(
        self,
        cfg="yolo11x.yaml",
        ch=3,
        nc=None,
        swin_variant="swin_base_patch4_window12_384_in22k",
        imgsz=1280,
        verbose=True,
    ):
        # Build standard YOLO architecture (backbone + neck + head)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Store expected image size for runtime adaptive resizing
        self.swin_imgsz = imgsz

        # --- Determine expected channel sizes from the built YOLO model ---
        # Rather than hardcode, inspect what SPPF (layer 9) expects and
        # what backbone layers 4/6 would output. Works for any YOLO scale.
        yolo_p5_ch = self._get_input_ch(self.model[9])   # SPPF input ch
        yolo_p4_ch = self._get_output_ch(self.model[6])   # backbone P4 ch
        yolo_p3_ch = self._get_output_ch(self.model[4])   # backbone P3 ch
        if verbose:
            print(f"  YOLO expects  : P3={yolo_p3_ch}, P4={yolo_p4_ch}, P5={yolo_p5_ch}")

        # --- Swin Transformer backbone ---
        # Create WITHOUT features_only to guarantee img_size reaches the
        # constructor (some timm versions don't forward it through the
        # features_only wrapper).  We extract multi-scale features manually.
        swin_full = timm.create_model(
            swin_variant,
            pretrained=True,
            img_size=imgsz,
        )

        # Keep only the pieces we need for feature extraction
        self.swin_patch_embed = swin_full.patch_embed
        # Allow variable input sizes (validation uses letterboxing → sizes like 1312)
        if hasattr(self.swin_patch_embed, 'strict_img_size'):
            self.swin_patch_embed.strict_img_size = False
        else:
            # Older timm: remove the fixed img_size so assertion is skipped
            self.swin_patch_embed.img_size = None
        self.swin_pos_drop = swin_full.pos_drop if hasattr(swin_full, "pos_drop") else nn.Identity()
        self.swin_layers = swin_full.layers   # nn.Sequential of 4 SwinTransformerStage
        self.swin_norms = nn.ModuleList()

        # Build per-stage norms (Swin uses LayerNorm on the last dim)
        embed_dim = swin_full.embed_dim
        for i in range(1, 4):  # stages 1, 2, 3
            dim = embed_dim * (2 ** i)
            self.swin_norms.append(nn.LayerNorm(dim))

        swin_ch = [embed_dim * (2 ** i) for i in range(1, 4)]  # e.g. [256, 512, 1024]

        # Clean up the full model reference
        del swin_full

        # --- Channel adapters: Swin channels → YOLO-expected channels ---
        self.adapt_p3 = self._make_adapter(swin_ch[0], yolo_p3_ch)
        self.adapt_p4 = self._make_adapter(swin_ch[1], yolo_p4_ch)
        self.adapt_p5 = self._make_adapter(swin_ch[2], yolo_p5_ch)

        # --- ImageNet normalisation buffers (Swin expects normalised input) ---
        self.register_buffer(
            "swin_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "swin_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # --- Freeze unused CNN backbone layers 0-8 ---
        for i in range(9):
            for p in self.model[i].parameters():
                p.requires_grad = False

        if verbose:
            n_swin = sum(
                p.numel()
                for mod in [self.swin_patch_embed, self.swin_layers, self.swin_norms]
                for p in mod.parameters()
            )
            n_adapt = sum(
                p.numel()
                for m in [self.adapt_p3, self.adapt_p4, self.adapt_p5]
                for p in m.parameters()
            )
            print(f"\n  Swin backbone : {swin_variant}")
            print(f"  Swin img_size : {imgsz}")
            print(f"  Swin channels : {swin_ch}")
            print(f"  Swin params   : {n_swin:,}")
            print(f"  Adapter params: {n_adapt:,}")
            print(f"  CNN layers 0-8: FROZEN (replaced by Swin)\n")

    # ------------------------------------------------------------------
    @staticmethod
    def _get_output_ch(module) -> int:
        """Get the output channels of a YOLO backbone layer."""
        # C3k2/C2f blocks have a cv2 output conv
        if hasattr(module, "cv2"):
            conv = module.cv2
            while hasattr(conv, "conv"):
                conv = conv.conv
            if hasattr(conv, "weight"):
                return conv.weight.shape[0]
        # Conv layers
        if hasattr(module, "conv") and hasattr(module.conv, "weight"):
            return module.conv.weight.shape[0]
        # Fallback: find last Conv2d
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is not None:
            return last.weight.shape[0]
        raise RuntimeError(f"Cannot determine output channels for {type(module)}")

    @staticmethod
    def _get_input_ch(module) -> int:
        """Get the input channels of a YOLO layer (e.g. SPPF)."""
        # SPPF has cv1 as the first conv
        if hasattr(module, "cv1"):
            conv = module.cv1
            while hasattr(conv, "conv"):
                conv = conv.conv
            if hasattr(conv, "weight"):
                return conv.weight.shape[1]
        # Fallback: find first Conv2d
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m.weight.shape[1]
        raise RuntimeError(f"Cannot determine input channels for {type(module)}")

    # ------------------------------------------------------------------
    @staticmethod
    def _make_adapter(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def _swin_features(self, x):
        """Extract multi-scale features from Swin stages 1, 2, 3.

        Handles input sizes that differ from self.swin_imgsz (e.g. during
        rectangular validation batching) by resizing to the expected size
        before PatchEmbed and scaling feature maps back afterwards.
        """
        H_in, W_in = x.shape[2], x.shape[3]
        need_resize = (H_in != self.swin_imgsz or W_in != self.swin_imgsz)
        if need_resize:
            x = F.interpolate(
                x, size=(self.swin_imgsz, self.swin_imgsz),
                mode="bilinear", align_corners=False,
            )

        x = self.swin_patch_embed(x)
        x = self.swin_pos_drop(x)

        feats = []
        for i, stage in enumerate(self.swin_layers):
            x = stage(x)
            if i >= 1:  # collect stages 1, 2, 3 (skip stage 0)
                # x is (B, H, W, C) — normalize and convert to (B, C, H, W)
                out = self.swin_norms[i - 1](x)
                out = out.permute(0, 3, 1, 2).contiguous()
                if need_resize:
                    # Scale feature maps back to match the original input
                    # resolution at this stride level
                    stride = 2 ** (i + 2)  # stages 1,2,3 → strides 8,16,32
                    target_h = H_in // stride
                    target_w = W_in // stride
                    out = F.interpolate(
                        out, size=(target_h, target_w),
                        mode="bilinear", align_corners=False,
                    )
                feats.append(out)
        return feats  # [P3 (1/8), P4 (1/16), P5 (1/32)]

    # ------------------------------------------------------------------
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward pass: Swin backbone → channel adapt → SPPF/C2PSA → YOLO head."""

        # During super().__init__(), a dummy forward pass runs to compute strides
        # before Swin is attached. Fall back to standard CNN forward in that case.
        if not hasattr(self, "swin_patch_embed"):
            return super()._predict_once(x, profile, visualize, embed)

        # 1. Normalise for Swin (YOLO preprocessing already scales to [0,1])
        x_norm = (x - self.swin_mean) / self.swin_std

        # 2. Swin multi-scale features at full resolution
        feats = self._swin_features(x_norm)
        p3 = self.adapt_p3(feats[0])      # 1/8  → YOLO P3 ch
        p4 = self.adapt_p4(feats[1])      # 1/16 → YOLO P4 ch
        p5_raw = self.adapt_p5(feats[2])  # 1/32 → YOLO P5 ch

        # 3. Apply SPPF (layer 9) + C2PSA (layer 10) on P5
        p5 = self.model[9](p5_raw)
        p5 = self.model[10](p5)

        # 4. Build feature cache — head references backbone indices 4, 6, 10
        y = [None] * 11  # slots 0-10 (backbone)
        y[4] = p3
        y[6] = p4
        y[10] = p5

        # 5. Run head layers (index 11+)
        x = p5
        for m in self.model:
            if m.i <= 10:
                continue
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        return x


# ======================================================================
# Custom Trainer
# ======================================================================

class SwinTrainer(DetectionTrainer):
    """DetectionTrainer that builds a SwinDetectionModel."""

    swin_variant: str = "swin_base_patch4_window12_384_in22k"

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = SwinDetectionModel(
            cfg=cfg or "yolo11x.yaml",
            nc=self.data["nc"],
            swin_variant=self.__class__.swin_variant,
            imgsz=self.args.imgsz,
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        return model


