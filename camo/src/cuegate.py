"""CueGate-SAM: classical cues injected into SAM2-UNet's adapters.

The arms differ only in which cues are used and how they are weighted:

  B : no cues, just SAM2-UNet trained with corruption augmentation
  C : detail cue only
  D : all four cues with equal weights
  E : all four cues, weighted per image by the quality gate (our model)

third_party/ is not edited. The blocks are wrapped after the checkpoint loads.
Every projection starts at zero, so a new model equals SAM2-UNet exactly.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from cues import CHANNELS, compute_cues
from quality import quality_vector

CUE_NAMES = list(CHANNELS)
ARM_CUES = {"B": [], "C": ["detail"], "D": CUE_NAMES, "E": CUE_NAMES}
FEATURE_DIM = 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class QualityGate(nn.Module):
    """Map the z-scored quality vector q to one weight per cue."""

    def __init__(self, q_mean, q_std):
        super().__init__()
        self.register_buffer("q_mean", torch.tensor(q_mean, dtype=torch.float32))
        self.register_buffer("q_std", torch.tensor(q_std, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(len(q_mean), 32), nn.GELU(), nn.Linear(32, len(CUE_NAMES)))

    def forward(self, q):
        z = ((q - self.q_mean) / self.q_std).clamp(-4, 4)
        return self.mlp(z).softmax(dim=-1)


def cue_encoder(channels):
    """Small conv encoder: cue map to a feature map at 1/4 of the input size."""
    return nn.Sequential(
        nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(16, FEATURE_DIM, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 3, padding=1),
    )


class CueAdapter(nn.Module):
    """Wrap one SAM2-UNet Adapter and add the cue feature to its prompt."""

    def __init__(self, adapter, proj, context):
        super().__init__()
        self.adapter = adapter
        self.proj = proj
        self.context = [context]    # a list, so it isn't registered as a submodule

    def forward(self, x):           # Hiera blocks use x = (B, H, W, C)
        feature = F.interpolate(self.context[0].feature, size=x.shape[1:3], mode="bilinear",
                                align_corners=False)
        cue = self.proj(feature.permute(0, 2, 3, 1))
        return self.adapter.block(x + self.adapter.prompt_learn(x) + cue)


class CueGateSAM(nn.Module):
    """SAM2-UNet plus the parts each arm adds."""

    def __init__(self, net, arm, q_mean=None, q_std=None):
        super().__init__()
        self.net, self.arm, self.cue_names = net, arm, ARM_CUES[arm]
        self.feature, self.weights, self.quality = None, None, None
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.gate = QualityGate(q_mean, q_std) if arm == "E" else None
        if not self.cue_names:
            return
        self.encoders = nn.ModuleDict({c: cue_encoder(CHANNELS[c]) for c in self.cue_names})
        # Hiera-L has 48 blocks but only four feature sizes, so blocks share a projection.
        dims = sorted({blk.prompt_learn[0].in_features for blk in net.encoder.blocks})
        self.projs = nn.ModuleDict({str(d): nn.Linear(FEATURE_DIM, d) for d in dims})
        for proj in self.projs.values():
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        net.encoder.blocks = nn.Sequential(*[
            CueAdapter(blk, self.projs[str(blk.prompt_learn[0].in_features)], self)
            for blk in net.encoder.blocks])

    def cue_weights(self, rgb):
        """One weight per cue, per image."""
        if self.gate is not None:
            self.quality = quality_vector(rgb)
            return self.gate(self.quality)
        return torch.full((rgb.shape[0], len(self.cue_names)), 1 / len(self.cue_names), device=rgb.device)

    def forward(self, x, drop_cue=None):
        """Return the three SAM2-UNet outputs. drop_cue zeroes one cue for analysis."""
        if self.cue_names:
            with torch.autocast("cuda", enabled=False):
                rgb = (x.float() * self.std + self.mean).clamp(0, 1)
                cues = compute_cues(rgb)
            self.weights = self.cue_weights(rgb)
            parts = []
            for i, name in enumerate(self.cue_names):
                weight = self.weights[:, i].view(-1, 1, 1, 1)
                if name == drop_cue:
                    weight = weight * 0
                parts.append(weight * self.encoders[name](cues[name]))
            self.feature = sum(parts)
        return self.net(x)


def trainable_parameters(model):
    """Encoder weights stay frozen. Everything else trains."""
    return [p for p in model.parameters() if p.requires_grad]


def load_q_stats(path):
    with open(path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def build_model(arm, device, ckpt=None):
    """Build an arm from the published checkpoint, then load a trained one if given."""
    from models import ROOT, new_sam2unet
    q_mean, q_std = (None, None)
    if arm == "E":
        q_mean, q_std = load_q_stats(os.path.join(ROOT, "results", "q_stats.json"))
    model = CueGateSAM(new_sam2unet(), arm, q_mean, q_std)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    return model.to(device)
