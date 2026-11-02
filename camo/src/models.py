"""Load each detector behind one interface: logits = net(batch).

A batch is a float tensor of normalised images. The output is raw logits at the
model's input size, before any sigmoid or min-max stretching, because
scripts/benchmark.py scores both raw and stretched predictions.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(ROOT, "third_party")
WEIGHTS = os.path.join(ROOT, "weights")


def new_sam2unet(ckpt=None):
    """Build SAM2-UNet on the CPU with the published camouflage weights."""
    sys.path.insert(0, os.path.join(THIRD_PARTY, "SAM2-UNet"))
    from SAM2UNet import SAM2UNet
    ckpt = ckpt or os.path.join(WEIGHTS, "sam2unet_cod", "SAM2UNet-COD.pth")
    net = SAM2UNet()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return net


def load_sam2unet(device, ckpt=None):
    net = new_sam2unet(ckpt).to(device).eval()

    def forward(x):
        return net(x)[0]    # first output is the final prediction
    return forward


def load_sinet(device, ckpt=None):
    sys.path.insert(0, os.path.join(THIRD_PARTY, "SINet-V2"))
    from lib.Network_Res2Net_GRA_NCD import Network
    ckpt = ckpt or os.path.join(WEIGHTS, "sinetv2", "Net_epoch_best.pth")
    net = Network(imagenet_pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    net = net.to(device).eval()

    def forward(x):
        return net(x)[3]    # res2, the last output, as in the official MyTesting.py
    return forward


def load_arm(arm):
    """Return a loader for a trained arm, kept in weights/runs/<arm>/best.pth."""
    def load(device, ckpt=None):
        from cuegate import build_model
        ckpt = ckpt or os.path.join(WEIGHTS, "runs", arm, "best.pth")
        net = build_model(arm, device, ckpt).eval()

        def forward(x):
            return net(x)[0]
        return forward
    return load


LOADERS = {"sam2unet": load_sam2unet, "sinet": load_sinet,
           **{f"arm_{a}": load_arm(a) for a in "BCDE"}}
