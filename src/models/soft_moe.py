"""Soft mixture-of-experts restoration (Task 3).

    g = softmax(gate(x) / T)            weights for [identity, salt, blur, occlusion]
    y = g0*x + g1*salt(x) + g2*blur(x) + g3*occ(x)

Fully differentiable, so the gate and the experts are trained jointly through the final
reconstruction error. The gate starts from the Task 2 classifier, the experts from the three
Task 2 specialists.
"""
import torch
import torch.nn as nn

BRANCHES = ["identity", "salt", "blur", "occlusion"]


class SoftMoE(nn.Module):
    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.temperature = float(temperature)

    def weights(self, logits):
        return torch.softmax(logits / self.temperature, dim=1)

    def forward(self, x):
        """Returns (restored, routing weights (B,4), gate logits (B,4))."""
        logits = self.gate(x)
        w = self.weights(logits)
        outs = [x] + [e(x) for e in self.experts]
        y = sum(w[:, i].view(-1, 1, 1, 1).to(o.dtype) * o for i, o in enumerate(outs))
        return y, w, logits

    def set_experts_trainable(self, flag):
        for p in self.experts.parameters():
            p.requires_grad_(flag)


class SoftMoEExport(nn.Module):
    """ONNX-friendly wrapper: image in -> (restored image, routing weights) out."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        y, w, _ = self.moe(x)
        return y, w
