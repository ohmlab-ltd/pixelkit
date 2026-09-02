"""Simulated int8 quantisation: QAT fine-tuning and direct QDQ export.

PixelKit addition (not part of upstream YOLOX). Measured on VOC2007:
post-training quantisation of this depthwise + SiLU backbone loses
8-17 AP@0.5 points however it is calibrated, because per-tensor
activation ranges are dominated by outlier channels; the weights
themselves quantise for free. Fine-tuning with the quantisation
simulated in the graph lets the network adapt to those ranges.

ActFQ    per-tensor affine int8 on an activation. While training it
         tracks a moving-average min/max; `fixed=(lo, hi)` pins a known
         range (the [0,1] input image, sigmoid outputs).
fq_weight per-output-channel symmetric int8 on a conv weight.

Both use torch.fake_quantize_*, which trains with a straight-through
estimator and exports to ONNX as QuantizeLinear + DequantizeLinear
pairs: the QDQ pattern ST Edge AI already accepts.
"""
import torch
import torch.nn as nn

QMIN, QMAX = -128, 127


class ActFQ(nn.Module):
    def __init__(self, fixed=None, momentum=0.01):
        super().__init__()
        self.enabled = False
        # Observe ranges in eval mode too: the head's output fake quant
        # sits on the inference-only branch and would otherwise never
        # see data (range 0 -> every prediction saturates -> AP 0).
        self.calibrate = False
        self.momentum = momentum
        self.fixed = fixed
        self.register_buffer("lo", torch.tensor(0.0))
        self.register_buffer("hi", torch.tensor(0.0))
        self.register_buffer("seen", torch.tensor(False))
        self.register_buffer("n", torch.tensor(0.0))

    @torch.no_grad()
    def observe(self, x):
        mn, mx = torch.aminmax(x.detach().float())
        if not bool(self.seen):
            self.lo.copy_(mn)
            self.hi.copy_(mx)
            self.seen.fill_(True)
            self.n.fill_(1.0)
        elif self.calibrate:
            # Seeding: the MEAN of per-batch min/max over the calibration
            # images (ORT's moving-average calibrator, its best of the
            # MinMax family here). A global min/max lets one spike inflate
            # a range and quantise everything else coarsely; a single
            # batch's range clips everything outside it, and clipped
            # values get zero STE gradient so fine-tuning cannot recover.
            self.n.add_(1.0)
            self.lo.add_((mn - self.lo) / self.n)
            self.hi.add_((mx - self.hi) / self.n)
        else:
            # Training: track drift slowly.
            m = self.momentum
            self.lo.mul_(1 - m).add_(mn * m)
            self.hi.mul_(1 - m).add_(mx * m)

    def qparams(self):
        if self.fixed is not None:
            lo, hi = self.fixed
        else:
            lo, hi = float(self.lo), float(self.hi)
        lo, hi = min(lo, 0.0), max(hi, 0.0)  # affine range must contain zero
        scale = max((hi - lo) / float(QMAX - QMIN), 1e-8)
        zp = int(round(QMIN - lo / scale))
        return scale, max(QMIN, min(QMAX, zp))

    def forward(self, x):
        if not self.enabled:
            return x
        if (self.training or self.calibrate) and self.fixed is None:
            self.observe(x)
        scale, zp = self.qparams()
        return torch.fake_quantize_per_tensor_affine(x, scale, zp, QMIN, QMAX)


def _const(t):
    """While tracing for ONNX export, bake a derived tensor (a folded
    weight, a weight scale) as a graph CONSTANT. Left as ops, the
    exporter emits Abs/ReduceMax/Div chains feeding the Q/DQ scale
    inputs, and ST Edge AI, which resolves those as initializers, fails
    with "'NoneType' object has no attribute 'tolist'"."""
    if torch.jit.is_tracing():
        return torch.from_numpy(t.detach().cpu().numpy().copy()).to(t.device)
    return t


def fq_weight(w):
    """Per-output-channel symmetric int8, scale = max|w| / 127."""
    amax = w.detach().abs().amax(dim=tuple(range(1, w.dim()))).clamp(min=1e-8)
    scale = _const(amax / float(QMAX))
    zp = _const(torch.zeros_like(scale, dtype=torch.int32))
    return torch.fake_quantize_per_channel_affine(_const(w), scale, zp, 0, QMIN, QMAX)


def fold_bn(conv, bn):
    """BatchNorm folded into the conv's weight and bias with the RUNNING
    statistics, i.e. BN frozen - standard for a short QAT phase."""
    g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * g.view(-1, *([1] * (conv.weight.dim() - 1)))
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    b = (b - bn.running_mean) * g + bn.bias
    return _const(w), _const(b)


def set_qat(model, on: bool) -> int:
    """Switch simulated quantisation on/off everywhere. Returns the
    number of fake-quant points affected."""
    n = 0
    for m in model.modules():
        if isinstance(m, ActFQ):
            m.enabled = on
            n += 1
        if hasattr(m, "qat") and not isinstance(m, ActFQ):
            m.qat = on
    return n


def set_calibrate(model, on: bool) -> None:
    """Let every fake-quant point observe ranges regardless of
    train/eval mode (used for the seeding passes before fine-tuning)."""
    for m in model.modules():
        if isinstance(m, ActFQ):
            m.calibrate = on


def unseeded(model) -> list:
    """Names of learned fake-quant points that never saw data - a range
    of zero there saturates everything downstream."""
    return [name for name, m in model.named_modules()
            if isinstance(m, ActFQ) and m.enabled and m.fixed is None and not bool(m.seen)]
