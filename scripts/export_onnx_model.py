#!/usr/bin/env python3
"""Export the trained width-3 model plus its layer-0-post probes for the browser."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch
from torch import nn

SOURCE = Path("/Users/diegovalverdegarro/workspace/projects/addition_gpt2_transformer_lens")
sys.path.insert(0, str(SOURCE))
from addition_gpt.model import ModelConfig, build_model

CHECKPOINT = SOURCE / "checkpoints/widths-3-seed-1-step-1000.pt"
FIRST_DIGIT = SOURCE / "artifacts/first_digit_probes/baseline/first_digit_probe_weights.pt"
CARRY = SOURCE / "artifacts/units_carry_out_probes/baseline/first_digit_probe_weights.pt"
HOOK = "blocks.0.hook_resid_post"

class AdditionProbeWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.model = build_model(ModelConfig(**saved["model_config"]), "cpu")
        self.model.load_state_dict(saved["model_state_dict"]); self.model.eval()
        self.first_digit = self._probe(FIRST_DIGIT, 10)
        self.carry = self._probe(CARRY, 2)
        self.activation: torch.Tensor | None = None
        self.handle = self.model.hook_dict[HOOK].register_forward_hook(self.capture)
    def _probe(self, path: Path, outputs: int) -> nn.Linear:
        state = torch.load(path, map_location="cpu", weights_only=False)["baseline"][HOOK]
        probe = nn.Linear(128, outputs); probe.load_state_dict(state); probe.eval(); return probe
    def capture(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activation = output
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.activation = None
        self.model(tokens, return_type="logits")
        if self.activation is None: raise RuntimeError("Layer-0 post activation missing")
        residual = self.activation[:, -1, :]
        return self.first_digit(residual), self.carry(residual)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("public/model/addition-layer0-probes.onnx")); args = parser.parse_args()
    wrapper = AdditionProbeWrapper(); args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.tensor([[12, 7, 4, 2, 10, 6, 8, 5, 11]], dtype=torch.long)
    torch.onnx.export(wrapper, (example,), args.output, input_names=["tokens"], output_names=["first_digit_logits", "carry_logits"], dynamic_axes={"tokens": {0:"batch", 1:"position"}, "first_digit_logits": {0:"batch"}, "carry_logits": {0:"batch"}}, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {args.output}")
if __name__ == "__main__": main()
