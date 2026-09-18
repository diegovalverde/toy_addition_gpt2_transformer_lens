#!/usr/bin/env python3
"""Export the trained width-3 model plus its layer-0 and layer-1 probes."""
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
SECOND_DIGIT = SOURCE / "artifacts/second_digit_probes/baseline/after-first-digit_second_digit_probe_weights.pt"
SECOND_CARRY_IN = SOURCE / "artifacts/second_digit_probes/baseline/after-first-digit_carry_in_probe_weights.pt"
SECOND_CARRY_OUT = SOURCE / "artifacts/second_digit_probes/baseline/after-first-digit_carry_out_probe_weights.pt"
THIRD_DIGIT = SOURCE / "artifacts/third_digit_probes/baseline/after-second-digit_third_digit_probe_weights.pt"
THIRD_CARRY_IN = SOURCE / "artifacts/third_digit_probes/baseline/after-second-digit_carry_in_probe_weights.pt"
THIRD_CARRY_OUT = SOURCE / "artifacts/third_digit_probes/baseline/after-second-digit_carry_out_probe_weights.pt"
LAYER0_HOOK = "blocks.0.hook_resid_post"
LAYER1_HOOK = "blocks.1.hook_resid_post"

class AdditionProbeWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.model = build_model(ModelConfig(**saved["model_config"]), "cpu")
        self.model.load_state_dict(saved["model_state_dict"]); self.model.eval()
        self.first_digit = self._probe(FIRST_DIGIT, LAYER0_HOOK, 10)
        self.carry = self._probe(CARRY, LAYER0_HOOK, 2)
        self.second_digit = self._probe(SECOND_DIGIT, LAYER1_HOOK, 10)
        self.second_carry_in = self._probe(SECOND_CARRY_IN, LAYER1_HOOK, 2)
        self.second_carry_out = self._probe(SECOND_CARRY_OUT, LAYER1_HOOK, 2)
        self.third_digit = self._probe(THIRD_DIGIT, LAYER1_HOOK, 10)
        self.third_carry_in = self._probe(THIRD_CARRY_IN, LAYER1_HOOK, 2)
        self.third_carry_out = self._probe(THIRD_CARRY_OUT, LAYER1_HOOK, 2)
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = [self.model.hook_dict[name].register_forward_hook(self.capture(name)) for name in (LAYER0_HOOK, LAYER1_HOOK)]
    def _probe(self, path: Path, hook: str, outputs: int) -> nn.Linear:
        state = torch.load(path, map_location="cpu", weights_only=False)["baseline"][hook]
        probe = nn.Linear(128, outputs); probe.load_state_dict(state); probe.eval(); return probe
    def capture(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations[name] = output
        return hook
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.activations = {}
        self.model(tokens, return_type="logits")
        if LAYER0_HOOK not in self.activations or LAYER1_HOOK not in self.activations: raise RuntimeError("Probe activation missing")
        # Units and tens are teacher-forced; causal attention preserves earlier states.
        layer0_equals = self.activations[LAYER0_HOOK][:, -3, :]
        layer1_second_digit = self.activations[LAYER1_HOOK][:, -2, :]
        layer1_third_digit = self.activations[LAYER1_HOOK][:, -1, :]
        return (self.first_digit(layer0_equals), self.carry(layer0_equals), self.second_digit(layer1_second_digit), self.second_carry_in(layer1_second_digit), self.second_carry_out(layer1_second_digit), self.third_digit(layer1_third_digit), self.third_carry_in(layer1_third_digit), self.third_carry_out(layer1_third_digit))

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("public/model/addition-probes.onnx")); args = parser.parse_args()
    wrapper = AdditionProbeWrapper(); args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.tensor([[12, 7, 4, 2, 10, 6, 8, 5, 11, 3, 3]], dtype=torch.long)
    outputs = ["first_digit_logits", "carry_logits", "second_digit_logits", "second_carry_in_logits", "second_carry_out_logits", "third_digit_logits", "third_carry_in_logits", "third_carry_out_logits"]
    torch.onnx.export(wrapper, (example,), args.output, input_names=["tokens"], output_names=outputs, dynamic_axes={"tokens": {0:"batch", 1:"position"}, **{name: {0:"batch"} for name in outputs}}, opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {args.output}")
if __name__ == "__main__": main()
