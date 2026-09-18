# Addition-GPT Probe Toy

Static browser UI for the trained width-3 addition model’s `blocks.0.hook_resid_post` linear probes:

- next output digit (the units digit)
- units-column carry-out
- tens digit, units carry-in, and tens carry-out from the teacher-forced second-step `blocks.1.hook_resid_post` state

The browser loads the checked-in `public/model/addition-layer0-probes.onnx`, which contains the transformer and the two learned linear heads. It does not use a server or calculate probe outcomes in JavaScript.

## Run

```bash
npm install
npm run dev
```

To regenerate the ONNX asset from the source experiment’s checkpoint and probe weights, run `python scripts/export_onnx_model.py` using that experiment’s Python environment.
