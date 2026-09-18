import * as ort from "onnxruntime-web/wasm";

const BASE = import.meta.env.BASE_URL;
let sessionPromise: Promise<ort.InferenceSession> | null = null;
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = { wasm: `${BASE}ort/ort-wasm-simd-threaded.wasm` };

export type ProbeResult = {
  left: number;
  right: number;
  firstDigitScores: number[];
  carryScores: number[];
};

function softmax(scores: number[]) {
  const max = Math.max(...scores);
  const values = scores.map((score) => Math.exp(score - max));
  const total = values.reduce((sum, value) => sum + value, 0);
  return values.map((value) => value / total);
}

function toReversedDigits(value: number) {
  return value.toString().padStart(3, "0").split("").reverse().map(Number);
}

function modelTokens(left: number, right: number) {
  // <bos>, three least-significant-first digits, +, three digits, =
  return [12, ...toReversedDigits(left), 10, ...toReversedDigits(right), 11];
}

async function session() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(`${BASE}model/addition-layer0-probes.onnx`, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

function output(outputs: ort.InferenceSession.OnnxValueMapType, name: string) {
  const value = outputs[name];
  if (!(value instanceof ort.Tensor) || !(value.data instanceof Float32Array)) {
    throw new Error(`Model output ${name} was unavailable.`);
  }
  return Array.from(value.data);
}

export async function runProbes(left: number, right: number): Promise<ProbeResult> {
  if (!Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < 0 || left > 999 || right > 999) {
    throw new Error("Enter whole numbers from 0 through 999.");
  }
  const runtime = await session();
  const tokens = modelTokens(left, right);
  const input = new ort.Tensor("int64", BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
  const outputs = await runtime.run({ tokens: input });
  return { left, right, firstDigitScores: softmax(output(outputs, "first_digit_logits")), carryScores: softmax(output(outputs, "carry_logits")) };
}
