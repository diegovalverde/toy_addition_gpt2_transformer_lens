import { cp, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm");
const destination = resolve(root, "public/ort/ort-wasm-simd-threaded.wasm");
await mkdir(dirname(destination), { recursive: true });
await cp(source, destination);
