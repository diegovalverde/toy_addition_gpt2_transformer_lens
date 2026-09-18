import { useEffect, useState } from "react";
import { Activity, ArrowRight, CircleCheck, CircleX, Sigma } from "lucide-react";
import { ProbeRail } from "../components/ProbeRail";
import { ScoreBars } from "../components/ScoreBars";
import { runProbes, type ProbeResult } from "../inference/browserRuntime";

const DEFAULT_LEFT = "247";
const DEFAULT_RIGHT = "586";
const digits = Array.from({ length: 10 }, (_, index) => String(index));

function winner(scores: number[]) { return scores.indexOf(Math.max(...scores)); }
function reverseTokens(value: string) { return value.padStart(3, "0").split("").reverse().join(" "); }

export function App() {
  const [left, setLeft] = useState(DEFAULT_LEFT); const [right, setRight] = useState(DEFAULT_RIGHT);
  const [result, setResult] = useState<ProbeResult | null>(null); const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false); const [active, setActive] = useState("l0post");
  useEffect(() => { const timer = window.setTimeout(async () => {
    setLoading(true); const parsedLeft = Number(left), parsedRight = Number(right);
    try { setResult(await runProbes(parsedLeft, parsedRight)); setError(null); } catch (caught) { setResult(null); setError(caught instanceof Error ? caught.message : "Could not run the probes."); }
    setLoading(false);
  }, 300); return () => window.clearTimeout(timer); }, [left, right]);

  const digitPrediction = result ? winner(result.firstDigitScores) : null;
  const carryPrediction = result ? winner(result.carryScores) : null;
  const truthDigit = result ? (result.left + result.right) % 10 : null;
  const truthCarry = result ? Number((result.left % 10) + (result.right % 10) >= 10) : null;
  const sum = result ? result.left + result.right : null;
  return <main className="app-shell">
    <header className="app-header"><div className="brand-block"><div className="brand-mark"><Sigma size={26} /></div><div><h1>Addition-GPT</h1><p>Probe Toy</p></div></div>
      <div className="addition-form"><label>Three-digit addition prompt</label><div className="number-inputs"><input aria-label="First addend" value={left} inputMode="numeric" onChange={(e) => setLeft(e.target.value.replace(/\D/g, "").slice(0, 3))} /><span>+</span><input aria-label="Second addend" value={right} inputMode="numeric" onChange={(e) => setRight(e.target.value.replace(/\D/g, "").slice(0, 3))} /><ArrowRight size={19} /><output>{sum ?? "—"}</output></div>{error && <p className="input-error">{error}</p>}</div></header>
    {loading && <p className="runtime-loading">Loading model assets and running layer-0 probes…</p>}
    {result && <><section className="status-strip"><span title="Runs fully in this browser"><Activity size={15} /> Browser ONNX runtime</span><span>prompt: <code>&lt;bos&gt; {reverseTokens(left)} + {reverseTokens(right)} =</code></span><span className="status-good">layer-0 post residual</span></section>
    <ProbeRail active={active} onSelect={setActive} />
    <section className="main-grid"><div className="arithmetic-panel"><div className="panel-heading"><span className="eyebrow">Input columns</span><h2>What reaches the <code>=</code> token</h2></div><div className="column-table"><div className="column-label">hundreds</div><div className="column-label">tens</div><div className="column-label">units</div><div className="column-label equals">=</div>
      {[result.left, result.right].map((value, row) => <div className="digit-row" key={row}>{String(value).padStart(3, "0").split("").map((digit, index) => <span key={index}>{digit}</span>)}<b>{row === 0 ? "+" : ""}</b></div>)}
      <div className="sum-row">{String(sum).padStart(4, "0").split("").map((digit, index) => <span className={index === 3 ? "units-answer" : ""} key={index}>{digit}</span>)}</div></div>
      <div className="residual-card"><span className="probe-module-label">blocks.0.hook_resid_post</span><strong>128-d residual at <code>=</code></strong><p>The probes read this state before the model produces its first answer token.</p></div></div>
      <div className="side-stack"><section className="probe-card digit-card"><div className="card-title"><div><span className="eyebrow">Next-digit probe</span><h2>First answer digit</h2></div><span className="prediction-badge">{digitPrediction}</span></div><p>Predicts the units digit: <strong>{truthDigit}</strong> is the arithmetic target.</p><ScoreBars scores={result.firstDigitScores} selected={digitPrediction!} tone="digit" labels={digits} /><div className={`verdict ${digitPrediction === truthDigit ? "correct" : "incorrect"}`}>{digitPrediction === truthDigit ? <CircleCheck size={18} /> : <CircleX size={18} />} Probe predicts {digitPrediction} {digitPrediction === truthDigit ? "correctly" : `; target is ${truthDigit}`}</div></section>
      <section className="probe-card carry-card"><div className="card-title"><div><span className="eyebrow">Carry probe</span><h2>Units carry-out</h2></div><span className="prediction-badge">{carryPrediction ? "carry" : "no carry"}</span></div><p>Will the units column write a carry into tens? Arithmetic target: <strong>{truthCarry ? "carry" : "no carry"}</strong>.</p><ScoreBars scores={result.carryScores} selected={carryPrediction!} tone="carry" labels={["no carry", "carry"]} /><div className={`verdict ${carryPrediction === truthCarry ? "correct" : "incorrect"}`}>{carryPrediction === truthCarry ? <CircleCheck size={18} /> : <CircleX size={18} />} Probe predicts {carryPrediction ? "carry" : "no carry"}</div></section></div></section></>}
  </main>;
}
