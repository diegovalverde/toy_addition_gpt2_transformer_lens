import { useState } from "react";

const layers = [
  { id: "l0post", title: "Layer 0 post", detail: "units digit · carry-out", selectable: true },
  { id: "l1post", title: "Layer 1 post", detail: "tens / hundreds probes", selectable: true },
  { id: "l2post", title: "Layer 2 post", detail: "no probes", selectable: false },
  { id: "l3post", title: "Layer 3 post", detail: "no probes", selectable: false },
] as const;

export type LayerId = "l0post" | "l1post";

export function ProbeRail({ active, onSelect, prompt }: { active: LayerId; onSelect: (id: LayerId) => void; prompt: string }) {
  const [showInfo, setShowInfo] = useState(false);
  return <nav className="layer-selector" aria-label="Transformer layers"><div className="input-prompt">{prompt}</div><span className="path-arrow" aria-hidden="true">→</span><div className="layer-buttons">{layers.map((layer) => layer.selectable ? <button key={layer.id} type="button" onClick={() => onSelect(layer.id)} className={active === layer.id ? "is-active" : ""} aria-pressed={active === layer.id}><strong>{layer.title}</strong><small>{layer.detail}</small></button> : <div className="layer-disabled" key={layer.id} aria-label={`${layer.title}: ${layer.detail}`}><strong>{layer.title}</strong><small>{layer.detail}</small></div>)}</div><div className="network-help"><button type="button" className="help-button" aria-label="About the network" aria-expanded={showInfo} onClick={() => setShowInfo(!showInfo)}>?</button>{showInfo && <article className="help-popover network-popover"><strong>About this network</strong><p>This is a small GPT-2-style causal transformer: 4 layers, 4 attention heads, 128 residual dimensions, 512 MLP dimensions, and a 15-token vocabulary.</p><p>Its input is a token string, not numeric scalars: <code>&lt;bos&gt; 7 4 2 + 6 8 5 =</code> represents 247 + 586 with digits least-significant first.</p><p>We already have held-out structured tests and matched residual/component patches that change answers to the predicted carry counterfactuals. That is causal evidence for a carry-dependent circuit. The linear heads shown here remain readouts, though: their high accuracy alone does not prove that the particular probe direction is causally used.</p><p>The model still generalizes poorly to unseen four-digit width, so the evidence supports a learned local addition mechanism, not a fully length-general decimal algorithm.</p></article>}</div></nav>;
}
