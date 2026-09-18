const layers = [
  { id: "l0post", title: "Layer 0 post", detail: "units digit · carry-out", selectable: true },
  { id: "l1post", title: "Layer 1 post", detail: "tens / hundreds probes", selectable: true },
  { id: "l2post", title: "Layer 2 post", detail: "no probes", selectable: false },
  { id: "l3post", title: "Layer 3 post", detail: "no probes", selectable: false },
] as const;

export type LayerId = "l0post" | "l1post";

export function ProbeRail({ active, onSelect, prompt }: { active: LayerId; onSelect: (id: LayerId) => void; prompt: string }) {
  return <nav className="layer-selector" aria-label="Transformer layers"><div className="input-prompt">{prompt}</div><span className="path-arrow" aria-hidden="true">→</span><div className="layer-buttons">{layers.map((layer) => layer.selectable ? <button key={layer.id} type="button" onClick={() => onSelect(layer.id)} className={active === layer.id ? "is-active" : ""} aria-pressed={active === layer.id}><strong>{layer.title}</strong><small>{layer.detail}</small></button> : <div className="layer-disabled" key={layer.id} aria-label={`${layer.title}: ${layer.detail}`}><strong>{layer.title}</strong><small>{layer.detail}</small></div>)}</div></nav>;
}
