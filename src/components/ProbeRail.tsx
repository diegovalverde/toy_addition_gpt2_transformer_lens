const stages = [
  { id: "input", label: "Input", note: "tokens", tone: "neutral" },
  { id: "l0pre", label: "L0", note: "resid pre", tone: "neutral" },
  { id: "l0post", label: "L0 post", note: "residual", tone: "active" },
  { id: "digit", label: "Digit", note: "linear probe", tone: "digit" },
  { id: "carry", label: "Carry", note: "linear probe", tone: "carry" },
];

export function ProbeRail({ active, onSelect }: { active: string; onSelect: (id: string) => void }) {
  return <section className="probe-rail" aria-label="Addition transformer probe rail">
    <div className="wire-field" aria-hidden="true"><span className="wire wire-digit" /><span className="wire wire-carry" /></div>
    <div className="rail-track">{stages.map((stage, index) => <div className="rail-stage-wrap" key={stage.id}>
      <button type="button" onClick={() => onSelect(stage.id)} className={`rail-stage tone-${stage.tone} ${active === stage.id ? "active" : ""}`}>{stage.label}</button>
      <p>{stage.note}</p>{index < stages.length - 1 && <span className="rail-connector" />}
    </div>)}</div>
  </section>;
}
