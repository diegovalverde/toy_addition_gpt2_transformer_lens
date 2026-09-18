export function ScoreBars({ scores, selected, tone, labels }: { scores: number[]; selected: number; tone: "digit" | "carry" | "tens"; labels: string[] }) {
  return <div className="score-bars">{scores.map((score, index) => <div className={`score-row ${index === selected ? "is-selected" : ""}`} key={labels[index]}>
    <span className="score-label">{labels[index]}</span><div className="score-track"><span className={`score-fill ${tone}`} style={{ width: `${score * 100}%` }} /></div><strong>{(score * 100).toFixed(1)}%</strong>
  </div>)}</div>;
}
