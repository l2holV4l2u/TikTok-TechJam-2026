import type { RunData } from "../lib/types";
import { Card, Empty } from "./ui";

export default function Files({ data }: { data: RunData }) {
  const blocks: { title: string; body: string }[] = [];
  if (data.eda) blocks.push({ title: "eda_report.txt — the agent's own look at the data", body: data.eda });
  if (data.console) blocks.push({ title: "console.log — what the run printed", body: data.console });
  if (data.knowledgeMd) blocks.push({ title: "knowledge.md", body: data.knowledgeMd });

  return (
    <>
      {blocks.length === 0 && <Empty>This run left no EDA report or console log.</Empty>}
      {blocks.map((b) => (
        <Card title={b.title} key={b.title}>
          <pre className="text">{b.body.trimEnd()}</pre>
        </Card>
      ))}

      <Card title="Files in this run folder">
        <ul className="mono secondary" style={{ margin: 0, paddingLeft: 20, fontSize: 12.5 }}>
          {data.manifest.files.map((f) => (
            <li key={f}>{f}</li>
          ))}
          {data.manifest.scripts.length > 0 && <li>scripts/ ({data.manifest.scripts.length} files)</li>}
        </ul>
      </Card>
    </>
  );
}
