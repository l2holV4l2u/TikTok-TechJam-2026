import { File } from "lucide-react";
import type { RunData } from "../lib/types";
import { Panel, Empty, Pre } from "@/components/common";

export default function Files({ data }: { data: RunData }) {
  const blocks: { title: string; body: string }[] = [];
  if (data.eda) blocks.push({ title: "eda_report.txt - the agent's own look at the data", body: data.eda });
  if (data.console) blocks.push({ title: "console.log - what the run printed", body: data.console });
  if (data.knowledgeMd) blocks.push({ title: "knowledge.md", body: data.knowledgeMd });

  return (
    <>
      {blocks.length === 0 && <Empty>This run left no EDA report or console log.</Empty>}
      {blocks.map((b) => (
        <Panel title={b.title} key={b.title} className="mb-4">
          <Pre className="max-h-[560px] overflow-auto">{b.body.trimEnd()}</Pre>
        </Panel>
      ))}

      <Panel title="Files in this run folder">
        <ul className="text-ink-2 grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-x-4 gap-y-1 font-mono text-xs">
          {data.manifest.files.map((f) => (
            <li key={f} className="flex items-center gap-1.5 truncate" title={f}>
              <File className="size-3.5 shrink-0" />
              {f}
            </li>
          ))}
          {data.manifest.scripts.length > 0 && (
            <li className="flex items-center gap-1.5 truncate">
              <File className="size-3.5 shrink-0" />
              scripts/ ({data.manifest.scripts.length} files)
            </li>
          )}
        </ul>
      </Panel>
    </>
  );
}
