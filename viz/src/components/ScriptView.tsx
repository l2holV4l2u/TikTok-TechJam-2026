import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { structuredPatch } from "diff";

hljs.registerLanguage("python", python);

/** Highlighting a very long script blocks the main thread for no real gain. */
const HIGHLIGHT_LIMIT = 400_000;

function Highlighted({ source }: { source: string }) {
  const html = useMemo(() => {
    if (source.length > HIGHLIGHT_LIMIT) return null;
    try {
      return hljs.highlight(source, { language: "python" }).value;
    } catch {
      return null;
    }
  }, [source]);

  return (
    <pre className="code">
      {html === null ? <code>{source}</code> : <code dangerouslySetInnerHTML={{ __html: html }} />}
    </pre>
  );
}

function Diff({ from, to, fromLabel, toLabel }: { from: string; to: string; fromLabel: string; toLabel: string }) {
  const hunks = useMemo(
    () => structuredPatch(fromLabel, toLabel, from, to, "", "", { context: 4 }).hunks,
    [from, to, fromLabel, toLabel],
  );

  if (!hunks.length) return <p className="muted">Identical to {fromLabel}.</p>;

  return (
    <pre className="code diff">
      <code>
        {hunks.map((h, hi) => (
          <span key={hi}>
            <span className="diff-line hunk">
              @@ -{h.oldStart},{h.oldLines} +{h.newStart},{h.newLines} @@
            </span>
            {h.lines.map((line, li) => (
              <span
                key={li}
                className={`diff-line${line.startsWith("+") ? " add" : line.startsWith("-") ? " del" : ""}`}
              >
                {line === "" ? " " : line}
              </span>
            ))}
          </span>
        ))}
      </code>
    </pre>
  );
}

export default function ScriptView({
  source,
  path,
  parentSource,
  parentLabel,
}: {
  source?: string;
  path?: string;
  parentSource?: string;
  parentLabel?: string;
}) {
  const [mode, setMode] = useState<"source" | "diff">("source");

  if (!source) return <p className="muted">No script was recorded for this iteration.</p>;

  const canDiff = Boolean(parentSource && parentLabel);
  const lines = source.split("\n").length;

  return (
    <>
      <div className="code-head">
        <span className="path">{path ?? "from ledger.jsonl"}</span>
        <span className="muted" style={{ fontSize: 12 }}>
          {lines.toLocaleString()} lines
        </span>
        {canDiff && (
          <button
            type="button"
            className="toggle-table"
            onClick={() => setMode((m) => (m === "source" ? "diff" : "source"))}
          >
            {mode === "source" ? `Diff vs ${parentLabel}` : "Full source"}
          </button>
        )}
      </div>
      {mode === "diff" && canDiff ? (
        <Diff from={parentSource!} to={source} fromLabel={parentLabel!} toLabel="this iteration" />
      ) : (
        <Highlighted source={source} />
      )}
    </>
  );
}
