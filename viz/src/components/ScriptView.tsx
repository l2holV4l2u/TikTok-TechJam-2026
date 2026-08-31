import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { structuredPatch } from "diff";
import { FileCode, GitCompare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

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
    <pre className="bg-secondary max-h-[70vh] overflow-auto rounded-lg border font-mono text-xs leading-relaxed">
      {html === null ? (
        <code className="block p-3.5">{source}</code>
      ) : (
        <code className="block p-3.5" dangerouslySetInnerHTML={{ __html: html }} />
      )}
    </pre>
  );
}

function Diff({ from, to, fromLabel, toLabel }: { from: string; to: string; fromLabel: string; toLabel: string }) {
  const hunks = useMemo(
    () => structuredPatch(fromLabel, toLabel, from, to, "", "", { context: 4 }).hunks,
    [from, to, fromLabel, toLabel],
  );

  if (!hunks.length) return <p className="text-muted-foreground text-sm">Identical to {fromLabel}.</p>;

  return (
    <pre className="bg-secondary max-h-[70vh] overflow-auto rounded-lg border font-mono text-xs leading-relaxed">
      <code className="block py-3.5">
        {hunks.map((h, hi) => (
          <span key={hi}>
            <span className="text-muted-foreground bg-card block px-3.5 whitespace-pre">
              @@ -{h.oldStart},{h.oldLines} +{h.newStart},{h.newLines} @@
            </span>
            {h.lines.map((line, li) => (
              <span
                key={li}
                className={cn(
                  "block px-3.5 whitespace-pre",
                  line.startsWith("+") && "bg-[color-mix(in_srgb,var(--good)_15%,transparent)]",
                  line.startsWith("-") && "bg-[color-mix(in_srgb,var(--critical)_15%,transparent)]",
                )}
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

  if (!source) return <p className="text-muted-foreground text-sm">No script was recorded for this iteration.</p>;

  const canDiff = Boolean(parentSource && parentLabel);
  const lines = source.split("\n").length;

  return (
    <>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground font-mono text-xs">{path ?? "from ledger.jsonl"}</span>
        <span className="text-muted-foreground text-xs">{lines.toLocaleString()} lines</span>
        {canDiff && (
          <Button
            type="button"
            variant="outline"
            size="xs"
            className="ml-auto"
            onClick={() => setMode((m) => (m === "source" ? "diff" : "source"))}
          >
            {mode === "source" ? <GitCompare /> : <FileCode />}
            {mode === "source" ? `Diff vs ${parentLabel}` : "Full source"}
          </Button>
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
