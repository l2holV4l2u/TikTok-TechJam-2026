import { useEffect, useMemo, useState } from "react";
import runConfig from "../run.config.json";
import { loadRun } from "./lib/loadRun";
import type { RunData } from "./lib/types";
import Overview from "./components/Overview";
import SearchTree from "./components/SearchTree";
import IterationDetail from "./components/IterationDetail";
import LlmCalls from "./components/LlmCalls";
import Knowledge from "./components/Knowledge";
import Files from "./components/Files";

type View = "overview" | "tree" | "llm" | "knowledge" | "files";

const VIEWS: { id: View; label: string; blurb: string }[] = [
  {
    id: "overview",
    label: "Overview",
    blurb: "What the run was, what it cost, and how validation moved across its iterations.",
  },
  {
    id: "tree",
    label: "Search tree",
    blurb:
      "Every iteration the agent ran, linked parent to child. Pick a node to read its hypothesis, metrics and the script it executed.",
  },
  {
    id: "llm",
    label: "LLM calls",
    blurb: "Every prompt the agent sent and every response it got back, in order.",
  },
  {
    id: "knowledge",
    label: "Knowledge",
    blurb: "What the agent came to believe, and the iterations it read that from.",
  },
  { id: "files", label: "EDA & logs", blurb: "The raw text the run left behind." },
];

export default function App() {
  const run = (runConfig as { run: string }).run;
  const [data, setData] = useState<RunData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    setData(null);
    setError(null);
    loadRun(run)
      .then((d) => {
        if (!live) return;
        setData(d);
        // Open on the submitted iteration when the run names one, else the last one.
        const fallback = d.iterations.length ? d.iterations[d.iterations.length - 1].iter_id : null;
        const submitted = d.meta?.submission?.iter_id;
        setSelected(submitted !== undefined && d.byId.has(submitted) ? submitted : fallback);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [run]);

  const counts = useMemo(() => {
    if (!data) return {} as Record<View, number | undefined>;
    return {
      overview: undefined,
      tree: data.iterations.length,
      llm: data.llmCalls.length,
      knowledge: data.beliefs.length || undefined,
      files: undefined,
    } as Record<View, number | undefined>;
  }, [data]);

  /** Jumping to an iteration from anywhere lands on the tree view with it selected. */
  const goToIteration = (id: number) => {
    setSelected(id);
    setView("tree");
  };

  if (error) {
    return (
      <div className="fatal">
        <p>{error}</p>
        <p className="secondary">
          Edit <code>viz/run.config.json</code> and set <code>"run"</code> to a folder inside{" "}
          <code>runs/</code>, then save.
        </p>
      </div>
    );
  }
  if (!data) return <div className="loading muted">Loading runs/{run}…</div>;

  const current = VIEWS.find((v) => v.id === view)!;
  const iteration = selected !== null ? data.byId.get(selected) : undefined;

  return (
    <div className="app">
      <aside className="rail">
        <h1>Run viewer</h1>
        <p className="run-id">{data.run}</p>
        <p className="run-sub">
          {data.meta?.dataset ?? "dataset unknown"}
          {data.meta?.model ? ` · ${data.meta.model}` : ""}
        </p>
        <nav>
          {VIEWS.map((v) => (
            <button
              key={v.id}
              type="button"
              aria-current={view === v.id}
              onClick={() => setView(v.id)}
            >
              <span>{v.label}</span>
              {counts[v.id] !== undefined && <span className="count">{counts[v.id]}</span>}
            </button>
          ))}
        </nav>
        <p className="hint">
          Showing one run at a time. To switch, edit <code>run</code> in{" "}
          <code>viz/run.config.json</code> — the page reloads on save.
        </p>
      </aside>

      <main className="main">
        <header>
          <h2>{current.label}</h2>
          <p>{current.blurb}</p>
        </header>

        {data.warnings.length > 0 && view === "overview" && (
          <div className="warnings">
            <strong>Notes on this run's files</strong>
            <ul>
              {data.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {view === "overview" && <Overview data={data} onSelectIteration={goToIteration} />}

        {view === "tree" && (
          <div className="split">
            <SearchTree data={data} selected={selected} onSelect={setSelected} />
            <IterationDetail data={data} iteration={iteration} />
          </div>
        )}

        {view === "llm" && <LlmCalls data={data} onSelectIteration={goToIteration} />}
        {view === "knowledge" && <Knowledge data={data} onSelectIteration={goToIteration} />}
        {view === "files" && <Files data={data} />}
      </main>
    </div>
  );
}
