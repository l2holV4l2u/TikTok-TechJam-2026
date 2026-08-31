import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  FileText,
  GitBranch,
  LayoutDashboard,
  Loader2,
  MessagesSquare,
  Monitor,
  Moon,
  ScaleIcon,
  Sun,
} from "lucide-react";
import runConfig from "../run.config.json";
import { loadRun } from "./lib/loadRun";
import { loadRunIndex, type RunIndex } from "./lib/runIndex";
import type { RunData } from "./lib/types";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Note } from "@/components/common";
import Verdict from "./components/Verdict";
import Overview from "./components/Overview";
import SearchTree from "./components/SearchTree";
import TreeGraph from "./components/TreeGraph";
import IterationDetail from "./components/IterationDetail";
import LlmCalls from "./components/LlmCalls";
import Knowledge from "./components/Knowledge";
import Files from "./components/Files";

type View = "evidence" | "overview" | "tree" | "llm" | "knowledge" | "files";

const VIEWS: { id: View; label: string; blurb: string }[] = [
  {
    id: "evidence",
    label: "Evidence",
    blurb:
      "The submitted result and what it cost. Every figure is derived from this run's records, not from the write-up.",
  },
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

const VIEW_ICONS: Record<View, typeof LayoutDashboard> = {
  evidence: ScaleIcon,
  overview: LayoutDashboard,
  tree: GitBranch,
  llm: MessagesSquare,
  knowledge: BookOpen,
  files: FileText,
};

type Theme = "system" | "light" | "dark";
const THEME_ORDER: Theme[] = ["system", "light", "dark"];
const THEME_ICONS: Record<Theme, typeof Sun> = { system: Monitor, light: Sun, dark: Moon };

/** Cycles system -> light -> dark, stamping [data-theme] on <html> so styles.css can react. */
function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("theme") as Theme | null) ?? "system";
    } catch {
      return "system";
    }
  });

  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // storage unavailable (private mode, etc.) -- theme just won't persist
    }
  }, [theme]);

  const Icon = THEME_ICONS[theme];
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={`Theme: ${theme}. Click to change.`}
      onClick={() => setTheme(THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length])}
    >
      <Icon className="size-4" />
    </Button>
  );
}

export default function App() {
  const configured = (runConfig as { run: string }).run;
  // The site shows one run: whichever one submission_best.csv came out of.
  const [index, setIndex] = useState<RunIndex | null>(null);
  const run = index?.submittedRun ?? configured;
  const [data, setData] = useState<RunData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("evidence");
  const [selected, setSelected] = useState<number | null>(null);
  const [treeMode, setTreeMode] = useState<"graph" | "list">("graph");

  useEffect(() => {
    loadRunIndex()
      .then(setIndex)
      .catch(() => setIndex(null)); // the index is an extra; the single-run views work without it
  }, []);

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
    const base = { evidence: undefined } as Record<View, number | undefined>;
    if (!data) return base;
    return {
      ...base,
      overview: undefined,
      tree: data.iterations.length,
      llm: data.llmCalls.length,
      knowledge: data.beliefs.length || undefined,
      files: undefined,
    };
  }, [data, index]);

  /** Jumping to an iteration from anywhere lands on the tree view with it selected. */
  const goToIteration = (id: number) => {
    setSelected(id);
    setView("tree");
  };

  const summary = index?.runs.find((r) => r.run === run) ?? null;
  const current = VIEWS.find((v) => v.id === view)!;
  const iteration = selected !== null ? data?.byId.get(selected) : undefined;
  // Evidence reads the cross-run index, so it stays usable while one run fails to load.
  const blocked = view !== "evidence" && (error || !data);

  return (
    <div className="min-h-screen lg:flex">
      <aside className="bg-sidebar border-b lg:sticky lg:top-0 lg:h-screen lg:w-[232px] lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0">
        <div className="flex flex-wrap items-start justify-between gap-x-8 gap-y-3 px-5 py-4 lg:flex-col lg:items-stretch lg:gap-3 lg:py-6">
          <div className="min-w-0 lg:w-full">
            <p className="text-muted-foreground text-[11px] font-semibold tracking-[0.08em] uppercase">
              TikTok TechJam 2026
            </p>
            <p className="mt-1.5 text-sm font-semibold">Final submission</p>
            <p className="text-muted-foreground mt-1 truncate text-xs">
              {[summary?.dataset ?? data?.meta?.dataset, summary?.model]
                .filter(Boolean)
                .join(" · ")}
            </p>
            <p className="text-muted-foreground/70 mt-1 truncate font-mono text-[11px]">
              runs/{run}
            </p>
          </div>

          <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto lg:mt-2 lg:flex-none lg:flex-col lg:items-stretch lg:overflow-visible">
            {VIEWS.map((v) => {
              const Icon = VIEW_ICONS[v.id];
              const active = view === v.id;
              return (
                <Button
                  key={v.id}
                  type="button"
                  variant={active ? "secondary" : "ghost"}
                  size="sm"
                  aria-current={active}
                  onClick={() => setView(v.id)}
                  className="shrink-0 justify-start gap-2 lg:w-full lg:shrink"
                >
                  <Icon className="size-4" />
                  <span className="flex-1 text-left">{v.label}</span>
                  {counts[v.id] !== undefined && (
                    <span className="text-muted-foreground font-mono text-[11px]">
                      {counts[v.id]}
                    </span>
                  )}
                </Button>
              );
            })}
          </nav>
        </div>

        <div className="px-5 pb-4 lg:pb-6">
          <Separator className="mb-3" />
          <p className="text-muted-foreground text-[11px] leading-relaxed">
            Every view reads the submitted run - the one whose predictions are
            <code> submission_best.csv</code>. It follows that file, so shipping a new
            submission moves the whole site with it.
          </p>
        </div>
      </aside>

      <main className="min-w-0 max-w-[1400px] flex-1 px-8 py-8 pb-20">
        <div className="flex flex-col gap-6">
          <header className="flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">{current.label}</h2>
              <p className="text-ink-2 mt-1 max-w-[72ch] text-sm">{current.blurb}</p>
            </div>
            <ThemeToggle />
          </header>

          {view === "evidence" &&
            (index ? (
              <Verdict index={index} onReadHypotheses={() => setView("tree")} />
            ) : (
              <Note tone="critical">
                The cross-run index could not be loaded. It is served by the dev server at{" "}
                <code>/rundata/_index.json</code>; a static export only carries it if the build
                wrote one.
              </Note>
            ))}

          {blocked && (
            <Note tone="critical">
              {error ?? `Loading runs/${run}…`}
              {error && (
                <>
                  {" "}
                  Pick a different run above, or check that <code>runs/{run}</code> exists.
                </>
              )}
            </Note>
          )}

          {!blocked && !data && view !== "evidence" && (
            <div className="text-muted-foreground flex items-center gap-2 py-12 text-sm">
              <Loader2 className="size-4 animate-spin" />
              Loading runs/{run}…
            </div>
          )}

          {data && (
            <>
              {view === "overview" && (
                <Overview data={data} summary={summary} onSelectIteration={goToIteration} />
              )}

              {view === "tree" && (
                <>
                  <ToggleGroup
                    type="single"
                    value={treeMode}
                    onValueChange={(v) => v && setTreeMode(v as "graph" | "list")}
                  >
                    <ToggleGroupItem value="graph">diagram</ToggleGroupItem>
                    <ToggleGroupItem value="list">list</ToggleGroupItem>
                  </ToggleGroup>
                  {treeMode === "graph" ? (
                    <>
                      <TreeGraph data={data} selected={selected} onSelect={setSelected} />
                      <IterationDetail data={data} iteration={iteration} />
                    </>
                  ) : (
                    <div className="grid items-start gap-5 xl:grid-cols-[minmax(300px,400px)_minmax(0,1fr)]">
                      <SearchTree data={data} selected={selected} onSelect={setSelected} />
                      <IterationDetail data={data} iteration={iteration} />
                    </div>
                  )}
                </>
              )}

              {view === "llm" && <LlmCalls data={data} onSelectIteration={goToIteration} />}
              {view === "knowledge" && <Knowledge data={data} onSelectIteration={goToIteration} />}
              {view === "files" && <Files data={data} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
