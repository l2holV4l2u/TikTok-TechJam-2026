import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const HERE = path.dirname(fileURLToPath(import.meta.url));
/** Repo root: viz/ sits directly inside it, and runs/ is its sibling. */
const REPO_ROOT = path.resolve(HERE, "..");
const RUNS_DIR = path.join(REPO_ROOT, "runs");
const CONFIG_FILE = path.join(HERE, "run.config.json");

/** Everything a run folder may contain. Only ledger.jsonl and scripts/ are present in every
 * run -- the rest come and go with the harness version, so the manifest reports what exists
 * and the UI degrades to an empty state for the rest. */
export const RUN_FILES = [
  "run_meta.json",
  "ledger.jsonl",
  "llm_calls.jsonl",
  "candidates.jsonl",
  "diagnostics.jsonl",
  "harness_ensembles.jsonl",
  "knowledge.json",
  "knowledge.md",
  "reflections.md",
  "eda_report.txt",
  "search_tree.txt",
  "console.log",
];

const MIME: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".jsonl": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".log": "text/plain; charset=utf-8",
  ".md": "text/plain; charset=utf-8",
  ".py": "text/plain; charset=utf-8",
};

export function activeRun(): string {
  const raw = JSON.parse(fs.readFileSync(CONFIG_FILE, "utf-8"));
  const run = String(raw.run || "").trim();
  if (!run) throw new Error("viz/run.config.json: \"run\" is empty");
  return run;
}

export function buildManifest(run: string) {
  const dir = path.join(RUNS_DIR, run);
  if (!fs.existsSync(dir)) {
    return { run, exists: false, files: [] as string[], scripts: [] as string[] };
  }
  const files = RUN_FILES.filter((f) => fs.existsSync(path.join(dir, f)));
  const scriptDir = path.join(dir, "scripts");
  const scripts = fs.existsSync(scriptDir)
    ? fs.readdirSync(scriptDir).filter((f) => f.endsWith(".py")).sort()
    : [];
  return { run, exists: true, files, scripts };
}

/** Resolve a request path inside runs/, refusing anything that escapes it. */
function safeJoin(rel: string): string | null {
  const target = path.resolve(RUNS_DIR, rel);
  const prefix = RUNS_DIR + path.sep;
  return target === RUNS_DIR || target.startsWith(prefix) ? target : null;
}

function copyDir(src: string, dest: string) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) copyDir(from, to);
    else if (entry.isFile()) fs.copyFileSync(from, to);
  }
}

/**
 * Serves the repo's runs/ folder to the app.
 *
 * dev   -- straight off disk, so a finished run shows up on refresh with no rebuild.
 * build -- copies only the active run into dist/rundata/, giving a self-contained
 *          static export of that one run.
 */
export function runsPlugin(): Plugin {
  return {
    name: "run-viewer-runs",

    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url || "").split("?")[0];
        if (!url.startsWith("/rundata/")) return next();

        const rel = decodeURIComponent(url.slice("/rundata/".length));

        // /rundata/<run>/_manifest.json -- what this run folder actually contains.
        if (rel.endsWith("/_manifest.json")) {
          const run = rel.slice(0, -"/_manifest.json".length);
          res.setHeader("Content-Type", MIME[".json"]);
          res.setHeader("Cache-Control", "no-store");
          res.end(JSON.stringify(buildManifest(run)));
          return;
        }

        const file = safeJoin(rel);
        if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          res.statusCode = 404;
          res.end("not found");
          return;
        }
        res.setHeader("Content-Type", MIME[path.extname(file)] ?? "application/octet-stream");
        res.setHeader("Cache-Control", "no-store");
        fs.createReadStream(file).pipe(res);
      });

      // Editing run.config.json is the way to switch runs, so make it reload the page.
      server.watcher.add(CONFIG_FILE);
      server.watcher.on("change", (changed) => {
        if (path.resolve(changed) === CONFIG_FILE) {
          server.ws.send({ type: "full-reload", path: "*" });
        }
      });
    },

    writeBundle(options) {
      const run = activeRun();
      const src = path.join(RUNS_DIR, run);
      if (!fs.existsSync(src)) {
        this.warn(`runs/${run} does not exist -- static export will have no data`);
        return;
      }
      const outDir = options.dir ?? path.join(HERE, "dist");
      const dest = path.join(outDir, "rundata", run);
      copyDir(src, dest);
      fs.writeFileSync(
        path.join(outDir, "rundata", run, "_manifest.json"),
        JSON.stringify(buildManifest(run)),
      );
    },
  };
}
