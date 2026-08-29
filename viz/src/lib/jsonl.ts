/** Parse JSONL the way report_run.py does: skip a line that will not parse rather than
 * losing the whole file to one truncated write at the end of an interrupted run. */
export function parseJsonl<T = unknown>(text: string): { rows: T[]; skipped: number } {
  const rows: T[] = [];
  let skipped = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line) as T);
    } catch {
      skipped += 1;
    }
  }
  return { rows, skipped };
}

/** iter_id is a number in ledger.jsonl but a string in candidates/diagnostics/ensembles. */
export function asIterId(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

/** Those same files also store their payloads as Python-repr strings in places. */
export function asRecord(v: unknown): Record<string, number> | undefined {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    const out: Record<string, number> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
      const n = typeof val === "number" ? val : Number(val);
      if (Number.isFinite(n)) out[k] = n;
    }
    return out;
  }
  if (typeof v === "string") {
    try {
      return asRecord(JSON.parse(v.replace(/'/g, '"')));
    } catch {
      return undefined;
    }
  }
  return undefined;
}

export function asNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}
