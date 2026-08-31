import { ChevronsUpDown } from "lucide-react";
import {
  ELIGIBILITY_LABEL,
  ELIGIBILITY_ORDER,
  type Eligibility,
  type RunIndex,
} from "@/lib/runIndex";

/**
 * A native select, grouped by eligibility. 72 runs with optgroup headings beat a custom
 * combobox here: the grouping is the point, and keyboard and screen-reader behaviour is free.
 */
export default function RunSwitcher({
  index,
  value,
  onChange,
}: {
  index: RunIndex | null;
  value: string;
  onChange: (run: string) => void;
}) {
  if (!index) return null;

  const groups = ELIGIBILITY_ORDER.map((e) => ({
    eligibility: e as Eligibility,
    runs: index.runs.filter((r) => r.eligibility === e),
  })).filter((g) => g.runs.length > 0);

  // A run named in run.config.json that the index has not seen still has to be selectable.
  const known = index.runs.some((r) => r.run === value);

  return (
    <label className="relative block">
      <span className="sr-only">Run</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border-border bg-card hover:bg-secondary focus-visible:ring-ring w-full cursor-pointer appearance-none rounded-md border py-1.5 pr-8 pl-2.5 font-mono text-sm outline-none focus-visible:ring-[3px]"
      >
        {!known && <option value={value}>{value}</option>}
        {groups.map((g) => (
          <optgroup key={g.eligibility} label={`${ELIGIBILITY_LABEL[g.eligibility]} (${g.runs.length})`}>
            {g.runs.map((r) => (
              <option key={r.run} value={r.run}>
                {r.run}
                {r.bestValid !== null ? `  ·  ${r.bestValid.toFixed(4)}` : ""}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <ChevronsUpDown className="text-muted-foreground pointer-events-none absolute top-1/2 right-2 size-3.5 -translate-y-1/2" />
    </label>
  );
}
