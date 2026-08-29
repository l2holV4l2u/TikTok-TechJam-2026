/** Score formatting: four decimals everywhere, matching the write-ups and report_run.py. */
export const score = (v: number | undefined | null, digits = 4): string =>
  v === undefined || v === null || !Number.isFinite(v) ? "--" : v.toFixed(digits);

export const signed = (v: number | undefined, digits = 4): string =>
  v === undefined || !Number.isFinite(v) ? "--" : (v >= 0 ? "+" : "") + v.toFixed(digits);

export const int = (v: number | undefined | null): string =>
  v === undefined || v === null || !Number.isFinite(v) ? "--" : Math.round(v).toLocaleString();

export function duration(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds)) return "--";
  if (seconds < 90) return `${seconds.toFixed(0)} s`;
  const min = seconds / 60;
  if (min < 90) return `${min.toFixed(1)} min`;
  return `${(min / 60).toFixed(2)} h`;
}

export function timestamp(ts: number | undefined): string {
  if (ts === undefined || !Number.isFinite(ts)) return "--";
  return new Date(ts * 1000).toLocaleString();
}

export function truncate(text: string, n: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= n ? clean : `${clean.slice(0, n - 1)}…`;
}
