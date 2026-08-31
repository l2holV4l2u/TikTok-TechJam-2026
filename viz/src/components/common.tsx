import type { ReactNode } from "react";
import { Circle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const STATUS_VARIANT = {
  kept: "good",
  active: "good",
  ok: "info",
  reverted: "serious",
  qualified: "serious",
  failed: "critical",
  retired: "muted",
} as const;

/** One iteration or belief status, coloured by what it means for the run. */
export function StatusBadge({ status, infra }: { status?: string; infra?: boolean }) {
  if (infra) {
    return (
      <Badge variant="warning" title="the LLM call failed; the agent's code never ran">
        <Dot />
        api outage
      </Badge>
    );
  }
  const s = (status ?? "unknown").toLowerCase();
  const variant = STATUS_VARIANT[s as keyof typeof STATUS_VARIANT] ?? "muted";
  return (
    <Badge variant={variant}>
      <Dot />
      {s}
    </Badge>
  );
}

/** lucide's Circle, filled, so the status dot is a real icon rather than a styled span. */
export function Dot({ className }: { className?: string }) {
  return <Circle className={cn("size-2 shrink-0 fill-current stroke-none", className)} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="text-muted-foreground bg-card rounded-xl border border-dashed px-6 py-8 text-center text-sm">
      {children}
    </div>
  );
}

/** A card with the small uppercase eyebrow the viewer uses for section titles. */
export function Panel({
  title,
  action,
  className,
  contentClassName,
  children,
}: {
  title?: ReactNode;
  action?: ReactNode;
  className?: string;
  contentClassName?: string;
  children: ReactNode;
}) {
  return (
    <Card className={cn("gap-5 py-6", className)}>
      {(title || action) && (
        <CardHeader className="flex flex-row items-center justify-between gap-3 px-6">
          {title && (
            <CardTitle className="text-muted-foreground text-[11px] font-semibold tracking-[0.06em] uppercase">
              {title}
            </CardTitle>
          )}
          {action}
        </CardHeader>
      )}
      {/* A column gap by default: panels stack several blocks, and without it they touch. */}
      <CardContent className={cn("flex flex-col gap-4 px-6", contentClassName)}>
        {children}
      </CardContent>
    </Card>
  );
}

/** A strip of figures. Borderless on purpose: it lives inside one Panel, so boxing each
 *  number again would stack a card inside a card. */
export function StatGrid({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-x-6 gap-y-5",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: ReactNode;
  note?: ReactNode;
  tone?: "good" | "bad";
}) {
  return (
    <div className="min-w-0">
      <div className="text-muted-foreground text-[11px] font-medium tracking-[0.04em] uppercase">
        {label}
      </div>
      <div
        className={cn(
          "tnum mt-1 font-mono text-[22px] leading-tight font-semibold",
          tone === "good" && "text-good-ink",
          tone === "bad" && "text-critical",
        )}
      >
        {value}
      </div>
      {note !== undefined && (
        <div className="text-muted-foreground mt-1 text-xs leading-snug">{note}</div>
      )}
    </div>
  );
}

/** Label/value rows. Keys are muted and shrink-to-fit; values take the rest. */
export function KeyValue({ children }: { children: ReactNode }) {
  return <dl className="tnum grid grid-cols-[max-content_minmax(0,1fr)] gap-x-6">{children}</dl>;
}

export function Row({ k, children }: { k: ReactNode; children: ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground border-line-strong border-b py-1.5 text-sm whitespace-nowrap last:border-0">
        {k}
      </dt>
      <dd className="border-line-strong m-0 border-b py-1.5 text-sm last:border-0">{children}</dd>
    </>
  );
}

/** Preformatted run output: logs, diagnostics, prompts. */
export function Pre({
  children,
  className,
  wrap,
}: {
  children: ReactNode;
  className?: string;
  wrap?: boolean;
}) {
  return (
    <pre
      className={cn(
        "bg-secondary text-foreground overflow-x-auto rounded-lg border p-3.5 font-mono text-xs leading-relaxed",
        wrap ? "break-words whitespace-pre-wrap" : "whitespace-pre",
        className,
      )}
    >
      {children}
    </pre>
  );
}

export function Note({ children, tone }: { children: ReactNode; tone?: "warning" | "critical" }) {
  return (
    <div
      className={cn(
        "bg-secondary text-ink-2 rounded-r-lg border-l-[3px] px-3.5 py-2.5 text-xs leading-relaxed",
        tone === "critical" ? "border-critical" : "border-warning",
        "[&_code]:bg-card [&_code]:rounded [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono",
      )}
    >
      {children}
    </div>
  );
}

/** A small pill button that jumps to an iteration. */
export function IterLink({ id, onClick }: { id: number; onClick: (id: number) => void }) {
  return (
    <button
      type="button"
      onClick={() => onClick(id)}
      className="bg-secondary text-muted-foreground hover:text-foreground hover:border-axis cursor-pointer rounded-md border px-1.5 py-0.5 font-mono text-[11px] transition-colors"
    >
      #{id}
    </button>
  );
}
