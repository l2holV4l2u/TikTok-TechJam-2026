import type { ReactNode } from "react";

export function StatusPill({ status, infra }: { status?: string; infra?: boolean }) {
  if (infra) {
    return (
      <span className="pill infra" title="the LLM call failed; the agent's code never ran">
        <span className="dot" />
        api outage
      </span>
    );
  }
  const s = (status ?? "unknown").toLowerCase();
  return (
    <span className={`pill ${s}`}>
      <span className="dot" />
      {s}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <section className="card">
      {title && <h3>{title}</h3>}
      {children}
    </section>
  );
}

export function Tile({
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
    <div className={`tile${tone ? ` ${tone}` : ""}`}>
      <div className="label">{label}</div>
      <div className="value mono">{value}</div>
      {note !== undefined && <div className="note">{note}</div>}
    </div>
  );
}
