# Architecture segment

~95 seconds, ~235 words. Cues in **bold** point at the diagram.
The loop first, then the three things that make it ours.

---

## The loop — 20s

**[Harness → GPT 5.6 sol → code script → Score]**

> The loop itself is simple. The harness gives the model the task and the tools —
> and nothing about what works on this data. It writes a complete experiment.
> We run it, we score it. It reads the result and decides what to try next.
>
> What sits under that is where the work went. Three things.

## One — it doesn't grade itself — 20s

**[code script → Score]**

> We throw away the score it reports, and re-score from its raw predictions.
> And it can't see the test answers at all —
> not *we asked it not to*, the data physically isn't there.
>
> The agent proposes. Only the harness judges.

## Two — it can change its mind — 25s

**[DOOM'S TREE]**

> After every experiment it rewrites what it believes.
> And a belief can be marked **wrong**.
>
> Most agents just pile up notes, so one bad early conclusion follows them
> for the rest of the run. Ours can be overturned by later evidence —
> which means it stops defending a dead idea and moves on.

## Three — three lines, and losers come back — 25s

**[DOOMS → slots → DOOM'S HISTORY]**

> We run three ideas at once, under one shared clock.
> Losers don't die — they go to DOOM'S HISTORY.
> When we revive one, we take the most **different** one, not the best one.
> A second opinion that agrees with you is worth nothing.
>
> And one round is one *script*, not one model — it can compare a dozen ideas
> inside a single round. Eighty of them, in six rounds. That's why it finished so fast.

---

## Delivery

- Pause before *"Only the harness judges."* and before *"marked **wrong**."*
- **Trim to ~70s:** cut the "most agents pile up notes" comparison (−10s) and the
  eighty-ideas line (−12s). Keep all three headings — the structure is the argument.
- Don't say "belief set" or "archive" out loud. *What it believes* and *DOOM'S HISTORY*
  match the slide and need no explaining.

## Numbers used

Eighty candidates, six iterations: `runs/r79/run_meta.json`
(`candidates_evaluated: 80`, `iterations: 6`).
