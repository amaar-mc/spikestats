# Charter

## Purpose

`spikestats` computes the standard rate and variability statistics of a spike train from a
plain list of spike times, with no runtime dependencies.

## Scope

In scope:
- Rate: firing rate, spike counts per bin.
- Intervals: inter-spike intervals.
- Variability: coefficient of variation of the ISIs, CV2, local variation Lv, and the
  refractory-corrected LvR.
- Dispersion of counts: the Fano factor.

Out of scope:
- Anything that needs array libraries, plotting, or data containers. Those are well served
  by Elephant and spiketools.
- Spike-train generation (see spikegen) and spike-train distances (see spikedist).
- Multi-unit synchrony and correlation measures, unless they can be expressed cleanly on
  plain lists without new dependencies.

## Design principles

- Zero runtime dependencies: standard library `statistics` and `math` only.
- Plain data: a spike train is a `list[float]` of times, shared with spikegen and spikedist.
- Each metric implements a cited definition and is covered by an exact-value test.
- Keyword-only parameters with no default values.
- Explicit, helpful errors for input that cannot yield a defined statistic.

## Non-goals

- Performance on very large arrays. The library favors clarity and zero dependencies. A
  NumPy fast path may be added later as an optional extra, never as a required dependency.
