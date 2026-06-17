"""Compute spike-train statistics for a regular train and a Poisson train.

Run with: python examples/statistics.py

The trains are built here with the standard library so the example has no dependencies. In
practice you can pass the output of spikegen straight into these functions, since both use a
plain list of spike times.
"""

import math
import random

import spikestats as ss


def regular_train(rate: float, duration: float) -> list[float]:
    n = int(duration * rate)
    return [i / rate for i in range(n)]


def poisson_train(rate: float, duration: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    times: list[float] = []
    t = 0.0
    while True:
        t += -math.log(1.0 - rng.random()) / rate
        if t >= duration:
            break
        times.append(t)
    return times


def summarize(name: str, spikes: list[float], duration: float) -> None:
    print(f"{name}: {len(spikes)} spikes over {duration} s")
    print(f"  firing rate : {ss.firing_rate(spikes, duration=duration):.2f} Hz")
    print(f"  CV(ISI)     : {ss.cv_isi(spikes):.3f}")
    print(f"  CV2         : {ss.cv2(spikes):.3f}")
    print(f"  Lv          : {ss.lv(spikes):.3f}")
    print(f"  LvR (2 ms)  : {ss.lvr(spikes, refractory=0.002):.3f}")
    print(f"  Fano (100 ms): {ss.fano_factor(spikes, duration=duration, bin_width=0.1):.3f}")


def main() -> None:
    duration = 20.0
    summarize("regular 50 Hz", regular_train(50.0, duration), duration)
    print()
    summarize("Poisson 50 Hz", poisson_train(50.0, duration, seed=0), duration)


if __name__ == "__main__":
    main()
