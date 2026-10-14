# `cb_einsum_ml_cpsat97.py` — context for the next agent

A CP-SAT model that finds the tiling, loop order, parallelism and memory placement for a single einsum on a multi-level memory hierarchy. It is an extension of the single-level model in `cb_einsum_cpsat97.py`. The main pieces are:
- `make_problem`: parses the architecture arguments into a `Problem`, which holds the units, memories, parents and multicast sets. The model and the evaluator both use it.
- `cb_einsum_ml`: builds and solves the CP-SAT model and returns a `Schedule` NamedTuple. `r[0]` through `r[5]` still work for older callers.
- `evaluate_schedule(problem, order, factors, par_rows)`: an independent plain-Python check of a schedule. It checks validity and computes tiles, traffic, energy, delay and port times.

## Running it
- It needs `ortools==9.7.2996` (see `requirements.txt`). On the original machine this lived in the conda env `solvers`, run as `~/miniconda3/envs/solvers/bin/python`. The default `python` there did not have ortools.
- `python cb_einsum_ml_cpsat97.py` runs the attention-projection example in `__main__`: a 3-level hierarchy with a 128×128 L0 array, multicast and EDP. It solves to OPTIMAL in about 5 s.
- After every solve, the solver's tiles, traffic, energy and delay are asserted equal to `evaluate_schedule`'s. A failed `assert` means the model and its intended semantics disagree. Treat that as a bug, not noise.
- **Every reformulation must keep these asserts passing.** That includes the performance ideas below. Keep `evaluate_schedule` independent of the model's encoding, or it stops being a check.

## Model in brief (read the docstring for the full semantics)
- **Units and slots.**
  - A unit is an `(operand, level)` pair. Each unit has one load/store slot in a single global order.
  - Buckets of tiling factors sit between consecutive slots.
  - A unit's tile (its spatial cost) is the product of the factors inside its slot, for the dims it uses.
  - Its traffic is that tile × the product of all factors outside its slot.
- **Memories.** `capacities[l]` is either an int (one shared memory) or a list of `(cap, ops[, MemCosts])` entries.
  - An operand missing from every memory at a level bypasses that level.
  - `MemCosts` holds integer read/write energy and time. DRAM uses `dram_costs`.
- **Objective.**
  - Energy = Σ (read energy + write energy) over all transfers, plus compute accesses, plus `compute_energy × iterations`.
  - Delay = the busiest port across all memories, plus a compute "port" equal to `compute_time × ceil(iterations / parallelism)`. Each memory has separate read and write ports, and port times are per instance.
  - `'edp'` scales E and D to at most 2^30 each before multiplying, because the raw product overflows int64.
- **Parallelism.** `fanouts[l]` replicates level `l` (its memories and the compute below). A tuple means a multi-dimensional array.
  - Parallel factors sit in the bucket just outside level `l`'s first slot.
  - If a level uses any parallelism, every slot of the levels above it must come before every slot at or below it. That keeps the parallel bucket at a fixed index (`par_bucket[l]`).
  - The effective factors are the temporal factors times the parallel factors (`eff_vars`), so the existing cost formulas apply unchanged.
- **Multicast.** `multicast[l]` is a bool, or one entry per array dimension (a flat operand list is allowed only for a 1-D fanout).
  - Inputs: a parent read is shared across instances that differ only in dims the operand lacks.
  - Output: partial sums are reduced on the way to the parent.
  - Allowed parallel dims per array dimension: reduction dims if the output is multicast there; output dims if an input is multicast there or nothing is. `True` is the same as listing every operand.
- **Placement rules.** Two rules restrict where factors go: the optimal-placement rules, and "no temporal reduction outside the output's innermost slot" (no partial spilling).
- **Partial sums (`partial_sums=True`).** This drops the no-spilling rule and charges fills instead.
  - For each output unit, fills = `parent_traffic − first visits`. First visits are the padded output size times the reduction parallelism at or above the parent, since each parent copy starts from zero.
  - Each fill is one parent read and one write into the unit's memory. Compute's read-modify-write needs no change.
  - `evaluate_schedule` always charges fills. They are exactly zero for schedules that obey the rule.
  - On TCM's AV schedule, the read and write counts at every memory match AccelForge's (`../comparison/check_partial_sums.py`).
  - With it on, AV's optimum improves by 1.07% in AccelForge, matching TCM's. The other GPT-3 einsums are unchanged.
  - The search space is larger: without the fast options, V, K and Q stop FEASIBLE at 120 s, and Z takes about 100 s. With `redundant_bounds`, `sparse_domains` and `merge_dims`, all eight solve OPTIMAL in under 0.5 s.
  - A tie-break orders adjacent slots that have only empty buckets between them.
  - These rules act only on temporal factors (`factor_vars`), never on parallel ones.

## Behaviour that surprised us
- **Blocking under parallelism can make parallelism worthless.** In matmul, the best schedule interleaves L1 loads inside the L0 tile. Forcing P=4 cost about 27× more energy. The rejected alternative (allowing shared slots inside the parallel loop) would need per-instance concurrent tiles in the shared memory.
- **`multicast` semantics drive feasibility.** A level with `True` allows reduction parallelism, which changes the problem. One apparent "slowdown" turned out to be a different (worse) optimum, not slower search.
- **Run-to-run variance is large.** CP-SAT's multi-worker search is nondeterministic, so fix the seed before comparing runs.

## Regression values (2-level matmul, no fanout)
These use `mm = {'A': ('m','k'), 'B': ('k','n'), 'C': ('m','n')}` with `{'m': 4096, 'k': 65536, 'n': 4096}`, `capacities=[4096, 524288]`, `objective='energy'` and default costs:
- With `compute_accesses=False`: energy is 140,694,749,184.
- With the default `compute_accesses=True`: energy is 4,538,724,483,072. That is the same schedule plus the constant 4·2^40 − 2^24.

## Performance ideas, in suggested order
Items 1–4 are implemented as opt-in arguments, all off by default: `seed`, `num_workers`, `linearization_level`, `redundant_bounds`, `sparse_domains` and `merge_dims`. The benchmark ran the eight GPT-3 6.7B einsums with the `__main__` TPU-v4i parameters on 16 cores, taking the median of 3 seeds; see `../comparison/results/bench_cpsat.md`. Every run reached OPTIMAL. The summed CP-SAT walltime was:
- 22.7 s for the baseline;
- 25.2 s with `linearization_level=2`, so no gain;
- 19.3 s with `redundant_bounds`;
- 7.0 s with `sparse_domains`;
- 2.3 s with `merge_dims`;
- 0.52 s with `redundant_bounds`, `sparse_domains` and `merge_dims` together, about 0.06 s per einsum;
- 0.50 s with all four.

These numbers use the rescaled costs of 2026-10-09; see `../HANDOFF.md`.

No run changed the optimum. The matmul regression values below still hold with the options on, though that tiny padded case is slightly slower with them (0.28 s vs. 0.21 s).

How `merge_dims` works:
- `cb_einsum_ml` calls itself on the merged problem. Merged dims are named by joining the group with `'*'`, e.g. `b*m`.
- It splits the result back with `split_merged`, prime by prime, as item 3 describes.
- It re-checks the split schedule with `evaluate_schedule` on the original problem.
- With `verbose` on, it prints both the merged loop nest and the split one.

The compute port's bound is capped at `DELAY_CAP = 2**61`, because einsums with 2^48 or more iterations otherwise exceed CP-SAT's int64max / 2 limit. A warning fires if the cap could have excluded a better schedule.
- Energy is split into one variable per memory, each capped at `ENERGY_CAP / (number of memories)`, with `ENERGY_CAP = 2**61`. This is needed because with the rescaled costs, the bound on QK's and AV's energy expression exceeds the int64max / 2 limit CP-SAT enforces per linear constraint. A warning fires if the cap could have excluded a better schedule.

Benchmark each change on its own against `__main__`. Record time-to-OPTIMAL, or the final gap between the objective and `BestObjectiveBound()` at a fixed time limit. You can capture both by wrapping `cp_model.CpSolver.Solve` with a solution callback; that's how the original comparisons were done.

1. **Make runs reproducible.** Set `solver.parameters.random_seed` and `num_workers`. Also try `linearization_level=2`, which can help with the weak bounds that products produce.
2. **Tighten the proven bound with redundant constraints.** Observed runs found good solutions in about 1 s, then spent the rest of the time on a bound stuck around 40% of the objective. Candidates:
   - An inner unit's traffic is at least its next outer unit's traffic (this is linear and holds with multicast).
   - Each operand's inner tile is at most its outer tile.
   - Delay is at least `compute_time × ceil(iters / total_fan)`, and at least DRAM's time to move each operand once.
   - Energy is at least the cost of moving each operand into each level it lives in once.
3. **Merge interchangeable dims** (when `perfect_division=True`). *Implemented as `merge_dims`; the padded case is not handled.*
   - Two dims are interchangeable if they appear in exactly the same operand set. In `__main__` that's `b` with `m`, and `h` with `e`.
   - Costs depend only on the product of such dims' factors in each bucket. So merge them into one dim, solve, then split back.
   - To split back, go prime by prime: walk the buckets from outermost to innermost, filling one dim's exponent first.
   - This removes the symmetry entirely. Prefer it over an ordering constraint like "exhaust `m` before `b`", which needs extra product chains.
   - It isn't exact with padding (`perfect_division=False`). There, either skip merging or add the ordering constraint.
4. **Use sparse factor domains.** Under perfect division, give factors and parallel variables `Domain.FromValues(divisors(size))`.
   - **With padding:** restrict temporal factors to `{ceil(size/k) : k ≥ 1}`. That set has about 2√size values and already includes every divisor.
   - **Why it's sound:** every cost is non-decreasing in each temporal factor. Tile and traffic products contain it; energy and port times sum those products; compute terms use the unpadded iteration count. So any optimum can shrink each temporal factor f until `(f−1)·R < size ≤ f·R`, where R is the product of that dim's other factors, including parallel ones. Then `f = ceil(size/R)`.
   - **Limits:**
     - **Temporal factors only.** Lowering a parallel factor can raise the compute or port times.
     - **Recheck if costs change.** The argument must be redone if any cost becomes non-monotone in a temporal factor, e.g. if padded iterations start to count.
5. **Break array-dimension symmetry.**
   - Interchangeable array dimensions have equal size and the same multicast entry. Order their parallel-factor rows, e.g. lexicographically.
   - Have each dim fill an earlier interchangeable array dimension before a later one.
6. **Warm-start EDP.** Do a short `objective='energy'` solve, then `model.AddHint` its schedule into the EDP solve.
7. **Tighten variable bounds** in the `add_mul_chain` intermediates. Many are bounded by `max_iters` or `dim × total_fan`, where capacity or dim-size products would be tighter. Tighter bounds also shrink the EDP scale factors and their rounding.
8. **Log2 (exponent) reformulation** — the largest change, do not do this before talking to the user about it. All example sizes are powers of two.
   - Factor = 2^x, so every product chain becomes a linear sum. Lookup tables (`AddElement`) map exponents back to values for energy and delay.
   - `ft5_log2_cpsat97.py` in this repo is a precedent.
   - It makes the item 3 ordering constraint linear, so it also covers the padded case.
