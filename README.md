# Tensor Program DSE

Tensor-Program-DSE is a research prototype for exploring how tensor programs
should be tiled, ordered, fused, and mapped onto limited memory. It expresses
the scheduling problem as a constraint program and uses Google OR-Tools CP-SAT
to search for a low-cost schedule.

The repository has two main scheduling paths:

- [`cb_einsum_ml_cpsat97.py`](cb_einsum_ml_cpsat97.py) schedules one einsum on
  a multi-level memory hierarchy and can optimize energy, delay, or
  energy-delay product (EDP).
- [`general_dag_scheduler.py`](general_dag_scheduler.py) schedules a DAG of
  einsums, jointly choosing execution order, loop-tree structure, tiling,
  buffer lifetimes, fusion or spilling, and parallel factors.

## Environment setup

Python 3.11 is recommended for the pinned OR-Tools release. From the repository
root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pandas
```

Verify the solver version:

```bash
python -c "import ortools, pandas; print(ortools.__version__)"
```

The expected OR-Tools version is `9.7.2996`.
[`requirements.txt`](requirements.txt) pins both OR-Tools and Matplotlib, which
is used by the optional SVG schedule visualizations. The CP-SAT module in this
legacy OR-Tools release imports pandas, but the wheel does not declare it as a
dependency, so the setup installs pandas explicitly. Clang is not needed to
build or run the CP-SAT models; it is only needed for the optional generated-C
benchmarks.

```bash
export PYTHONPATH=./:$PYTHONPATH
```

## Quick start: one einsum

The following small matrix multiplication uses two shared memory levels. The
levels in `capacities` are listed from innermost to outermost, and DRAM is
implicit above the final level.

```bash
python - <<'PY'
from cb_einsum_ml_cpsat97 import cb_einsum_ml

matmul = {
    "A": ("m", "k"),
    "B": ("k", "n"),
    "C": ("m", "n"),
}

schedule = cb_einsum_ml(
    operand_dims=matmul,
    output_operand="C",
    dim_sizes={"m": 16, "k": 16, "n": 16},
    capacities=[64, 512],
    objective="energy",
    perfect_division=True,
    time_limit=30.0,
    seed=0,
    num_workers=1,
    verbose=True,
)

if schedule is None:
    raise RuntimeError("CP-SAT did not find a schedule")

print("Chosen energy:", schedule.energy)
print("Chosen delay:", schedule.delay)
PY
```

The solver prints CP-SAT statistics, Python-like loop pseudocode with directional
transfers and per-placement tile sizes, per-level occupancy and traffic, port
times, and final energy/delay metrics. Before returning, it checks the chosen
schedule with the independent plain-Python
`evaluate_schedule` implementation; a failed assertion indicates a mismatch in
the solver model rather than a harmless warning.

### `cb_einsum_ml` interface

The most important arguments are:

- `operand_dims`: tensor name to tuple of dimension names. The output tensor
  must also appear here.
- `output_operand`: the tensor produced by the einsum.
- `dim_sizes`: full problem size for every dimension.
- `capacities`: memory levels from innermost to outermost. At a level, an integer is syntax sugar for one memory of that size shared by every operand; a list of `(capacity, operands)` entries describes one or more separate memories at that level.
- `level_costs` and `dram_costs`: optional `MemCosts` values for read/write
  energy and time.
- `objective`: `"energy"`, `"delay"`, or `"edp"`.
- `fanouts`: avaliable topology, nested grids. `fanouts = [(128, 128), 4, 1]`: 1 L2, 4 L1, 128*128 L0 with compute instances. (`len(fanouts) == len(capacities)`)
- `multicast`: operands that can be broadcast (inputs) or reduced (the output) along each fanout axis. (`len(multicast) == len(capacities)`, forall i in len(multicast): len(multicast[i] == len(fanouts[i]))). `[(('I',), ('Q',)), ('I', 'WQ'), False]`: in L0, `'I'` replicate on axis0, `'Q'`(output) reduce on axis 1; in L1, `'I', 'WQ'` replicate on the only axis; in L2, no replication
- `perfect_division`: require factors to multiply exactly to each dimension;
  otherwise padded coverage is allowed.
- `time_limit`, `seed`, and `num_workers`: CP-SAT search controls.

For example, if the complete operand set is `A`, `B`, and output `C`, these are
equivalent:

```python
capacities = [64, 512]

capacities = [
    [(64, ("A", "B", "C"))],
    [(512, ("A", "B", "C"))],
]
```

The integer therefore grants one *combined* capacity to all operands; it does
not grant that capacity separately to each operand. The L0 constraint above is
`tile(A) + tile(B) + tile(C) <= 64`.

A level can instead contain separate memories:

```python
capacities = [
    [(64, ("A",)), (64, ("B",))],
    512,
]
```

This produces independent L0 constraints `tile(A) <= 64` and
`tile(B) <= 64`, whereas `[(128, ("A", "B"))]` produces the shared constraint
`tile(A) + tile(B) <= 128`. Separate memories also have separate modeled
read/write ports. An operand may belong to at most one memory within a level,
but it may appear at multiple levels; if it is absent from a level, it bypasses
that level. These operand-to-memory memberships are architecture inputs, not
placements selected by the solver.

The result is a `Schedule` containing the objective, load/store order, temporal
factors, energy, delay, aggregate parallel factors, and per-array-dimension
parallel factors. It returns `None` if the time-limited solver finds no feasible
schedule.

Running `python cb_einsum_ml_cpsat97.py` executes the larger TPU-style attention
projection example at the bottom of the file. Start with the compact example
above before using that workload.

### Visualize the memory model and solution

The small experiment in [`exp/single_einsum.py`](exp/single_einsum.py) is the
recommended entry point for the newer `src/` single-operation workflow. From
the repository root, it can be run directly after activating the environment:

```bash
conda activate allo
python3 exp/single_einsum.py
```

It writes two searchable-text SVG files:

- `exp/plots/single_einsum_memory_model.svg` shows the configured DRAM, L1,
  L0, and compute paths for every operand.
- `exp/plots/single_einsum_solution.svg` overlays the solver's per-instance
  allocations and total traffic, and shows temporal buckets, load/store order,
  parallel factors, memory-port pressure, energy, and delay.

The distinction between the two diagrams is important: `capacities` fixes
which operands reside in each memory before the solver runs. CP-SAT chooses the
tile sizes and the positions of each load/store in the loop nest; it does not
move an operand to a different memory level.

The plotting functions can also be called directly:

```python
from src.single_op_multi_level import make_problem
from src.single_op_visualization import plot_memory_model, plot_schedule_solution

problem = make_problem(
    operand_dims=matmul,
    output_operand="C",
    dim_sizes={"m": 16, "k": 16, "n": 16},
    capacities=[64, 512],
)
plot_memory_model(problem, output_path="memory_model.svg")
plot_schedule_solution(problem, schedule, output_path="solution.svg")
```

Both functions return a Matplotlib `Figure`, do not open a GUI, and create the
parent directory of an SVG output path when necessary. The `Problem` and
`Schedule` must describe the same solver invocation; the solution plot checks
that relationship with `evaluate_schedule` before drawing it.

