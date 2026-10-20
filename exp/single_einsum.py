from pathlib import Path
import sys


# Keep this file directly runnable as ``python3 exp/single_einsum.py`` from the
# repository root without requiring the project to be installed as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.single_op_multi_level import cb_einsum_ml, make_problem, MemCosts
from src.single_op_visualization import plot_memory_model, plot_schedule_solution

matmul = {
    "A": ("m", "k"),
    "B": ("k", "n"),
    "C": (
        "m",
        "n",
    ),  # 'k' do not exist in the output dimensions -> `k` is the reduction dimension
}

architecture = dict(
    operand_dims=matmul,  # name -> tuple (symbolic tensor shape)
    output_operand="C",
    dim_sizes={"m": 16, "k": 16, "n": 16},  # symbol -> real shape
    capacities=[
        [(64, ("A",)), (64, ("B",))],
        512,
    ],  # shared memory levels from innermost to outermost: [L0, L1, ...]
    dram_costs=MemCosts(670, 670, 10, 10),
)

plot_dir = Path(__file__).resolve().parent / "plots"
problem = make_problem(**architecture)
plot_memory_model(
    problem,
    output_path=plot_dir / "single_einsum_memory_model.svg",
    title="Matrix Multiplication: Configured Memory Model",
)

schedule = cb_einsum_ml(
    **architecture,
    objective="energy",  # "energy", "delay", or "edp"
    perfect_division=True,
    # --- for cp-sat ---
    time_limit=30.0,
    seed=0,
    num_workers=1,
    verbose=True,
)

if schedule is None:
    raise RuntimeError("CP-SAT did not find a schedule")

plot_schedule_solution(
    problem,
    schedule,
    output_path=plot_dir / "single_einsum_solution.svg",
    title="Matrix Multiplication: Solved Schedule",
)

print("Chosen energy:", schedule.energy)
print("Chosen delay:", schedule.delay)
print("Memory model:", plot_dir / "single_einsum_memory_model.svg")
print("Solved schedule:", plot_dir / "single_einsum_solution.svg")
