"""SVG visualizations for the multi-level single-einsum scheduler.

The memory hierarchy is an input to :func:`cb_einsum_ml`; the solver does not
choose which level holds an operand.  These plots therefore distinguish the
configured operand paths from the solved loop/load-store placement.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

from .single_op_multi_level import Problem, Schedule, Unit, evaluate_schedule

__all__ = ["plot_memory_model", "plot_schedule_solution"]


_COMPUTE = ("compute", 0)


def _operand_colors(problem: Problem) -> dict[str, tuple[float, float, float, float]]:
    palette = mpl.colormaps["tab20"]
    count = max(1, len(problem.operand_dims))
    return {
        op: palette(i / max(1, min(count, 20) - 1))
        for i, op in enumerate(problem.operand_dims)
    }


def _fmt(value: int) -> str:
    return f"{value:,}"


def _cost_text(problem: Problem, key: tuple[int, int]) -> str:
    costs = problem.mem_costs[key]
    return (
        f"read e/t={costs.read_energy}/{costs.read_time}; "
        f"write e/t={costs.write_energy}/{costs.write_time}"
    )


def _multicast_text(problem: Problem, level: int) -> str:
    fanout = " x ".join(str(size) for size in problem.fanouts[level])
    groups = []
    for array_dim, operands in enumerate(problem.mc_ops[level]):
        if operands:
            groups.append(f"axis {array_dim}: {{{', '.join(sorted(operands))}}}")
    multicast = "; ".join(groups) if groups else "none"
    return f"fanout {fanout}; multicast {multicast}"


def _problem_side_lines(problem: Problem) -> list[str]:
    lines = [
        "Configured operand paths",
        "",
        "Memory membership is fixed by capacities.",
        "Capacities and sizes are element counts.",
        "",
    ]
    all_levels = set(range(len(problem.memories)))
    for op, dims in problem.operand_dims.items():
        role = "OUTPUT" if op == problem.output else "INPUT"
        shape = " x ".join(f"{dim}={problem.dim_sizes[dim]}" for dim in dims)
        size = prod(problem.dim_sizes[dim] for dim in dims)
        inner_to_outer = problem.op_levels[op]
        memories = [
            problem.label(problem.unit_mem[Unit(op, level)])
            for level in reversed(inner_to_outer)
        ]
        if op == problem.output:
            path = " -> ".join(["compute", *reversed(memories), "DRAM"])
        else:
            path = " -> ".join(["DRAM", *memories, "compute"])
        skipped = sorted(all_levels - set(inner_to_outer))
        lines.extend(
            [
                f"{op}  [{role}]",
                f"  shape: {shape}  ({_fmt(size)} elements)",
                f"  path:  {path}",
            ]
        )
        if skipped:
            lines.append("  bypasses: " + ", ".join(f"L{level}" for level in skipped))
        lines.append("")
    lines.extend(
        [
            "Arrow direction shows data flow:",
            "inputs move toward compute; the output",
            "moves outward toward DRAM.",
        ]
    )
    return lines


def _validate_schedule(problem: Problem, schedule: Schedule | None) -> Mapping:
    if schedule is None:
        raise ValueError(
            "cannot plot a solution because no feasible schedule was found"
        )
    if Counter(schedule.order) != Counter(problem.units):
        raise ValueError("schedule order does not contain exactly the problem's units")
    if set(schedule.factors) != set(problem.dim_sizes):
        raise ValueError("schedule factors do not match the problem dimensions")
    bucket_count = len(schedule.order) + 1
    if any(len(factors) != bucket_count for factors in schedule.factors.values()):
        raise ValueError("each schedule dimension needs one factor per loop bucket")
    if any(factor < 1 for factors in schedule.factors.values() for factor in factors):
        raise ValueError("schedule factors must be positive")
    for level, rows in schedule.par_rows.items():
        if level < 0 or level >= len(problem.memories):
            raise ValueError(
                f"parallel factors reference unknown memory level L{level}"
            )
        if len(rows) != len(problem.fanouts[level]):
            raise ValueError(f"parallel rows do not match L{level}'s fanout axes")
        for row in rows:
            if not set(row).issubset(problem.dim_sizes):
                raise ValueError("parallel factors reference unknown dimensions")
            if any(factor < 1 for factor in row.values()):
                raise ValueError("parallel factors must be positive")
    try:
        analysis = evaluate_schedule(
            problem, schedule.order, schedule.factors, schedule.par_rows
        )
    except (AssertionError, KeyError, TypeError) as exc:
        raise ValueError(f"schedule is not valid for this problem: {exc}") from exc
    if (analysis["energy"], analysis["delay"]) != (
        schedule.energy,
        schedule.delay,
    ):
        raise ValueError("schedule metrics do not match evaluation for this problem")
    return analysis


def _schedule_side_lines(
    problem: Problem, schedule: Schedule, analysis: Mapping
) -> list[str]:
    lines = [
        "Solved loop/load-store placement",
        "",
        "Residency is fixed by the input Problem.",
        "Placement below means position in the loop nest.",
        "",
        "Metrics",
        f"  objective value: {_fmt(schedule.objective)}",
        f"  energy:          {_fmt(schedule.energy)}",
        f"  delay:           {_fmt(schedule.delay)}",
        f"  energy x delay:  {_fmt(schedule.energy * schedule.delay)}",
        "",
        "Buckets (outermost -> innermost)",
    ]

    parallel_bucket = {
        level: sum(1 for unit in schedule.order if unit.lvl > level)
        for level in schedule.par_rows
    }
    for bucket in range(len(schedule.order) + 1):
        factors = [
            f"{dim} x{schedule.factors[dim][bucket]}"
            for dim in problem.dim_sizes
            if schedule.factors[dim][bucket] > 1
        ]
        lines.append(f"  bucket {bucket}: " + (", ".join(factors) if factors else "1"))
        for level in sorted(schedule.par_rows, reverse=True):
            if parallel_bucket[level] != bucket:
                continue
            for array_dim, row in enumerate(schedule.par_rows[level]):
                active = [
                    f"{dim} x{factor}" for dim, factor in row.items() if factor > 1
                ]
                if active:
                    lines.append(
                        f"    L{level}.axis{array_dim} par: " + ", ".join(active)
                    )
        if bucket < len(schedule.order):
            unit = schedule.order[bucket]
            direction = "ST" if unit.op == problem.output else "LD"
            lines.append(f"    {direction} {unit.op} @ L{unit.lvl}")

    lines.extend(["", "Parallel factors"])
    active_parallel = False
    for level in sorted(schedule.parallel, reverse=True):
        factors = [
            f"{dim} x{factor}"
            for dim, factor in schedule.parallel[level].items()
            if factor > 1
        ]
        if factors:
            active_parallel = True
            lines.append(f"  L{level}: " + ", ".join(factors))
    if not active_parallel:
        lines.append("  none")

    lines.extend(["", "Memory port pressure (time / instance)"])
    keys = sorted(
        problem.mem_costs,
        key=lambda key: (key[0], key[1]),
        reverse=True,
    )
    for key in keys:
        read_time = analysis["ports"][(key, "read")]
        write_time = analysis["ports"][(key, "write")]
        lines.append(
            f"  {problem.label(key)}: read {_fmt(read_time)}, write {_fmt(write_time)}"
        )

    if problem.compute_time > 0:
        iterations = prod(problem.dim_sizes.values())
        parallelism = prod(
            factor
            for rows in schedule.par_rows.values()
            for row in rows
            for factor in row.values()
        )
        compute_time = problem.compute_time * -(-iterations // parallelism)
        lines.append(f"  compute: {_fmt(compute_time)}")
    return lines


def _make_figure(problem: Problem, side_lines: list[str], title: str) -> tuple:
    hierarchy_height = 2.2 + 1.75 * (len(problem.memories) + 2)
    text_height = 1.8 + 0.225 * len(side_lines)
    figure = Figure(
        figsize=(16, max(7.0, hierarchy_height, text_height)),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.75, 1.0))
    hierarchy_ax = figure.add_subplot(grid[0, 0])
    detail_ax = figure.add_subplot(grid[0, 1])
    figure.suptitle(title, fontsize=16, fontweight="bold")
    detail_ax.axis("off")
    detail_ax.text(
        0.0,
        1.0,
        "\n".join(side_lines),
        transform=detail_ax.transAxes,
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.5,
        linespacing=1.25,
    )
    return figure, hierarchy_ax


def _box_port(
    box: tuple[float, float, float, float], operand_index: int, operand_count: int
) -> tuple[float, float]:
    x, y, width, _ = box
    return x + width * (operand_index + 1) / (operand_count + 1), y


def _boundary_points(
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    source_x: float,
    target_x: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    _, source_y, _, source_height = source
    _, target_y, _, target_height = target
    if source_y > target_y:
        return (source_x, source_y - source_height / 2), (
            target_x,
            target_y + target_height / 2,
        )
    return (source_x, source_y + source_height / 2), (
        target_x,
        target_y - target_height / 2,
    )


def _draw_hierarchy(
    ax,
    problem: Problem,
    colors: Mapping[str, tuple],
    analysis: Mapping | None,
) -> None:
    operand_names = list(problem.operand_dims)
    operand_index = {op: index for index, op in enumerate(operand_names)}
    operand_count = len(operand_names)
    level_count = len(problem.memories)
    box_height = 1.22
    row_gap = 1.78
    compute_y = 0.85
    level_y = {level: compute_y + row_gap * (level + 1) for level in range(level_count)}
    dram_y = compute_y + row_gap * (level_count + 1)
    boxes: dict[object, tuple[float, float, float, float]] = {}

    boxes[_COMPUTE] = (0.07, compute_y, 0.86, box_height)
    dram_key = (level_count, 0)
    boxes[dram_key] = (0.07, dram_y, 0.86, box_height)
    for level, memories in enumerate(problem.memories):
        gap = 0.018
        width = (0.86 - gap * (len(memories) - 1)) / max(1, len(memories))
        for memory_index, _ in enumerate(memories):
            boxes[(level, memory_index)] = (
                0.07 + memory_index * (width + gap),
                level_y[level],
                width,
                box_height,
            )

    # Routes are drawn before the boxes so bypassing edges pass behind levels.
    for unit in problem.units:
        parent_key = problem.unit_src[unit]
        child_key = problem.unit_mem[unit]
        if unit.op == problem.output:
            source_key, target_key = child_key, parent_key
        else:
            source_key, target_key = parent_key, child_key
        source = boxes[source_key]
        target = boxes[target_key]
        source_x, _ = _box_port(source, operand_index[unit.op], operand_count)
        target_x, _ = _box_port(target, operand_index[unit.op], operand_count)
        start, end = _boundary_points(source, target, source_x, target_x)
        skipped_level = abs(parent_key[0] - child_key[0]) > 1
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.6,
            linestyle="--" if skipped_level else "-",
            color=colors[unit.op],
            alpha=0.82,
            zorder=2,
        )
        ax.add_patch(arrow)
        if analysis is not None:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{unit.op}: {_fmt(analysis['traffic'][unit])}",
                ha="center",
                va="center",
                fontsize=6.8,
                color=colors[unit.op],
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.88,
                },
                zorder=5,
            )

    for op in operand_names:
        memory_key = problem.comp_mem[op]
        if op == problem.output:
            source_key, target_key = _COMPUTE, memory_key
        else:
            source_key, target_key = memory_key, _COMPUTE
        source = boxes[source_key]
        target = boxes[target_key]
        source_x, _ = _box_port(source, operand_index[op], operand_count)
        target_x, _ = _box_port(target, operand_index[op], operand_count)
        start, end = _boundary_points(source, target, source_x, target_x)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.6,
                color=colors[op],
                alpha=0.82,
                zorder=2,
            )
        )

    # DRAM and compute endpoints.
    dram_x, dram_center, dram_width, dram_height = boxes[dram_key]
    ax.add_patch(
        Rectangle(
            (dram_x, dram_center - dram_height / 2),
            dram_width,
            dram_height,
            facecolor="#eef2f7",
            edgecolor="#34495e",
            linewidth=1.7,
            zorder=3,
        )
    )
    ax.text(
        dram_x + dram_width / 2,
        dram_center + 0.24,
        "DRAM (implicit, unbounded)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        dram_x + dram_width / 2,
        dram_center - 0.18,
        _cost_text(problem, dram_key),
        ha="center",
        va="center",
        fontsize=8,
        zorder=4,
    )

    compute_x, compute_center, compute_width, compute_height = boxes[_COMPUTE]
    ax.add_patch(
        Rectangle(
            (compute_x, compute_center - compute_height / 2),
            compute_width,
            compute_height,
            facecolor="#f4f6f6",
            edgecolor="#34495e",
            linewidth=1.7,
            zorder=3,
        )
    )
    ax.text(
        compute_x + compute_width / 2,
        compute_center + 0.22,
        "COMPUTE",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=4,
    )
    access_note = (
        "compute accesses charged at each operand's innermost memory"
        if problem.compute_accesses
        else "compute memory accesses disabled"
    )
    ax.text(
        compute_x + compute_width / 2,
        compute_center - 0.18,
        f"{access_note}; time={problem.compute_time}, energy={problem.compute_energy} / iteration",
        ha="center",
        va="center",
        fontsize=7.7,
        zorder=4,
    )

    # Finite memories and, for a solution, their per-instance allocations.
    for level in reversed(range(level_count)):
        ax.text(
            0.07,
            level_y[level] + box_height / 2 + 0.16,
            f"L{level}  ({_multicast_text(problem, level)})",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#2c3e50",
            zorder=4,
        )
        for memory_index, memory in enumerate(problem.memories[level]):
            key = (level, memory_index)
            x, center, width, height = boxes[key]
            ax.add_patch(
                Rectangle(
                    (x, center - height / 2),
                    width,
                    height,
                    facecolor="#ffffff",
                    edgecolor="#34495e",
                    linewidth=1.4,
                    zorder=3,
                )
            )
            ax.text(
                x + width / 2,
                center + 0.39,
                f"{problem.label(key)} | capacity {_fmt(memory.cap)} / instance",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                zorder=4,
            )
            ax.text(
                x + width / 2,
                center + 0.13,
                _cost_text(problem, key),
                ha="center",
                va="center",
                fontsize=6.8,
                zorder=4,
            )
            if analysis is None:
                configured = ", ".join(memory.ops) if memory.ops else "none"
                ax.text(
                    x + width / 2,
                    center - 0.25,
                    f"configured operands: {configured}",
                    ha="center",
                    va="center",
                    fontsize=7.7,
                    zorder=4,
                )
                continue

            bar_x = x + width * 0.04
            bar_y = center - 0.22
            bar_width = width * 0.92
            bar_height = 0.18
            offset = 0.0
            allocations = []
            for op in memory.ops:
                amount = analysis["spatial"][Unit(op, level)]
                allocations.append((op, amount))
                segment = bar_width * amount / memory.cap if memory.cap else 0
                ax.add_patch(
                    Rectangle(
                        (bar_x + offset, bar_y),
                        segment,
                        bar_height,
                        facecolor=colors[op],
                        edgecolor="#ffffff",
                        linewidth=0.6,
                        hatch="///" if op == problem.output else None,
                        zorder=4,
                    )
                )
                offset += segment
            ax.add_patch(
                Rectangle(
                    (bar_x, bar_y),
                    bar_width,
                    bar_height,
                    fill=False,
                    edgecolor="#34495e",
                    linewidth=0.8,
                    zorder=5,
                )
            )
            used = sum(amount for _, amount in allocations)
            allocation_text = " + ".join(
                f"{op} {_fmt(amount)}" for op, amount in allocations
            )
            ax.text(
                x + width / 2,
                center - 0.43,
                f"allocation: {allocation_text} = {_fmt(used)}; free {_fmt(memory.cap - used)}",
                ha="center",
                va="center",
                fontsize=6.7,
                zorder=4,
            )

    legend_handles = [
        Patch(
            facecolor=colors[op],
            edgecolor="#34495e",
            hatch="///" if op == problem.output else None,
            label=f"{op} ({'output' if op == problem.output else 'input'})",
        )
        for op in operand_names
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, min(4, len(legend_handles))),
        frameon=False,
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, dram_y + box_height / 2 + 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        (
            "Configured hierarchy"
            if analysis is None
            else "Configured hierarchy with solved allocation"
        ),
        fontsize=11,
        pad=10,
    )


def _save_svg(figure: Figure, output_path: str | Path | None, title: str) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    if path.suffix.lower() != ".svg":
        raise ValueError("output_path must end in .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"svg.fonttype": "none"}):
        figure.savefig(
            path,
            format="svg",
            bbox_inches="tight",
            metadata={"Title": title},
        )


def plot_memory_model(
    problem: Problem,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the configured operand paths and memory hierarchy.

    If ``output_path`` is given, its parent directories are created and the
    figure is written as searchable-text SVG.  The returned figure is not
    shown or closed, so callers may customize it further.
    """

    plot_title = title or "Single-Einsum Memory Model"
    side_lines = _problem_side_lines(problem)
    figure, hierarchy_ax = _make_figure(problem, side_lines, plot_title)
    colors = _operand_colors(problem)
    _draw_hierarchy(hierarchy_ax, problem, colors, analysis=None)
    _save_svg(figure, output_path, plot_title)
    return figure


def plot_schedule_solution(
    problem: Problem,
    schedule: Schedule | None,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a validated solution's allocations, traffic, and loop placement."""

    analysis = _validate_schedule(problem, schedule)
    assert schedule is not None  # narrowed by _validate_schedule
    plot_title = title or "Single-Einsum Solved Schedule"
    side_lines = _schedule_side_lines(problem, schedule, analysis)
    figure, hierarchy_ax = _make_figure(problem, side_lines, plot_title)
    colors = _operand_colors(problem)
    _draw_hierarchy(hierarchy_ax, problem, colors, analysis=analysis)
    _save_svg(figure, output_path, plot_title)
    return figure
