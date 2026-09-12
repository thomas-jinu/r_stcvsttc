#!/usr/bin/env python3
"""Generate and collect a causal two-time tDMRG grid for DMRG++.

The initial implementation supports the greater impurity Green function

    G^>(t,t') = -i <psi(t)|c|phi_+(t;t')>,
    |phi_+(t';t')> = c' |psi(t')>.

All time-grid decisions use integer indices.  The generated DMRG++ inputs keep
the reference and insertion branches in one restart record and evolve them
synchronously.  Raw matrix elements are retained by the collector.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import stat
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


class UserError(Exception):
    """An input or generated-data error suitable for a concise CLI message."""


@dataclass(frozen=True)
class Grid:
    initial: Decimal
    final: Decimal
    step: Decimal
    first_index: int
    last_index: int

    def time(self, index: int) -> Decimal:
        return index * self.step


def decimal_arg(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"not a decimal number: {text}") from error
    if not value.is_finite():
        raise argparse.ArgumentTypeError(f"time must be finite: {text}")
    return value


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def make_grid(initial: Decimal, final: Decimal, step: Decimal) -> Grid:
    if step <= 0:
        raise UserError("--dt must be positive")
    if initial < 0:
        raise UserError("--ti must not precede the t=0 initial state")
    if final < initial:
        raise UserError("--tf must not precede --ti")
    first_index = initial / step
    last_index = final / step
    if first_index != first_index.to_integral_value():
        raise UserError("--ti must be an integer multiple of --dt")
    if last_index != last_index.to_integral_value():
        raise UserError("--tf must be an integer multiple of --dt")
    return Grid(initial, final, step, int(first_index), int(last_index))


def require(config: dict[str, Any], key: str, expected: type) -> Any:
    if key not in config:
        raise UserError(f"configuration is missing {key!r}")
    value = config[key]
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise UserError(f"configuration field {key!r} must be {expected.__name__}")
    return value


def number_list(config: dict[str, Any], key: str, length: int) -> list[float | int]:
    values = require(config, key, list)
    invalid = any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(float(x))
        for x in values
    )
    if len(values) != length or invalid:
        raise UserError(f"configuration field {key!r} must contain {length} numbers")
    return values


def ainur_number(value: float | int) -> str:
    if isinstance(value, bool):
        raise UserError("boolean found where a model number was expected")
    return str(value)


def ainur_vector(values: Iterable[float | int]) -> str:
    return "[" + ", ".join(ainur_number(value) for value in values) + "]"


def finite_loops(count: int, kept_states: int, flag: int) -> str:
    if count < 1:
        raise UserError("each generated DMRG++ stage needs at least one finite loop")
    row = f"[@auto, {kept_states}, {flag}]"
    return "[" + ", ".join(row for _ in range(count)) + "]"


def stage_potential(
    config: dict[str, Any], stage: str, sites: int
) -> list[float | int]:
    stage_key = f"potential_v_{stage}"
    key = stage_key if stage_key in config else "potential_v"
    return number_list(config, key, 2 * sites)


def common_input(
    config: dict[str, Any],
    interaction: list[float | int],
    potential: list[float | int],
) -> str:
    sites = require(config, "sites", int)
    if isinstance(sites, bool) or sites < 2:
        raise UserError("configuration field 'sites' must be at least 2")
    connectors = number_list(config, "connectors", sites - 1)
    if len(interaction) != sites:
        raise UserError(f"interaction vectors must contain {sites} numbers")
    model = require(config, "model", str)
    geometry = require(config, "geometry", str)
    geometry_options = require(config, "geometry_options", str)
    up = require(config, "target_electrons_up", int)
    down = require(config, "target_electrons_down", int)
    return f"""TotalNumberOfSites={sites};
NumberOfTerms=1;
DegreesOfFreedom=1;
GeometryKind={geometry};
GeometryOptions={geometry_options};
hubbardU={ainur_vector(interaction)};
Model={model};
TargetElectronsUp={up};
TargetElectronsDown={down};
dir0:Connectors={ainur_vector(connectors)};
potentialV={ainur_vector(potential)};
"""


def target_input(
    config: dict[str, Any],
    *,
    version: str,
    output_file: str,
    restart_file: str,
    loop_count: int,
    expressions: list[str],
    restart_mapping: list[int] | None = None,
    source_tv_for_psi: int | None = None,
) -> str:
    sites = require(config, "sites", int)
    interaction = number_list(config, "interaction_final", sites)
    potential = stage_potential(config, "final", sites)
    kept = require(config, "kept_states", int)
    if isinstance(kept, bool) or kept < 1:
        raise UserError("configuration field 'kept_states' must be a positive integer")
    loop_flag = config.get("restart_loop_flag", 2)
    if isinstance(loop_flag, bool) or not isinstance(loop_flag, int):
        raise UserError("configuration field 'restart_loop_flag' must be an integer")
    lines = [
        "##Ainur1.0",
        "",
        common_input(config, interaction, potential).rstrip(),
        "SolverOptions=twositedmrg,geometryallinsystem,TargetingExpression,restart,usecomplex;",
        f"Version={version};",
        f"OutputFile={output_file};",
        f"InfiniteLoopKeptStates={kept};",
        f"FiniteLoops={finite_loops(loop_count, kept, loop_flag)};",
        f"RestartFilename={restart_file};",
    ]
    if source_tv_for_psi is not None:
        # This label is not in Ainur's fixed schema and therefore needs a type.
        lines.append(f"integer RestartSourceTvForPsi={source_tv_for_psi};")
    if restart_mapping is not None:
        lines.append(f"RestartMappingTvs={ainur_vector(restart_mapping)};")
    lines.append("RestartMapStages=0;")
    lines.extend(expressions)
    lines.append(f"GsWeight={config.get('gs_weight', 0.1)};")
    lines.append("")
    return "\n".join(lines)


def ground_state_input(config: dict[str, Any]) -> str:
    sites = require(config, "sites", int)
    interaction = number_list(config, "interaction_initial", sites)
    potential = stage_potential(config, "initial", sites)
    kept = require(config, "kept_states", int)
    loops = config.get("ground_state_loops", 4)
    if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
        raise UserError(
            "configuration field 'ground_state_loops' must be a positive integer"
        )
    return "\n".join(
        [
            "##Ainur1.0",
            "",
            "# Initial-state ground-state checkpoint.",
            common_input(config, interaction, potential).rstrip(),
            "SolverOptions=twositedmrg,geometryallinsystem;",
            "Version=TwoTimeGrid-ground-state;",
            "OutputFile=data;",
            f"InfiniteLoopKeptStates={kept};",
            f"FiniteLoops={finite_loops(loops, kept, 0)};",
            "",
        ]
    )


def time_evolve(config: dict[str, Any], dt: Decimal, source: str) -> str:
    time_options = config.get("time_evolution", {})
    if not isinstance(time_options, dict):
        raise UserError("configuration field 'time_evolution' must be an object")
    steps = time_options.get("steps", 5)
    advance_each = time_options.get("advance_each", 4)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise UserError("time_evolution.steps must be a positive integer")
    if (
        isinstance(advance_each, bool)
        or not isinstance(advance_each, int)
        or advance_each < 1
    ):
        raise UserError("time_evolution.advance_each must be a positive integer")
    return (
        f'"TimeEvolve{{tau={decimal_text(dt)},steps={steps},'
        f'advanceEach={advance_each}}}*|{source}>"'
    )


def parse_indices(text: str | None, minimum: int, maximum: int) -> list[int]:
    if text is None:
        return list(range(minimum, maximum + 1))
    try:
        indices = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise UserError(
            "--insertion-indices must be comma-separated integers"
        ) from error
    if not indices:
        raise UserError("--insertion-indices must not be empty")
    if len(indices) != len(set(indices)):
        raise UserError("--insertion-indices contains a duplicate")
    if any(index < minimum or index > maximum for index in indices):
        raise UserError(f"insertion indices must lie between {minimum} and {maximum}")
    return sorted(indices)


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def generate(args: argparse.Namespace) -> None:
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"cannot read configuration {args.config}: {error}") from error
    if not isinstance(config, dict):
        raise UserError("configuration root must be a JSON object")
    grid = make_grid(args.ti, args.tf, args.dt)
    indices = parse_indices(args.insertion_indices, grid.first_index, grid.last_index)
    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        raise UserError(f"output path exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise UserError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    insertion_site = config.get("insertion_site", 0)
    measurement_site = config.get("measurement_site", insertion_site)
    if (
        isinstance(insertion_site, bool)
        or isinstance(measurement_site, bool)
        or not isinstance(insertion_site, int)
        or not isinstance(measurement_site, int)
    ):
        raise UserError("insertion_site and measurement_site must be integers")
    sites = require(config, "sites", int)
    if not (0 <= insertion_site < sites and 0 <= measurement_site < sites):
        raise UserError(
            "insertion_site and measurement_site must be valid site indices"
        )
    insertion_operator = f"c'[{insertion_site}]"
    diagnostic = "<P2.last|P2>"

    insertion_loops = config.get("insertion_loops", 2)
    if (
        isinstance(insertion_loops, bool)
        or not isinstance(insertion_loops, int)
        or insertion_loops < 1
    ):
        raise UserError(
            "configuration field 'insertion_loops' must be a positive integer"
        )

    write_text(output / "gs" / "input.inp", ground_state_input(config))
    jobs: list[dict[str, Any]] = [
        {
            "id": "gs",
            "directory": "gs",
            "input": "input.inp",
            "checkpoint": "gs/data.hd5",
            "restart_checkpoint": None,
            "log": "gs/runForinput.cout",
            "depends_on": [],
            "arguments": [],
        }
    ]
    columns: list[dict[str, Any]] = []

    for insertion_index in indices:
        tag = f"t{insertion_index:03d}"
        column_dir = Path(f"column-{tag}")
        insertion_time = grid.time(insertion_index)
        reference_id = "gs"
        reference_checkpoint = "gs/data.hd5"

        if insertion_index > 0:
            reference_dir = Path(f"reference-{tag}")
            reference_expressions = [
                f"string P0={time_evolve(config, grid.step, 'gs')};"
            ]
            write_text(
                output / reference_dir / "input.inp",
                target_input(
                    config,
                    version=f"TwoTimeGrid-reference-{tag}",
                    output_file="data",
                    restart_file="../gs/data",
                    loop_count=insertion_index + 1,
                    expressions=reference_expressions,
                ),
            )
            reference_id = f"reference-{tag}"
            reference_checkpoint = f"{reference_dir}/data.hd5"
            jobs.append(
                {
                    "id": reference_id,
                    "directory": str(reference_dir),
                    "input": "input.inp",
                    "checkpoint": reference_checkpoint,
                    "restart_checkpoint": "gs/data.hd5",
                    "log": f"{reference_dir}/runForinput.cout",
                    "depends_on": ["gs"],
                    "arguments": ["<P0|P0>"],
                    "restart_mapping_tvs": None,
                    "target_expressions": [
                        f"P0={reference_expressions[0].split('=', 1)[1][:-1]}"
                    ],
                    "expected_final_time": decimal_text(insertion_time),
                }
            )

        if insertion_index == 0:
            # DMRG++ treats P0=|gs> as a self-assignment and does not serialize
            # a target vector. Use the established t'=0 layout instead: save
            # only the inserted branch here and evolve |gs> beside it below.
            insert_expressions = [f'string P0="{insertion_operator}*|gs>";']
            source_tv = None
            mapping = None
            restart_name = "../gs/data"
        else:
            insert_expressions = [
                "string P0=|P0>;",
                f'string P1="{insertion_operator}*|gs>";',
            ]
            # TargetingExpression currently determines a changed symmetry sector
            # from |gs>, not from |P0>. Make the evolved reference target the
            # stage's psi and map the same vector to P0.
            source_tv = 0
            mapping = [0, -1]
            restart_name = f"../reference-{tag}/data"

        # A nonzero-time reference can restart at either sweep edge. Three
        # movements ensure the inserted target is available when site 0 is
        # measured, independent of that orientation.
        stage_insertion_loops = (
            insertion_loops if insertion_index == 0 else max(insertion_loops, 3)
        )
        write_text(
            output / column_dir / "insert.inp",
            target_input(
                config,
                version=f"TwoTimeGrid-insert-{tag}",
                output_file="insert",
                restart_file=restart_name,
                loop_count=stage_insertion_loops,
                expressions=insert_expressions,
                restart_mapping=mapping,
                source_tv_for_psi=source_tv,
            ),
        )
        insert_id = f"insert-{tag}"
        jobs.append(
            {
                "id": insert_id,
                "directory": str(column_dir),
                "input": "insert.inp",
                "checkpoint": f"{column_dir}/insert.hd5",
                "restart_checkpoint": reference_checkpoint,
                "log": f"{column_dir}/runForinsert.cout",
                "depends_on": [reference_id],
                "arguments": ["<P0|P0>" if insertion_index == 0 else "<P0|P0>,<P1|P1>"],
                "restart_mapping_tvs": mapping,
                "restart_source_tv_for_psi": source_tv,
                "finite_loop_count": stage_insertion_loops,
                "target_expressions": insert_expressions,
            }
        )

        relative_steps = grid.last_index - insertion_index
        if insertion_index == 0:
            evolve_expressions = [
                "string P0=|P0>;",
                f"string P1={time_evolve(config, grid.step, 'P0')};",
                f"string P2={time_evolve(config, grid.step, 'gs')};",
            ]
            evolve_mapping = [0, -1, -1]
            measurement = "<P2|c|P1>"
        else:
            evolve_expressions = [
                "string P0=|P0>;",
                "string P1=|P1>;",
                f"string P2={time_evolve(config, grid.step, 'P0')};",
                f"string P3={time_evolve(config, grid.step, 'P1')};",
            ]
            evolve_mapping = [0, 1, -1, -1]
            measurement = "<P2|c|P3>"
        write_text(
            output / column_dir / "evolve.inp",
            target_input(
                config,
                version=f"TwoTimeGrid-evolve-{tag}",
                output_file="evolve",
                restart_file="insert",
                loop_count=relative_steps + 1,
                expressions=evolve_expressions,
                restart_mapping=evolve_mapping,
            )
            + f"\n# Run with arguments: {measurement},{diagnostic}\n",
        )
        evolve_id = f"evolve-{tag}"
        jobs.append(
            {
                "id": evolve_id,
                "directory": str(column_dir),
                "input": "evolve.inp",
                "checkpoint": f"{column_dir}/evolve.hd5",
                "restart_checkpoint": f"{column_dir}/insert.hd5",
                "log": f"{column_dir}/runForevolve.cout",
                "depends_on": [insert_id],
                "arguments": [f"{measurement},{diagnostic}"],
                "restart_mapping_tvs": evolve_mapping,
                "target_expressions": evolve_expressions,
            }
        )
        columns.append(
            {
                "insertion_index": insertion_index,
                "insertion_time": decimal_text(insertion_time),
                "relative_step_count": relative_steps,
                "relative_time_min": "0",
                "relative_time_max": decimal_text(relative_steps * grid.step),
                "reference_checkpoint": reference_checkpoint,
                "insertion_checkpoint": f"{column_dir}/insert.hd5",
                "insertion_log": f"{column_dir}/runForinsert.cout",
                "diagonal_label": "<P0|P0>" if insertion_index == 0 else "<P1|P1>",
                "evolution_checkpoint": f"{column_dir}/evolve.hd5",
                "evolution_log": f"{column_dir}/runForevolve.cout",
                "restart_mapping_tvs": evolve_mapping,
                "insertion_operator": insertion_operator,
                "measurement_label": measurement,
                "diagnostic_label": diagnostic,
            }
        )

    manifest = {
        "format": "dmrgpp-two-time-grid-v1",
        "component": args.component,
        "grid": {
            "initial_time": decimal_text(grid.initial),
            "final_time": decimal_text(grid.final),
            "time_step": decimal_text(grid.step),
            "first_index": grid.first_index,
            "last_index": grid.last_index,
        },
        "measurement_site": measurement_site,
        "runner_policy": {
            "accepted_nonzero_exit": 134,
            "required_completion_marker": "Written sys. and env. stacks to disk.",
            "requires_nonempty_checkpoint": True,
            "hdf5_header_check_when_available": True,
            "disable_environment_variable": "DMRG_ALLOW_VALIDATED_ABORT=0",
        },
        "model": config,
        "columns": columns,
        "jobs": jobs,
    }
    write_text(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    commands = []
    for job in jobs:
        fields = [
            job["directory"],
            job["input"],
            job["checkpoint"],
            job["log"],
            *job["arguments"],
        ]
        commands.append("run " + " ".join(shlex.quote(field) for field in fields))
    run_script = (
        """#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${DMRG:?Set DMRG to the absolute path of the dmrg executable}"
DMRG_ALLOW_VALIDATED_ABORT=${DMRG_ALLOW_VALIDATED_ABORT:-1}

validate_completed_abort() {
    checkpoint=$1
    log=$2
    test -s "$checkpoint" || return 1
    test -s "$log" || return 1
    grep -Eq 'Checkpoint .*: Written sys\\. and env\\. stacks to disk\\.$' "$log" || return 1
    if command -v h5dump >/dev/null 2>&1; then
        h5dump -H "$checkpoint" >/dev/null 2>&1 || return 1
    fi
}

run() {
    directory=$1
    input=$2
    checkpoint=$ROOT/$3
    log=$ROOT/$4
    shift 4
    echo "==> $directory/$input"

    # Never mistake products from an earlier invocation for successful output.
    rm -f "$checkpoint" "$log"
    set +e
    (cd "$ROOT/$directory" && "$DMRG" -f "$input" "$@")
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        return 0
    fi

    if [ "$status" -eq 134 ] && [ "$DMRG_ALLOW_VALIDATED_ABORT" = 1 ] \\
        && validate_completed_abort "$checkpoint" "$log"; then
        echo "WARNING: validated DMRG++ shutdown abort for $directory/$input; continuing." >&2
        return 0
    fi

    echo "ERROR: DMRG++ failed for $directory/$input (exit $status)." >&2
    return "$status"
}

"""
        + "\n".join(commands)
        + "\n"
    )
    write_text(output / "run_all.sh", run_script, executable=True)
    print(f"Generated {len(columns)} column(s) and {len(jobs)} job(s) in {output}")


def parse_log_values(
    path: Path, labels: list[str], site: int
) -> dict[str, dict[Decimal, complex]]:
    label_alternation = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"^\s*(?P<site>\d+)\s+"
        rf"\((?P<real>{FLOAT_PATTERN})\s*,\s*(?P<imag>{FLOAT_PATTERN})\)\s+"
        rf"(?P<time>{FLOAT_PATTERN})\s+(?P<label>{label_alternation})(?:\s|$)"
    )
    values: dict[str, dict[Decimal, complex]] = {label: {} for label in labels}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise UserError(f"cannot read evolution log {path}: {error}") from error
    for line in lines:
        match = pattern.match(line)
        if match is None or int(match.group("site")) != site:
            continue
        label = match.group("label")
        time = Decimal(match.group("time"))
        value = complex(float(match.group("real")), float(match.group("imag")))
        old = values[label].get(time)
        if old is not None and abs(old - value) > 2e-6:
            raise UserError(
                f"conflicting duplicate {label} values at time {time} in {path}"
            )
        values[label][time] = value
    return values


def collect(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"cannot read manifest {manifest_path}: {error}") from error
    if manifest.get("format") != "dmrgpp-two-time-grid-v1":
        raise UserError("unsupported or missing manifest format")
    root = manifest_path.parent
    dt = Decimal(manifest["grid"]["time_step"])
    site = int(manifest["measurement_site"])
    rows: list[tuple[int, int, Decimal, Decimal, complex, complex | None]] = []

    for column in manifest["columns"]:
        insertion_index = int(column["insertion_index"])
        insertion_time = Decimal(column["insertion_time"])
        relative_steps = int(column["relative_step_count"])

        # Use the insertion norm for the diagonal. Restarted expression stages
        # can inherit a stale physical-time label, but this equal-time identity
        # is independent of the label: <P1|P1> = <psi|c c'|psi>.
        diagonal_label = column["diagonal_label"]
        insertion_log = root / column["insertion_log"]
        diagonal_values = parse_log_values(insertion_log, [diagonal_label], site)[
            diagonal_label
        ]
        if not diagonal_values:
            raise UserError(
                f"missing diagonal {diagonal_label} for site {site} in {insertion_log}"
            )
        diagonal = diagonal_values[max(diagonal_values)]
        rows.append(
            (
                insertion_index,
                insertion_index,
                insertion_time,
                insertion_time,
                diagonal,
                None,
            )
        )

        if relative_steps == 0:
            continue
        measurement = column["measurement_label"]
        diagnostic = column["diagnostic_label"]
        log = root / column["evolution_log"]
        parsed = parse_log_values(log, [measurement, diagnostic], site)
        samples = sorted(parsed[measurement].items())
        if len(samples) < relative_steps:
            raise UserError(
                f"found only {len(samples)} of {relative_steps} post-insertion "
                f"samples for {measurement}, site {site}, in {log}"
            )

        # DMRG++ can report evolution times with an offset inherited from the
        # restart. The samples remain causally ordered, so discard any initial
        # duplicate of the separately collected diagonal and map by rank.
        for relative_index, (reported_time, raw) in enumerate(
            samples[-relative_steps:], start=1
        ):
            relative_time = relative_index * dt
            rows.append(
                (
                    insertion_index + relative_index,
                    insertion_index,
                    insertion_time + relative_time,
                    insertion_time,
                    raw,
                    parsed[diagnostic].get(reported_time),
                )
            )

    rows.sort(key=lambda row: (row[0], row[1]))
    destination = args.output
    output_lines = [
        "# t\ttprime\tm_real\tm_imag\tg_real\tg_imag\toverlap_real\toverlap_imag\tcomponent"
    ]
    for _, _, time, insertion_time, raw, overlap in rows:
        # G^> = -i M: (a + ib) -> b - ia.
        green = complex(raw.imag, -raw.real)
        overlap_real = "nan" if overlap is None else f"{overlap.real:.16g}"
        overlap_imag = "nan" if overlap is None else f"{overlap.imag:.16g}"
        output_lines.append(
            "\t".join(
                [
                    decimal_text(time),
                    decimal_text(insertion_time),
                    f"{raw.real:.16g}",
                    f"{raw.imag:.16g}",
                    f"{green.real:.16g}",
                    f"{green.imag:.16g}",
                    overlap_real,
                    overlap_imag,
                    manifest["component"],
                ]
            )
        )
    text = "\n".join(output_lines) + "\n"
    if destination is None:
        sys.stdout.write(text)
    else:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        except OSError as error:
            raise UserError(
                f"cannot write collected table {destination}: {error}"
            ) from error
        print(f"Collected {len(rows)} point(s) in {destination}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate DMRG++ runs")
    generate_parser.add_argument("--config", required=True, type=Path)
    generate_parser.add_argument(
        "--ti",
        required=True,
        type=decimal_arg,
        help="first requested insertion time on the global dt grid",
    )
    generate_parser.add_argument("--tf", required=True, type=decimal_arg)
    generate_parser.add_argument("--dt", required=True, type=decimal_arg)
    generate_parser.add_argument(
        "--component",
        choices=("greater",),
        default="greater",
        help="Green-function component (currently only greater)",
    )
    generate_parser.add_argument(
        "--insertion-indices",
        help="comma-separated global indices with tprime = index*dt",
    )
    generate_parser.add_argument("--output", required=True, type=Path)
    generate_parser.set_defaults(function=generate)

    collect_parser = subparsers.add_parser("collect", help="collect a generated grid")
    collect_parser.add_argument("--manifest", required=True, type=Path)
    collect_parser.add_argument("--output", type=Path)
    collect_parser.set_defaults(function=collect)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (UserError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
