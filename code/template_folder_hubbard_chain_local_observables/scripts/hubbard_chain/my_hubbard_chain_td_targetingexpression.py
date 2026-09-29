"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DMRG_EXECUTABLES = {
    "local": Path("/global/common/software/m5228/dmrgpp/installdir/bin/dmrg"),
    "nersc": Path("/global/common/software/m5228/dmrgpp/installdir/bin/dmrg"),
}

DMRG_PRECISION = 12
DEFINE_OPERATORS_DMRGPP = "double:nup*ndown,hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,parity:identity+(-2.0)*n+4.0*nup*ndown,local_moment:0.75*n+(-1.5)*nup*ndown"
OPERATOR_LABELS = {
    "<P0|n|P0>",
    "<gs|n|gs>",
    "<P0|sz|P0>",
    "<gs|sz|gs>",
    "<P0|local_moment|P0>",
    "<gs|local_moment|gs>",
    "<P0|hole|P0>",
    "<gs|hole|gs>",
    "<P0|parity|P0>",
    "<gs|parity|gs>",
    "<P0|double|P0>",
    "<gs|double|gs>",
    "<P0|n*n|P0>",
    "<gs|n*n|gs>",
}
COLLECTION_MARKER = "FiniteLoops printing ends"

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPERATOR_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<site>\d+)\s+
    \(
        (?P<real>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),
        (?P<imag>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \)\s+
    (?P<time>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+
    (?P<label><[^>]+>)
    """,
    re.VERBOSE,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int, help="Number of spin-up electrons")
    parser.add_argument("down", type=int, help="Number of spin-down electrons")
    parser.add_argument("t", type=float, help="Hopping parameter t")
    parser.add_argument("U", type=float, help="Hubbard interaction U")
    parser.add_argument("potentialV", type=float, help="Potential V")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument(
        "finite_kept", type=int, help="Number of states in finite loops"
    )
    parser.add_argument(
        "TSPAdvanceEach",
        type=int,
        help="Multiples of N-2",
    )
    parser.add_argument("TSPTau", type=float, help="Time step size for the pump table")
    parser.add_argument("Pump_Amplitude", type=float)
    parser.add_argument("Pump_Frequency", type=float)
    parser.add_argument("Pump_time_delay", type=float)
    parser.add_argument("Pump_pulse_width", type=float)
    parser.add_argument("Pump_time_steps", type=int)
    parser.add_argument("cluster", choices=DMRG_EXECUTABLES)
    parser.add_argument("--frequency-start", type=float, default=0.0)
    parser.add_argument("--frequency-stop", type=float, default=10.0)
    parser.add_argument("--frequency-step", type=float, default=0.25)
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPU cores assigned for this run.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate inputs, run DMRG++, and process the results.",
    )
    parser.add_argument(
        "--launcher",
        choices=["local", "srun"],
        default="local",
        help="The launcher to use for running the simulation.",
    )
    mode.add_argument(
        "--process",
        type=Path,
        metavar="OUTPUT_FILE",
        help="Process an existing output file.",
    )

    return parser.parse_args()


def build_input_td(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
) -> str:
    """Build the time-dependent reference-state input."""

    # Number of finite loops is determined by the number of pump time steps and the TSPAdvanceEach parameter.
    finite_loops = args.Pump_time_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

    # Finite rows is the ordering of loops in DMRG++ input.
    # Numeric value 3 = 1 + 2; 1: save and 2: usefastwft
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 3]" for _ in range(finite_loops)
    )

    # Build the AversusTime table for the pump values.
    aversus_t_table = "\n".join(
        f"    [{time:.15g}, {pump:.15g}]," for time, pump in zip(time_axis, pump_axis)
    ).rstrip(",")

    return (
        "\n\n".join(
            [
                "##Ainur1.0",
                "\n".join(
                    [
                        "# --- Model parameters ---",
                        f"TotalNumberOfSites = {args.sites};",
                        "NumberOfTerms = 1;",
                        "DegreesOfFreedom = 1;",
                        'GeometryKind = "chain";',
                        'GeometryOptions = "ConstantValues";',
                        f"dir0:Connectors = [{args.t}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Pump parameters --- #",
                        "matrix AversusTime = [",
                        f"{aversus_t_table}",
                        "];",
                        "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {args.up};",
                        f"TargetElectronsDown = {args.down};",
                        f'DefineOperators="{DEFINE_OPERATORS_DMRGPP}";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                        "RestartMapStages=0;",
                        f'string P0="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|gs>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,minimizedisk";',
                        'Version = "stc_vs_ttc";',
                        # f'string RecoverySave = "%l%%2,@keep,@M={args.Pump_time_steps}";',
                        f'OutputFile = "{run_name}";',
                        f'RestartFilename = "../{Path(args.gs_filename).resolve().name}";',
                        "GsWeight = 0.1;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def create_time_axis(nsteps: int, step_size: float) -> list[float]:
    """Create the time values used by the pump table."""
    return [round(index * step_size, 9) for index in range(nsteps)]


def create_pump_axis(
    time_axis: list[float],
    amplitude: float,
    omega_pump: float,
    t_delay: float,
    sigma: float,
) -> list[float]:
    """Create Gaussian-envelope cosine pump values."""
    return [
        (rounded_value if rounded_value != 0 else 0.0)
        for rounded_value in (
            round(
                amplitude
                * math.exp(-0.5 * ((time - t_delay) / sigma) ** 2)
                * math.cos(omega_pump * (time - t_delay)),
                7,
            )
            for time in time_axis
        )
    ]


def write_pump_table(
    pump_table: Iterable[tuple[float, float]],
    output_path: Path,
) -> None:
    """Write time and pump values to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "pump"])
        writer.writerows((f"{time:.7f}", f"{pump:.7f}") for time, pump in pump_table)


def read_operator_output(
    output_path: Path,
) -> dict[str, list[tuple[int, complex, float]]]:
    """Read operator values grouped by operator label."""

    operators = {label: [] for label in OPERATOR_LABELS}

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = OPERATOR_PATTERN.match(line)

            if match is None:
                continue

            label = match.group("label")

            if label not in operators:
                continue

            site = int(match.group("site"))
            value = complex(
                float(match.group("real")),
                float(match.group("imag")),
            )
            time = float(match.group("time"))

            operators[label].append((site, value, time))

    return operators


def collect_td_data(
    output_path: Path,
) -> dict[str, dict[tuple[float, int], complex]]:
    """
    Return:
        {
            operator_label: {
                (time, site): complex_value
            }
        }

    Repeated keys retain the last value.
    """
    data = {label: {} for label in OPERATOR_LABELS}

    step_data = read_operator_output(output_path)

    for label, rows in step_data.items():
        for site, value, time in rows:
            key = (float(time), int(site))
            data[label][key] = value

    return data


def save_td_csv(
    data: dict[str, dict[tuple[float, int], complex]],
    output_path: Path,
) -> None:
    """Save all operator values in one wide CSV."""

    labels = sorted(data)

    all_keys = {key for operator_data in data.values() for key in operator_data}

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = ["t", "site_index"]

        for label in labels:
            header.extend(
                [
                    f"{label}_real",
                    f"{label}_imag",
                ]
            )

        writer.writerow(header)

        for time, site in sorted(all_keys):
            row = [f"{time:.15g}", site]

            for label in labels:
                value = data[label].get((time, site), complex(float("nan")))

                row.extend(
                    [
                        f"{value.real:.15g}",
                        f"{value.imag:.15g}",
                    ]
                )

            writer.writerow(row)


def process_td_results(
    output_path: Path,
    number_of_steps: int,
    time_axis,
) -> Path:
    """Process an existing DMRG++ output file."""

    if not output_path.is_file():
        raise FileNotFoundError(f"Output file not found: {output_path}")

    if number_of_steps > len(time_axis):
        raise ValueError("number_of_steps cannot exceed the length of time_axis")

    data = collect_td_data(output_path)

    csv_path = output_path.with_name(f"{output_path.stem}_operators.csv")

    save_td_csv(
        data=data,
        output_path=csv_path,
    )

    return csv_path


def validate_args(args: argparse.Namespace) -> None:
    """Validate constraints required by the finite-loop construction."""
    if args.sites <= 2:
        raise ValueError("sites must be greater than 2")
    if args.Pump_time_steps <= 0 or args.Pump_time_steps % 2 == 0:
        raise ValueError("Pump_time_steps must be a positive odd number")
    if args.TSPAdvanceEach <= 0:
        raise ValueError("TSPAdvanceEach must be positive")
    if args.TSPAdvanceEach % (args.sites - 2) != 0:
        raise ValueError("TSPAdvanceEach must be a multiple of sites - 2")
    if args.Pump_pulse_width == 0:
        raise ValueError("Pump_pulse_width must be nonzero")


def save_arguments(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments as JSON."""
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)


def run_dmrg(
    working_directory: Path,
    input_path: Path,
    operator: str | None = None,
    launcher: str = "local",
    cpus_per_task: int = 1,
) -> None:
    dmrg_command = [
        "./dmrg",
        "-f",
        input_path.name,
        "-p",
        str(DMRG_PRECISION),
    ]

    if operator is not None:
        dmrg_command.append(operator)

    if launcher == "srun":
        command = [
            "srun",
            "--exclusive",
            "--ntasks=1",
            f"--cpus-per-task={cpus_per_task}",
            "--wait",
            *dmrg_command,
        ]
    else:
        command = dmrg_command

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpus_per_task)

    subprocess.run(
        command,
        cwd=working_directory,
        check=True,
        env=environment,
    )


def frequency_tag(frequency: float) -> str:
    """Convert a frequency into a filesystem-safe tag."""
    return f"freq_{frequency:g}".replace(".", "p")


def main() -> int:
    """Generate inputs, optionally run DMRG++, and collect the results."""
    args = parse_args()
    restart_path = args.gs_filename.resolve()

    time_axis = create_time_axis(
        nsteps=args.Pump_time_steps,
        step_size=args.TSPTau,
    )

    pump_axis = create_pump_axis(
        time_axis,
        amplitude=args.Pump_Amplitude,
        omega_pump=args.Pump_Frequency,
        t_delay=args.Pump_time_delay,
        sigma=args.Pump_pulse_width,
    )

    if args.process:
        if args.Pump_time_steps <= 0:
            raise ValueError("Pump_time_steps must be positive")

        csv_path = process_td_results(
            output_path=args.process,
            number_of_steps=args.Pump_time_steps,
            time_axis=time_axis,
        )
        print(f"Wrote post-processed data: {csv_path}")
        return 0

    validate_args(args)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    executable_path = DMRG_EXECUTABLES[args.cluster]

    if not restart_path.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {restart_path}")
    if not executable_path.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable_path}")

    args_path = restart_path.parent / f"input_args_twotime_{run_name}.json"
    save_arguments(args, args_path)
    print(f"Wrote arguments: {args_path}")

    # Stage 1: create the time-dependent reference states.
    frequency_tag_value = frequency_tag(args.Pump_Frequency)
    run_folder_td = restart_path.parent / f"td_{run_name}_{frequency_tag_value}"
    run_folder_td.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_td}")

    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder_td / "dmrg")

    write_pump_table(
        pump_table=zip(time_axis, pump_axis),
        output_path=run_folder_td / "pump_table.csv",
    )

    input_path = run_folder_td / f"input_{run_name}.ain"
    input_path.write_text(
        build_input_td(args, run_name, time_axis, pump_axis), encoding="utf-8"
    )
    print(f"Wrote Ainur input: {input_path}")

    if args.run:
        run_dmrg(
            run_folder_td,
            input_path,
            launcher=args.launcher,
            cpus_per_task=args.cpus_per_task,
            operator=",".join(sorted(OPERATOR_LABELS)),
        )

        csv_path = process_td_results(
            output_path=run_folder_td / f"runForinput_{run_name}.cout",
            number_of_steps=args.Pump_time_steps,
            time_axis=time_axis,
        )
        print(f"Wrote post-processed data: {csv_path}")

    elif args.cluster == "nersc":
        slurm_path = run_folder_td / f"batch_{run_name}.slurm"

        body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --job-name=dmrg_frequency
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module reset
module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

conda activate dmrg

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"

SCRIPT="{Path(__file__).resolve()}"
GS_FILE="{restart_path}"

for frequency in $(seq {args.frequency_start} {args.frequency_step} {args.frequency_stop}); do
    frequency_tag="${{frequency//./p}}"

    srun \\
        --exclusive \\
        --ntasks=1 \\
        --cpus-per-task="${{SLURM_CPUS_PER_TASK}}" \\
        --output="frequency_${{frequency_tag}}.out" \\
        --error="frequency_${{frequency_tag}}.err" \\
        python "$SCRIPT" \\
            {args.sites} {args.up} {args.down} {args.t} {args.U} {args.potentialV} \\
            "$GS_FILE" \\
            {args.finite_kept} {args.TSPAdvanceEach} {args.TSPTau} {args.Pump_Amplitude} \\
            "$frequency" \\
            {args.Pump_time_delay} {args.Pump_pulse_width} {args.Pump_time_steps} \\
            {args.cluster} \\
            --launcher local \\
            --cpus-per-task="${{SLURM_CPUS_PER_TASK}}" \\
            --run &
done

wait

echo "All frequency jobs completed."
"""
        slurm_path.write_text(body, encoding="utf-8")
        print(f"Wrote batch script: {slurm_path}")
    else:
        print("Inputs generated. Use --run to execute DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
