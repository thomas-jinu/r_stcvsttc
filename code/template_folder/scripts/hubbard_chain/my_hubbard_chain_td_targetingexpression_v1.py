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
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    ),
    "isaac": Path(
        "/nfs/home/jthom214/dmrgpp/programs_08192026/dmrgpp/installdir/bin/dmrg"
    ),
    "nersc": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    ),
}

DMRG_PRECISION = 12
OPERATOR_LABEL = "<P0|n|P0>"
COLLECTION_MARKER = "FiniteLoops printing ends"

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPERATOR_PATTERN = re.compile(
    rf"^\s*(\d+)\s+"
    rf"\(\s*({NUMBER_PATTERN})\s*,\s*({NUMBER_PATTERN})\s*\)\s+"
    rf"({NUMBER_PATTERN})\s+"
    rf"{re.escape(OPERATOR_LABEL)}"
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
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--parallel-steps",
        type=int,
        default=1,
        help="Number of independent time steps to run simultaneously.",
    )
    parser.add_argument(
        "--launcher",
        choices=["local", "srun"],
        default="local",
        help="Run processes directly or launch them with srun.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPU cores assigned to each DMRG++ process.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate inputs, run DMRG++, and process the results.",
    )
    mode.add_argument(
        "--process",
        metavar="RUN_NAME",
        help="Only process an existing twotime_RUN_NAME directory.",
    )

    return parser.parse_args()


def build_input_td(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
) -> str:
    """Build the time-dependent reference-state input."""
    finite_loops = args.Pump_time_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

    # Finite rows is the ordering of loops in DMRG++ input.
    # Numeric value 3 = 1 + 2; 1: save and 2: usefastwft
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 3]" for _ in range(finite_loops)
    )

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


def find_td_output(td_folder: Path, run_name: str) -> Path:
    """Find the DMRG++ evolve output file for one step."""
    candidates = (td_folder / f"runForinput_{run_name}.cout",)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No output found in {td_folder}")


def read_operator_output(output_path: Path) -> np.ndarray:
    """Read operator data after ``FiniteLoops printing ends``."""
    rows_by_key = {}
    collecting = False

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if COLLECTION_MARKER in line:
                collecting = True
                continue
            if not collecting or OPERATOR_LABEL not in line:
                continue

            match = OPERATOR_PATTERN.search(line)
            if match is None:
                continue

            site_index = int(match.group(1))
            real_part = float(match.group(2))
            imaginary_part = float(match.group(3))
            time = float(match.group(4))

            # If the same site/time appears again, keep the last value.
            rows_by_key[(site_index, time)] = [
                site_index,
                real_part,
                imaginary_part,
                time,
            ]

    if not rows_by_key:
        return np.empty((0, 4), dtype=float)

    return np.asarray(list(rows_by_key.values()), dtype=float)


def collect_td_data(
    td_folder: Path,
    run_name: str,
) -> dict[tuple[float, int], complex]:
    """
    Return:
        {
            (t, site): complex_green_value
        }

    Repeated keys retain the last value.
    """
    data = {}

    output_path = find_td_output(td_folder, run_name)
    step_data = read_operator_output(output_path)

    for site, real_part, imaginary_part, t in step_data:
        key = (
            float(t),
            int(site),
        )

        data[key] = complex(
            real_part,
            imaginary_part,
        )

    return data


def save_td_csv(
    data: dict[tuple[float, int], complex],
    output_path: Path,
) -> None:
    """
    Save two-time Green's-function data to CSV.

    Dictionary format:
        {
            (t, site): complex_value
        }
    """
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "t",
                "site_index",
                "real_part",
                "imaginary_part",
            ]
        )

        for (t, site), value in sorted(data.items()):
            writer.writerow(
                [
                    f"{t:.15g}",
                    f"{site}",
                    f"{value.real:.15g}",
                    f"{value.imag:.15g}",
                ]
            )


def process_td_results(
    base_directory: Path,
    run_name: str,
    number_of_steps: int,
    time_axis,
) -> Path:
    """Collect an existing run and write its combined CSV file."""
    td_folder = base_directory / f"td_{run_name}"

    if not td_folder.is_dir():
        raise FileNotFoundError(f"TD run folder not found: {td_folder}")
    if number_of_steps > len(time_axis):
        raise ValueError("number_of_steps cannot exceed the length of time_axis")

    data = collect_td_data(
        td_folder=td_folder,
        run_name=run_name,
    )

    csv_path = td_folder / f"{run_name}_{OPERATOR_LABEL}.csv"

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
            base_directory=restart_path.parent,
            run_name=args.process,
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
    run_folder_td = restart_path.parent / f"td_{run_name}"
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
            operator=OPERATOR_LABEL,
        )

        csv_path = process_td_results(
            base_directory=restart_path.parent,
            run_name=run_name,
            number_of_steps=args.Pump_time_steps,
            time_axis=time_axis,
        )
        print(f"Wrote post-processed data: {csv_path}")

    elif args.cluster == "nersc":
        slurm_path = run_folder_td / f"batch_{run_name}.slurm"

        body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks={args.parallel_steps}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --time=03:00:00
#SBATCH --job-name=dmrg_twotime
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module reset
module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

export CC=cc
export CXX=CC
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export BASE=/global/common/software/m5228
export LOCAL="$BASE/local"

date

# Stage 1: generate the recovery states.
cd "{run_folder_td}"

srun \\
    --exclusive \\
    --ntasks=1 \\
    --cpus-per-task={args.cpus_per_task} \\
    ./dmrg \\
    -f "{input_path.name}" \\
    -p "{DMRG_PRECISION}"


date
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
