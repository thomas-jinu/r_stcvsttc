"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import math
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
    "nersc-cpu": Path("/global/common/software/m5228/dmrgpp_cpu/installdir/bin/dmrg"),
    "nersc-gpu": Path("/global/common/software/m5228/dmrgpp/builddir-cuda/dmrg/dmrg"),
}

DMRG_PRECISION = 12
OPERATOR_LABEL = "<P2|c'|P3>"
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
    parser.add_argument("center_site", type=int, help="Index of the center site")
    parser.add_argument("t", type=float, help="Hopping parameter t")
    parser.add_argument("U", type=float, help="Hubbard interaction U")
    parser.add_argument("potentialV", type=float, help="Potential V")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument(
        "finite_kept", type=int, help="Number of states in finite loops"
    )
    parser.add_argument(
        "finite_loops_apply",
        type=int,
        help="Number of finite loops for applying the operator step",
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
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression";',
                        'Version = "stc_vs_ttc";',
                        f'string RecoverySave = "%l%%2,@keep,@M={args.Pump_time_steps}";',
                        f'OutputFile = "{run_name}";',
                        f'RestartFilename = "../{Path(args.gs_filename).resolve().name}";',
                        "GsWeight = 0.1;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def build_input_apply(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    step: int,
    restart_filename_apply: str,
) -> str:
    """Build the input that applies the operator at one time step."""

    # Finite rows for the apply step.
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(args.finite_loops_apply)
    )

    # Generate the pump table at the specific time step.
    aversus_t_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

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
                        f"GeometryFactor = exp:*:1.0i:!readTableAversusTime,{time_axis[step]:.15g};",
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
                        "string P0 = |P0>;",
                        f'string P1 = "c[{args.center_site}]*|P0>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression";',
                        'Version = "stc_vs_ttc_apply";',
                        f'OutputFile = "{run_name}_apply";',
                        f'RestartFilename = "{restart_filename_apply}";',
                        "GsWeight = 0.1;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def build_input_evolve(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    step: int,
    restart_filename_evolve: str,
) -> str:
    """Build the time-evolution input after applying the operator."""
    remaining_steps = len(time_axis) - step

    if remaining_steps == 1:
        finite_loops = args.TSPAdvanceEach // (args.sites - 2)

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        aversus_t_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

        targeting_expression = [
            "RestartMappingTvs=[0, 1, -1, -1];",
            'string P0="|P0>";',
            'string P1="|P1>";',
            'string P2="|P0>";',
            'string P3="|P1>";',
        ]

    else:
        finite_loops = remaining_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        aversus_t_table = "\n".join(
            f"    [{time:.15g}, {pump:.15g}],"
            for time, pump in zip(time_axis[step:], pump_axis[step:])
        ).rstrip(",")

        targeting_expression = [
            "RestartMappingTvs=[0, 1, -1, -1];",
            'string P0="|P0>";',
            'string P1="|P1>";',
            f'string P2="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P0>";',
            f'string P3="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P1>";',
        ]

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
                        *targeting_expression,
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,minimizedisk";',
                        'Version = "stc_vs_ttc_apply_evolve";',
                        f'OutputFile = "{run_name}_evolve";',
                        f'RestartFilename = "{restart_filename_evolve}";',
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
                9,
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
        writer.writerows((f"{time:.9f}", f"{pump:.9f}") for time, pump in pump_table)


def find_evolve_output(step_folder: Path, run_name: str) -> Path:
    """Find the DMRG++ evolve output file for one step."""
    candidates = (step_folder / f"runForinput_{run_name}_evolve.cout",)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No evolve output found in {step_folder}")


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


def collect_twotime_data(
    twotime_folder: Path,
    run_name: str,
    number_of_steps: int,
    time_axis: list[float],
    center_site: int,
) -> dict[tuple[float, float, int, int], complex]:
    """
    Return:
        {
            (t, tprime, center_site, site): complex_green_value
        }

    Repeated keys retain the last value.
    """
    data = {}

    for step in range(number_of_steps):
        step_folder = twotime_folder / f"step_{step:04d}"
        output_path = find_evolve_output(step_folder, run_name)
        step_data = read_operator_output(output_path)

        t = float(time_axis[step])

        for site, real_part, imaginary_part, tprime in step_data:
            key = (
                t,
                float(tprime),
                int(center_site),
                int(site),
            )

            data[key] = complex(
                real_part,
                imaginary_part,
            )

    return data


def save_twotime_csv(
    data: dict[tuple[float, float, int, int], complex],
    output_path: Path,
) -> None:
    """
    Save two-time Green's-function data to CSV.

    Dictionary format:
        {
            (t, tprime, center_site, site): complex_value
        }
    """
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "t",
                "t_prime",
                "center_site",
                "site_index",
                "real_part",
                "imaginary_part",
            ]
        )

        for (t, tprime, center_site, site), value in sorted(data.items()):
            writer.writerow(
                [
                    f"{t:.15g}",
                    f"{tprime:.15g}",
                    center_site,
                    site,
                    f"{value.real:.15g}",
                    f"{value.imag:.15g}",
                ]
            )


def process_twotime_results(
    base_directory: Path,
    run_name: str,
    number_of_steps: int,
    time_axis,
    center_site: int,
) -> Path:
    """Collect an existing run and write its combined CSV file."""
    twotime_folder = base_directory / f"twotime_{run_name}"

    if not twotime_folder.is_dir():
        raise FileNotFoundError(f"Two-time run folder not found: {twotime_folder}")
    if number_of_steps > len(time_axis):
        raise ValueError("number_of_steps cannot exceed the length of time_axis")

    data = collect_twotime_data(
        twotime_folder=twotime_folder,
        run_name=run_name,
        number_of_steps=number_of_steps,
        time_axis=time_axis,
        center_site=center_site,
    )

    csv_path = twotime_folder / f"{run_name}_P2_c_P3_centersite={center_site}.csv"

    save_twotime_csv(
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
) -> None:
    """Run the copied DMRG++ executable for one input file."""
    command = ["./dmrg", "-f", input_path.name, "-p", str(DMRG_PRECISION)]
    if operator is not None:
        command.append(operator)

    subprocess.run(command, cwd=working_directory, check=True)


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
        csv_path = process_twotime_results(
            base_directory=restart_path.parent,
            run_name=args.process,
            number_of_steps=args.Pump_time_steps,
            time_axis=time_axis,
            center_site=args.center_site,
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
        run_dmrg(run_folder_td, input_path)

    # Stage 2: create and optionally run each two-time calculation.
    run_folder_twotime = restart_path.parent / f"twotime_{run_name}"
    run_folder_twotime.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_twotime}")

    for step in range(args.Pump_time_steps):
        step_folder = run_folder_twotime / f"step_{step:04d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created step folder: {step_folder}")
        shutil.copy2(executable_path, step_folder / "dmrg")

        restart_filename = f"../../td_{run_name}/Recovery{step}{run_name}"

        input_path_apply = step_folder / f"input_{run_name}_apply.ain"
        input_path_apply.write_text(
            build_input_apply(
                args,
                run_name,
                step=step,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_apply=restart_filename,
            ),
            encoding="utf-8",
        )

        input_path_evolve = step_folder / f"input_{run_name}_evolve.ain"
        input_path_evolve.write_text(
            build_input_evolve(
                args,
                run_name,
                step=step,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_evolve=run_name + "_apply",
            ),
            encoding="utf-8",
        )

        print(f"Wrote Ainur input: {input_path_apply}")
        print(f"Wrote Ainur input: {input_path_evolve}")

        if args.run:
            run_dmrg(step_folder, input_path_apply)
            run_dmrg(step_folder, input_path_evolve, OPERATOR_LABEL)

    if args.run:
        csv_path = process_twotime_results(
            base_directory=restart_path.parent,
            run_name=run_name,
            center_site=args.center_site,
            number_of_steps=args.Pump_time_steps,
            time_axis=time_axis,
        )
        print(f"Wrote post-processed data: {csv_path}")
    else:
        print("Inputs generated. Use --run to execute DMRG++.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
