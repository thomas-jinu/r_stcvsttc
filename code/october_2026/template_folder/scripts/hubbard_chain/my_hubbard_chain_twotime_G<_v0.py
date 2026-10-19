"""
Generate and optionally run a two-time Green's function calculation for the Hubbard chain using DMRG++.

Run this script from above the directory containing the ``dmrgpp`` scripts,
or adjust the project-root discovery below if your package layout differs.
"""  # noqa: EXE002

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

"""
Resolve paths so that scripts live inside scripts/hubbard_chain, and the project root is two levels up.
"""
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # two level up to the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


"""
Location of the current DMRG++ executables for different clusters.
"""
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


def parse_args_input() -> argparse.Namespace:
    """
    Parse command-line arguments for the script, including the cluster choice and the --run flag.
    """

    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Order of positional arguments: sites up down t U potentialV infinite_kept finite_loops finite_kept cluster

        Example:
        python make_hubbard.py 32 16 16 -1 8 0 128 5 1000 local --run
        """,
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int, help="Number of spin-up electrons")
    parser.add_argument("down", type=int, help="Number of spin-down electrons")
    parser.add_argument("center_site", type=int, help="Index of the center site")
    parser.add_argument("t", type=float, help="Hopping parameter t")
    parser.add_argument("U", type=float, help="Hubbard interaction U")
    parser.add_argument("potentialV", type=float, help="Potential V")
    parser.add_argument("gs_filename", type=str, help="Ground file location")
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
    # parser.add_argument("Pump_tau", type=float) - Inherits TSP
    parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating the input."
    )

    return parser.parse_args()


def build_input(args: argparse.Namespace, run_name: str, time_axis, pump_axis) -> str:
    """
    Builds the time-dependent input for the DMRG++ calculation and uses recoveryenable read to create reference states for each time step.
    """

    # The time axis should be from [0, T_max] with odd number of timesteps to work with finiteloop settings in this function.
    assert args.Pump_time_steps % 2 == 1, "Pump_time_steps must be an odd number."

    # Total number of finite loops - it is 1 less than the number of timesteps, since the first step is the t=0 state.
    finite_loops = args.Pump_time_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

    # Finite rows is the ordering of loops in DMRG++ input.
    # Numeric value 3 = 1 + 2; 1: save and 2: usefastwft
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 3]" for _ in range(finite_loops)
    )

    AversusT_table = "\n".join(
        f"    [{time:.15g}, {pump:.15g}]," for time, pump in zip(time_axis, pump_axis)
    ).rstrip(",")

    # return the Ainur input as a string written to file in the main function.
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
                        f"{AversusT_table}",
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
    time_axis,
    pump_axis,
    step: int,
    restart_filename_apply: str,
) -> str:
    """
    Apply the operator to create the initial state for time evolution at a specific time step.
    """

    # Finite rows for the apply step.
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(args.finite_loops_apply)
    )

    # Generate the pump table at the specific time step.
    AversusT_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

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
                        f"{AversusT_table}",
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
    time_axis,
    pump_axis,
    step: int,
    restart_filename_evolve: str,
) -> str:
    """
    Build the input for the time evolution step after applying the operator.
    """

    # Logic to determine the number of finite loops and the targeting expression based on the current step in the time evolution.

    if len(time_axis[step:]) == 1:  # Last time step t=t'!
        finite_loops = args.TSPAdvanceEach // (args.sites - 2)

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        AversusT_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

        targeting_expression = [
            "RestartMappingTvs=[0, 1, -1, -1];",
            'string P0="|P0>";',
            'string P1="|P1>";',
            'string P2="|P0>";',
            'string P3="|P1>";',
        ]

    else:
        # The number of finite loops for the time evolution is determined by the remaining time steps after the current step.
        finite_loops = (
            len(time_axis[step:]) * (args.TSPAdvanceEach // (args.sites - 2)) - 1
        )  # Total number of finite loops

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        AversusT_table = "\n".join(
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
                        f"{AversusT_table}",
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


def main() -> int:  # type: ignore
    """
    Creates the scripts for the full two-time Green's function calculation, including the pump table and the input files for each step.
    Stage 1: Create reference states for each time step using recoveryenable read.
    Stage 2: Create folders for each time step, apply the operator to create the initial state for time evolution, and generate the input files for the time evolution.
    Stage 3: Time evolve to T_max for each time step, using the previously generated reference states and the pump table.
    """

    # Parse the command-line arguments for the script
    args = parse_args_input()

    # Create a unique run name based on the input parameters and the current timestamp.
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    restart_path = Path(args.gs_filename).resolve()

    args_dict = vars(args)
    with Path(restart_path.parent / f"input_args_twotime_{run_name}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(args_dict, file, indent=2)
    print(f"Wrote arguments to input_args_twotime_{run_name}.json")

    """
    Stage 1
    - Create reference states for each t^prime!
    - uses recoveryenableread.
    - recoveryenable read index 0 is for the first timestep = TspTau
    """

    # Create the folder;
    run_folder_td = restart_path.parent / f"td_{run_name}"
    run_folder_td.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_td}\n")

    # Copy the DMRG++ executable to the run folder;
    executable_path = DMRG_EXECUTABLES[args.cluster]
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder_td / "dmrg")

    # Create the pump table and write it to a CSV file in the run folder.
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

    write_pump_table(
        pump_table=zip(time_axis, pump_axis),
        output_path=run_folder_td / "pump_table.csv",
    )

    # Create stage 1 input file for Ainur and write it to the run folder.
    input_path = run_folder_td / f"input_{run_name}.ain"
    input_path.write_text(
        build_input(args, run_name, time_axis, pump_axis), encoding="utf-8"
    )
    print(f"Wrote Ainur input: {input_path}")

    # If --run is specified, execute DMRG++ with the generated input. If a cluster is specified, create a SLURM batch script for submission.
    if args.run:
        subprocess.run(
            ["./dmrg", "-f", str(input_path), "-p", "12"],
            cwd=run_folder_td,
            check=True,
        )
    elif args.cluster != "local":
        # args_slurm = parse_args_slurm()
        slurm_path = run_folder_td / f"batch_{run_name}.slurm"
        # body = f"\n./dmrg -f input_{run_name}.ain\n"
        # build_slurm_script(body, slurm_path, args.cluster, use_gpu=args.gpu)
        print(f"Wrote batch script: {slurm_path}")
    else:
        print("Input generated. Use --run to start DMRG++.")

    """
    Stage 2
    - Create folders for each t^prime inside the twotime folder.
    - Apply operator to create the initial state for time evolution.
    - Inputs to timeevolve to T_max.
    """

    # Create the twotime folder
    run_folder_twotime = restart_path.parent / f"twotime_{run_name}"
    run_folder_twotime.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_twotime}")

    # Loop over each time step, create a folder for that step, copy the DMRG++ executable, and generate the input files for applying the operator and evolving the state.
    for step in range(args.Pump_time_steps):
        step_folder = run_folder_twotime / f"step_{step:04d}"
        step_folder.mkdir(exist_ok=True)
        print(f"Created step folder: {step_folder}")
        executable_path = DMRG_EXECUTABLES[args.cluster]
        print(
            f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}"
        )
        shutil.copy2(executable_path, step_folder / "dmrg")

        recoveryindex = int(step)
        restart_filename = f"../../td_{run_name}/Recovery{recoveryindex}{run_name}"

        # Create the input files for applying the operator and evolving the state, and write them to the step folder.
        input_path_restart = step_folder / f"input_{run_name}_apply.ain"
        input_path_restart.write_text(
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

        print(f"Wrote Ainur input: {input_path_restart}")
        print(f"Wrote Ainur input: {input_path_evolve}")

        # If --run is specified, execute DMRG++ with the generated input files for applying the operator and evolving the state. If a cluster is specified, create a SLURM batch script for submission.
        if args.run:
            subprocess.run(
                ["./dmrg", "-f", str(input_path_restart), "-p", "12"],
                cwd=step_folder,
                check=True,
            )
            subprocess.run(
                ["./dmrg", "-f", str(input_path_evolve), "-p", "12", "<P2|c'|P3>"],
                cwd=step_folder,
                check=True,
            )
        elif args.cluster != "local":
            # args_slurm = parse_args_slurm()
            slurm_path = run_folder_twotime / f"batch_{run_name}.slurm"
            # body = f"\n./dmrg -f input_{run_name}.ain\n"
            # build_slurm_script(body, slurm_path, args.cluster, use_gpu=args.gpu)
            print(f"Wrote batch script: {slurm_path}")
        else:
            print("Input generated. Use --run to start DMRG++.")

    return 0


# def parse_args_slurm() -> argparse.Namespace:

#     parser = argparse.ArgumentParser(
#         description="Parse arguments for the Slurm script.",
#     )
#     parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
#     # parser.add_argument("--gpu", action="store_true", help="Use GPU.")
#     return parser.parse_args()


def myprocess_twotime(time_axis, args, run_name):
    """
    Function to obtain the two-time Green's function for the Hubbard chain using DMRG++.
    """

    # Read in a folder named twotime_{run_name} and the args file


"""
Helpful functions for creating the time axis and pump values, and writing the pump table to a CSV file.
"""


def create_time_axis(nsteps, step_size):
    """
    Create the time values used by the time-dependent pump table.
    """

    return [round(index * step_size, 9) for index in range(nsteps)]


def create_pump_axis(time_axis, amplitude, omega_pump, t_delay, sigma):
    """
    Create the Gaussian-envelope cosine pump values.
    """

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


def write_pump_table(pump_table, output_path):
    """
    Write time and pump values to a CSV file.

    The CSV contains one time-pump pair per row with column headers
    named "time" and "pump".
    """
    with Path(output_path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "pump"])

        for time, pump in pump_table:
            writer.writerow(
                [
                    f"{time:.9f}",
                    f"{pump:.9f}",
                ]
            )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")


LINE_PATTERN = re.compile(
    r"^\s*(\d+)\s+"
    r"\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+"
    r"<P2\|c\|P3>"
)


def read_operator_array(filename):
    rows = []
    collecting = False

    with Path(filename).open("r", encoding="utf-8") as file:
        for line in file:
            if "FiniteLoops printing ends" in line:
                collecting = True
                continue

            if not collecting or "<P2|c|P3>" not in line:
                continue

            match = LINE_PATTERN.search(line)
            if match is None:
                continue

            site_index = int(match.group(1))
            real_part = float(match.group(2))
            imaginary_part = float(match.group(3))
            time = float(match.group(4))

            rows.append(
                [
                    site_index,
                    real_part,
                    imaginary_part,
                    time,
                ]
            )

    return np.asarray(rows, dtype=float)
