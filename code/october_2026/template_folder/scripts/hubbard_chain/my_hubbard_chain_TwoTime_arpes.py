"""Generate and optionally run a DMRG++ Hubbard-chain ground state.

Run this script from above the directory containing the ``dmrgpp`` scripts,
or adjust the project-root discovery below if your package layout differs.
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Resolve paths so that scripts live inside scripts/hubbard_chain, and the project root is two levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # two level up to the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Location of the current DMRG++ executables for different clusters. Adjust these paths as needed. The location is printed when the script is completed.
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


# Define all the input arguments for the script, including the cluster choice and the --run flag.
def parse_args_input() -> argparse.Namespace:
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
    parser.add_argument("up", type=int)
    parser.add_argument("down", type=int)
    parser.add_argument("center_site", type=int)
    parser.add_argument("t", type=float)
    parser.add_argument("U", type=float)
    parser.add_argument("potential_V", type=float)
    parser.add_argument(
        "gs_filename", type=str, help="Ground state filename for the calculation"
    )
    parser.add_argument("finite_kept", type=int)
    parser.add_argument("finite_loops_insert", type=int)
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


def build_input(args: argparse.Namespace, run_name: str) -> str:

    # If GS finite loop is even, add 2 more. If it is add just add 1 more.
    # First loop is from time 0 to
    finite_loops = (
        args.Pump_time_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1
    )  # Total number of finite loops

    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 3]" for _ in range(finite_loops)
    )

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

    AversusT_table = "\n".join(
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
                        f"potentialV = [{args.potential_V}, ...];",
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
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead";',
                        'Version = "stc_vs_ttc";',
                        f'string RecoverySave = "%l%%1,@keep,@M={args.Pump_time_steps}";',
                        f'OutputFile = "{run_name}";',
                        f'RestartFilename = "../{Path(args.gs_filename).resolve().name}";',
                        "GsWeight = 0.2;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def build_input_insert(
    args: argparse.Namespace,
    run_name: str,
    time_axis,
    pump_axis,
    step: int,
    restart_filename_insert: str,
) -> str:

    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(args.finite_loops_insert)
    )

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
                        f"potentialV = [{args.potential_V}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Pump parameters --- #",
                        f"GeometryFactor = exp:*:1.0i:{pump_axis[step]};",
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
                        "RestartMappingTvs=[0, -1];",
                        "RestartMapStages=0;",
                        "string P0 =|P0>;",
                        f'string P1= "c\'[{args.center_site}]*|P0>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression";',
                        'Version = "stc_vs_ttc_insert";',
                        f'OutputFile = "{run_name}_insert";',
                        f'RestartFilename = "{restart_filename_insert}";',
                        "GsWeight = 0.2;",
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

    finite_loops = (
        len(time_axis) * (args.TSPAdvanceEach // (args.sites - 2)) - 1
    )  # Total number of finite loops

    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
    )

    AversusT_table = "\n".join(
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
                        f"potentialV = [{args.potential_V}, ...];",
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
                        "RestartMappingTvs=[0, 1, -1, -1];",
                        'string P0="|P0>";',
                        'string P1="|P1>";',
                        f'string P2="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P0>";',
                        f'string P3="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P1>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,minimizedisk";',
                        'Version = "stc_vs_ttc_insert_evolve";',
                        f'OutputFile = "{run_name}_evolve";',
                        f'RestartFilename = "{restart_filename_evolve}";',
                        "GsWeight = 0.2;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def parse_args_slurm() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Parse arguments for the Slurm script.",
    )
    parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
    # parser.add_argument("--gpu", action="store_true", help="Use GPU.")
    return parser.parse_args()


def main() -> int:  # type: ignore

    args = parse_args_input()

    """
    Stage 1
    - Create reference states for each t^prime!
    - uses recoveryenableread.
    - recoveryenable read index 0 is for the first timestep = TspTau
    """

    # Create Reference states for each t^prime

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    restart_path = Path(args.gs_filename).resolve()
    run_folder_td = restart_path.parent / "td"
    run_folder_td.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_td}")

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

    input_path = run_folder_td / f"input_{run_name}.ain"
    input_path.write_text(build_input(args, run_name), encoding="utf-8")
    print(f"Wrote Ainur input: {input_path}")

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
    - Create folders for each t^prime.
    - Apply operator to create the initial state for time evolution.
    - Inputs to timeevolve to T_max.
    """
    run_folder_twotime = restart_path.parent / "twotime"
    run_folder_twotime.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_twotime}")

    for step in range(args.Pump_time_steps):
        step_folder = run_folder_twotime / f"step_{step:04d}"
        step_folder.mkdir(exist_ok=True)
        # shutil.copy2(restart_path, step_folder / restart_path.name)
        # print(f"Created step folder: {step_folder} and copied restart file.")

        executable_path = DMRG_EXECUTABLES[args.cluster]
        print(
            f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}"
        )
        shutil.copy2(executable_path, step_folder / "dmrg")

        input_path_restart = step_folder / f"input_{run_name}.ain"

        if step != 0:
            recoveryindex = int(step - 1)
            restart_filename = f"../../td/Recovery{recoveryindex}{run_name}"
        else:
            restart_filename = f"../../{Path(args.gs_filename).resolve().name}"

        input_path_restart.write_text(
            build_input_insert(
                args,
                run_name,
                step=step,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_insert=restart_filename,
            ),
            encoding="utf-8",
        )

        input_path_evolve = step_folder / f"input_{run_name}_evolve.ain"

        input_path_evolve.write_text(
            build_input_evolve(
                args,
                run_name,
                step=step,
                time_axis=time_axis[step:],
                pump_axis=pump_axis[step:],
                restart_filename_evolve=run_name + "_insert",
            ),
            encoding="utf-8",
        )

        print(f"Wrote Ainur input: {input_path_restart}")
        print(f"Wrote Ainur input: {input_path_evolve}")

        if args.run:
            subprocess.run(
                ["./dmrg", "-f", str(input_path_restart), "-p", "12"],
                cwd=step_folder,
                check=True,
            )
            subprocess.run(
                ["./dmrg", "-f", str(input_path_evolve), "-p", "12"],
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


############ HELPERS FOR PUMP TABLE GENERATION ############


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
