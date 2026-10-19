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
    parser.add_argument("t", type=float)
    parser.add_argument("U", type=float)
    parser.add_argument("potential_v", type=float)

    parser.add_argument(
        "restart_filename", type=str, help="Restart filename for the calculation"
    )

    parser.add_argument("finite_kept", type=int)
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
    finite_loops = args.Pump_time_steps * (
        args.TSPAdvanceEach // (args.sites - 2) - 1
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
                        f"potentialV = [{args.potential_v}, ...];",
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
                        f"TSPAdvanceEach = {args.TSPAdvanceEach};",
                        f"TSPTau = {args.TSPTau};",
                        'TSPAlgorithm = "Krylov";',
                        "TSPTimeSteps = 5;",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TimeStepTargeting,recoveryEnableRead";',
                        'Version = "stc_vs_ttc";',
                        f'string RecoverySave = "%l%%2,@keep,@M={args.Pump_time_steps}";',
                        f'OutputFile = "{run_name}";',
                        f'RestartFilename = "../{Path(args.restart_filename).resolve().name}";',
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
    parser.add_argument("--gpu", action="store_true", help="Use GPU.")
    return parser.parse_args()


# def build_slurm_script(
#     body: str, slurm_path: Path, cluster: str, use_gpu: bool = False
# ) -> None:

#     if cluster == "isaac":
#         slurm_header = (
#             "#!/bin/bash\n"
#             "#SBATCH --job-name=dmrg_job\n"
#             "#SBATCH --output=dmrg_output_%j.txt\n"
#             "#SBATCH --error=dmrg_error_%j.txt\n"
#             "#SBATCH --time=01:00:00\n"
#             "#SBATCH --partition=compute\n"
#             "#SBATCH --nodes=1\n"
#             "#SBATCH --ntasks-per-node=1\n"
#             "#SBATCH --natasks=12\n"
#         )
#     elif cluster == "nersc":
#         slurm_header = (
#             "#!/bin/bash\n"
#             "#SBATCH --job-name=dmrg_job\n"
#             "#SBATCH --output=dmrg_output_%j.txt\n"
#             "#SBATCH --error=dmrg_error_%j.txt\n"
#             "#SBATCH --time=01:00:00\n"
#             "#SBATCH --partition=regular\n"
#             "#SBATCH --nodes=1\n"
#             "#SBATCH --ntasks-per-node=1\n"
#             "#SBATCH --cpus-per-task=12\n"
#         )
#         if use_gpu:
#             slurm_header += "#SBATCH --gres=gpu:1\n"
#     else:
#         raise ValueError(f"Unsupported cluster: {cluster}")


def main() -> int:

    args = parse_args_input()

    # Create the run name based on the input parameters, and prepare the run folder and executable path.

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    restart_path = Path(args.restart_filename).resolve()
    run_folder = restart_path.parent / "td"
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder}")

    executable_path = DMRG_EXECUTABLES[args.cluster]
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder / "dmrg")

    # Create input file for Ainur and write it to the run folder. If --run is specified, execute DMRG++ with the generated input. If a cluster is specified, create a SLURM batch script for submission.

    input_path = run_folder / f"input_{run_name}.ain"
    input_path.write_text(build_input(args, run_name), encoding="utf-8")
    print(f"Wrote Ainur input: {input_path}")

    if args.run:
        subprocess.run(
            ["./dmrg", "-f", str(input_path), "-p", "12"],
            cwd=run_folder,
            check=True,
        )
    elif args.cluster != "local":
        # args_slurm = parse_args_slurm()
        slurm_path = run_folder / f"batch_{run_name}.slurm"
        # body = f"\n./dmrg -f input_{run_name}.ain\n"
        # build_slurm_script(body, slurm_path, args.cluster, use_gpu=args.gpu)
        print(f"Wrote batch script: {slurm_path}")
    else:
        print("Input generated. Use --run to start DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")


############ HELPERS FOR PUMP TABLE GENERATION ############


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
