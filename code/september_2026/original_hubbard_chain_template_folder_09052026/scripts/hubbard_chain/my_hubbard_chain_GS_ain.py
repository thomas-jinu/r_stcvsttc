"""Generate and optionally run a DMRG++ Hubbard-chain ground state.

Run this script from above the directory containing the ``dmrgpp`` scripts,
or adjust the project-root discovery below if your package layout differs.
"""

import argparse
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
        Order of positional arguments: sites up down t U potentialV infinite_kept finite_loops finite_kept version cluster

        Example:
        python make_hubbard.py 32 16 16 -1 8 0 128 5 1000 local --run
        """,
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int)
    parser.add_argument("down", type=int)
    parser.add_argument("t", type=float)
    parser.add_argument("U", type=float)
    parser.add_argument("potentialV", type=float)
    parser.add_argument("infinite_kept", type=int)
    parser.add_argument("finite_loops", type=int)
    parser.add_argument("finite_kept", type=int)
    parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating the input."
    )
    return parser.parse_args()


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


def build_input(args: argparse.Namespace, up: int, down: int, run_name: str) -> str:
    import conda_dmrg  # type: ignore

    loops = args.finite_loops
    finite_rows = conda_dmrg.create_finite_loop_rows(loops, args.finite_kept)
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
                        f"dir0:Connectors = [{conda_dmrg.format_number(args.t)}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {up};",
                        f"TargetElectronsDown = {down};",
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        f"InfiniteLoopKeptStates = {args.infinite_kept};",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        "SolverOptions = twositedmrg,usecomplex",
                        "Version = stc_vs_ttc;",
                        f'OutputFile = "{run_name}";',
                    ]
                ),
            ]
        )
        + "\n"
    )


def main() -> int:
    args = parse_args_input()

    # Create the run name based on the input parameters, and prepare the run folder and executable path.

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    run_folder = PROJECT_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    print(f"Created run folder: {run_folder}")

    executable_path = DMRG_EXECUTABLES[args.cluster]
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder / "dmrg")

    # Create input file for Ainur and write it to the run folder. If --run is specified, execute DMRG++ with the generated input. If a cluster is specified, create a SLURM batch script for submission.

    input_path = run_folder / f"input_{run_name}.ain"
    input_path.write_text(
        build_input(args, args.up, args.down, run_name), encoding="utf-8"
    )
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
