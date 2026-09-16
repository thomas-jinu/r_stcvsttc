#!/usr/bin/env python3
"""Generate and optionally run a DMRG++ Hubbard-chain ground state.

Run this script from above the directory containing the ``dmrgpp`` scripts,
or adjust the project-root discovery below if your package layout differs.
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Order of positional arguments: sites up down t u potential infinite_kept finite_loops finite_kept version cluster

        Example:
        python make_hubbard.py 32 16 16 -1 8 0 128 5 1000 local --run
        """,
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int)
    parser.add_argument("down", type=int)
    parser.add_argument("t", type=float)
    parser.add_argument("u", type=float)
    parser.add_argument("potential", type=float)
    parser.add_argument("infinite_kept", type=int)
    parser.add_argument("finite_loops", type=int)
    parser.add_argument("finite_kept", type=int)
    parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating the input."
    )
    return parser.parse_args()


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
                        f"hubbardU = [{args.u}, ...];",
                        f"potentialV = [{args.potential}, ...];",
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
                        "SolverOptions = twositedmrg,usecomplex,BatchedGemm",
                        "Version = stc_vs_ttc;",
                        f'OutputFile = "{run_name}";',
                    ]
                ),
            ]
        )
        + "\n"
    )


def main() -> int:
    args = parse_args()
    import conda_dmrg  # type: ignore

    up = args.up
    down = args.down

    run_name = conda_dmrg.create_run_name(
        args.sites, up, down, "HubbardOneBand", "chain", code="dmrgpp"
    )

    run_folder, executable_path = conda_dmrg.prepare_run_folder(
        run_name, DMRG_EXECUTABLES[args.cluster], executable_name="dmrg"
    )

    input_path = run_folder / f"input_{run_name}.ain"
    input_path.write_text(build_input(args, up, down, run_name), encoding="utf-8")
    print(f"Created run folder: {run_folder}")
    print(f"Wrote Ainur input: {input_path}")

    if args.run:
        conda_dmrg.run_dmrgpp(input_path, executable_path=executable_path)
    elif args.cluster != "local":
        slurm_path = run_folder / f"batch_{run_name}.slurm"
        body = f"\n./dmrg -f input_{run_name}.ain\n"
        conda_dmrg.create_slurm_script(body, slurm_path, args.cluster, use_gpu=args.gpu)
        print(f"Wrote batch script: {slurm_path}")
    else:
        print("Input generated. Use --run to start DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
