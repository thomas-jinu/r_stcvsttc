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
    "local": Path("/Users/qqt/Documents/Codes/dmrgpp/installdir/bin/dmrg"),
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
Input summary:
  All value-bearing options are required.
  --sites             Chain length N
  --up                Target up-spin electrons
  --down              Target down-spin electrons
  --t                 Signed hopping amplitude
  --u                 Hubbard interaction U
  --potential         Onsite potential V
  --infinite-kept     Infinite-loop kept states
  --finite-loops      Number of finite loops
  --finite-kept       Finite-loop kept states
  --tolerance         Truncation tolerance
  --cluster           Execution target: local, isaac, or nersc
  --gpu               Use the NERSC GPU executable
  --run               Run DMRG++ immediately after generating the input

Example:
  python make_hubbard.py --sites 32 --up 16 --down 16 --u 8 --run

Generate only an input file:
  python make_hubbard.py --sites 32 --up 16 --down 16 --u 8
""",
    )
    parser.add_argument("--sites", type=int, required=True, help="Chain length N.")
    parser.add_argument("--terms", type=int, required=True)
    parser.add_argument("--dof", type=int, required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--geometry-options", required=True)
    parser.add_argument("--t", type=float, required=True, help="Signed hopping.")
    parser.add_argument("--u", type=float, required=True, help="Hubbard interaction.")
    parser.add_argument("--potential", type=float, required=True)
    parser.add_argument(
        "--up", type=int, required=True, help="Target up-spin electrons."
    )
    parser.add_argument(
        "--down", type=int, required=True, help="Target down-spin electrons."
    )
    parser.add_argument("--infinite-kept", type=int, required=True)
    parser.add_argument("--finite-loops", type=int, required=True)
    parser.add_argument("--finite-kept", type=int, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--solver-options", required=True)
    parser.add_argument("--cluster", choices=["local", "isaac", "nersc"], required=True)
    parser.add_argument(
        "--gpu", action="store_true", help="Use the NERSC GPU executable."
    )
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating the input."
    )
    return parser.parse_args()


def validate(args: argparse.Namespace, up: int, down: int) -> None:
    if args.sites < 1:
        raise ValueError("--sites must be positive")
    if min(args.terms, args.dof, args.infinite_kept, args.finite_kept) < 1:
        raise ValueError("terms, dof, and kept-state values must be positive")
    if args.finite_loops is not None and args.finite_loops < 1:
        raise ValueError("--finite-loops must be positive")
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be positive")
    if up < 0 or down < 0:
        raise ValueError("electron counts must be non-negative")
    if up + down > 2 * args.sites:
        raise ValueError("total electrons cannot exceed two per site")
    if args.gpu and args.cluster != "nersc":
        raise ValueError("--gpu is currently supported only with --cluster nersc")


def build_input(args: argparse.Namespace, up: int, down: int, run_name: str) -> str:
    import conda_dmrg  # type: ignore

    loops = args.finite_loops
    solver_options = args.solver_options
    if args.gpu and "BatchedGemm" not in solver_options:
        solver_options += ",BatchedGemm"
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
                        'GeometryKind = chain";',
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
                        f"TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        f'SolverOptions = "{solver_options}";',
                        f'Version = "{args.version}";',
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
    validate(args, up, down)

    run_name = conda_dmrg.create_run_name(
        args.sites, up, down, "HubbardOneBand", args.geometry, code="dmrgpp"
    )
    executable_key = (
        "nersc-gpu"
        if args.gpu
        else ("nersc-cpu" if args.cluster == "nersc" else args.cluster)
    )
    run_folder, executable_path = conda_dmrg.prepare_run_folder(
        run_name, DMRG_EXECUTABLES[executable_key], executable_name="dmrg"
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
