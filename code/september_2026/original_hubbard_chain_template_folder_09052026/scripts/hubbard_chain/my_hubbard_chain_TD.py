#!/usr/bin/env python3
"""Standalone positional-argument Hubbard-chain time-evolution input generator."""

import argparse
import math
import subprocess
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        "Order of positional arguments: sites up down t U potentialV"
        "Pump_A Pump_omega Pump_t0 Pump_sigma Pump_numberoftimesteps Pump_dt GS_weight"
        "TspTimeSteps TSPAdvanceEach FiniteLoopKeptStates FiniteLoopSave "
        "RepeatFiniteLoopsTimes TruncationTolerance Version SolverOptions Cluster",

        Example:
        python my_hubbard_chain_TD.py 32 16 16 -1 8 0 0.1 1.0 0.0 0.5 100 0.01 0.1 0.01 128 5 100
        local --run 

        """,
    )

    return parser.parse_args()


def build_input(args: argparse.Namespace, up: int, down: int, run_name: str) -> str:
    import conda_dmrg  # type: ignore

    # Create the table of pump values versus time.
    Pump_table = conda_dmrg.create_time_pump_axis(
        Pump_A,
        Pump_omega,
        Pump_t0,
        Pump_sigma,
        nsteps=Pump_numberoftimesteps,
        step_size=Pump_dt,
    )
    AversusTime = conda_dmrg.create_AversusTime_rows(Pump_table)

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
                        "# --- Pump parameters --- #",
                        "matrix AversusTime = [",
                        f"    {AversusTime}",
                        "];",
                        "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
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


if __name__ == "__main__":
    args = parse_args()
    main()
