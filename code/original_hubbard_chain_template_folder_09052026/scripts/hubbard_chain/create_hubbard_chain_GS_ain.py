"""
Generate and optionally run a DMRG++ Hubbard-chain ground state.
Always run this above the scripts dmrgpp folder
"""

import sys
from pathlib import Path

# Add parent directory to path to enable imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # Up one folder
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import my local package.
import conda_dmrg  # type: ignore

# Default path to the DMRG++ executable (update as needed for your system)
DMRG_EXECUTABLE_SOURCE_local = Path(
    "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
)
DMRG_EXECUTABLE_SOURCE_isaac = Path(
    "/nfs/home/jthom214/dmrgpp/programs_08192026/dmrgpp/installdir/bin/dmrg"
)
DMRG_EXECUTABLE_SOURCE_nersc_gpu = Path(
    "/global/common/software/m5228/dmrgpp/builddir-cuda/dmrg/dmrg"
)
DMRG_EXECUTABLE_SOURCE_nersc_cpu = Path(
    "/global/common/software/m5228/dmrgpp_cpu/installdir/bin/dmrg"
)


def main():
    """
    Collect model parameters and write an Ainur input file.
    """

    print("Generate an Ainur 1.0 input file for Hubbard-chain ground state")
    print(
        "Values in brackets are the default values. Press Enter to accept the default."
    )

    write_ainur_header = "##Ainur1.0"

    """
    Geometry / model
    """
    print("=== Chain geometry and Hamiltonian ===")

    model = conda_dmrg.ask_text("Model (Model)", "HubbardOneBand")
    total_sites = conda_dmrg.ask_int("Chain length N (TotalNumberOfSites)", 4)
    number_of_terms = conda_dmrg.ask_int("Number of hopping terms (NumberOfTerms)", 1)
    degrees_of_freedom = conda_dmrg.ask_int("Hopping directions (DegreesOfFreedom)", 1)
    geometry_kind = conda_dmrg.ask_text("Geometry type (GeometryKind)", "chain")
    geometry_options = conda_dmrg.ask_text(
        "Geometry options (GeometryOptions)", "ConstantValues"
    )
    connector = conda_dmrg.ask_float("Hopping t, signed (dir0:Connectors)", -1.0)
    hubbard_u = conda_dmrg.ask_float("Hubbard interaction U (hubbardU)", 8.0)
    potential_v = conda_dmrg.ask_float("Onsite potential V (potentialV)", 0.0)

    write_model = "\n".join(
        (
            "# --- Model parameters ---",
            f"TotalNumberOfSites = {total_sites};",
            f"NumberOfTerms      = {number_of_terms};",
            f"DegreesOfFreedom   = {degrees_of_freedom};",
            f'GeometryKind       = "{geometry_kind}";',
            f'GeometryOptions    = "{geometry_options}";',
            f"dir0:Connectors    = [{conda_dmrg.format_number(connector)}];",
            f"hubbardU           = [{hubbard_u}, ...];",
            f"potentialV         = [{potential_v}, ...];",
            f'Model              = "{model}";',
        )
    )

    """
    Fock space parameters
    """
    print("\n=== Fock Space ===")

    n_half = int(total_sites / 2)
    target_up = conda_dmrg.ask_int("Up electrons (TargetElectronsUp)", n_half)
    target_down = conda_dmrg.ask_int("Down electrons (TargetElectronsDown)", n_half)

    # Few checks to ensure the user input is valid
    assert target_up >= 0, "TargetElectronsUp must be non-negative"
    assert target_down >= 0, "TargetElectronsDown must be non-negative"
    assert target_up + target_down <= 2 * total_sites, (
        "Total electrons cannot exceed total sites"
    )

    write_fock_space = "\n".join(
        (
            "# --- Fock Space parameters --- #",
            f"TargetElectronsUp   = {target_up};",
            f"TargetElectronsDown = {target_down};",
        )
    )

    """
    DMRG control parameters
    """
    print("=== DMRG control parameters ===")

    infinite_loop_states = conda_dmrg.ask_int(
        "Infinite-loop kept states (InfiniteLoopKeptStates)", 128
    )

    if total_sites % 2 == 1:
        finite_loop_count = conda_dmrg.ask_int(
            "Number of finite loops (Keep even for odd sites)", 6
        )
    else:
        finite_loop_count = conda_dmrg.ask_int(
            "Number of finite loops (Keep odd for even sites)", 5
        )

    finite_loop_states = conda_dmrg.ask_int(
        "Finite-loop kept states (FiniteLoopKeptStates)", 1000
    )
    truncation_tolerance = conda_dmrg.ask_float(
        "Truncation tolerance (TruncationTolerance)", 1e-9
    )

    finite_loop_rows = conda_dmrg.create_finite_loop_rows(
        finite_loop_count,
        finite_loop_states,
    )

    write_dmrg = "\n".join(
        (
            "# --- DMRG++ control parameters --- #",
            f"InfiniteLoopKeptStates = {infinite_loop_states};",
            f"TruncationTolerance = {truncation_tolerance};",
            "FiniteLoops = [",
            finite_loop_rows,
            "];",
        )
    )

    """
    Solver/Output parameters
    """
    print("\n=== Solver Options ===")

    run_name = conda_dmrg.create_run_name(
        total_sites,
        target_up,
        target_down,
        model,
        geometry_kind,
        code="dmrgpp",  # Specify the code for run name generation
    )

    output_file = run_name

    version = conda_dmrg.ask_text("DMRG++ version/profile (Version)", "Pierls")

    if total_sites % 2 == 1:
        solver_options = conda_dmrg.ask_text(
            "Solver options, comma-separated (SolverOptions)",
            "twositedmrg,usecomplex,geometryallinsystem",
        )
    else:
        solver_options = conda_dmrg.ask_text(
            "Solver options, comma-separated (SolverOptions)",
            "twositedmrg,usecomplex",
        )

    use_gpu = conda_dmrg.ask_yes_no("Use GPU ?", default=False)

    if use_gpu:
        solver_options += ",BatchedGemm"

    write_solver = "\n".join(
        (
            "# --- Solver / run control -- #",
            f'SolverOptions   = "{solver_options}";',
            f'Version         = "{version}";',
            f'OutputFile      = "{output_file}";',
        )
    )

    """
    Bring everything together and write the Ainur input file
    """
    write_all = (
        write_ainur_header,
        write_model,
        write_fock_space,
        write_dmrg,
        write_solver,
    )

    print("\n=== Run/output options ===")

    cluster = conda_dmrg.ask_text("Cluster", "local").lower()

    if cluster == "local":
        DMRG_EXECUTABLE_SOURCE = DMRG_EXECUTABLE_SOURCE_local
    elif cluster == "isaac":
        DMRG_EXECUTABLE_SOURCE = DMRG_EXECUTABLE_SOURCE_isaac
    elif cluster == "nersc":
        DMRG_EXECUTABLE_SOURCE = (
            DMRG_EXECUTABLE_SOURCE_nersc_cpu
            if not use_gpu
            else DMRG_EXECUTABLE_SOURCE_nersc_gpu
        )
    else:
        raise ValueError(f"Unknown cluster: {cluster}")

    run_folder, executable_path = conda_dmrg.prepare_run_folder(
        run_name,
        DMRG_EXECUTABLE_SOURCE,
        executable_name="dmrg",
    )
    output_path = run_folder / f"input_{run_name}.ain"
    with open(output_path, "w", encoding="utf-8") as output:
        output.write("\n\n".join(write_all) + "\n")

    print(f"\nCreated run folder: {run_folder}")
    print(f"Copied DMRG++ executable: {executable_path}")
    print(f"Wrote Ainur input file: {output_path}")

    if not conda_dmrg.ask_yes_no("Run DMRG++ now?", default=False):
        print("Generating a batch script to run DMRG++ later...")

        if cluster == "isaac":
            body = f"""
module load boost
module load openblas
module load gcc/13.4.0
module load hdf5/1.13.1-gcc
module load openmpi
module load cmake/
module load gsl/2.7-gcc

./dmrg -f  input_{run_name}.ain
            """

            slurm_script_path = run_folder / f"batch_{run_name}.slurm"

            conda_dmrg.create_slurm_script(
                body, slurm_script_path, cluster, use_gpu=use_gpu
            )

            return

        elif cluster == "nersc":
            body = f"""
#!/bin/bash


./dmrg -f  input_{run_name}.ain
            """

            slurm_script_path = run_folder / f"batch_{run_name}.slurm"

            conda_dmrg.create_slurm_script(
                body, slurm_script_path, cluster, use_gpu=use_gpu
            )

            return
    else:
        conda_dmrg.run_dmrgpp(output_path, executable_path=executable_path)


if __name__ == "__main__":
    main()
