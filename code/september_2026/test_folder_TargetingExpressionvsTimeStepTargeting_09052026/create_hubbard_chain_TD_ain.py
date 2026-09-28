"""
Generate a simple Ainur 1.0 input file for a DMRG++ Hubbard chain time evolution.
Always run this from the topmost level (above src).
"""

import math
import sys
from pathlib import Path

# Add parent directory to path to enable imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # Up to redo_template_folder_code_08112026
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import conda_dmrg  # type: ignore

# Default path to the DMRG++ executable (update as needed for your system)
DMRG_EXECUTABLE_SOURCE_local = Path(
    "/Users/qqt/Documents/Codes/dmrgpp/installdir/bin/dmrg"
)
DMRG_EXECUTABLE_SOURCE_isaac = Path(
    "/nfs/home/jthom214/dmrgpp/programs_08192026/dmrgpp/installdir/bin/dmrg"
)
DMRG_EXECUTABLE_SOURCE_nersc_gpu = Path(
    "/global/common/software/m5228/dmrgpp/builddir/dmrg/dmrg"
)
DMRG_EXECUTABLE_SOURCE_nersc_cpu = Path(
    "/global/common/software/m5228/dmrgpp/builddir/dmrg/dmrg"
)


def main():
    """
    Collect model parameters and write an Ainur input file.
    """

    print("Generate an Ainur 1.0 input file for time evolution of a Hubbard chain.")
    print("Values in brackets are the default values.")

    write_ainur_header = "##Ainur1.0"

    """
    Define regex patterns for extracting parameters from an inherited Ainur input file.
    """
    patterns = {
        "total_sites": r"^\s*TotalNumberOfSites\s*=\s*(\d+)\s*;",
        "number_of_terms": r"^\s*NumberOfTerms\s*=\s*(\d+)\s*;",
        "degrees_of_freedom": r"^\s*DegreesOfFreedom\s*=\s*(\d+)\s*;",
        "geometry_kind": r"^\s*GeometryKind\s*=\s*\"([^\"]+)\"\s*;?",
        "geometry_options": r"^\s*GeometryOptions\s*=\s*\"([^\"]+)\"\s*;?",
        "connector": r"^\s*dir0:Connectors\s*=\s*\[\s*([-+0-9.eE]+)",
        "model": r"^\s*Model\s*=\s*\"([^\"]+)\"\s*;?",
        "hubbard_u": r"^\s*hubbardU\s*=\s*\[\s*([-+0-9.eE]+)",
        "potential_v": r"^\s*potentialV\s*=\s*\[\s*([-+0-9.eE]+)",
        "target_up": r"^\s*TargetElectronsUp\s*=\s*(\d+)\s*;?",
        "target_down": r"^\s*TargetElectronsDown\s*=\s*(\d+)\s*;?",
    }

    integer_fields = {
        "total_sites",
        "number_of_terms",
        "degrees_of_freedom",
        "target_up",
        "target_down",
    }

    text_fields = {"geometry_kind", "geometry_options", "model"}

    inherited, inherited_input_path = conda_dmrg.ask_inherited_parameters(
        return_path=True,
        patterns=patterns,
        integer_fields=integer_fields,
        text_fields=text_fields,
    )

    """
    Geometry / model
    """
    print("=== Chain geometry and Hamiltonian ===")

    model = conda_dmrg.inherited_or_prompt(
        inherited, "model", "Model (Model)", "HubbardOneBand", conda_dmrg.ask_text
    )
    total_sites = conda_dmrg.inherited_or_prompt(
        inherited,
        "total_sites",
        "Chain length N (TotalNumberOfSites)",
        4,
        conda_dmrg.ask_int,
    )
    number_of_terms = conda_dmrg.inherited_or_prompt(
        inherited,
        "number_of_terms",
        "Number of hopping terms (NumberOfTerms)",
        1,
        conda_dmrg.ask_int,
    )
    degrees_of_freedom = conda_dmrg.inherited_or_prompt(
        inherited,
        "degrees_of_freedom",
        "Hopping directions (DegreesOfFreedom)",
        1,
        conda_dmrg.ask_int,
    )
    geometry_kind = conda_dmrg.inherited_or_prompt(
        inherited,
        "geometry_kind",
        "Geometry type (GeometryKind)",
        "chain",
        conda_dmrg.ask_text,
    )
    geometry_options = conda_dmrg.inherited_or_prompt(
        inherited,
        "geometry_options",
        "Geometry options (GeometryOptions)",
        "ConstantValues",
        conda_dmrg.ask_text,
    )
    connector = conda_dmrg.inherited_or_prompt(
        inherited,
        "connector",
        "Hopping t, signed (dir0:Connectors)",
        -1.0,
        conda_dmrg.ask_float,
    )
    hubbard_u = conda_dmrg.inherited_or_prompt(
        inherited,
        "hubbard_u",
        "Hubbard interaction U (hubbardU)",
        8.0,
        conda_dmrg.ask_float,
    )
    potential_v = conda_dmrg.inherited_or_prompt(
        inherited,
        "potential_v",
        "Onsite potential V (potentialV)",
        0.0,
        conda_dmrg.ask_float,
    )

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

    target_up = conda_dmrg.inherited_or_prompt(
        inherited,
        "target_up",
        "Up electrons (TargetElectronsUp)",
        n_half,
        conda_dmrg.ask_int,
    )
    assert target_up >= 0, "TargetElectronsUp must be non-negative"
    target_down = conda_dmrg.inherited_or_prompt(
        inherited,
        "target_down",
        "Down electrons (TargetElectronsDown)",
        n_half,
        conda_dmrg.ask_int,
    )
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
    Pump and time parameters
    """
    print("\n=== Pump and time grid ===")

    Pump_A = conda_dmrg.ask_float("Pump amplitude A (Pump_A)", 1.0)
    Pump_omega = conda_dmrg.ask_float("Pump frequency Omega (Pump_omega)", 4.0)
    assert Pump_omega > 0, "Pump frequency must be positive"

    Pump_t0 = conda_dmrg.ask_float("Pump center time t0 (Pump_t0)", 20.0)
    Pump_sigma = conda_dmrg.ask_float("Pump width sigma (Pump_sigma)", 4.0)
    assert Pump_sigma > 0, "Pump sigma must be positive"

    Pump_numberoftimesteps = conda_dmrg.ask_int(
        "Number of physical time steps, including t=0 (Pump_numberoftimesteps)", 1001
    )
    Pump_dt = conda_dmrg.ask_float("Physical time step dt (Pump_dt)", 0.1)
    assert Pump_dt > 0, "Physical time step must be positive"

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

    write_pump = "\n".join(  # noqa: FLY002
        (
            "# --- Pump parameters --- #",
            "matrix AversusTime = [",
            AversusTime,
            "];",
            "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
        )
    )

    """
    Solver/Output parameters
    """
    print("\n=== Solver Options ===")

    # Get the restart file from the inherited input path if available, otherwise prompt the user.
    restart_file, restart_checkpoint_path, restart_stem = (
        conda_dmrg.resolve_restart_info(inherited_input_path)
    )

    run_name = (
        f"{restart_stem}_td".lower()
        if inherited_input_path is not None
        else conda_dmrg.create_run_name(
            total_sites,
            target_up,
            target_down,
            model,
            geometry_kind,
            code="dmrgpp",
        )
    )

    version = conda_dmrg.ask_text("DMRG++ version/profile (Version)", "Pierls")
    GS_weight = conda_dmrg.ask_float("Ground-state weight (GS_weight)", 0.2)

    if total_sites % 2 == 1:
        solver_options = conda_dmrg.ask_text(
            "Solver options, comma-separated (SolverOptions)",
            "twositedmrg,TimeStepTargeting,restart,usecomplex,geometryallinsystem",
        )
    else:
        solver_options = conda_dmrg.ask_text(
            "Solver options, comma-separated (SolverOptions)",
            "twositedmrg,TimeStepTargeting,restart,usecomplex",
        )

    use_gpu = conda_dmrg.ask_yes_no("Use GPU ?", default=False)

    if use_gpu:
        solver_options += ",BatchedGemm"

    output_file = run_name

    write_solver = "\n".join(
        (
            "# --- Solver / run control -- #",
            f'SolverOptions   = "{solver_options}";',
            f'Version         = "{version}";',
            f'OutputFile      = "{output_file}";',
            f'RestartFilename = "{restart_file}";',
            f"GsWeight        = {GS_weight};",
        )
    )

    """
    DMRG control parameters
    """
    print("=== DMRG control parameters (have internal overlaps with pump) ===")

    TSPAlgorithm = conda_dmrg.ask_text(
        "Time-stepping algorithm (TSPAlgorithm)", "Krylov"
    )
    Tsptau = Pump_dt  # DMRG++ internal time step size must match physical time step
    TspTimeSteps = conda_dmrg.ask_int(
        "DMRG++ internal substeps per dt (TspTimeSteps)", 5
    )

    if (
        total_sites % 2 == 1
    ):  # Odd sites require TSPAdvanceEach to be a multiple of (TotalNumberOfSites - 1)
        TSPAdvanceEach = conda_dmrg.ask_int(
            "Advance after how many sweeps (TSPAdvanceEach, multiple of Sites-1)",
            int(total_sites - 1),
        )

        sweeps_per_timestep = TSPAdvanceEach / (total_sites - 1)

    else:
        TSPAdvanceEach = conda_dmrg.ask_int(
            "Advance after how many sweeps (TSPAdvanceEach, multiple of Sites-2)",
            int(total_sites - 2),
        )

        sweeps_per_timestep = TSPAdvanceEach / (total_sites - 2)

    finite_loop_states = conda_dmrg.ask_int(
        "Finite-loop kept states (FiniteLoopKeptStates)", 1000
    )
    finite_loop_save = conda_dmrg.ask_text(
        "Finite-loop save setting (FiniteLoopSave)", "3"
    )
    finite_loop_rows = conda_dmrg.create_restart_finite_loop_rows(
        restart_checkpoint_path,
        total_sites,
        finite_loop_states,
        finite_loop_save=finite_loop_save,
        use_autoflag=True,  # Set to False to avoid automatic adjustments
    )

    initial_finite_loop_count = sum(
        1 for line in finite_loop_rows.splitlines() if line.strip().startswith("[")
    )

    # This is the logic to get the number of times to repeat the finite loops to match the number of physical time steps.
    min_RepeatFiniteLoopsTimes = int(
        (Pump_numberoftimesteps - 1) // initial_finite_loop_count * sweeps_per_timestep
    )

    RepeatFiniteLoopsTimes = conda_dmrg.ask_int(
        "Repeat finite loops times (RepeatFiniteLoopsTimes)", min_RepeatFiniteLoopsTimes
    )

    assert (
        # The first time step is t=0, so we subtract the number of sweeps for t=0
        RepeatFiniteLoopsTimes * initial_finite_loop_count
    ) * sweeps_per_timestep * Tsptau == Pump_table[-1][0], (
        print(
            f"Computed last time value: "
            f"{(RepeatFiniteLoopsTimes * initial_finite_loop_count) * sweeps_per_timestep * Tsptau}, "
            f"Expected last time value: {Pump_table[-1][0]}"
        ),
    )

    truncation_tolerance = conda_dmrg.ask_float(
        "Truncation tolerance (TruncationTolerance)", 1e-9
    )

    write_dmrg = "\n".join(
        (
            "# --- DMRG++ control parameters --- #",
            f"TruncationTolerance = {truncation_tolerance};",
            "FiniteLoops = [",
            finite_loop_rows,
            "];",
            f"RepeatFiniteLoopsTimes = {RepeatFiniteLoopsTimes};",
            f"TSPTau = {Tsptau};",
            f"TSPTimeSteps = {TspTimeSteps};",
            f"TSPAdvanceEach = {TSPAdvanceEach};",
            f"TSPAlgorithm = {TSPAlgorithm};",
        )
    )

    """
    Bring everything together and write the Ainur input file
    """
    write_all = (
        write_ainur_header,
        write_model,
        write_pump,
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

    if inherited_input_path is not None:
        run_folder, executable_path = conda_dmrg.prepare_run_folder(
            "td",
            executable_source=DMRG_EXECUTABLE_SOURCE,
            root=inherited_input_path.parent,
            executable_name="dmrg",
        )
    else:
        run_folder, executable_path = conda_dmrg.prepare_run_folder(
            run_name,
            executable_source=DMRG_EXECUTABLE_SOURCE,
            executable_name="dmrg",
        )

    output_path = (
        run_folder / f"input_{run_name}.ain"
        if run_folder is not None
        else Path(f"input_{run_name}.ain")
    )
    with open(output_path, "w", encoding="utf-8") as output:
        output.write("\n\n".join(write_all) + "\n")

    PUMP_OUTPUT_FILENAME = "pump_dmrgpp.csv"

    if run_folder is not None:
        pump_output_path = run_folder / PUMP_OUTPUT_FILENAME
        conda_dmrg.write_pump_table(Pump_table, pump_output_path)
        print(f"\nCreated TD run folder: {run_folder}")
        print(f"Copied DMRG++ executable: {executable_path}")
        print(f"Wrote pump table: {pump_output_path}")
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
