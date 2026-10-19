"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Location of the current DMRG++ executables for different clusters. Adjust these paths as needed. The location is printed when the script is completed.
DMRG_EXECUTABLES = {
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    ),
    "nersc": Path("/global/common/software/m5228/dmrgpp/installdir/bin/dmrg"),
    # "nersc": Path(
    #     "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    # ),
}


# DMRG Settings
DMRG_PRECISION = 12

TWOTIME_OPERATOR_LABEL = "<P2|c'|P3>"

TWOTIME_COLLECTION_MARKER = "FiniteLoops printing ends"

TWOTIME_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

TWOTIME_OPERATOR_PATTERN = re.compile(
    rf"^\s*(\d+)\s+"
    rf"\(\s*({TWOTIME_NUMBER_PATTERN})\s*,\s*({TWOTIME_NUMBER_PATTERN})\s*\)\s+"
    rf"({TWOTIME_NUMBER_PATTERN})\s+"
    rf"{re.escape(TWOTIME_OPERATOR_LABEL)}"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
    )
    parser.add_argument("center_site", type=int, help="Center site index")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument("pump_file", type=Path, help="Path to the pump file")
    parser.add_argument(
        "process_folder",
        type=Path,
        help="Only process an existing twotime_RUN_NAME directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of output files to parse concurrently (default: CPU count).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def read_pump_file(pump_file_path: Path) -> tuple[list[float], list[float]]:
    """Read the pump file and return the list of pump values."""
    time = []
    pump = []

    with open(pump_file_path, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            time.append(float(row["time"]))
            pump.append(float(row["pump"]))

    return time, pump


def save_arguments(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments as JSON."""
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)


def find_evolve_output(step_folder: Path, run_name: str) -> Path:
    """Find the DMRG++ evolve output file for one step."""
    candidates = (step_folder / f"runForinput_{run_name}_evolve.cout",)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No evolve output found in {step_folder}")


def read_operator_output(output_path: Path) -> dict[tuple[int, float], complex]:
    """Read operator data after ``FiniteLoops printing ends``.

    A dictionary avoids creating an intermediate NumPy array only to iterate over
    it later.  Its key also preserves the original "last occurrence wins" rule.
    """
    values = {}
    collecting = False

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if TWOTIME_COLLECTION_MARKER in line:
                collecting = True
                continue
            if not collecting or TWOTIME_OPERATOR_LABEL not in line:
                continue

            match = TWOTIME_OPERATOR_PATTERN.search(line)
            if match is None:
                continue

            site_index = int(match.group(1))
            time = float(match.group(4))
            values[(site_index, time)] = complex(
                float(match.group(2)), float(match.group(3))
            )

    return values


def collect_twotime_data(
    twotime_folder: Path,
    run_name: str,
    number_of_steps: int,
    time_axis: list[float],
    center_site: int,
    workers: int,
) -> dict[tuple[float, float, int, int], complex]:
    """
    Return:
        {
            (t_prime, t, center_site, site): complex_green_value
        }

    Repeated keys retain the last value.
    """
    # Locate files before starting workers so missing-output errors remain clear.
    output_paths = [
        find_evolve_output(twotime_folder / f"step_{step:04d}", run_name)
        for step in range(number_of_steps)
    ]
    data = {}
    if not output_paths:
        return data

    # Parsing DMRG output is CPU-heavy (regular expressions and float conversion),
    # so processes bypass the GIL. executor.map returns results in step order,
    # retaining the original behavior if time_axis contains duplicate values.
    max_workers = min(workers, number_of_steps)
    if max_workers == 1:
        parsed_steps = map(read_operator_output, output_paths)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        chunk_size = max(1, number_of_steps // (max_workers * 4))
        parsed_steps = executor.map(
            read_operator_output, output_paths, chunksize=chunk_size
        )

    try:
        for step, step_data in enumerate(parsed_steps):
            t_prime = float(time_axis[step])
            for (site, time), value in step_data.items():
                data[(t_prime, time, center_site, site)] = value
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    return data


def save_twotime_csv(
    data: dict[tuple[float, float, int, int], complex],
    output_path: Path,
) -> None:
    """
    Save two-time Green's-function data to CSV.

    Dictionary format:
        {
            (t_prime, t, center_site, site): complex_value
        }
    """
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "t_prime",
                "t",
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
    time_axis: list[float],
    center_site: int,
    workers: int,
) -> Path:
    """Collect an existing run and write its combined CSV file."""
    twotime_folder = base_directory / f"{run_name}"

    if not twotime_folder.is_dir():
        raise FileNotFoundError(f"Two-time run folder not found: {twotime_folder}")
    if number_of_steps > len(time_axis):
        raise ValueError("number_of_steps cannot exceed the length of time_axis")

    data = collect_twotime_data(
        twotime_folder=twotime_folder,
        run_name=run_name.split("_", 1)[-1],  # Remove Gless from name,
        number_of_steps=number_of_steps,
        time_axis=time_axis,
        center_site=center_site,
        workers=workers,
    )

    csv_path = base_directory / f"{run_name}_P2_c_P3_centersite={center_site}.csv"

    save_twotime_csv(
        data=data,
        output_path=csv_path,
    )

    return csv_path


def main() -> int:
    """Generate inputs, optionally run DMRG++, and collect the results."""
    args = parse_args()

    restart_path = args.gs_filename.resolve()
    time_axis, pump_axis = read_pump_file(args.pump_file)

    csv_path = process_twotime_results(
        base_directory=restart_path.parent,
        run_name=args.process_folder.name,
        number_of_steps=len(time_axis),
        time_axis=time_axis,
        center_site=args.center_site,
        workers=args.workers,
    )
    print(f"Wrote post-processed data: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")