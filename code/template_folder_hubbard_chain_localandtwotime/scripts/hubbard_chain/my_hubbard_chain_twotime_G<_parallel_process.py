"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

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
    return parser.parse_args()


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


def read_operator_output(output_path: Path) -> np.ndarray:
    """Read operator data after ``FiniteLoops printing ends``."""
    rows_by_key = {}
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
            real_part = float(match.group(2))
            imaginary_part = float(match.group(3))
            time = float(match.group(4))

            # If the same site/time appears again, keep the last value.
            rows_by_key[(site_index, time)] = [
                site_index,
                real_part,
                imaginary_part,
                time,
            ]

    if not rows_by_key:
        return np.empty((0, 4), dtype=float)

    return np.asarray(list(rows_by_key.values()), dtype=float)


def collect_twotime_data(
    twotime_folder: Path,
    run_name: str,
    number_of_steps: int,
    time_axis: list[float],
    center_site: int,
) -> dict[tuple[float, float, int, int], complex]:
    """
    Return:
        {
            (t_prime, t, center_site, site): complex_green_value
        }

    Repeated keys retain the last value.
    """
    data = {}

    for step in range(number_of_steps):
        step_folder = twotime_folder / f"step_{step:04d}"
        output_path = find_evolve_output(step_folder, run_name)
        step_data = read_operator_output(output_path)

        t_prime = float(time_axis[step])

        for site, real_part, imaginary_part, t in step_data:
            key = (
                t_prime,
                float(t),
                int(center_site),
                int(site),
            )

            data[key] = complex(
                real_part,
                imaginary_part,
            )

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
    time_axis,
    center_site: int,
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
    )
    print(f"Wrote post-processed data: {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
