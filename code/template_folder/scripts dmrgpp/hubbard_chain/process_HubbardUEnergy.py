"""
Obtain Simulations Times from a DMRG++ .cout File
This script processes a DMRG++ time-dependent .cout file to extract simulation times and observables, and writes them to a CSV file.
"""

from pathlib import Path

import conda_dmrg  # type: ignore

output_filename = "hubbardU_table.csv"

OBSERVE_EXECUTABLE_SOURCE = Path(
    "/Users/qqt/Documents/codes/dmrgpp/builddir/dmrg/observe"
)


def main():
    """
    TODO
    """

    print("Process a DMRG++ time-dependent .cout file\n")

    cout_filename = conda_dmrg.ask_path("Input DMRG++ .cout filename")
    if not cout_filename:
        print("No input filename provided.")
        return

    try:
        cout_path = Path(cout_filename).expanduser().resolve()
        required_columns, data = conda_dmrg.extract_simulation_times(
            cout_path, "dmrgpp"
        )
        conda_dmrg.write_to_csv(
            data=data,
            column_names=required_columns,
            output_csv=cout_path.parent / output_filename,
        )
        output_path = cout_path.parent / output_filename
        print(f"Wrote observables table: {output_path}")
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Could not process .cout file: {error}")
        return


if __name__ == "__main__":
    main()
