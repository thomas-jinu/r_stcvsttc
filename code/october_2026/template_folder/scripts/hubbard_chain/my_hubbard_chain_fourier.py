"""Direct Fourier transforms for DMRG++ two-time Green's functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "step",
    "site_index",
    "real_part",
    "imaginary_part",
    "time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct Fourier transform of a Green's-function CSV file."
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument(
        "--mode",
        choices=("omega", "komega"),
        default="omega",
        help="Local frequency transform or momentum-frequency transform.",
    )
    parser.add_argument(
        "--center-site",
        type=int,
        help="Reference site; required if the file contains multiple sites.",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Step to use in komega mode; defaults to the first step.",
    )
    parser.add_argument("--omega-min", type=float, default=-10.0)
    parser.add_argument("--omega-max", type=float, default=10.0)
    parser.add_argument("--omega-points", type=int, default=1001)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Core-hole damping factor for omega mode.",
    )
    parser.add_argument(
        "--eta_lorentzian",
        type=float,
        default=0.1,
        help="Core-hole damping factor for omega mode.",
    )
    parser.add_argument(
        "--eta_gaussian",
        type=float,
        default=0.1,
        help="Gaussian damping factor for komega mode.",
    )
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def apply_dampening_corehole(
    green: np.ndarray,
    time: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply exp(-gamma*time) core-hole damping."""
    return green * np.exp(-gamma * abs(time))


def apply_dampening_gaussian(
    green: np.ndarray,
    time: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Apply exp(-eta*time^2) Gaussian damping along the time axis."""
    return green * np.exp(-eta * eta * time**2)[None, :]


def compute_kgrid_chain_pbc(number_of_sites: int) -> np.ndarray:
    """Return the periodic-chain momentum grid in [-pi, pi)."""
    return np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(number_of_sites))


def compute_fourier_transform_omega_chain(
    green_t: np.ndarray,
    time: np.ndarray,
    omega: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute the direct local transform."""
    if green_t.ndim != 1 or time.ndim != 1:
        raise ValueError("green_t and time must be one-dimensional")
    if green_t.size != time.size:
        raise ValueError("green_t and time must have the same length")

    dampened = apply_dampening_corehole(green_t, time, gamma)
    phase = np.outer(omega, time)
    transform = np.cos(phase) @ dampened.real - np.sin(phase) @ dampened.imag
    return transform / np.pi


def compute_fourier_transform_komega_chain(
    green_it: np.ndarray,
    time: np.ndarray,
    omega: np.ndarray,
    number_of_sites: int,
    center_site: int,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the direct momentum-frequency transform."""
    if green_it.shape != (number_of_sites, time.size):
        raise ValueError("green_it must have shape (number_of_sites, number_of_times)")

    dampened = apply_dampening_gaussian(green_it, time, eta)
    phase = np.outer(omega, time)
    site_omega = dampened.real @ np.cos(phase).T - dampened.imag @ np.sin(phase).T

    k_grid = compute_kgrid_chain_pbc(number_of_sites)
    site_offsets = np.arange(number_of_sites) - center_site
    spatial_factor = np.cos(np.outer(k_grid, site_offsets))
    transform = spatial_factor @ site_omega
    return k_grid / np.pi, transform / (np.pi * number_of_sites)


def load_csv(csv_path: Path) -> np.ndarray:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None)
    data = np.atleast_1d(data)
    if data.dtype.names is None or not REQUIRED_COLUMNS.issubset(data.dtype.names):
        columns = ", ".join(sorted(REQUIRED_COLUMNS))
        raise ValueError(f"CSV must contain these columns: {columns}")
    if data.size == 0:
        raise ValueError(f"CSV contains no data: {csv_path}")
    return data


def extract_columns(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = np.asarray(data["step"], dtype=int)
    sites = np.asarray(data["site_index"], dtype=int)
    times = np.asarray(data["time"], dtype=float)
    values = np.asarray(data["real_part"], dtype=float) + 1j * np.asarray(
        data["imaginary_part"], dtype=float
    )
    return steps, sites, times, values


def keep_last_value_per_time(
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time and overwrite repeated times with their final value."""
    values_by_time: dict[float, complex] = {}
    for time_value, green_value in zip(time, values):
        values_by_time[float(time_value)] = complex(green_value)

    unique_time = np.asarray(sorted(values_by_time), dtype=float)
    unique_values = np.asarray([values_by_time[value] for value in unique_time])
    return unique_time, unique_values


def compute_omega_map(
    data: np.ndarray,
    omega: np.ndarray,
    center_site: int | None,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute G(omega,t') for every step at one site."""
    steps, sites, times, values = extract_columns(data)
    available_sites = np.unique(sites)

    if center_site is None:
        if available_sites.size != 1:
            listed = ", ".join(str(site) for site in available_sites)
            raise ValueError(f"Multiple sites found ({listed}); use --center-site")
        center_site = int(available_sites[0])

    site_mask = sites == center_site
    if not np.any(site_mask):
        raise ValueError(f"Site {center_site} is not present in the CSV")

    steps = steps[site_mask]
    times = times[site_mask]
    values = values[site_mask]
    unique_steps = np.unique(steps)
    probe_times = np.empty(unique_steps.size)
    spectra = np.empty((unique_steps.size, omega.size))

    for index, step in enumerate(unique_steps):
        mask = steps == step
        absolute_time, green_t = keep_last_value_per_time(times[mask], values[mask])
        probe_times[index] = absolute_time[0]
        relative_time = absolute_time - absolute_time[0]
        spectra[index] = compute_fourier_transform_omega_chain(
            green_t, relative_time, omega, gamma
        )

    return probe_times, spectra, center_site


def compute_komega_map(
    data: np.ndarray,
    omega: np.ndarray,
    requested_step: int | None,
    center_site: int | None,
    eta: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build G(i,t) for one step and compute G(k,omega)."""
    steps, sites, times, values = extract_columns(data)
    available_steps = np.unique(steps)
    step = int(available_steps[0] if requested_step is None else requested_step)
    if step not in available_steps:
        raise ValueError(f"Step {step} is not present in the CSV")

    step_mask = steps == step
    sites = sites[step_mask]
    times = times[step_mask]
    values = values[step_mask]
    available_sites = np.unique(sites)

    if not np.array_equal(available_sites, np.arange(available_sites.size)):
        raise ValueError("komega mode requires every site indexed from 0 through N-1")

    number_of_sites = available_sites.size
    if center_site is None:
        center_site = number_of_sites // 2
    if center_site not in available_sites:
        raise ValueError(f"Center site {center_site} is not present in the CSV")

    common_time: np.ndarray | None = None
    site_traces = []
    for site in available_sites:
        mask = sites == site
        site_time, site_values = keep_last_value_per_time(times[mask], values[mask])
        if common_time is None:
            common_time = site_time
        elif common_time.shape != site_time.shape or not np.allclose(
            common_time, site_time
        ):
            raise ValueError("All sites must use the same time grid in komega mode")
        site_traces.append(site_values)

    assert common_time is not None
    relative_time = common_time - common_time[0]
    green_it = np.vstack(site_traces)
    k_over_pi, spectrum = compute_fourier_transform_komega_chain(
        green_it,
        relative_time,
        omega,
        number_of_sites,
        center_site,
        eta,
    )
    return k_over_pi, spectrum, step, center_site


def save_figure(figure: plt.Figure, output_path: Path, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_omega_map(
    omega: np.ndarray,
    probe_times: np.ndarray,
    spectrum: np.ndarray,
    center_site: int,
    output_path: Path,
    title: str | None,
    show: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if probe_times.size == 1:
        axis.plot(omega, spectrum[0], linewidth=1.5)
        axis.set_ylabel(r"$G(\omega)$")
    else:
        image = axis.pcolormesh(
            omega, probe_times, spectrum, shading="auto", cmap="RdBu_r"
        )
        axis.set_ylabel(r"Probe time $t'$")
        figure.colorbar(image, ax=axis, label=r"$G(\omega,t')$")

    axis.set_xlabel(r"Angular frequency $\omega$")
    axis.set_title(title or f"Frequency transform at site {center_site}")
    save_figure(figure, output_path, show)


def plot_komega_map(
    omega: np.ndarray,
    k_over_pi: np.ndarray,
    spectrum: np.ndarray,
    step: int,
    output_path: Path,
    title: str | None,
    show: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = axis.pcolormesh(omega, k_over_pi, spectrum, shading="auto", cmap="RdBu_r")
    axis.set_xlabel(r"Angular frequency $\omega$")
    axis.set_ylabel(r"$k/\pi$")
    axis.set_title(title or f"Momentum-frequency transform, step {step}")
    figure.colorbar(image, ax=axis, label=r"$G(k,\omega)$")
    save_figure(figure, output_path, show)


def main() -> int:
    args = parse_args()
    if args.omega_points < 2:
        raise ValueError("--omega-points must be at least 2")
    if args.omega_max <= args.omega_min:
        raise ValueError("--omega-max must be greater than --omega-min")
    if args.gamma < 0 or args.eta < 0:
        raise ValueError("Damping factors must be nonnegative")

    csv_path = args.csv_file.resolve()
    data = load_csv(csv_path)
    omega = np.linspace(args.omega_min, args.omega_max, args.omega_points)
    output_path = (
        args.output.resolve()
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_{args.mode}.png")
    )

    if args.mode == "omega":
        probe_times, spectrum, center_site = compute_omega_map(
            data, omega, args.center_site, args.gamma
        )
        plot_omega_map(
            omega,
            probe_times,
            spectrum,
            center_site,
            output_path,
            args.title,
            args.show,
        )
    else:
        k_over_pi, spectrum, step, _ = compute_komega_map(
            data, omega, args.step, args.center_site, args.eta
        )
        plot_komega_map(
            omega,
            k_over_pi,
            spectrum,
            step,
            output_path,
            args.title,
            args.show,
        )

    print(f"Wrote Fourier-transform plot: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
