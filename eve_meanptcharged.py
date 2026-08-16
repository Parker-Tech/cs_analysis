"""
Gamma-distribution fit to the event-by-event mean-pT spectra from a
Trajectum HDF5 output file.

Reads:  eventbyeventmeanptcharged/STARTPC200MeV/centralitybinned
Fits:   A * gamma.pdf(x, a, loc, scale)  to each centrality bin
Plots:  the fitted curves only (log-y), zoomed to where the data lives
"""

import h5py
import numpy as np
import matplotlib
#matplotlib.use("Agg")  # remove this line if running interactively
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from scipy.stats import gamma

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
H5_PATH = "src/juneptstudy/200GeV_ptStudy_06.10.26.h5"
DATA_GROUP = "eventbyeventmeanptcharged/STARTPC200MeV/centralitybinned"
OUTPUT_PNG = "graphs/eventbyevent_meanpt_dist.png"

XLIM = (0.45, 0.85)
YLIM = (1e-4, 40)
COLORS = ["#d62728", "#1f77b4"]


def gamma_func(x, A, a, loc, scale):
    """Scaled gamma PDF used as the fit model."""
    return A * gamma.pdf(x, a, loc=loc, scale=scale)


def fit_gamma_to_bin(binc, y):
    """Fit a scaled gamma PDF to one centrality bin's distribution."""
    mask = y > 1e-6  # ignore near-zero/noise bins for the fit
    mean_guess = binc[np.argmax(y)]

    p0 = [1.0, 100, 0.0, mean_guess / 100]
    bounds = (
        [0, 1, -0.5, 1e-5],          # lower bounds: A, a, loc, scale
        [1000, 10000, binc.min(), 1.0],  # upper bounds
    )

    popt, pcov = curve_fit(
        gamma_func, binc[mask], y[mask], p0=p0, bounds=bounds, maxfev=50000
    )
    return popt  # A, a, loc, scale


def main():
    with h5py.File(H5_PATH, "r") as f:
        g = f[DATA_GROUP]
        binc = g["bin"][:]            # <pT> bin centers [GeV]
        vals = g["values"][:]         # shape (n_centrality_bins, 1, n_bins)
        cent = f["centrality"][:]     # bin centers [%]
        dcent = f["dcentrality"][:]   # bin half-widths [%]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(vals.shape[0]):
        y = vals[i, 0, :]
        c_lo = cent[i] - dcent[i]
        c_hi = cent[i] + dcent[i]
        label = f"{c_lo:.0f}-{c_hi:.0f}%"

        popt = fit_gamma_to_bin(binc, y)
        A, a, loc, scale = popt
        mean = loc + a * scale
        sigma = np.sqrt(a) * scale
        skew = 2 / np.sqrt(a)
        print(f"{label}: A={A:.4f}  a={a:.2f}  loc={loc:.5f}  scale={scale:.6f}"
              f"  -> mean={mean:.4f} GeV, sigma={sigma:.4f} GeV, skew={skew:.4f}")

        xfit = np.linspace(binc.min(), binc.max(), 1000)
        ax.plot(xfit, gamma_func(xfit, *popt), color=COLORS[i % len(COLORS)],
                 lw=2, label=label)

    ax.set_xlabel(r"$\langle p_T \rangle$ [GeV]", fontsize=13)
    ax.set_ylabel(r"P($\langle p_T \rangle$) [GeV$^{-1}$]", fontsize=13)
    ax.set_title(
        "Event-by-Event Mean $p_T$ Distribution (charged, 0.2-2 GeV, "
        "|$\\eta$|<0.5)\nAu+Au 200 GeV - Gamma Fit",
        fontsize=12,
    )
    ax.legend(title="Centrality")

    ax.set_xlim(*XLIM)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    ax.tick_params(which="major", length=7)
    ax.tick_params(which="minor", length=4)

    ax.grid(which="major", alpha=0.3)
    ax.grid(which="minor", alpha=0.15)

    
    plt.tight_layout()
    #plt.show()

    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"saved -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()