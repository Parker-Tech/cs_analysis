import h5py
import numpy as np
import re
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. List of files you want to scan (each with some MeV cut(s))
# ------------------------------------------------------------------
filepaths = [
    '/Users/justin/code/trajectum/cs_analysis/src/200GeV_0_800_MeVpTThresholds_SMASH.h5',
    '/Users/justin/code/trajectum/cs_analysis/src/200GeV_200_MeVpTThresholds_SMASH 1.h5',
    '/Users/justin/code/trajectum/cs_analysis/src/200GeV_400_500_MeVpTThresholds_SMASH 1.h5'
]

# ------------------------------------------------------------------
# 2. Helper: process one file & return curves for each MeV cut
# ------------------------------------------------------------------
def get_fluctuation_curves_from_file(filepath):
    curves = []  # each element: (mev_label, centrality, y, yerr)

    with h5py.File(filepath, "r") as hdf:
        # centrality axis (common to all cuts in this file)
        centrality = hdf["centrality"][:].squeeze()

        # find all STARTPCxxxMeV groups under meanptcharged
        if "meanptcharged" not in hdf:
            raise KeyError(f"'meanptcharged' group not found in {filepath}")

        startpc_groups = [
            key for key in hdf["meanptcharged"].keys()
            if key.startswith("STARTPC")
        ]

        for grp in startpc_groups:
            # parse the MeV value from something like "STARTPC200MeV"
            m = re.search(r"(\d+)MeV", grp)
            mev_label = m.group(1) if m else grp  # "200" from "STARTPC200MeV"

            mean_base = f"meanptcharged/{grp}/centralitybinned"
            fluc_base = f"ptfluctuationscharged/{grp}/centralitybinned"

            # mean pT charged
            meanpTcharged_values = hdf[f"{mean_base}/values"][:].squeeze()
            meanpTcharged_uerr   = hdf[f"{mean_base}/uppererrors"][:].squeeze()
            meanpTcharged_lerr   = hdf[f"{mean_base}/lowererrors"][:].squeeze()
            meanpTcharged_symerr = 0.5 * (meanpTcharged_uerr + meanpTcharged_lerr)

            # delta pT
            deltapT_values = hdf[f"{fluc_base}/values"][:].squeeze()
            deltapT_uerr   = hdf[f"{fluc_base}/uppererrors"][:].squeeze()
            deltapT_lerr   = hdf[f"{fluc_base}/lowererrors"][:].squeeze()
            deltapT_symerr = 0.5 * (deltapT_uerr + deltapT_lerr)

            # δpT / ⟨pT⟩ in percent
            normalizedfluctuation = (deltapT_values / meanpTcharged_values) * 100.0

            # error propagation
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_err = normalizedfluctuation * np.sqrt(
                    (deltapT_symerr / deltapT_values)**2 +
                    (meanpTcharged_symerr / meanpTcharged_values)**2
                )

            # clean bad points
            mask = (
                np.isfinite(normalizedfluctuation) &
                np.isfinite(ratio_err) &
                (ratio_err >= 0)
            )

            curves.append((
                mev_label,
                centrality[mask],
                normalizedfluctuation[mask],
                ratio_err[mask],
            ))

    return curves

# ------------------------------------------------------------------
# 3. Build plot: loop over files and MeV cuts
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))

for fp in filepaths:
    curves = get_fluctuation_curves_from_file(fp)

    for mev_label, x, y, yerr in curves:
        # interpolation for smooth curve
        if len(x) > 1:
            f = interp1d(x, y, kind="linear")
            x_new = np.linspace(x.min(), x.max(), 1000)
            y_new = f(x_new)

            plt.plot(x_new, y_new, "-", alpha=0.7)

        # data points with error bars
        plt.errorbar(
            x, y, yerr=yerr,
            fmt="o", capsize=3,
            label=fr"{mev_label} MeV cut"
        )

# ------------------------------------------------------------------
# 4. Formatting (STAR style, log x, etc.)
# ------------------------------------------------------------------
plt.xscale("log")          # if you want log centrality like before
plt.xlabel("Centrality (%)", fontsize=14)
plt.ylabel(r"$\delta p_T / \langle p_T \rangle$ [%]", fontsize=14)
plt.title(r"$\delta p_T / \langle p_T \rangle$ vs Centrality", fontsize=16)
plt.ylim(0.5,2)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.6)
ax.spines['bottom'].set_linewidth(1.6)
ax.tick_params(axis='both', which='both', width=1.4, labelsize=12)

plt.legend()
plt.tight_layout()
plt.show()
