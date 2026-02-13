
import numpy as np, h5py
from scipy.optimize import curve_fit

def boltzmann_model(pt, A, Teff):
    return A * np.exp(-pt / Teff)

def load_multiplicity_pt_spectrum(h5file, cent_idx=0, startpc="STARTPC200MeV"):
    with h5py.File(h5file, "r") as h:
        base = f"/multiplicityptbinned/{startpc}/centralitybinned"
        pT = h[f"{base}/bin"][()]
        y = h[f"{base}/values"][cent_idx, 0, :]
        err_lo = h[f"{base}/lowererrors"][cent_idx, 0, :]
        err_hi = h[f"{base}/uppererrors"][cent_idx, 0, :]
    yerr = 0.5 * (np.abs(err_lo) + np.abs(err_hi))
    return pT.astype(float), y.astype(float), yerr.astype(float)

def fit_teff(pt, y, yerr=None, pt_min=None, pt_max=None):
    mask = np.isfinite(pt) & np.isfinite(y) & (y > 0)
    if pt_min is not None:
        mask &= pt >= pt_min
    if pt_max is not None:
        mask &= pt <= pt_max
    pt = pt[mask]; y = y[mask]
    if yerr is not None:
        yerr = yerr[mask]
    m, b = np.polyfit(pt, np.log(y), 1)
    Teff0 = max(1e-6, -1.0 / m)
    A0 = np.exp(b)
    if yerr is not None:
        popt, pcov = curve_fit(boltzmann_model, pt, y, p0=(A0, Teff0),
                               sigma=yerr, absolute_sigma=True,
                               bounds=((0,0),(np.inf,np.inf)), maxfev=20000)
    else:
        popt, pcov = curve_fit(boltzmann_model, pt, y, p0=(A0, Teff0),
                               bounds=((0,0),(np.inf,np.inf)), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
