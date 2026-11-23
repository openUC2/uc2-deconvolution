"""3D deconvolution: a stack, the optics and the parameters in, the deconvolved stack out.

numpy and scipy only. The same module runs in ImSwitch (preview and small stacks on the
microscope's Raspberry Pi), in the processing worker (``uc2deconvolution.worker``, the container
image) and in benchmark/benchmark.py.

    out = deconvolve(stack, Optics(pixel_um=0.12, z_step_um=0.5, na=0.75), Parameters())

Four solvers. ``rl`` is Richardson-Lucy. ``map`` ports DeconvOptim.jl's map_cj_lean (branch
new_methods, src/map_cj_lean.jl, MIT licence): a data term (Poisson deviance, Anscombe or Gauss)
plus a penalty (Good's roughness or the Frobenius norm of the Hessian), object = x**2,
Fletcher-Reeves conjugate gradients, one Newton step per iteration with Armijo backtracking, three
FFT convolutions per iteration. ``lbfgs`` ports DeconvOptim's default deconvolution()
(src/deconvolution.jl, generic_invert.jl): data normalised to mean 1, object = x**2, Optim's
L-BFGS with a backtracking line search; ``admm`` ports src/admm.jl: x >= 0 by projection, no
derivatives. Both take the Poisson loss and total variation (src/regularizer.jl's TV), lbfgs also
Good's roughness, both the Hessian norm. test_deconvolution.py checks them against stored
DeconvOptim.jl output.

Arrays are (z, y, x) float32 and a PSF is centred at index (0, 0, 0). ``padding`` is one fraction
per axis, as in DeconvOptim.jl: the grid grows by that fraction and the border is left unconstrained
by the data, which removes FFT wrap-around (wide-field light leaves the stack along z). Planes wider
than ``Parameters.tile`` are deconvolved in tiles that overlap by the PSF's reach.

PSF models (``Optics.psf_model``): ``scalar``, the pupil propagated plane by plane (Fourier
optics); ``vectorial``, Richards-Wolf for an isotropic emitter (the integrals cgohlke/psf evaluates
in C, here in numpy); ``gaussian``, widths from the optics or given; ``measured``, an array such as
``distill_psf`` makes from a bead stack (PSFDistiller.jl's approach), resampled to the stack's
voxels.
"""
import base64
import itertools
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import fft, ndimage, spatial, special

NAME, VERSION = "deconvolution", "1"
WORKERS = os.cpu_count() or 4        # FFT threads
EPS = np.float32(1e-7)
PADDING = (0.25, 0, 0)               # z only: light from planes outside the stack
METHODS = ("map", "rl", "lbfgs", "admm")
LOSSES = ("poisson", "anscombe", "gauss")
PENALTIES = ("good", "hessian", "tv")
# what each method takes (rl ignores both). lbfgs and admm normalise the data to mean 1 as
# DeconvOptim does; with the Poisson loss that leaves the weight of the penalty unchanged
SUPPORTED = {"map": (LOSSES, ("good", "hessian")), "rl": (LOSSES, PENALTIES),
             "lbfgs": (("poisson",), ("tv", "good", "hessian")),
             "admm": (("poisson",), ("tv", "hessian"))}
ANSCOMBE = np.float32(3 / 8)          # offset of the Anscombe transform, in counts
MODALITIES = ("widefield", "lightsheet")
PSF_MODELS = ("scalar", "vectorial", "gaussian", "measured")
LIMITS = {"iterations": (1, 500), "regularization": (0.0, 10.0), "tile": (64, 16384)}
# peak memory of deconvolve() per voxel of a padded tile (tracemalloc, 2026-11-23; the same for
# every shape tried) times 1.25 for the FFT's own buffers. Measured: rl 32, map 61 (with the
# Hessian 100), lbfgs 142 (167; 80 of it the 10 steps L-BFGS keeps), admm 81 (105) bytes
TILE_BYTES = {"rl": 40, "map": 75, "map_hessian": 125, "lbfgs": 180, "lbfgs_hessian": 210,
              "admm": 100, "admm_hessian": 130}


class Cancelled(Exception):
    """The progress callback asked deconvolve() to stop."""


def _check(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass
class Optics:
    """What the PSF is computed from. Lengths in µm, in the sample."""
    pixel_um: float
    z_step_um: float = 1.0
    na: float = 0.75
    emission_um: float = 0.52
    immersion_n: float = 1.0          # air 1.0, water 1.33, oil 1.515
    modality: str = "widefield"       # or "lightsheet"
    sheet_um: float = 2.0             # light-sheet thickness (FWHM along z); light sheet only
    psf_model: str = "scalar"         # one of PSF_MODELS
    fwhm_xy_um: Optional[float] = None    # gaussian: lateral FWHM; None: expected_fwhm
    fwhm_z_um: Optional[float] = None     # gaussian: axial FWHM; None: expected_fwhm
    psf_path: str = ""                # measured: the PSF dataset; the caller reads it

    def __post_init__(self):
        for name in ("pixel_um", "z_step_um", "na", "emission_um", "immersion_n", "sheet_um"):
            value = float(getattr(self, name))
            _check(math.isfinite(value) and value > 0, f"{name} must be a positive number")
            setattr(self, name, value)
        _check(1.0 <= self.immersion_n <= 2.0, "immersion_n must be between 1 and 2")
        _check(self.na < self.immersion_n, f"NA {self.na} must be below the immersion index "
                                           f"{self.immersion_n}")
        _check(0.2 <= self.emission_um <= 2.0, "emission_um must be between 0.2 and 2 µm")
        _check(self.modality in MODALITIES, f"modality must be one of {MODALITIES}")
        _check(self.psf_model in PSF_MODELS, f"psf_model must be one of {PSF_MODELS}")
        for name in ("fwhm_xy_um", "fwhm_z_um"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                _check(math.isfinite(value) and value > 0, f"{name} must be a positive number")
                setattr(self, name, value)
        self.psf_path = str(self.psf_path or "")


@dataclass
class Parameters:
    method: str = "map"                   # "map", "lbfgs", "admm" (regularised) or "rl"
                                          # (Richardson-Lucy, fast)
    iterations: int = 30
    regularization: float = 1e-4          # weight of the penalty in measured counts. lbfgs and
                                          # admm: DeconvOptim's lambda, the same weight (the
                                          # Poisson loss and the penalties scale with the counts)
    loss: str = "poisson"                 # map: "poisson", "anscombe" or "gauss" (the noise
                                          # model); lbfgs, admm: "poisson"
    penalty: str = "good"                 # "good" (Good's roughness; map, lbfgs), "hessian"
                                          # (Frobenius norm of the Hessian: keeps edges and fine
                                          # lines) or "tv" (total variation; lbfgs, admm)
    background: Optional[float] = None    # camera offset + dark counts; None: estimated
    tile: int = 512                       # planes wider than this are split into tiles; a
                                          # 64-plane tile then needs about 3 GB

    def __post_init__(self):
        _check(self.method in METHODS, f"method must be one of {METHODS}")
        _check(self.loss in LOSSES, f"loss must be one of {LOSSES}")
        _check(self.penalty in PENALTIES, f"penalty must be one of {PENALTIES}")
        losses, penalties = SUPPORTED[self.method]
        _check(self.loss in losses, f"method {self.method} takes the loss {' or '.join(losses)}")
        _check(self.penalty in penalties,
               f"method {self.method} takes the penalty {' or '.join(penalties)}")
        self.iterations, self.tile = int(self.iterations), int(self.tile)
        self.regularization = float(self.regularization)
        for name, (lo, hi) in LIMITS.items():
            _check(lo <= getattr(self, name) <= hi, f"{name} must be between {lo} and {hi}")
        if self.background is not None:
            self.background = float(self.background)
            _check(math.isfinite(self.background), "background must be a number")


def parse(cls, data):
    """Optics or Parameters from a dict (a request body); ValueError for unknown or bad fields."""
    try:
        return cls(**(data or {}))
    except TypeError as e:
        raise ValueError(str(e)) from e


def emission_um(channel: str) -> float:
    """A guess at the emission wavelength from a channel name (excitation line or dye)."""
    name = (channel or "").lower()
    for keys, um in ((("405", "dapi", "hoechst"), 0.46), (("488", "gfp", "fitc", "alexa488"), 0.52),
                     (("561", "532", "rfp", "mcherry", "tritc", "cy3"), 0.60),
                     (("635", "640", "cy5", "alexa647"), 0.67)):
        if any(k in name for k in keys):
            return um
    return 0.52


# --- point spread functions -------------------------------------------------------------------

def widefield_psf(shape, px_um, dz_um, na, wavelength_um=0.52, n=1.0):
    """Scalar wide-field PSF |ifft2(pupil * defocus)|**2 per plane, sum 1."""
    nz, ny, nx = shape
    fy, fx = np.meshgrid(fft.fftfreq(ny, px_um), fft.fftfreq(nx, px_um), indexing="ij")
    kr2 = fx ** 2 + fy ** 2
    pupil = kr2 <= (na / wavelength_um) ** 2
    kz = np.sqrt(np.maximum((n / wavelength_um) ** 2 - kr2, 0))
    z = fft.fftfreq(nz, 1 / nz) * dz_um                       # planes 0, 1, .., -1
    field = fft.ifft2(pupil * np.exp(2j * np.pi * kz * z[:, None, None]), workers=WORKERS)
    psf = np.abs(field) ** 2
    return (psf / psf.sum()).astype(np.float32)


def vectorial_psf(shape, px_um, dz_um, na, wavelength_um=0.52, n=1.0):
    """Richards-Wolf PSF of an isotropic emitter, |I0|**2 + 2 |I1|**2 + |I2|**2, sum 1.

    I0, I1, I2 are the integrals over the aperture angle with the aplanatic sqrt(cos) apodisation
    that cgohlke/psf evaluates in C; here Gauss-Legendre in numpy on a radial profile per plane,
    interpolated onto the grid. Differs from ``widefield_psf`` mostly above NA 0.9.
    """
    nz, ny, nx = shape
    k = 2 * np.pi * n / wavelength_um
    alpha = math.asin(min(na / n, 0.9999))
    z = fft.fftfreq(nz, 1 / nz) * dz_um
    r = np.hypot(*np.meshgrid(fft.fftfreq(ny, 1 / ny) * px_um, fft.fftfreq(nx, 1 / nx) * px_um,
                              indexing="ij"))
    dr = px_um / 4
    radii = np.arange(0, r.max() + 2 * dr, dr)
    cycles = k * (np.abs(z).max() * (1 - math.cos(alpha)) + radii[-1] * math.sin(alpha)) / 2 / np.pi
    nodes, weights = np.polynomial.legendre.leggauss(int(min(32 + 6 * cycles, 4000)))
    theta = (nodes + 1) * alpha / 2
    s, c = np.sin(theta), np.cos(theta)
    a = weights * alpha / 2 * np.sqrt(c) * s
    arg = k * radii[:, None] * s
    phase = np.exp(1j * k * np.outer(c, z))                       # (angles, planes)
    i0 = (special.j0(arg) * (a * (1 + c))) @ phase                # (radii, planes)
    i1 = (special.j1(arg) * (a * s)) @ phase
    i2 = (special.jv(2, arg) * (a * (1 - c))) @ phase
    profile = (np.abs(i0) ** 2 + 2 * np.abs(i1) ** 2 + np.abs(i2) ** 2).astype(np.float32)
    at = r / dr
    i = at.astype(int)
    f = (at - i).astype(np.float32)
    out = np.empty(shape, np.float32)
    for j in range(nz):
        out[j] = profile[i, j] * (1 - f) + profile[i + 1, j] * f
    return out / out.sum()


def gaussian_psf(shape, px_um, dz_um, fwhm_xy_um, fwhm_z_um):
    """3D Gaussian of the given FWHM (µm), sum 1. It has no missing cone: wide-field haze stays."""
    z, y, x = (fft.fftfreq(n, 1 / n) * d for n, d in zip(shape, (dz_um, px_um, px_um)))
    g = [np.exp(-4 * np.log(2) * (u / w) ** 2)
         for u, w in ((z, fwhm_z_um), (y, fwhm_xy_um), (x, fwhm_xy_um))]
    p = g[0][:, None, None] * g[1][None, :, None] * g[2][None, None, :]
    return (p / p.sum()).astype(np.float32)


def expected_fwhm(optics: Optics):
    """(lateral, axial) FWHM in µm of the wide-field detection PSF, 0.51 lambda / NA and
    0.88 lambda / (n - sqrt(n**2 - NA**2)): the Gaussian model's widths unless given."""
    o = optics
    return (0.51 * o.emission_um / o.na,
            0.88 * o.emission_um / (o.immersion_n - math.sqrt(o.immersion_n ** 2 - o.na ** 2)))


def place_psf(array, voxel_um, optics: Optics, shape):
    """A measured PSF, centred in its (z, y, x) array with voxels ``voxel_um`` (µm), resampled to
    the optics' voxels and put on a grid of ``shape`` centred at (0, 0, 0), sum 1."""
    a = np.maximum(np.asarray(array, np.float64), 0)
    _check(a.ndim == 3 and a.sum() > 0, "a measured PSF is a non-empty (z, y, x) array")
    target = (optics.z_step_um, optics.pixel_um, optics.pixel_um)
    if not np.allclose(voxel_um, target, rtol=0.01):
        # ponytail: linear point sampling; box-filter first if beads come much finer than data
        axes = [n // 2 + np.arange(-int(n // 2 * v / t), int(n // 2 * v / t) + 1) * t / v
                for n, v, t in zip(a.shape, voxel_um, target)]
        a = ndimage.map_coordinates(a, np.meshgrid(*axes, indexing="ij"), order=1)
    out = np.zeros(shape)
    src, dst = [], []
    for n, m in zip(a.shape, shape):
        lo, hi = max(0, n // 2 - m // 2), min(n, n // 2 + m - m // 2)
        src.append(slice(lo, hi))
        dst.append(slice(m // 2 - (n // 2 - lo), m // 2 + (hi - n // 2)))
    out[tuple(dst)] = a[tuple(src)]
    _check(out.sum() > 0, "the measured PSF is empty around its centre")
    return (fft.ifftshift(out) / out.sum()).astype(np.float32)


def psf(optics: Optics, shape, measured=None) -> np.ndarray:
    """The PSF of ``optics`` on a (z, y, x) grid of ``shape``, centred at (0, 0, 0), sum 1.

    ``measured`` is (array, voxel_um) for the measured model, see ``place_psf``; a measured PSF
    already holds the light sheet.
    """
    o = optics
    if o.psf_model == "measured":
        _check(measured is not None, "the measured PSF model needs a PSF (psf_path)")
        return place_psf(*measured, o, shape)
    if o.psf_model == "gaussian":
        fxy, fz = expected_fwhm(o)
        p = gaussian_psf(shape, o.pixel_um, o.z_step_um, o.fwhm_xy_um or fxy, o.fwhm_z_um or fz)
    elif o.psf_model == "vectorial":
        p = vectorial_psf(shape, o.pixel_um, o.z_step_um, o.na, o.emission_um, o.immersion_n)
    else:
        p = widefield_psf(shape, o.pixel_um, o.z_step_um, o.na, o.emission_um, o.immersion_n)
    if o.modality == "lightsheet":        # detection PSF times the sheet (FWHM along z)
        z = fft.fftfreq(shape[0], 1 / shape[0]) * o.z_step_um
        p = p * np.exp(-4 * np.log(2) * (z / o.sheet_um) ** 2).astype(np.float32)[:, None, None]
        p /= p.sum()
    return p


def fwhm(profile, step=1.0) -> float:
    """Full width at half maximum of a peaked 1D profile (linear interpolation), times ``step``."""
    p = np.asarray(profile, np.float64)
    i, half = int(p.argmax()), p.max() / 2

    def side(q):                    # samples from the peak to the half-maximum crossing
        below = np.nonzero(q < half)[0]
        if not len(below):
            return float(len(q) - 1)    # never drops to half: a lower bound
        j = below[0]
        return j - 1 + (q[j - 1] - half) / (q[j - 1] - q[j])
    return float((side(p[i:]) + side(p[i::-1])) * step)


def widths(centred, optics: Optics):
    """(lateral, axial) FWHM in µm of a PSF centred in its array, through its brightest voxel."""
    zc, yc, xc = np.unravel_index(int(np.argmax(centred)), centred.shape)
    return fwhm(centred[zc, yc], optics.pixel_um), fwhm(centred[:, yc, xc], optics.z_step_um)


def encode_psf(array, voxel_um) -> dict:
    """A measured PSF as JSON for a spec: base64 float32 little endian, shape, voxel (µm)."""
    a = np.ascontiguousarray(array, "<f4")
    return {"shape": list(a.shape), "voxel_um": [float(v) for v in voxel_um],
            "data": base64.b64encode(a.tobytes()).decode()}


def decode_psf(data: dict):
    """(array, voxel_um) from ``encode_psf``'s dict; ValueError when it does not fit."""
    try:
        a = np.frombuffer(base64.b64decode(data["data"]), "<f4").reshape(data["shape"])
        voxel = tuple(float(v) for v in data["voxel_um"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"bad measured PSF: {e}") from e
    _check(a.ndim == 3 and len(voxel) == 3, "a measured PSF is (z, y, x) with three voxel sizes")
    return a, voxel


def distill_psf(stack, window_px=17, sigma=1.3, threshold=0.1, brightest=1.6):
    """Average the beads of a bead stack into a measured PSF, the way PSFDistiller.jl does.

    Beads are local maxima of the smoothed maximum projection above ``threshold`` of the
    brightest. A bead is used when no other one lies within ``window_px``, its window fits in the
    image, it is at least 2 planes from the top and bottom, and it is at most ``brightest`` times
    the median bead (brighter ones are clumps). Each window (``window_px`` wide, the stack's depth
    around the bead's plane) loses its border median, is shifted sub-pixel so the centroid of its
    in-focus plane sits in the middle (Fourier shift), is scaled to the light of the three planes
    around focus and averaged; a plane counts the beads the stack holds it for.

    Returns (psf, info): the PSF centred in an odd-sized (z, y, x) float32 array, sum 1, and
    {"found", "used", "crowded", "edge", "bright", "shallow", "beads": [[z, y, x], ..]}.
    """
    data = np.asarray(stack, np.float32)
    _check(data.ndim == 3 and data.shape[0] >= 5, "a bead stack is (z, y, x) with 5 planes or more")
    nz, ny, nx = data.shape
    h = int(window_px) // 2
    _check(h >= 2 and ny > 2 * h and nx > 2 * h, "the bead window must fit in the image")
    mip = ndimage.gaussian_filter(data.max(0), sigma)
    floor = float(np.median(mip))
    peaks = (mip == ndimage.maximum_filter(mip, size=3)) & (mip - floor > threshold
                                                           * (mip.max() - floor))
    ys, xs = np.nonzero(peaks)
    _check(len(ys) > 0, "no beads found")
    points = np.c_[ys, xs]
    gap = spatial.cKDTree(points).query(points, k=2)[0][:, 1] if len(ys) > 1 else np.full(1, 1e9)
    height = mip[ys, xs] - floor
    zs = ndimage.gaussian_filter1d(data[:, ys, xs], 1.0, axis=0).argmax(0)
    reasons = {
        "crowded": gap <= 2 * h,
        "edge": (ys < h) | (ys >= ny - h) | (xs < h) | (xs >= nx - h),
        "bright": height > brightest * np.median(height),
        "shallow": (zs < 2) | (zs > nz - 3),
    }
    use = ~np.any(list(reasons.values()), axis=0)
    counts = {k: int(v.sum()) for k, v in reasons.items()}
    _check(use.any(), f"none of the {len(ys)} beads found is usable: {counts['crowded']} have a "
                      f"neighbour within {2 * h} px, {counts['edge']} are at the edge, "
                      f"{counts['bright']} are clumps, {counts['shallow']} are at the top or "
                      f"bottom")
    hz = nz // 2
    total = np.zeros((2 * hz + 1, 2 * h + 1, 2 * h + 1))
    count = np.zeros(2 * hz + 1)
    for y, x, z in zip(ys[use], xs[use], zs[use]):
        w = data[:, y - h:y + h + 1, x - h:x + h + 1].astype(np.float64)
        w -= np.median(np.concatenate([w[:, [0, -1]].ravel(), w[:, :, [0, -1]].ravel()]))
        core = w[z]
        cy, cx = ndimage.center_of_mass(np.where(core > core.max() / 2, core, 0))
        w = np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(w), (0, h - cy, h - cx))).real
        w /= max(w[z - 1:z + 2].sum(), 1e-12)
        lo, hi = max(0, z - hz), min(nz, z + hz + 1)
        total[lo - z + hz:hi - z + hz] += w[lo:hi]
        count[lo - z + hz:hi - z + hz] += 1
    reach = next(d for d in range(hz, -1, -1) if count[hz - d:hz + d + 1].min() > 0)
    p = (total / np.maximum(count, 1)[:, None, None])[hz - reach:hz + reach + 1]
    p = np.maximum(p - np.median(np.concatenate([p[:, [0, -1]].ravel(), p[:, :, [0, -1]].ravel()])),
                   0)
    info = {"found": int(len(ys)), "used": int(use.sum()), **counts,
            "beads": [[int(z), int(y), int(x)] for y, x, z in zip(ys[use], xs[use], zs[use])]}
    return (p / p.sum()).astype(np.float32), info


def overlap_px(optics: Optics, nz: int, measured=None) -> int:
    """How far (pixels) light spreads sideways within the stack: the margin a tile needs.

    The defocus cone over half the stack depth (half the sheet for a light sheet) plus two Airy
    radii; 1.5 FWHM of a Gaussian; half the width of a measured PSF.
    """
    if optics.psf_model == "measured" and measured is not None:
        return int(math.ceil(np.shape(measured[0])[2] // 2 * measured[1][2] / optics.pixel_um))
    if optics.psf_model == "gaussian":
        return int(math.ceil(1.5 * (optics.fwhm_xy_um or expected_fwhm(optics)[0])
                             / optics.pixel_um))
    depth = nz * optics.z_step_um / 2
    if optics.modality == "lightsheet":
        depth = min(depth, optics.sheet_um)
    cone = math.tan(math.asin(min(optics.na / optics.immersion_n, 0.99)))
    return int(math.ceil((depth * cone + optics.emission_um / optics.na) / optics.pixel_um))


def estimate_background(stack) -> float:
    """Camera offset plus dark counts: the 5th percentile after 8 x 8 lateral averaging.

    The averaging beats the noise. Too high when the sample fills the whole field; set the value
    by hand then.
    """
    planes = np.asarray(stack[::max(1, len(stack) // 8)], np.float32)
    return float(np.percentile(ndimage.uniform_filter(planes, size=(1, 8, 8)), 5))


# --- grid helpers -----------------------------------------------------------------------------

def _padded_shape(shape, padding):
    return tuple(n if (n == 1 or p == 0)
                 else fft.next_fast_len(n + max(4, 2 * round(n * abs(p))), real=True)
                 for n, p in zip(shape, padding))


def _centre(big, small):
    return tuple(slice(b // 2 - s // 2, b // 2 - s // 2 + s) for b, s in zip(big, small))


def _setup(measured, psf, padding):
    """Data on the padded grid (border = -1), its mask, and conv(a, adjoint) with the PSF;
    conv.otf is the PSF's rfft."""
    g = np.asarray(measured, np.float32)
    big = _padded_shape(g.shape, padding)
    data = np.full(big, -1, np.float32)
    data[_centre(big, g.shape)] = g
    kernel = np.zeros(big, np.float32)
    kernel[_centre(big, psf.shape)] = fft.fftshift(psf)
    otf = fft.rfftn(fft.ifftshift(kernel), workers=WORKERS) / np.float32(psf.sum())

    def conv(a, adjoint=False):
        return fft.irfftn(fft.rfftn(a, workers=WORKERS) * (np.conj(otf) if adjoint else otf),
                          s=big, workers=WORKERS)
    conv.otf = otf
    return g, data, data >= 0, conv


def _tiles(n, tile, margin):
    """(window start, window stop, core start, core stop) along one axis; windows of one length.

    Cores tile [0, n). Each window holds its core plus up to ``margin`` on both sides, shifted
    inward at the ends, so every tile uses the same PSF.
    """
    if n <= tile:
        return [(0, n, 0, n)]
    size = min(n, tile + 2 * margin)
    out = []
    for c0 in range(0, n, tile):
        w0 = min(max(c0 - margin, 0), n - size)
        out.append((w0, w0 + size, c0, min(c0 + tile, n)))
    return out


# --- solvers ----------------------------------------------------------------------------------

def richardson_lucy(measured, psf, iterations=30, background=0.0, padding=(0, 0, 0), callback=None):
    """Classic Richardson-Lucy with a constant background and an unconstrained padding border."""
    g, data, mask, conv = _setup(measured, psf, padding)
    b = np.float32(background)
    ratio = np.zeros_like(data)
    norm = np.maximum(conv(mask.astype(np.float32), True), EPS)
    est = np.full(data.shape, max(float(g.mean()) - background, 1e-7), np.float32)
    centre = _centre(data.shape, g.shape)
    for it in range(1, iterations + 1):
        np.divide(data, np.maximum(conv(est) + b, EPS), out=ratio, where=mask)
        est *= conv(ratio, True) / norm
        if callback is not None and callback(it, est[centre]):
            break
    return est[centre]


def _good(x, weights):
    """Good's roughness 4 * sum_d w_d * sum (x[n+1] - x[n])**2."""
    return 4 * sum(w * float(np.sum(np.diff(x, axis=d) ** 2, dtype=np.float64))
                   for d, w in enumerate(weights))


def _laplacian(x, weights):
    """Weighted Laplacian with Neumann borders."""
    lap = np.zeros_like(x)
    for d, w in enumerate(weights):
        diff = np.float32(w) * np.diff(x, axis=d)
        hi = [slice(None)] * x.ndim
        lo = [slice(None)] * x.ndim
        hi[d], lo[d] = slice(1, None), slice(None, -1)
        lap[tuple(lo)] += diff
        lap[tuple(hi)] -= diff
    return lap


def _poisson(data, mask, b):
    """Poisson noise as a deviance (Float32 sums stay accurate at high counts)."""
    logdata = np.log(np.where(data > 0, data, 1))

    def value(mu):
        mt = np.maximum(mu + b, EPS)
        dev = np.where(data > 0, mt - data + data * (logdata - np.log(mt)), mt)
        return float(np.sum(dev, where=mask, dtype=np.float64))

    def residual(mu):
        return np.where(mask, 1 - data / np.maximum(mu + b, EPS), 0).astype(np.float32)

    def second(mu):
        return np.where(mask, data / np.maximum(mu + b, EPS) ** 2, 0)
    return value, residual, second


def _anscombe(data, mask, b):
    """sum (sqrt(mu + b + 3/8) - sqrt(data + 3/8))**2."""
    root = np.sqrt(np.maximum(data, 0) + ANSCOMBE)

    def value(mu):
        err = np.sqrt(np.maximum(mu + b, 0) + ANSCOMBE) - root
        return float(np.sum(err * err, where=mask, dtype=np.float64))

    def residual(mu):
        return np.where(mask, 1 - root / np.sqrt(np.maximum(mu + b, 0) + ANSCOMBE),
                        0).astype(np.float32)

    def second(mu):
        m = np.maximum(mu + b, 0) + ANSCOMBE
        return np.where(mask, root / (2 * m * np.sqrt(m)), 0)
    return value, residual, second


def _gauss(data, mask, b):
    """sum (mu + b - data)**2; the noise variance is in gamma."""
    def value(mu):
        err = mu + b - data
        return float(np.sum(err * err, where=mask, dtype=np.float64))

    def residual(mu):
        return np.where(mask, 2 * (mu + b - data), 0).astype(np.float32)

    def second(mu):
        return np.where(mask, np.float32(2), np.float32(0))
    return value, residual, second


# the data terms of map_cj_lean.jl: value(mu), residual(mu) = dL/dmu and second(mu) = d2L/dmu2,
# zero in the padding border (mask False)
DATA_TERMS = {"poisson": _poisson, "anscombe": _anscombe, "gauss": _gauss}


def _along(ndim, axis, s):
    return tuple(s if k == axis else slice(None) for k in range(ndim))


def _hessian_pairs(shape, weights):
    """(axis a, axis b, stencil weight, weight in the norm) of the Hessian entries, as
    map_cj.jl's _mappr_hessian_pairs: the diagonal once, each mixed entry twice."""
    axes = [a for a, n in enumerate(shape) if n > 1]
    return ([(a, a, float(weights[a]), 1.0) for a in axes]
            + [(a, c, math.sqrt(weights[a] * weights[c]), 2.0)
               for i, a in enumerate(axes) for c in axes[i + 1:]])


def _hessian_entry(f, a, c, w):
    """One weighted Hessian entry: the second difference along a with clamped ends (a == c),
    else the forward mixed difference, zero on the last row along a or c."""
    out = np.zeros_like(f)
    if a == c:
        n = f.ndim
        d = np.diff(f, axis=a)                  # f[i+1] - f[i]; the ends take one side only
        out[_along(n, a, slice(None, -1))] = d
        out[_along(n, a, slice(1, None))] -= d
        out *= np.float32(w)
        return out
    inner = tuple(slice(None, -1) if k in (a, c) else slice(None) for k in range(f.ndim))
    out[inner] = np.float32(w) * np.diff(np.diff(f, axis=a), axis=c)
    return out


def _hessian_adjoint(q, a, c, w):
    """The transpose of _hessian_entry (the clamped second difference is symmetric)."""
    if a == c:
        return _hessian_entry(q, a, a, w)
    v = q[tuple(slice(None, -1) if k in (a, c) else slice(None) for k in range(q.ndim))]
    for axis in (a, c):
        v = -np.diff(np.pad(v, [(1, 1) if k == axis else (0, 0) for k in range(v.ndim)]),
                     axis=axis)
    return np.float32(w) * v


def _hessian_norm(f, pairs, eps2, entries=None):
    """r = sqrt(eps**2 + ||Hessian(f)||_F**2) per voxel; the penalty is its sum. ``entries``,
    a list, receives the Hessian entries."""
    s = np.full(f.shape, eps2, f.dtype)
    for a, c, w, pw in pairs:
        h = _hessian_entry(f, a, c, w)
        s += np.float32(pw) * h * h
        if entries is not None:
            entries.append(h)
    return np.sqrt(s, out=s)


def _hessian_gradient(r, entries, pairs):
    """d/df of sum(_hessian_norm(f)), from the r and entries _hessian_norm gave for f."""
    out = np.zeros_like(r)
    for (a, c, w, pw), h in zip(pairs, entries):
        out += _hessian_adjoint(np.float32(pw) * h / r, a, c, w)
    return out


def _hessian_curvature(r, entries, e, pairs):
    """Second derivative of sum(_hessian_norm(f + t e)) at t = 0, from f's r and entries, without
    the term with the second derivative of f itself (map_lean adds that through the object-space
    gradient)."""
    cross, slope = np.zeros_like(r), np.zeros_like(r)
    for (a, c, w, pw), h in zip(pairs, entries):
        hp = _hessian_entry(e, a, c, w)
        cross += np.float32(pw) * h * hp
        slope += np.float32(pw) * hp * hp
    return float(np.sum(slope / r - cross * cross / (r * r * r), dtype=np.float64))


def map_lean(measured, psf, gamma=0.0, iterations=60, background=0.0, padding=(0, 0, 0),  # noqa: C901
             weights=(1, 1, 1), saddle_escape=1.0, escape_threshold=1e-4, callback=None,
             loss="poisson", penalty="good", epsilon=None):
    # one function, as in map_cj_lean.jl, so the two can be read side by side
    """MAP deconvolution, data term ``loss`` + gamma * ``penalty``, object = x**2 >= 0.

    loss: "poisson" (deviance), "anscombe" or "gauss"; penalty: "good" (Good's roughness) or
    "hessian" (sum sqrt(epsilon**2 + ||Hessian(object)||_F**2), map_cj_lean's
    :hessian_frobenius; epsilon defaults to sqrt of the float32 machine epsilon). gamma is in
    units of the measured counts (the data are not normalised). Returns the object.
    callback(iteration, object) may return True to stop early.
    """
    g, data, mask, conv = _setup(measured, psf, padding)
    b = np.float32(background)
    centre = _centre(data.shape, g.shape)
    x = np.full(data.shape, np.sqrt(max(float(g.mean()) - background, 1e-7)), np.float32)
    value, residual, second = DATA_TERMS[loss](data, mask, b)
    hessian = penalty == "hessian" and gamma > 0
    good = penalty == "good" and gamma > 0
    pairs = _hessian_pairs(data.shape, weights) if hessian else []
    eps = math.sqrt(float(np.finfo(np.float32).eps)) if epsilon is None else float(epsilon)
    eps2 = np.float32(eps * eps)

    def objective(x, mu):
        if hessian:
            return value(mu) + gamma * float(np.sum(_hessian_norm(x * x, pairs, eps2),
                                                    dtype=np.float64))
        return value(mu) + (gamma * _good(x, weights) if good else 0.0)

    mu = conv(x * x)
    psi = objective(x, mu)
    direction = np.zeros_like(x)
    gnorm_prev, alpha_prev, restart = 0.0, 1.0, True
    for it in range(1, iterations + 1):
        # object-space gradient g_f (with the Hessian penalty's part), then the one in x
        g_f = conv(residual(mu), True)
        if hessian:     # ponytail: the trial step's entries are recomputed here; keep them if
            entries = []    # the Hessian penalty's run time (about 3x Good's) matters
            r = _hessian_norm(x * x, pairs, eps2, entries)
            g_f += np.float32(gamma) * _hessian_gradient(r, entries, pairs)
        grad = 2 * x * g_f
        if good:
            grad -= np.float32(8 * gamma) * _laplacian(x, weights)
        gnorm = float(np.sum(grad * grad, dtype=np.float64))
        beta = 0.0 if (restart or gnorm_prev == 0) else gnorm / gnorm_prev
        direction = np.float32(beta) * direction - grad
        steepest = beta == 0
        if float(np.sum(grad * direction, dtype=np.float64)) >= 0:
            direction, steepest = -grad, True
        gnorm_prev, restart = gnorm, False

        if saddle_escape > 0:       # voxels stuck at x = 0 although the data ask them to grow
            shift = 8 * gamma * sum(weights) if good else 0.0
            pull = np.where((x * x <= escape_threshold * float((x * x).max())) & (g_f + shift < 0),
                            -(g_f + shift), 0)
            rms = float(np.sqrt(np.mean(direction * direction, dtype=np.float64)))
            if pull.max() > 0 and rms > 0:
                direction += (np.where(x < 0, -1, 1) * np.float32(saddle_escape * rms)
                              * np.sqrt(pull / pull.max()))

        dphi = float(np.sum(grad * direction, dtype=np.float64))
        h = conv(2 * x * direction)
        curvature = float(np.sum(2 * g_f * direction * direction, dtype=np.float64))
        curvature += float(np.sum(second(mu) * h * h, dtype=np.float64))
        if good:
            curvature += gamma * 2 * _good(direction, weights)
        elif hessian:
            curvature += gamma * _hessian_curvature(r, entries, 2 * x * direction, pairs)
            del entries, r
        alpha = -dphi / curvature if (np.isfinite(curvature) and curvature > 0) else 2 * alpha_prev
        if not (np.isfinite(alpha) and alpha > 0):
            alpha = alpha_prev

        x += np.float32(alpha) * direction        # Armijo backtracking, one convolution per trial
        mu = conv(x * x)
        psi_new = objective(x, mu)
        for _ in range(20):
            if np.isfinite(psi_new) and psi_new <= psi + 1e-4 * alpha * dphi:
                break
            alpha /= 2
            x -= np.float32(alpha) * direction
            mu = conv(x * x)
            psi_new = objective(x, mu)
        if np.isfinite(psi_new) and psi_new < psi:
            psi, alpha_prev = psi_new, alpha
        else:                                     # no decrease: undo, restart with steepest descent
            x -= np.float32(alpha) * direction
            mu = conv(x * x)
            if steepest:
                break
            restart = True
        if callback is not None and callback(it, (x * x)[centre]):
            break
    return (x * x)[centre]


# --- DeconvOptim's deconvolution(): total variation, L-BFGS, ADMM ------------------------------

def _dot(a, b):
    return float(np.vdot(a, b))


def _tv(f, weights=(1, 1, 1), epsilon=1e-8, central=False, gradient=False):
    """DeconvOptim's TV() (src/regularizer.jl): the sum, over the voxels where every difference
    exists, of sqrt(epsilon + sum_d w_d (f[i + e_d] - f[i])**2); ``central``: f[i + e_d] -
    f[i - e_d]. Axes of length 1 are left out (Julia's plane guard). ``gradient``: (value, d/df).
    """
    axes = [a for a, n in enumerate(f.shape) if n > 1]
    lo = int(central)
    core = [slice(lo, n - 1) if a in axes else slice(None) for a, n in enumerate(f.shape)]

    def ends(a):            # the core shifted to i + e_a and to i - lo * e_a
        return (tuple(slice(lo + 1, None) if k == a else s for k, s in enumerate(core)),
                tuple(slice(0, f.shape[a] - 1 - lo) if k == a else s for k, s in enumerate(core)))
    r = np.full(f[tuple(core)].shape, epsilon, f.dtype)
    for a in axes:
        hi, lo_ = ends(a)
        d = f[hi] - f[lo_]
        r += np.float32(weights[a]) * d * d
    r = np.sqrt(r, out=r)
    value = float(np.sum(r, dtype=np.float64))
    if not gradient:
        return value
    g = np.zeros_like(f)
    for a in axes:
        hi, lo_ = ends(a)
        q = np.float32(weights[a]) * (f[hi] - f[lo_]) / r
        g[hi] += q
        g[lo_] -= q
    return value, g


def _penalty(name, x, weights, epsilon, gradient=False, central=False):
    """A penalty of the object x**2, as a function of x: "tv" (_tv), "hessian" (sum
    sqrt(epsilon + ||Hessian||_F**2)) or "good" (map_lean's 4 sum (x[n+1] - x[n])**2).
    ``gradient``: (value, d/dx)."""
    if name == "good":
        value = _good(x, weights)
        return (value, -8 * _laplacian(x, weights)) if gradient else value
    f = x * x
    if name == "tv":
        out = _tv(f, weights, epsilon, central, gradient)
    else:
        pairs, entries = _hessian_pairs(f.shape, weights), []
        r = _hessian_norm(f, pairs, np.float32(epsilon), entries if gradient else None)
        value = float(np.sum(r, dtype=np.float64))
        out = (value, _hessian_gradient(r, entries, pairs)) if gradient else value
    return (out[0], 2 * x * out[1]) if gradient else out


def _backtracking(phi, phi_0, dphi_0, alpha=1.0, c_1=1e-4, rho_hi=0.5, rho_lo=0.1,
                  iterations=1000):
    """LineSearches.jl's BackTracking(order=3): alpha shrinks by quadratic, then cubic
    interpolation until phi(alpha) <= phi_0 + c_1 alpha dphi_0. (alpha, phi(alpha)), or None where
    Julia throws a LineSearchException."""
    if not (math.isfinite(phi_0) and math.isfinite(dphi_0)) or dphi_0 > 0:
        return None
    a_1 = a_2 = np.float64(alpha)
    phi_x0, phi_x1 = phi_0, phi(a_2)
    for _ in range(52):                     # halve until the value is finite
        if math.isfinite(phi_x1):
            break
        a_2 /= 2
        phi_x1 = phi(a_2)
    iteration = 0
    with np.errstate(divide="ignore", invalid="ignore"):     # NaN and Inf as in Julia
        while math.isfinite(phi_x1) and phi_x1 > phi_0 + c_1 * a_2 * dphi_0:
            iteration += 1
            if iteration > iterations:
                return None
            if iteration == 1:              # quadratic through phi(0), phi'(0), phi(a_2)
                a_tmp = -(dphi_0 * a_2 ** 2) / (2 * (phi_x1 - phi_0 - dphi_0 * a_2))
            else:                           # cubic, also through phi(a_1)
                u, v = phi_x1 - phi_0 - dphi_0 * a_2, phi_x0 - phi_0 - dphi_0 * a_1
                div = 1 / (a_1 ** 2 * a_2 ** 2 * (a_2 - a_1))
                a = (a_1 ** 2 * u - a_2 ** 2 * v) * div
                b = (-a_1 ** 3 * u + a_2 ** 3 * v) * div
                a_tmp = (-dphi_0 / (2 * b) if a == 0
                         else -dphi_0 / (b + np.sqrt(max(b * b - 3 * a * dphi_0, 0))))
            a_1, a_2 = a_2, np.fmax(np.fmin(a_tmp, a_2 * rho_hi), a_2 * rho_lo)
            phi_x0, phi_x1 = phi_x1, phi(a_2)
            while not math.isfinite(phi_x1) and iteration < iterations:
                iteration += 1
                a_2 /= 2
                phi_x1 = phi(a_2)
    return (float(a_2), phi_x1) if math.isfinite(phi_x1) else None


def _twoloop(grad, dxs, dgs, rhos, pseudo):
    """Optim's twoloop!: the L-BFGS direction -H grad from the last steps dx and gradient changes
    dg (slot (i - 1) % m holds step i); H0 = 1, or dx'dg / dg'dg of the last pair."""
    m = len(rhos)
    used = [(i - 1) % m for i in range(max(1, pseudo - m), pseudo)]      # oldest first
    q, alphas = grad.copy(), {}
    for k in reversed(used):
        alphas[k] = rhos[k] * _dot(dxs[k], q)
        q -= np.float32(alphas[k]) * dgs[k]
    if pseudo > 1:
        k = (pseudo - 2) % m
        q *= np.float32(_dot(dxs[k], dgs[k]) / _dot(dgs[k], dgs[k]))
    for k in used:
        q += np.float32(alphas[k] - rhos[k] * _dot(dgs[k], q)) * dxs[k]
    q *= -1
    return q


def lbfgs(measured, psf, gamma=0.05, iterations=20, background=0.0, padding=(0, 0, 0),
          weights=(1, 1, 1), penalty="tv", epsilon=1e-8, central=False, memory=10,
          callback=None):
    """DeconvOptim's default deconvolution() (src/deconvolution.jl and generic_invert.jl):
    Poisson loss + gamma * penalty of the object x**2 (Non_negative), from x = 1, minimised by
    Optim's LBFGS(m=10) with LineSearches' BackTracking (order 3, first step 1).

    The data are divided by their mean (normalize_data) and the background with them, so gamma
    is Julia's lambda; as the Poisson loss and the penalties scale linearly with the counts, it
    weighs the penalty as map_lean's gamma in counts does. penalty: see _penalty; epsilon is the
    constant under its square root (TV() and TH() in Julia: 1f-8). callback(iteration, object)
    may return True to stop early.
    """
    g, data, mask, conv = _setup(measured, psf, padding)
    scale = max(float(g.mean()), 1e-7)
    data /= np.float32(scale)
    value, residual, _ = _poisson(data, mask, np.float32(background / scale))
    centre = _centre(data.shape, g.shape)

    def fg(x, gradient=True):               # total_loss of generic_invert.jl, its gradient in x
        mu = conv(x * x)
        if not gradient:
            return value(mu) + gamma * _penalty(penalty, x, weights, epsilon, False, central)
        pen, dpen = _penalty(penalty, x, weights, epsilon, True, central)
        grad = 2 * x * conv(residual(mu), True)
        grad += np.float32(gamma) * dpen
        return value(mu) + gamma * pen, grad

    def phi(alpha):                         # the objective along the search direction s
        np.add(x, np.float32(alpha) * s, out=x_ls)
        return fg(x_ls, False)

    x, x_ls = np.ones(data.shape, np.float32), np.empty(data.shape, np.float32)
    f, grad = fg(x)
    dxs, dgs, rhos = [None] * memory, [None] * memory, [0.0] * memory
    pseudo = 0
    for it in range(1, iterations + 1):
        pseudo += 1
        s = _twoloop(grad, dxs, dgs, rhos, pseudo)
        if _dot(grad, s) >= 0:              # not a descent direction: reset_search_direction!
            pseudo, s = 1, -grad
        found = _backtracking(phi, f, _dot(grad, s))
        if found is None:                   # perform_linesearch!: reset and try once more
            pseudo, s = 1, -grad
            found = _backtracking(phi, f, _dot(grad, s))
        if found is None:
            break
        f_new, g_new = fg(x_ls)             # finite: the line search found it so
        dx, dg = x_ls - x, g_new - grad     # accept_step!
        x, x_ls, f, grad = x_ls, x, f_new, g_new
        curvature = _dot(dx, dg)
        if curvature == 0:                  # 1 / dot is Inf: forget the history
            pseudo = 0
        else:
            k = (pseudo - 1) % memory
            dxs[k], dgs[k], rhos[k] = dx, dg, 1 / curvature
        if callback is not None and callback(it, np.square(x[centre]) * np.float32(scale)):
            break
    return np.square(x[centre]) * np.float32(scale)


def _stencil(st, x, adjoint=False):
    """admm.jl's Stencil: (S x)[i] = sum c x[i + o] over st = [(o, c), ..], periodic; the
    adjoint takes x[i - o]."""
    out = np.zeros_like(x)
    for offset, c in st:
        out += np.float32(c) * np.roll(x, [o if adjoint else -o for o in offset],
                                       tuple(range(x.ndim)))
    return out


def _admm_stencils(penalty, shape, weights, central=False):
    """admm_reg: the stencils K_k whose group norm sqrt(sum_k (K_k x)**2) is the penalty. "tv":
    forward (central) differences times sqrt(w); "hessian" (Julia's TH): second differences times
    w and mixed forward differences times sqrt(2 w_a w_b)."""
    axes = [a for a, n in enumerate(shape) if n > 1]
    zero = (0,) * len(shape)

    def e(*steps):                          # offset of (axis, step) pairs
        return tuple(sum(s for a, s in steps if a == k) for k in range(len(shape)))
    if penalty == "tv":
        return [[(e((a, 1)), math.sqrt(weights[a])),
                 (e((a, -1)) if central else zero, -math.sqrt(weights[a]))] for a in axes]
    out = [[(e((a, 1)), weights[a]), (e((a, -1)), weights[a]), (zero, -2 * weights[a])]
           for a in axes]
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            c = math.sqrt(2 * weights[a] * weights[b])
            out.append([(e((a, 1), (b, 1)), c), (e((a, 1)), -c), (e((b, 1)), -c), (zero, c)])
    return out


def _residuals(kx, z, z_old, w):
    """|Kx - z|**2, |z - z_old|**2, |Kx|**2, |z|**2 and |w|**2 of one split."""
    return np.array([_dot(a, a) for a in (kx - z, z - z_old, kx, z, w)])


def admm(measured, psf, gamma=0.05, iterations=100, background=0.0, padding=(0, 0, 0),  # noqa: C901
         weights=(1, 1, 1), penalty="tv", epsilon=1e-8, central=False, rho=None,
         adaptive_rho=True, check_every=10, rho_balance=10.0, rho_factor=2.0, callback=None):
    # one function, as admm_deconvolution in admm.jl, so the two can be read side by side
    """DeconvOptim's ADMM (src/admm.jl): Poisson loss + gamma * penalty, x >= 0, no derivatives.

    Splits z1 = h * x (Poisson prox per voxel; free in the padding border, Almeida-Figueiredo),
    z_k = K_k x (the penalty's differences, _admm_stencils, periodic; group shrinkage with a
    Huber threshold sqrt(epsilon)) and z3 = x (projection on x >= 0); the x-update is one division
    in Fourier space. Data normalised to mean 1 as in lbfgs, gamma is Julia's lambda. rho
    defaults to 0.1 / mean; every ``check_every`` iterations it doubles (halves) when the primal
    residual exceeds the dual one by ``rho_balance`` (or the reverse). Returns z3.
    callback(iteration, object) may return True to stop early.
    """
    g, y, mask, conv = _setup(measured, psf, padding)
    scale = max(float(g.mean()), 1e-7)
    y = np.maximum(y, 0) / np.float32(scale)          # zero in the border, as Julia's y
    bg, shape, otf = np.float32(background / scale), y.shape, conv.otf
    centre = _centre(shape, g.shape)
    stencils = _admm_stencils(penalty, shape, weights, central)
    delta = np.zeros(shape, np.float32)
    delta[(0,) * len(shape)] = 1
    denom = np.abs(otf) ** 2 + np.float32(1)          # H'H + sum K'K + I in Fourier space
    for st in stencils:
        denom += np.abs(fft.rfftn(_stencil(st, delta), workers=WORKERS)) ** 2
    del delta
    rho = 0.1 / float(y[centre].mean()) if rho is None else rho      # default_ρ(PoissonProx)
    thr = np.float32(math.sqrt(epsilon))
    x = np.ones(shape, np.float32)
    z1, w1 = conv(x), np.zeros_like(x)
    zs, ws = [_stencil(st, x) for st in stencils], [np.zeros_like(x) for _ in stencils]
    z3, w3 = np.maximum(x, 0), np.zeros_like(x)
    for it in range(1, iterations + 1):
        check = it % check_every == 0
        # x-update: x = (H'H + sum K'K + I)^-1 [H'(z1 - w1) + sum K'(z - w) + (z3 - w3)]
        r = z3 - w3
        for st, z, w in zip(stencils, zs, ws):
            r += _stencil(st, z - w, adjoint=True)
        spectrum = fft.rfftn(z1 - w1, workers=WORKERS) * np.conj(otf)
        spectrum += fft.rfftn(r, workers=WORKERS)
        spectrum /= denom
        del r
        hx = fft.irfftn(otf * spectrum, s=shape, workers=WORKERS)
        x = fft.irfftn(spectrum, s=shape, workers=WORKERS)
        del spectrum
        # z/w-updates; at a check also the primal |Kx - z| and dual |z - z_old| residuals
        res = np.zeros(5)
        rho32 = np.float32(rho)
        t = hx + w1             # data: argmin_u f(u + bg) + rho/2 (u - t)**2, f(mu) = mu - y log mu
        a = rho32 * (t + bg) - 1
        z = np.where(mask, (a + np.sqrt(a * a + 4 * rho32 * y)) / (2 * rho32) - bg, t)
        w1 = t - z
        if check:
            res += _residuals(hx, z, z1, w1)
        z1 = z
        del a, t, hx
        for st, w in zip(stencils, ws):     # penalty: w holds v = Kx + w during the prox
            w += _stencil(st, x)
        norm = np.sqrt(sum(w * w for w in ws))
        kappa = np.float32(gamma / rho)
        shrink = np.where(norm > thr + kappa, 1 - kappa / np.maximum(norm, thr + kappa),
                          thr / (thr + kappa))
        del norm
        for k, (st, w) in enumerate(zip(stencils, ws)):
            z = shrink * w
            w -= z
            if check:
                res += _residuals(_stencil(st, x), z, zs[k], w)
            zs[k] = z
        del shrink
        t = x + w3                          # positivity: projection on x >= 0
        z = np.maximum(t, 0)
        w3 = t - z
        if check:
            res += _residuals(x, z, z3, w3)
        z3 = z
        del t, z
        if callback is not None and callback(it, z3[centre] * np.float32(scale)):
            break
        if check and adaptive_rho:          # residual balancing (Boyd et al. 2011, 3.4.1)
            r_rel = math.sqrt(res[0]) / max(math.sqrt(res[2]), math.sqrt(res[3]), 2.2e-16)
            s_rel = math.sqrt(res[1]) / max(math.sqrt(res[4]), 2.2e-16)
            fac = (rho_factor if r_rel > rho_balance * s_rel
                   else 1 / rho_factor if s_rel > rho_balance * r_rel else 1.0)
            if fac != 1.0:
                rho *= fac
                for w in (w1, w3, *ws):
                    w /= np.float32(fac)
    return z3[centre] * np.float32(scale)


# --- the recipe -------------------------------------------------------------------------------

def memory_bytes(shape, optics: Optics, parameters: Optional[Parameters] = None, itemsize=2,
                 measured=None) -> int:
    """About the peak memory deconvolve() needs for a (z, y, x) stack of ``itemsize`` bytes per
    voxel: the stack, its float32 copy and the result, plus the working set of one tile."""
    p = parameters or Parameters()
    nz, ny, nx = shape
    margin = overlap_px(optics, nz, measured)
    ys, xs = _tiles(ny, p.tile, margin), _tiles(nx, p.tile, margin)
    tile = _padded_shape((nz, ys[0][1] - ys[0][0], xs[0][1] - xs[0][0]), PADDING)
    kind = p.method + ("_hessian" if p.penalty == "hessian" and p.method != "rl" else "")
    return nz * ny * nx * (itemsize + 8) + math.prod(tile) * TILE_BYTES[kind]


def deconvolve(stack, optics: Optics, parameters: Optional[Parameters] = None, progress=None,
               measured=None):
    """Deconvolve a (z, y, x) stack, or one (y, x) plane; float32 of the same shape.

    ``progress(fraction)`` is called after every iteration; when it returns True the run stops
    and Cancelled is raised. ``measured`` = (array, voxel_um) for the measured PSF model.
    ValueError for data or settings it cannot use.
    """
    p = parameters or Parameters()
    data = np.asarray(stack, np.float32)
    plane = data.ndim == 2
    data = data[None] if plane else data
    _check(data.ndim == 3 and data.size > 0, f"expected a (z, y, x) stack, got shape {data.shape}")
    _check(bool(np.isfinite(data).all()), "the stack holds NaN or infinite values")
    background = estimate_background(data) if p.background is None else p.background
    # the Gauss loss is about 2 * counts times the Poisson deviance: scale gamma by that so one
    # regularisation value means about the same with every noise model
    gamma = p.regularization * (2 * max(float(data.mean()) - background, 1.0)
                                if p.loss == "gauss" else 1.0)
    nz, ny, nx = data.shape
    margin = overlap_px(optics, nz, measured)
    ys, xs = _tiles(ny, p.tile, margin), _tiles(nx, p.tile, margin)
    kernel = psf(optics, (nz, ys[0][1] - ys[0][0], xs[0][1] - xs[0][0]), measured)
    out = np.empty_like(data)
    tiles = list(itertools.product(ys, xs))
    for k, ((y0, y1, cy0, cy1), (x0, x1, cx0, cx1)) in enumerate(tiles):
        def step(it, _, k=k):
            if progress is not None and progress((k + it / p.iterations) / len(tiles)):
                raise Cancelled()
        window = data[:, y0:y1, x0:x1]
        if p.method == "rl":
            result = richardson_lucy(window, kernel, p.iterations, background, PADDING, step)
        elif p.method == "map":
            result = map_lean(window, kernel, gamma, p.iterations, background, PADDING,
                              callback=step, loss=p.loss, penalty=p.penalty)
        else:       # Poisson only: gamma = regularization = DeconvOptim's lambda
            result = (lbfgs if p.method == "lbfgs" else admm)(
                window, kernel, gamma, p.iterations, background, PADDING, penalty=p.penalty,
                callback=step)
        out[:, cy0:cy1, cx0:cx1] = result[:, cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out[0] if plane else out


def run(stack, spec: dict, progress=None) -> np.ndarray:
    """The recipe signature every runner calls: spec = {"optics": {..}, "parameters": {..}},
    plus "psf": encode_psf(..) for the measured PSF model."""
    measured = decode_psf(spec["psf"]) if spec.get("psf") else None
    return deconvolve(stack, parse(Optics, spec.get("optics")),
                      parse(Parameters, spec.get("parameters")), progress, measured)


def memory(shape, itemsize: int, spec: dict) -> int:
    """About the peak memory of run() on a stack of this shape: what a runner checks first."""
    measured = decode_psf(spec["psf"]) if spec.get("psf") else None
    shape = tuple(shape) if len(shape) == 3 else (1,) + tuple(shape)
    return memory_bytes(shape, parse(Optics, spec.get("optics")),
                        parse(Parameters, spec.get("parameters")), itemsize, measured)


# --- test data --------------------------------------------------------------------------------

def synthetic(shape, optics: Optics, seed=1, photons=300.0, background=5.0):
    """(truth, measured): beads, filaments, a hollow shell and line pairs 3, 5 and 8 px apart,
    blurred with the PSF of ``optics`` and Poisson-sampled. The benchmark's and the Julia
    comparison's test object: keep it as it is, or the stored Julia output no longer matches.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    obj = np.zeros(shape, np.float32)
    for _ in range(40 * nx // 256):                                  # beads
        obj[rng.integers(nz // 4, 3 * nz // 4), rng.integers(8, ny - 8),
            rng.integers(8, nx - 8)] += rng.uniform(40, 120)
    t = np.linspace(0, 1, 4 * nx)
    for _ in range(6 * nx // 256):                       # filaments, slightly tilted in z
        p0 = np.array([rng.uniform(0.3, 0.7) * nz, rng.uniform(0.1, 0.9) * ny,
                       rng.uniform(0.1, 0.9) * nx])
        p1 = np.array([rng.uniform(0.3, 0.7) * nz, rng.uniform(0.1, 0.9) * ny,
                       rng.uniform(0.1, 0.9) * nx])
        z, y, x = (p0[:, None] + (p1 - p0)[:, None] * t).round().astype(int)
        obj[z, y, x] = 6
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]                             # hollow shell ("cell")
    r = np.sqrt(((zz - nz / 2) * optics.z_step_um / optics.pixel_um / 6) ** 2 + (yy - 0.3 * ny) ** 2
                + (xx - 0.7 * nx) ** 2)
    obj[np.abs(r - 0.12 * nx) < 1] += 2
    for i, gap in enumerate((3, 5, 8)):                              # line pairs
        x0 = int(0.2 * nx) + i * nx // 10
        obj[nz // 2, int(0.6 * ny):int(0.85 * ny), [x0, x0 + gap]] = 8
    big = (2 * nz, ny, nx)                               # light may leave the stack along z
    padded = np.zeros(big, np.float32)
    padded[nz // 2:nz // 2 + nz] = obj
    blurred = fft.irfftn(fft.rfftn(padded, workers=WORKERS)
                         * fft.rfftn(psf(optics, big), workers=WORKERS),
                         s=big, workers=WORKERS)[nz // 2:nz // 2 + nz]
    scale = photons / blurred.max()
    measured = rng.poisson(np.maximum(blurred * scale, 0) + background).astype(np.float32)
    return obj * scale, measured
