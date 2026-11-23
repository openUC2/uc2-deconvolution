# uc2-deconvolution

3D deconvolution for microscopy in numpy and scipy, nothing else: PSF models, PSFs from bead
stacks, Richardson–Lucy and the solvers of [DeconvOptim.jl](https://github.com/roflmaostc/DeconvOptim.jl),
ported and checked against its output. Plus a small HTTP worker that runs them for another
machine. [ImSwitch](https://github.com/openUC2/ImSwitch) uses it for the "Deconvolve" button of
its OME-Zarr viewer: previews run on the microscope's Raspberry Pi, whole stacks on a worker.

```bash
pip install "uc2-deconvolution @ git+https://github.com/openUC2/uc2-deconvolution@v0.1.0"
```

```python
from uc2deconvolution import Optics, Parameters, deconvolve

optics = Optics(pixel_um=0.12, z_step_um=0.5, na=0.75, emission_um=0.52, immersion_n=1.0)
result = deconvolve(stack, optics, Parameters(method="map", iterations=30))  # (z, y, x) -> float32
```

## What is in it

- **PSF models** (`Optics.psf_model`): `scalar` (the pupil propagated plane by plane),
  `vectorial` (Richards–Wolf), `gaussian` (widths from the optics or given), `measured` (an array
  such as `distill_psf` makes from a bead stack, resampled to the data's voxels).
- **Solvers** (`Parameters.method`), each checked against DeconvOptim.jl on the same stack
  (correlation of the maximum projections; tests/test_recipe.py):

  | method | DeconvOptim.jl | noise model | penalty | match |
  |---|---|---|---|---|
  | `rl` | `richardson_lucy_iterative` | Poisson | none | |
  | `map` | `map_cj_lean`: conjugate gradients, object = x² | Poisson, Anscombe, Gauss | Good's roughness, Hessian | 0.995 to 1.000 |
  | `lbfgs` | `deconvolution()`: L-BFGS (Optim.jl), object = x² | Poisson | total variation, Good's, Hessian | 1.000 |
  | `admm` | `deconvolution(opt=ADMM())` | Poisson | total variation, Hessian | 1.000 |

  `map` needs the least memory, `lbfgs` with total variation gives the best result per iteration
  on the test object, `admm` converges slowly on sparse objects (correlation 0.5 after 200
  iterations, against 0.98 for `lbfgs` after 60).
- **Tiles**: planes wider than `Parameters.tile` (512) are deconvolved in tiles that overlap by
  the PSF's reach, so a full sensor fits in a few GB. `memory_bytes()` estimates a run's peak.
- **Worker** (`uc2deconvolution.worker`): ImSwitch pushes a volume's planes in one POST (a JSON
  spec, then the raw pixels) and reads the result planes back; the worker never connects to the
  microscope. `run_remote()` is the client.

## Worker

```bash
uc2-deconvolution-worker --port 8090
docker run --rm -p 8090:8090 ghcr.io/openuc2/uc2-deconvolution
```

Then in ImSwitch: OME-Zarr viewer, Deconvolve, worker URL `http://<this computer>:8090`.

## Tests and benchmark

`pytest` compares the solvers with stored DeconvOptim.jl results (`tests/deconvolution_julia_reference.npz`);
`benchmark/benchmark.py` and `benchmark/benchmark.jl` produce them and time both sides.

## Licence

MIT. The solvers are ports of DeconvOptim.jl (MIT, © 2020 Felix Wechsler); see LICENSE.
