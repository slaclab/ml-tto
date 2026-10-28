Package for testing ML algorithms for transistion to operations work. Will be pulled into rhel7-dev environment during nightly builds.

## SLAC package dependencies

Python 3.11 or newer is required. Device and measurement definitions come from
the split SLAC packages rather than `lcls-tools`:

- `slac-devices`: magnets, screens, BPMs, TCAVs, and device readers.
- `slac-db`: device metadata and control-system accessor definitions consumed by
  the device readers. Use `slac_devices.reader.create_device("CQ01")` to build a
  device from the current database by its MAD name without a legacy YAML file.
- `slac-measurements`: beam-profile measurements, image processing/fitting,
  emittance measurements, and measurement result models.
- `slac-model`: beam-optics calculations and quadrupole strength conversions.

Install the package and its dependencies with `python -m pip install -e ".[dev]"`.
When developing against local SLAC checkouts, install those in the same environment.
Run unit tests with `python -m pytest ml_tto/tests`.

### Results and historical files

Beam energy supplied to the ML quadrupole scan remains in **eV**. Results follow
the `slac-measurements` schema: `emittance` is **normalized** emittance,
`geometric_emittance` is geometric emittance (both in **mm-mrad**),
`rms_beamsizes` is in **microns**, and `beam_matrix` is in **mm^2**.
`twiss_at_reconstruction` stores the upstream Twiss parameters.
Code that previously read geometric emittance from `result.emittance` should now
read `result.geometric_emittance`.

Historical HDF5 files remain supported by `ml_tto.io.H5Saver`. Use
`ml_tto.automatic_emittance.utils.quad_scan_result_from_dict` to upgrade legacy
result dictionaries to the new schema, or
`plot_emittance_measurement_from_file` to load and plot them directly.
The GPSR helpers in `ml_tto.gpsr.lcls_tools` retain their existing names for
compatibility but no longer import `lcls-tools`.

The removed ROI geometry and projection-plotting utilities are maintained locally:
`ml_tto.roi` supplies `ROI`/`CircularROI` for validation, and
`ml_tto.automatic_emittance.plotting` supplies `plot_image_projection_fit`.
