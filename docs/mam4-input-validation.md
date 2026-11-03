# MAM4 reconstruction inputs

Modes with exactly zero number concentration are omitted from initialization
and output reconstruction. All per-mode parameters are selected by the same
original mode indices. Unused composition in zero-number modes is ignored.
Every original number concentration is checked for missing, masked, nonfinite
or negative values before filtering. Invalid numbers are not inactive modes.

Active modes require finite nonnegative represented SO4/OC components and a
strictly positive dry total, including when wet sizes imply positive water.
The reader raises ValueError rather than inventing sulfate for an active mode
with no represented dry material. Files whose dry material exists only in
unrepresented native species need an extended reader, not a fallback mixture.

Initialization reads sulfate from `mfso4N` and organic material from `mfpomN`
in the namelist. Output reconstruction reads only `so4_aer` and `soa_aer`,
with optional water inferred from the diameter moments. Native BC, dust, salt
or other organic pools are not automatically represented by those two fields.
For the AMBRS box model compiled with `MODAL_AERO_4MODE`, this is a definite
primary-carbon output incompatibility: mode 4 contains POM and BC, with no SO4
or SOA. The driver initializes the SO4/SOA output arrays to zero and only fills
species that exist in a mode. Consequently, every output record with positive
mode-4 number concentration raises, in both dry and wet reconstruction. Positive
water inferred from wet diameter does not supply the missing dry material.

Initialization is different: a positive `mfpom4` can supply represented OC from
the namelist even though mode-4 output cannot. Successful initialization does
not establish that later records can be reconstructed. Exactly zero number in
mode 4 still permits its omission, with other modes kept aligned.

The regression uses synthetic NetCDF records with this driver's field layout
and absent mode-4 SO4/SOA; it is not a compiled native-model integration run.
Its initial-condition control uses POM only and makes no claim that a BC/POM
mixture's complete mass is represented by the OC-only initialization mapping.

Conversely, a positive represented total does not prove complete species mass
closure: additional unrepresented material is not detected by this guard.
Validate the full native composition before treating a successful read as a
physical round trip.

This intentionally differs from replacing an empty represented mixture with
unit sulfate. Such replacement invents chemical composition and can affect
downstream optics, hygroscopicity and freezing. For an unsupported native
composition, stop reconstruction and explicitly extend the species mapping
with reviewed mass, number and size checks. Do not silently drop a mode with
positive particle number. Existing example files passing validation do not
establish support for every native composition.

Negative values and wet diameters smaller than dry diameters are rejected
without clipping or an invented measurement tolerance. This validation does
not resolve native record timing, unsupported species or the same-GSD wet/dry
moment assumption. The public timestep contract remains 1 for initialization
and n >= 2 for output record n - 2.

For active output modes, pressure (Pa), temperature (K) and derived dry-air
density must be finite positive real scalars. Required dry diameters must be
finite positive and GSD must exceed one. Wet reconstruction additionally
requires finite positive wet diameters at least as large as dry diameters and
finite nonnegative reconstructed water. Dry-only reconstruction does not use
wet-diameter values. Initial fractions and output masses are checked before
normalization; required masked values and complex pressure/temperature inputs
are rejected rather than converted to plausible-looking real values.

## Native source basis and remaining support work

The native behavior above is pinned to AMBRS-project/MAM_box_model commit
`4150e2d582a8fa6debca5009d22ec000496bd405`:

- [MAM4 mode-4 species definition](https://github.com/AMBRS-project/MAM_box_model/blob/4150e2d582a8fa6debca5009d22ec000496bd405/e3sm_src/modal_aero_initialize_data.F90#L192): POM/BC only.
- [Output-array initialization](https://github.com/AMBRS-project/MAM_box_model/blob/4150e2d582a8fa6debca5009d22ec000496bd405/test_drivers/driver.F90#L1021) and [conditional species export](https://github.com/AMBRS-project/MAM_box_model/blob/4150e2d582a8fa6debca5009d22ec000496bd405/test_drivers/driver.F90#L1332): absent SO4/SOA remain zero.

Supporting this mode requires exporting its actual evolving dry components and
agreeing their mapping/properties, rather than reconstructing them from the
zero SO4/SOA fields or assuming the initial POM/BC ratio persists. Validate
initial and later records against native dry component mass, number and size,
including wet-state assumptions and records where mode 4 becomes empty.
Until that support exists, treat active primary-carbon output as unsupported.
