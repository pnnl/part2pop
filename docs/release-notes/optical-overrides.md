# Compatibility and numerical changes

An omitted or None species_modifications inherits population optical
overrides. An explicit mapping, including {}, replaces them; {} restores
species defaults. Default analyses preserve inherited settings. Changed
wavelength grids and supported RI parameters rebuild stale refractive indices.
Recheck optics computed with earlier releases and explicitly empty mappings.
Species objects can still carry shared mutable refractive-index state.
