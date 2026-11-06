# INSA distribution metadata

The raw distribution is number concentration per logarithmic surface-area
interval, so its units are m^-3. The logarithmic measure does not add an
inverse-length factor. The existing normalized INSA distribution is scaled to
unit peak and is dimensionless.

Each factory-built variable receives a copy of its metadata before normalized
labels and units are assigned. Constructing a normalized variable therefore
does not relabel existing raw variables or the class template. This change
does not alter the returned arrays or the normalization convention.

The diameter-distribution factory also copies metadata before assigning
normalized units or dry-diameter labels. Creating a normalized or dry variable
does not relabel existing or subsequently created raw/wet variables, or the
class template. Diameter normalization remains unit-integral in logarithmic
diameter; INSA normalization remains unit-peak.
