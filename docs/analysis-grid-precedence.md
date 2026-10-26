# Explicit grid precedence

An explicit `s_eval` or `wvls` overrides a lower-priority `s_grid` or `wvl_grid`
default. Within the same explicit layer, the existing canonical-key precedence
and consumer warnings are retained. Runtime `.modify()` settings have highest
priority. Explicit distribution bin/range controls suppress lower-priority center
grids; explicitly chosen centers or edges retain their priority. Distribution
state-line plots obtain coordinates and values from the same computation.

For histogram distributions, an explicit center grid is used to infer bin edges.
On irregular grids the reported coordinates are the geometric centers of the
actual bins, which can differ from the requested points. State-line plots use
those reported coordinates. KDE evaluates at the supplied points themselves.

Diameter state-line plots label their x-axis as wet or dry diameter using the
resolved distribution configuration (`wetsize=True` by default). The label is
local to the prepared plot; the shared diameter-grid metadata is not modified.

## Policy decision: repeated modify calls

Repeated `.modify()` calls update one shared runtime layer. Later values replace
the same key, but different aliases and grid representations remain together in
that layer, where the existing canonical-key and explicit-center precedence
still applies. Each call is not a new priority layer.

For example, `.modify(diam_grid=three_centers).modify(N_bins=7)` still uses the
three centers. To request generated bins explicitly, use
`.modify(diam_grid=None, edges=None, N_bins=7)`. To replace an alias group without
leaving conflicting values, set its canonical and alias keys to the same grid.

Whether each call should instead override earlier calls across aliases and grid
representations is a separate API policy decision. The current behavior is kept
deliberately; a last-call-wins policy would need explicit ordering tests.
