# Species overrides in the mass constructor

`make_particle_from_masses` resolves species through the runtime registry,
including its built-in aliases. An override keyed by the exact input label has
priority, including an explicit empty mapping. If that key is absent, an
override keyed by the returned species name is used. The mappings are selected
as a whole and are not merged.

For example, input `black carbon` accepts `{"BC": {"density": 1234}}`.
If `black carbon` also has an override, that exact-label mapping wins.
Registry templates and caller mappings are not modified. A runtime species
registered literally as `soot` still takes priority over the built-in `soot`
alias for BC; a BC override does not affect that distinct registered species.
