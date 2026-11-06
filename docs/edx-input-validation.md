# EDX input quality

Measured element mass fractions must be finite and nonnegative before
reconstruction or automatic percent/fraction inference. NaN, infinity and
negative values raise ValueError, even when their total could normalize to
one. Direct reconstruction helpers and the CSV path use the same check.
Missing numeric CSV cells therefore require explicit preprocessing.

This validation does not silently clip negative measurements, replace missing
values, subtract backgrounds or invent a noise threshold. Select and document
any such preprocessing scientifically before passing data to part2pop.
All-zero rows retain the existing invalid-total error. Valid fraction and
percent input retains the existing normalization and chemistry assumptions.
Input validation does not establish chemical speciation accuracy; the legacy
sulfate/oxide/organic scheme and its documented limitations remain unchanged.
