"""Constants for `underhood.tools`, travelling beside it so the package imports nothing internal."""

from __future__ import annotations

# Guard on `**` in the calculator: `2 ** 10` is arithmetic, `2 ** 10_000_000` a way to stall it.
MAX_EXPONENT = 1000.0

# Caps on one read and one write, which a tool's description tells the model.
MAX_FILE_READ_CHARS = 20_000
MAX_FILE_WRITE_CHARS = 20_000
