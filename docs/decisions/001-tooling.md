# 001 — Python tooling

## Rationale
uv and ruff collapse five tools into two with no configuration debt. Strict mypy
is cheap to adopt at project start and expensive to retrofit. pandas because
pvlib and LightGBM are pandas-native and mixing dataframe libraries would cost
more time.

## Consequences
Polars' performance advantage is forgone but is irrelevant at the envisaged :Wdata scales.
