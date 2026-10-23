# Project decisions

- Default robot constants kept at L1 = 1.0, L2 = 0.8.
- Grid resolution stays at 360 cells for the interactive C-space and tests can use smaller N.
- Potential-field tuning uses ZETA = 1.0, ETA = 2000.0, RHO0 = 25 to keep the dynamics stable while preserving local minima in trap scenarios.
- The project keeps mathematics in the `core/` package and UI logic in `ui/` so it is portable and test-friendly.
- Headless rendering is supported via `main.py --headless` and `export_figures.py`.
