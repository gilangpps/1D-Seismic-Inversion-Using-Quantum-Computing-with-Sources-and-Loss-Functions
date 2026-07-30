# PROMPT.md

## Goal

Improve the inversion result for the 1-D seismic inversion codebase, with emphasis on the right-side grid points (especially grid indices 5-7 in the recovered `mu` curve), while reducing the optimization loss. Do **not** confuse this with the independent Hamiltonian validation experiment. The validation overlap is a separate diagnostic and must remain untouched.

## What is already true in the codebase

- `main.py::build_source_from_sweep()` currently hardcodes the source position with `center_idx = nx // 2 + 1`, `width_idx = 2`, and `amplitude = 1.0`.
- The sweep dictionaries already contain `source_x`, but that value is currently **not used** by `build_source_from_sweep()`.
- `main.py::run_single_sweep()` creates:
  - `SeismicObjective(..., source_func=solver_source, engine=engine)`
  - `FiniteDifferenceGradient(..., delta_scale=1e-4, epsilon=1.0)`
  - `SeismicOptimizer(..., learning_rate=1e9, reg_weight=0.0, n_grad_avg=1, ma_window=5, max_iterations=env MAX_ITERATIONS or 250, convergence_tolerance=1e-5, early_stopping_patience=50)`
- `src/optimization/objective.py::_misfit_loss()` currently returns only the data misfit:
  - it computes `mean((u_fwd - u_ref)^2)` over time
  - it does **not** actually apply the regularization mentioned in its docstring
- `src/optimization/gradient.py::FiniteDifferenceGradient.compute_with_regularization()` already supports regularization, but the current fallback prior is effectively zero when `mu_prior` is not provided.
- `src/experiment/validate_hamiltonian.py` is an independent comparison of `exp(-iHt)` vs classical leapfrog. It must **not** be used as the inversion loss and must not be changed for this task.

## Required changes

### 1) Make source placement configurable and use `source_x`

Edit: `main.py::build_source_from_sweep()`

Replace the hardcoded source center logic with explicit parsing of `sweep["source_x"]`.

Implement these supported values:

- `"center"` → `center_idx = nx // 2 + 1`
- `"left"` → `center_idx = 2`
- `"right"` → `center_idx = nx - 1`
- `"right_bias"` → `center_idx = nx - 2`
- integer value → clamp to the valid source index range `[1, nx]`

Keep `width_idx = 2` for now unless the sweep explicitly provides a width value. Keep `amplitude = 1.0`.

Return `source_params` with these keys:

```python
{
    "kind": source_type,
    "center": center_idx,
    "width": width_idx,
    "amplitude": amplitude,
    "source_x": sweep.get("source_x", "center"),
}
```

Add support for an optional `source_width` sweep key if present. If not present, keep `width_idx = 2`.

### 2) Make the source more informative for the right-side grid

The recovery problem is weak on the right side, so move the source excitation away from the exact center.

For the current sweep entries:

- `gaussian_source`
  - set `source_x = "right_bias"`
  - set `sigma_t_s = 0.035`
  - keep `t0_s = 0.045`
- `ricker_wavelet_source`
  - set `source_x = "right_bias"`
  - set `f0_hz = 8.0`
  - keep `t0_s = 0.045`

The goal is to increase illumination of the higher-index grid points and reduce the "flat / collapsed" behavior on grid 5-7.

### 3) Fix the regularization path in the inversion objective

Edit: `src/optimization/objective.py`

The `_misfit_loss()` docstring claims spatial regularization exists, but the function currently returns `misfit` only.

Do one of these two approaches, but make the implementation real and consistent:

#### Preferred approach
Keep `_misfit_loss()` as the pure data term, and move model regularization into the optimizer layer where `mu_arr` is available.

That means:
- leave `_misfit_loss()` as the data misfit only
- update the optimizer so `reg_weight` applies to a model smoothness penalty on `mu_arr`
- do **not** pretend `_misfit_loss()` contains regularization if it does not

#### If you choose to implement regularization in the objective
Add a new objective method that receives `mu_arr` and computes:

```python
J_total = J_misfit + lambda_smooth * mean(diff(mu_arr)^2)
```

Then use that method from the optimizer. Do not leave `reg_weight` unused.

### 4) Use a real smoothness penalty for `mu`

Edit: `src/optimization/optimizer.py`

The current regularization path uses `compute_with_regularization()` with a zero prior when `mu_prior` is omitted. That is not a useful prior for this inversion.

Implement a smoothness-aware penalty instead of a zero-target penalty.

Recommended behavior:

- define a model smoothness loss on `mu_arr`:
  - `smoothness = mean((mu[i+1] - mu[i])^2)` for adjacent grid points
- add the corresponding gradient term analytically or through a small helper
- use `reg_weight` as the smoothness weight
- keep the weight small so it regularizes the curve without flattening it

Use this value for the current sweep:

```python
reg_weight = 1e-5
```

### 5) Stabilize the gradient estimate

Edit: `src/optimization/gradient.py`

Update the finite-difference configuration used in the sweep to:

```python
delta_scale = 5e-5
epsilon = 1.0
```

Keep the central finite-difference formula unchanged. The goal is slightly finer perturbation without changing the physical scale of `mu`.

### 6) Retune the optimizer for the inversion loop

Edit: `main.py::run_single_sweep()`

Update the `SeismicOptimizer(...)` settings to:

```python
max_iterations = 400
convergence_tolerance = 1e-6
early_stopping_patience = 80
learning_rate = 2e8
reg_weight = 1e-5
n_grad_avg = 5
ma_window = 9
use_deterministic = True
```

Also keep the gradient object aligned with:

```python
gradient_obj = FiniteDifferenceGradient(objective_fn=None, delta_scale=5e-5, epsilon=1.0)
```

Do not change the Hamiltonian validation block for this tuning pass.

## Files to touch

- `main.py`
- `src/optimization/objective.py`
- `src/optimization/gradient.py`
- `src/optimization/optimizer.py`

## What must stay unchanged

- `src/experiment/validate_hamiltonian.py`
- the validation plotting and reporting pipeline
- the forward-vs-quantum validation metric interpretation
- the `run_hamiltonian_validation()` logic
- the quantum reconstruction diagnostic path

## Acceptance criteria

After the changes:

1. The recovered `mu` curve should follow the true curve better at the right end of the domain.
2. Grid points 5-7 should no longer collapse toward the initial flat prior.
3. The inversion loss should decrease more than before.
4. The Hamiltonian validation overlap should still be reported independently and should not be used as the inversion objective.
5. The code should remain internally coherent:
   - if a parameter exists in a function signature, it must actually be used
   - if regularization is mentioned in the docstring, it must exist in the implementation
   - if a sweep key exists, the sweep builder must consume it

## Implementation note

Be strict about function reality. No dead parameters, no docstring fantasy, no "looks right" changes that do nothing. The code should reflect the actual physics and optimization path, not the decorative version humans love to ship when they are tired.
