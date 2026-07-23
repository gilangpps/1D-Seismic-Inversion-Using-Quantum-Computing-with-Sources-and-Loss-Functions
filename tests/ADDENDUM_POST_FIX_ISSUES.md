# ADDENDUM: Post-Fix Validation Issues

**Date**: 2026-07-16 17:54  
**Status**: FIX NOT APPLIED IN RUNTIME

---

## Issue: Fix Code Not Loaded

### Observation

After implementing IC fix in `main.py`, running `python main.py` still shows:

```
Config: {'u0': [0.32465..., 0.60653..., 0.88249..., 1.0, ...]}
Loss: 1.826266e-31
Gradient: 1.935e-37
```

These values correspond to **Gaussian IC** (old/buggy), not **multi-mode IC** (new/fixed).

Expected multi-mode IC would be:
```
u0: [0.0, 1.0, 0.25, 0.0, -0.25, -1.0, 0.0]  (has negative values!)
```

---

## Root Cause: Python Import Cache

Python caches compiled bytecode in `__pycache__/` directories. When source files are modified, cache may not be automatically invalidated, causing old code to execute.

---

## Solution

### Option 1: Clear Cache Manually

```bash
# Clear all cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Or on Windows:
python clear_cache.py

# Then restart Python process
python main.py
```

### Option 2: Force Reload

```bash
# Run with -B flag to ignore cache
python -B main.py
```

### Option 3: Delete .pyc Files

```bash
# Remove all compiled bytecode
find . -name "*.pyc" -delete
```

---

## Verification Steps

After clearing cache, verify IC is correct by checking:

1. **Config u0 values should have negative numbers**:
   ```
   u0: [..., -0.25, -1.0, ...]  ← Multi-mode (correct)
   ```
   NOT:
   ```
   u0: [0.32..., 0.60..., 0.88...]  ← Gaussian (buggy)
   ```

2. **Loss should be > 1e-10** (not 1e-31)

3. **Gradient should be > 1e-20** (not 1e-37)

---

## Secondary Issue: Hamiltonian Validation Low Overlap

Even after IC fix is properly applied, there's a **separate bug** in Hamiltonian validation:

```
Mean overlap: 0.351772 (threshold: >0.80)
[FAIL] VALIDATION FAILED
```

This indicates quantum evolution `exp(-iHt)` does NOT match classical leapfrog PDE.

### Possible Causes

1. **Mass-weighting mismatch**: Validation uses different encoding than forward simulation
2. **IC mismatch in validation**: Validation may still use old Gaussian IC
3. **Hamiltonian construction bug**: Hermitian dilation may have subtle error
4. **Time stepping mismatch**: dt or integration method differs

### Investigation Needed

Check `src/experiment/validate_hamiltonian.py`:
- Does it use same IC generation as main.py?
- Does it use same mass-weighting as objective.py?
- Are boundary conditions consistent?

---

## Action Items

- [ ] Clear Python cache
- [ ] Re-run `python main.py` 
- [ ] Verify IC in output shows negative values
- [ ] Verify loss > 1e-10 and gradient > 1e-20
- [ ] If still failing, investigate validation IC generation
- [ ] Fix Hamiltonian validation separately

---

## Status

**Fix implemented**: ✅ (code changed)  
**Fix validated**: ❌ (not loaded in runtime)  
**Cache cleared**: ⏳ USER ACTION REQUIRED  
**Hamiltonian validation**: ❌ SEPARATE BUG (pending investigation)

---

## Next Steps

1. User clears cache: `python clear_cache.py`
2. User re-runs: `python main.py`
3. If IC still wrong: Check if running from correct directory
4. If IC correct but loss still ~0: Investigate quantum evolution deeper
5. Fix Hamiltonian validation bug (may be unrelated to IC bug)
