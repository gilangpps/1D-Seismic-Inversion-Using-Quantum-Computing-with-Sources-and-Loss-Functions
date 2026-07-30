"""
Test diagnostic untuk mengidentifikasi kenapa quantum inversion menghasilkan
loss = 0 dan gradient = 0.

Hypothesis yang akan diuji:
1. Apakah quantum_forward_simulate() menghasilkan output yang berbeda untuk mu berbeda?
2. Apakah reference fields dan forward fields dibandingkan dengan benar?
3. Apakah Hamiltonian benar-benar depend on mu?
"""

import numpy as np
from src.optimization.objective import SeismicObjective
from src.hamiltonian import build_hamiltonian
from src.distributions import raised_cosine, homogeneous


def test_hamiltonian_depends_on_mu():
    """Test 1: Apakah H bergantung pada mu?"""
    print("\n" + "="*80)
    print("TEST 1: Hamiltonian dependence on mu")
    print("="*80)
    
    nx = 7
    dx = 63.0
    
    # Dua model berbeda
    mu1 = np.ones(nx + 2) * 1e10  # Homogeneous soft
    mu2 = np.ones(nx + 2) * 4e10  # Homogeneous stiff
    
    rho = np.ones(nx) * 2e3
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    H1, _, _, _, _ = build_hamiltonian(mu1, rho_bc, dx, nx)
    H2, _, _, _, _ = build_hamiltonian(mu2, rho_bc, dx, nx)
    
    diff = np.linalg.norm(H1 - H2, 'fro')
    print(f"  mu1 = {mu1[0]:.2e} Pa (homogeneous)")
    print(f"  mu2 = {mu2[0]:.2e} Pa (homogeneous)")
    print(f"  ||H1 - H2||_F = {diff:.6e}")
    
    if diff < 1e-10:
        print("  [FAIL] BUG FOUND: Hamiltonian tidak bergantung pada mu!")
        return False
    else:
        print("  [PASS] Hamiltonian berbeda untuk mu berbeda")
        return True


def test_quantum_forward_depends_on_mu():
    """Test 2: Apakah quantum forward menghasilkan trajectory berbeda untuk mu berbeda?"""
    print("\n" + "="*80)
    print("TEST 2: Quantum forward trajectory dependence on mu")
    print("="*80)
    
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 10  # Cukup pendek untuk test
    
    # Initial conditions
    u0 = np.sin(np.pi * np.arange(nx) / (nx - 1))
    v0 = np.zeros(nx)
    
    # Dua model berbeda
    mu_true = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    mu_init = homogeneous(0.5 * np.mean(mu_true), nx + 1)
    rho = np.ones(nx) * 2e3
    
    print(f"  mu_true: min={mu_true.min():.2e}, max={mu_true.max():.2e}, mean={mu_true.mean():.2e}")
    print(f"  mu_init: min={mu_init.min():.2e}, max={mu_init.max():.2e}, mean={mu_init.mean():.2e}")
    
    # Run quantum forward dengan mu_true
    objective_true = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    fields_true = objective_true.quantum_forward_simulate(mu_true, rho, u0, v0)
    
    # Run quantum forward dengan mu_init (model berbeda)
    objective_init = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    fields_init = objective_init.quantum_forward_simulate(mu_init, rho, u0, v0)
    
    # Bandingkan trajectory di timestep terakhir
    u_true_final = fields_true[-1][1:-1]
    u_init_final = fields_init[-1][1:-1]
    
    diff_l2 = np.linalg.norm(u_true_final - u_init_final)
    diff_rel = diff_l2 / (np.linalg.norm(u_true_final) + 1e-30)
    
    print(f"  u_true[-1]: {u_true_final}")
    print(f"  u_init[-1]: {u_init_final}")
    print(f"  L2 difference: {diff_l2:.6e}")
    print(f"  Relative difference: {diff_rel:.6e}")
    
    # Cek apakah ada evolusi sama sekali (bandingkan dengan IC)
    u_true_ic = fields_true[0][1:-1]
    diff_from_ic = np.linalg.norm(u_true_final - u_true_ic)
    print(f"  L2 difference from IC: {diff_from_ic:.6e}")
    
    if diff_l2 < 1e-10:
        print("  [FAIL] BUG FOUND: Quantum forward menghasilkan trajectory identik untuk mu berbeda!")
        return False
    elif diff_from_ic < 1e-10:
        print("  [FAIL] BUG FOUND: Quantum forward tidak mengubah state (hanya copy IC)!")
        return False
    else:
        print("  [PASS] Quantum forward menghasilkan trajectory berbeda")
        return True


def test_loss_computation():
    """Test 3: Apakah loss dihitung dengan benar (comparing against reference)?"""
    print("\n" + "="*80)
    print("TEST 3: Loss computation correctness")
    print("="*80)
    
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 10
    
    u0 = np.sin(np.pi * np.arange(nx) / (nx - 1))
    v0 = np.zeros(nx)
    
    mu_true = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    mu_init = homogeneous(0.5 * np.mean(mu_true), nx + 1)
    rho = np.ones(nx) * 2e3
    
    # Setup objective
    objective = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    
    # Compute reference fields dari mu_true
    ref_fields = objective.compute_reference_fields(mu_true, rho, u0, v0)
    print(f"  Reference computed: {len(ref_fields)} timesteps")
    
    # Compute loss untuk mu_true (seharusnya loss ≈ 0)
    loss_true = objective.compute(mu_true, rho, u0, v0)
    print(f"  Loss for mu_true (should be ~0): {loss_true:.6e}")
    
    # Compute loss untuk mu_init (seharusnya loss > 0)
    loss_init = objective.compute(mu_init, rho, u0, v0)
    print(f"  Loss for mu_init (should be >0): {loss_init:.6e}")
    
    if loss_true > 1e-6:
        print(f"  [FAIL] BUG FOUND: Loss untuk model yang sama dengan reference tidak nol!")
        print(f"     Kemungkinan: quantum forward non-deterministic atau reference salah")
        return False
    elif loss_init < 1e-6:
        print(f"  [FAIL] BUG FOUND: Loss untuk model berbeda juga nol!")
        print(f"     Kemungkinan: quantum forward selalu menghasilkan output yang sama")
        return False
    else:
        print(f"  [PASS] Loss computation correct (true=~0, init>0)")
        return True


def test_gradient_computation():
    """Test 4: Apakah gradient non-zero untuk mismatch case?"""
    print("\n" + "="*80)
    print("TEST 4: Gradient computation")
    print("="*80)
    
    from src.optimization.gradient import FiniteDifferenceGradient
    
    nx = 7
    dx = 63.0
    dt = 0.005
    steps = 10
    
    u0 = np.sin(np.pi * np.arange(nx) / (nx - 1))
    v0 = np.zeros(nx)
    
    mu_true = raised_cosine(3e10, nx + 1, nx, 6, 1e10)
    mu_init = homogeneous(0.5 * np.mean(mu_true), nx + 1)
    rho = np.ones(nx) * 2e3
    
    objective = SeismicObjective(
        nx=nx, dx=dx, dt=dt, steps=steps,
        measure_every=4, shots=1000, bc='dirichlet',
        seed=42, source_func=None, engine='quantum'
    )
    
    # Set reference
    ref_fields = objective.compute_reference_fields(mu_true, rho, u0, v0)
    
    # Setup gradient
    def obj_fn(mu, rho, u0, v0):
        loss = objective.compute(mu, rho, u0, v0)
        fields = objective.forward_simulate(mu, rho, u0, v0)
        return loss, fields
    
    gradient_obj = FiniteDifferenceGradient(objective_fn=obj_fn, delta_scale=1e-4, epsilon=1.0)
    
    # Compute gradient at mu_init
    grad = gradient_obj.compute(mu_init, rho, u0, v0)
    grad_norm = np.linalg.norm(grad)
    
    print(f"  mu_init: {mu_init[:3]}...")
    print(f"  Gradient: {grad[:3]}...")
    print(f"  ||gradient|| = {grad_norm:.6e}")
    
    if grad_norm < 1e-15:
        print(f"  [FAIL] BUG FOUND: Gradient identik nol!")
        print(f"     Kemungkinan: J(mu+delta) = J(mu-delta) untuk semua parameter")
        return False
    else:
        print(f"  [PASS] Gradient non-zero")
        return True


def main():
    """Run all diagnostic tests"""
    print("\n" + "#"*80)
    print("# QUANTUM INVERSION BUG DIAGNOSTIC")
    print("# Tujuan: Identifikasi kenapa loss=0 dan grad=0 dari awal")
    print("#"*80)
    
    results = {}
    results['hamiltonian_depends_on_mu'] = test_hamiltonian_depends_on_mu()
    results['quantum_forward_depends_on_mu'] = test_quantum_forward_depends_on_mu()
    results['loss_computation'] = test_loss_computation()
    results['gradient_computation'] = test_gradient_computation()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {test_name}")
    
    all_pass = all(results.values())
    if all_pass:
        print("\n  All tests passed. Bug is elsewhere (maybe in optimizer loop).")
    else:
        print("\n  Bug found! Check failed tests above.")
    
    return results


if __name__ == '__main__':
    main()
