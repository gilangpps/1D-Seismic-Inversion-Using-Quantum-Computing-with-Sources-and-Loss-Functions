"""
Diagnostic: Why is Hamiltonian validation overlap so low (0.35 vs expected >0.80)?

Test multiple hypotheses:
1. IC encoding mismatch
2. Evolution divergence grows over time
3. Amplitude encoding issue in overlap computation
4. Hamiltonian construction error
"""
import numpy as np
from scipy.linalg import expm
from src.hamiltonian import build_hamiltonian
from src.wave import evolve_1d_wave
from src.encoding import amplitude_encode

def diagnose_low_overlap():
    print("="*80)
    print("DIAGNOSTIC: Low Hamiltonian Validation Overlap")
    print("="*80)
    print()
    
    # Small test case
    nx = 3
    dx = 63.0
    dt = 0.005
    steps = 5  # Short
    
    # Use NEW multi-mode IC (not Gaussian!)
    x_norm = np.arange(nx) / (nx - 1)
    u0 = (0.5 * np.sin(2*np.pi*x_norm) + 
          0.3 * np.sin(4*np.pi*x_norm) + 
          0.2 * np.sin(6*np.pi*x_norm))
    u0 = u0 / (np.max(np.abs(u0)) + 1e-30)
    
    mu = np.ones(nx + 1) * 2e10
    rho = np.ones(nx) * 2e3
    
    c_typical = np.sqrt(np.mean(mu) / np.mean(rho))
    v0 = 0.1 * c_typical * np.cos(2*np.pi*x_norm)
    
    print(f"Test setup:")
    print(f"  nx={nx}, dx={dx}, dt={dt}, steps={steps}")
    print(f"  u0: {u0}")
    print(f"  v0: {v0}")
    print()
    
    # === CLASSICAL ===
    print("Running classical leapfrog...")
    u0_bc = np.zeros(nx + 2)
    u0_bc[1:-1] = u0
    u1_bc = u0_bc.copy()
    
    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho
    rho_bc[0] = rho[0]
    rho_bc[-1] = rho[-1]
    
    mu_bc = np.ones(nx + 2) * 2e10
    
    classical_traj = evolve_1d_wave(
        u0_bc, u1_bc,
        dx=dx, dt=dt,
        mu=mu_bc, rho=rho_bc,
        source_func=None,
        steps=steps,
        bc='dirichlet',
    )
    
    print(f"  Classical trajectory: {len(classical_traj)} timesteps")
    print(f"  u_classical[0]: {classical_traj[0][1:-1]}")
    print(f"  u_classical[-1]: {classical_traj[-1][1:-1]}")
    print()
    
    # === QUANTUM ===
    print("Running quantum exp(-iHt)...")
    H_mat, n_qubits, dim, phys_dim, _ = build_hamiltonian(mu_bc, rho_bc, dx, nx)
    U = expm(-1j * H_mat * dt)
    
    sqrt_rho = np.sqrt(rho)
    inv_sqrt_rho = 1.0 / sqrt_rho
    
    u_current = u0_bc.copy()
    v_current = np.zeros(nx + 2)
    v_current[1:-1] = v0
    
    quantum_traj = [u_current.copy()]
    
    for step in range(steps):
        u_interior = u_current[1:-1]
        v_interior = v_current[1:-1]
        
        u_weighted = sqrt_rho * u_interior
        v_weighted = inv_sqrt_rho * v_interior
        state_vec = np.concatenate([u_weighted, v_weighted])
        
        psi = np.zeros(dim, dtype=complex)
        psi[:phys_dim] = state_vec
        psi_norm = np.linalg.norm(psi)
        if psi_norm > 1e-15:
            psi = psi / psi_norm
        else:
            psi_norm = 1.0
        
        psi_evolved = U @ psi
        state_vec_evolved = psi_evolved[:phys_dim]
        
        u_weighted_evolved = np.real(state_vec_evolved[:nx]) * psi_norm
        u_decoded = u_weighted_evolved / (sqrt_rho + 1e-30)
        
        v_weighted_evolved = np.real(state_vec_evolved[nx:phys_dim]) * psi_norm
        v_decoded = v_weighted_evolved * (sqrt_rho + 1e-30)
        
        u_current = np.zeros(nx + 2)
        u_current[1:-1] = u_decoded
        v_current = np.zeros(nx + 2)
        v_current[1:-1] = v_decoded
        
        # Dirichlet BC
        u_current[0] = 0.0
        u_current[-1] = 0.0
        
        quantum_traj.append(u_current.copy())
    
    print(f"  Quantum trajectory: {len(quantum_traj)} timesteps")
    print(f"  u_quantum[0]: {quantum_traj[0][1:-1]}")
    print(f"  u_quantum[-1]: {quantum_traj[-1][1:-1]}")
    print()
    
    # === COMPARE ===
    print("Comparison:")
    print("-"*80)
    for i in range(min(len(classical_traj), len(quantum_traj))):
        u_c = classical_traj[i][1:-1]
        u_q = quantum_traj[i][1:-1]
        
        l2_err = np.linalg.norm(u_q - u_c) / (np.linalg.norm(u_c) + 1e-30)
        
        # Overlap via amplitude encoding
        sv_c, _, _ = amplitude_encode(classical_traj[i])
        sv_q, _, _ = amplitude_encode(quantum_traj[i])
        
        if sv_c is not None and sv_q is not None:
            L = max(len(sv_c), len(sv_q))
            a = np.zeros(L, dtype=complex)
            b = np.zeros(L, dtype=complex)
            a[:len(sv_c)] = sv_c
            b[:len(sv_q)] = sv_q
            overlap = float(abs(np.vdot(a, b)) ** 2)
        else:
            overlap = 0.0
        
        print(f"  Step {i}: L2_err={l2_err:.6f}, overlap={overlap:.6f}")
        
        if i == 0:
            print(f"    u_c: {u_c}")
            print(f"    u_q: {u_q}")
    
    print()
    
    # Diagnose if problem is in evolution or encoding
    final_l2 = np.linalg.norm(quantum_traj[-1][1:-1] - classical_traj[-1][1:-1])
    final_l2_rel = final_l2 / (np.linalg.norm(classical_traj[-1][1:-1]) + 1e-30)
    
    print("DIAGNOSIS:")
    print("-"*80)
    if final_l2_rel < 0.1:
        print("  ✓ Trajectories close (L2 error < 10%)")
        print("  → Low overlap likely due to amplitude_encode() sensitivity")
    elif final_l2_rel < 0.5:
        print("  ~ Trajectories moderately different (L2 error 10-50%)")
        print("  → Some physics mismatch, but not catastrophic")
    else:
        print("  ✗ Trajectories very different (L2 error > 50%)")
        print("  → Major bug in Hamiltonian or evolution implementation")
    
    print()
    print(f"Final L2 error (relative): {final_l2_rel:.4f}")
    print(f"Expected: <0.20 for good Hamiltonian")
    

if __name__ == '__main__':
    diagnose_low_overlap()
