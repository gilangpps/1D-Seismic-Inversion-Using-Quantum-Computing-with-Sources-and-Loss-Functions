"""
Update validate_hamiltonian.py to use direct expm(A*t) like objective.py
"""

NEW_QUANTUM_SECTION = '''    # ── 2. Run quantum simulation ────────────────────────────────────────
    print("[2/3] Running quantum exp(-iHt) evolution...")
    
    # Build K matrix directly (same as objective.py)
    K = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        mu_r = 0.5 * (mu_bc[min(i+1, len(mu_bc)-1)] + mu_bc[min(i+2, len(mu_bc)-1)])
        mu_l = 0.5 * (mu_bc[max(i, 0)] + mu_bc[min(i+1, len(mu_bc)-1)])
        rho_i = max(rho_bc[i+1], 1e-30)
        K[i,i] = -(mu_r + mu_l) / (rho_i * dx * dx)
        if i+1 < nx:
            K[i,i+1] = mu_r / (rho_i * dx * dx)
        if i > 0:
            K[i,i-1] = mu_l / (rho_i * dx * dx)
    
    # A matrix: [[0, I], [K, 0]]
    A = np.zeros((2*nx, 2*nx), dtype=float)
    A[:nx, nx:] = np.eye(nx)
    A[nx:, :nx] = K
    
    # Direct evolution (no dilation)
    U = expm(A * dt)
    
    print(f"  • System matrix A: {2*nx}×{2*nx} (direct, no dilation)")
    
    u_current = np.zeros(nx + 2)
    u_current[1:-1] = u0
    v_current = np.zeros(nx + 2)
    v_current[1:-1] = v0

    quantum_trajectory = [u_current.copy()]

    # Time evolution loop
    for step in range(steps):
        u_int = u_current[1:-1]
        v_int = v_current[1:-1]
        
        state = np.concatenate([u_int, v_int])
        state_new = U @ state
        
        u_new = state_new[:nx]
        v_new = state_new[nx:]
        
        u_current = np.zeros(nx+2)
        u_current[1:-1] = u_new
        v_current = np.zeros(nx+2)
        v_current[1:-1] = v_new
        
        if bc == 'dirichlet':
            u_current[0] = u_current[-1] = 0.0
        elif bc == 'neumann':
            u_current[0] = u_current[1]
            u_current[-1] = u_current[-2]
        
        quantum_trajectory.append(u_current.copy())
'''

# Read file
with open('src/experiment/validate_hamiltonian.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find section to replace
start_marker = '    # ── 2. Run quantum simulation ────────────────────────────────────────'
end_marker = '    print(f"  [OK] Quantum trajectory: {len(quantum_trajectory)} timesteps")'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print(f"ERROR: Markers not found")
    print(f"start_idx={start_idx}, end_idx={end_idx}")
    exit(1)

# Replace
new_content = content[:start_idx] + NEW_QUANTUM_SECTION + '\n\n    ' + content[end_idx:]

# Write
with open('src/experiment/validate_hamiltonian.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("SUCCESS: validate_hamiltonian.py updated to use direct expm(A*t)")
