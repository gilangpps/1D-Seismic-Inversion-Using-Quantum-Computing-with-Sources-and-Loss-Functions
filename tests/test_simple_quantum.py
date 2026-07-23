"""
Simplified quantum_forward_simulate - direct expm(A*t) without Hermitian dilation
"""

def quantum_forward_simulate_simple(
    mu_arr, rho_arr, u0, v0,
    nx, dx, dt, steps, bc='dirichlet', source_func=None
):
    """Direct exp(A*t) evolution without dilation complexity"""
    from scipy.linalg import expm
    import numpy as np
    
    # BC padding
    u_current = np.zeros(nx + 2)
    u_current[1:-1] = u0
    v_current = np.zeros(nx + 2)
    v_current[1:-1] = v0

    rho_bc = np.zeros(nx + 2)
    rho_bc[1:-1] = rho_arr
    rho_bc[0] = rho_arr[0]
    rho_bc[-1] = rho_arr[-1]

    mu_bc = np.zeros(nx + 2)
    mu_bc[1:min(len(mu_arr)+1, nx+2)] = mu_arr[:min(len(mu_arr), nx+1)]
    mu_bc[0] = mu_arr[0]
    mu_bc[-1] = mu_arr[-1]

    # Build K matrix
    K = np.zeros((nx, nx))
    for i in range(nx):
        mu_r = 0.5 * (mu_bc[i+1] + mu_bc[i+2]) if i+1 < len(mu_bc)-1 else mu_bc[-1]
        mu_l = 0.5 * (mu_bc[i] + mu_bc[i+1])
        rho_i = rho_bc[i+1]
        
        K[i,i] = -(mu_r + mu_l) / (rho_i * dx * dx)
        if i+1 < nx:
            K[i,i+1] = mu_r / (rho_i * dx * dx)
        if i > 0:
            K[i,i-1] = mu_l / (rho_i * dx * dx)
    
    # A matrix
    A = np.zeros((2*nx, 2*nx))
    A[:nx, nx:] = np.eye(nx)
    A[nx:, :nx] = K
    
    # Evolution operator
    U = expm(A * dt)
    
    trajectory = [u_current.copy()]
    
    for step in range(steps):
        t = step * dt
        
        u_int = u_current[1:-1]
        v_int = v_current[1:-1]
        
        # State [u, v]
        state = np.concatenate([u_int, v_int])
        
        # Evolve
        state_new = U @ state
        
        u_new = state_new[:nx]
        v_new = state_new[nx:]
        
        # Source
        if source_func:
            for i in range(nx):
                src = source_func(i+1, t)
                u_new[i] += (dt**2) * src / rho_bc[i+1]
        
        u_current = np.zeros(nx+2)
        u_current[1:-1] = u_new
        v_current = np.zeros(nx+2)
        v_current[1:-1] = v_new
        
        if bc == 'dirichlet':
            u_current[0] = u_current[-1] = 0
        
        trajectory.append(u_current.copy())
    
    return trajectory


# Test
if __name__ == '__main__':
    import numpy as np
    
    nx = 7
    x_norm = np.arange(nx) / (nx-1)
    u0 = 0.5*np.sin(2*np.pi*x_norm) + 0.3*np.sin(4*np.pi*x_norm)
    u0 = u0 / np.max(np.abs(u0))
    v0 = 10 * np.cos(2*np.pi*x_norm)
    
    mu1 = np.ones(nx+1) * 1e10
    mu2 = np.ones(nx+1) * 4e10
    rho = np.ones(nx) * 2e3
    
    traj1 = quantum_forward_simulate_simple(mu1, rho, u0, v0, nx, 63, 0.005, 5)
    traj2 = quantum_forward_simulate_simple(mu2, rho, u0, v0, nx, 63, 0.005, 5)
    
    diff = np.linalg.norm(traj1[-1][1:-1] - traj2[-1][1:-1])
    print(f"||traj1 - traj2|| = {diff:.6e}")
    print("Expected: >1e-6")
    print("Result:", "PASS" if diff > 1e-6 else "FAIL")
