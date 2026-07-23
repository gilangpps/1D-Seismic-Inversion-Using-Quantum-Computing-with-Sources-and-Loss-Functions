import sys

NEW_METHOD = '''    def quantum_forward_simulate(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """Direct exp(A*t) evolution without Hermitian dilation."""
        from scipy.linalg import expm
        
        nx = self.nx
        u_current = np.zeros(nx + 2)
        u_current[1:-1] = u0
        v_current = np.zeros(nx + 2)
        v_current[1:-1] = v0

        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0] = rho_arr[0]
        rho_bc[-1] = rho_arr[-1]
        if np.any(rho_bc == 0):
            rho_bc[rho_bc == 0] = float(np.mean(rho_arr))

        mu_bc = np.zeros(nx + 2)
        n_mu = len(mu_arr)
        mu_bc[1:min(n_mu+1, nx+2)] = mu_arr[:min(n_mu, nx+1)]
        mu_bc[0] = mu_arr[0]
        mu_bc[-1] = mu_arr[-1]

        K = np.zeros((nx, nx), dtype=float)
        for i in range(nx):
            mu_r = 0.5 * (mu_bc[min(i+1, len(mu_bc)-1)] + mu_bc[min(i+2, len(mu_bc)-1)])
            mu_l = 0.5 * (mu_bc[max(i, 0)] + mu_bc[min(i+1, len(mu_bc)-1)])
            rho_i = max(rho_bc[i+1], 1e-30)
            K[i,i] = -(mu_r + mu_l) / (rho_i * self.dx * self.dx)
            if i+1 < nx:
                K[i,i+1] = mu_r / (rho_i * self.dx * self.dx)
            if i > 0:
                K[i,i-1] = mu_l / (rho_i * self.dx * self.dx)
        
        A = np.zeros((2*nx, 2*nx), dtype=float)
        A[:nx, nx:] = np.eye(nx)
        A[nx:, :nx] = K
        U = expm(A * self.dt)
        
        if not hasattr(self, '_quantum_sim_logged'):
            print(f"[QUANTUM] Direct exp(A*t): {2*nx}x{2*nx}, steps={self.steps}")
            self._quantum_sim_logged = True
        
        trajectory = [u_current.copy()]
        for step in range(self.steps):
            t = step * self.dt
            u_int = u_current[1:-1]
            v_int = v_current[1:-1]
            state = np.concatenate([u_int, v_int])
            state_new = U @ state
            u_new = state_new[:nx]
            v_new = state_new[nx:]
            if self.source_func is not None:
                for i in range(nx):
                    src_val = self.source_func(i+1, t)
                    u_new[i] += (self.dt**2) * src_val / rho_bc[i+1]
            u_current = np.zeros(nx+2)
            u_current[1:-1] = u_new
            v_current = np.zeros(nx+2)
            v_current[1:-1] = v_new
            if self.bc == 'dirichlet':
                u_current[0] = u_current[-1] = 0.0
            elif self.bc == 'neumann':
                u_current[0] = u_current[1]
                u_current[-1] = u_current[-2]
            trajectory.append(u_current.copy())
        return trajectory
'''

with open('src/optimization/objective.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start = None
end = None
for i, line in enumerate(lines):
    if 'def quantum_forward_simulate(' in line:
        start = i
    if start is not None and 'return trajectory' in line and end is None:
        end = i + 1
        break

if start is None or end is None:
    print(f"ERROR: start={start}, end={end}")
    sys.exit(1)

print(f"Replacing lines {start+1}-{end} ({end-start} lines)")
new_lines = lines[:start] + [NEW_METHOD + '\n'] + lines[end:]

with open('src/optimization/objective.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("SUCCESS")
