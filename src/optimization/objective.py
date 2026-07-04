import numpy as np
from typing import Dict, Optional, Callable

from src.wave import evolve_1d_wave
from src.encoding import quantum_reconstruct


class SeismicObjective:
    """
    Objective function for seismic inversion.

    SCIENTIFIC DESIGN:
    ─────────────────
    The inversion objective must measure the misfit between:

        u_predicted(t; mu_current)   ← forward simulation with CURRENT model
        u_reference(t; mu_true)      ← reference wavefields from the TRUE model

    NOT between a field and its own quantum reconstruction.

    Bug 1 (FIXED):
        The original code computed:
            L = (1/T) Σ_t ||u_classical(t;mu) - u_quantum(t;mu)||²

        This is the QUANTUM RECONSTRUCTION ERROR for the SAME field,
        not a seismic inversion misfit. Since u_quantum ≈ u_classical (the
        quantum circuit perfectly reconstructs the input amplitude-encoded
        state in the noiseless case), this loss is near zero regardless of
        mu and its gradient w.r.t. mu is essentially zero. The optimizer
        cannot move because there is nothing meaningful to minimize.

        CORRECT objective:
            J(mu) = (1/T) Σ_t ||u_fwd(t; mu) - u_ref(t; mu_true)||²

        where u_ref(t; mu_true) is computed ONCE from the true/reference
        model, then held fixed throughout optimization. The optimizer then
        searches for mu that minimises the misfit to the reference.

    Bug 8 (FIXED):
        forward_simulate was called WITHOUT source_func, so the optimization
        iterated on a source-free (homogeneous IC decay) wave, while the
        experiment (run_experiment_1d) used a Gaussian/Ricker source. The two
        physics were inconsistent. The source_func is now stored and passed
        through to evolve_1d_wave in every forward call.

    Mathematical formulation (corrected):
        J(m) = (1/T) Σ_t (1/N) ||u_fwd(t; m) - u_ref(t)||²

    where:
        m      = elastic modulus array (mu)
        u_fwd  = forward wavefield computed with current model m
        u_ref  = reference wavefield computed with true model m_true
        T      = number of time steps
        N      = number of spatial grid points
    """

    def __init__(
        self,
        nx: int,
        dx: float,
        dt: float,
        steps: int,
        measure_every: int = 4,
        shots: int = 1000,
        bc: str = 'dirichlet',
        seed: int = 42,
        source_func: Optional[Callable] = None,
    ):
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.steps = steps
        self.measure_every = measure_every
        self.shots = shots
        self.bc = bc
        self.seed = seed
        # Bug 8 fix: store source_func so forward_simulate is physically
        # consistent with run_experiment_1d.
        self.source_func = source_func

        # Reference fields: set once with set_reference_fields() before
        # the optimisation loop begins.
        self._reference_fields: Optional[list] = None

    # ------------------------------------------------------------------ #
    #  Reference management                                               #
    # ------------------------------------------------------------------ #

    def set_reference_fields(self, reference_fields: list) -> None:
        """
        Store the reference (true-model) wavefields.

        These are computed ONCE from the true elastic modulus and held fixed
        for all subsequent objective evaluations. The optimiser minimises
        the misfit between the current-model fields and these reference fields.

        Args:
            reference_fields: List of wavefield snapshots from the reference
                              (true) model simulation.  Each snapshot is a
                              1-D numpy array of length nx+2 (with ghost nodes).
        """
        self._reference_fields = reference_fields

    def compute_reference_fields(
        self,
        mu_true: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """
        Compute reference fields from the true model and store them.

        Call this ONCE before running optimisation.

        Args:
            mu_true:  True elastic modulus (what the inversion is trying to recover)
            rho_arr:  Density array (held fixed)
            u0:       Initial displacement
            v0:       Initial velocity

        Returns:
            List of reference wavefield snapshots.
        """
        ref_fields = self.forward_simulate(mu_true, rho_arr, u0, v0)
        self.set_reference_fields(ref_fields)
        return ref_fields

    # ------------------------------------------------------------------ #
    #  Forward simulation                                                  #
    # ------------------------------------------------------------------ #

    def forward_simulate(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
    ) -> list:
        """
        Compute forward wavefield simulation.

        Bug 8 fix: source_func is now passed to evolve_1d_wave so that the
        optimizer and the experiment use identical physics.

        Args:
            mu_arr:  Elastic modulus at grid points (length nx+1)
            rho_arr: Density at grid points (length nx)
            u0:      Initial displacement field (length nx)
            v0:      Initial velocity field (length nx)  [currently unused by
                     evolve_1d_wave which uses u0 for both u_prev and u_curr,
                     kept for API consistency]

        Returns:
            List of wavefield snapshots at each time step (length steps+1).
            Each snapshot has length nx+2 (interior + 2 ghost nodes).
        """
        nx = self.nx

        # Pad interior fields to include ghost (boundary) nodes
        u0_bc = np.zeros(nx + 2)
        u0_bc[1:-1] = u0
        u1_bc = u0_bc.copy()

        rho_bc = np.zeros(nx + 2)
        rho_bc[1:-1] = rho_arr
        rho_bc[0] = rho_arr[0]
        rho_bc[-1] = rho_arr[-1]

        mu_bc = np.zeros(nx + 2)
        mu_bc[1:min(len(mu_arr) + 1, nx + 2)] = mu_arr[:min(len(mu_arr), nx + 1)]
        mu_bc[0] = mu_arr[0]
        mu_bc[-1] = mu_arr[-1]

        # Bug 8 fix: pass self.source_func (may be None for sourceless runs,
        # but must match whatever was used to produce the reference fields).
        fields = evolve_1d_wave(
            u0_bc, u1_bc, dx=self.dx, dt=self.dt,
            mu=mu_bc, rho=rho_bc,
            source_func=self.source_func,   # ← was hardcoded None before
            steps=self.steps, bc=self.bc
        )
        return fields

    # ------------------------------------------------------------------ #
    #  Loss computation                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        fields: list,
        use_deterministic: bool = False,
    ) -> float:
        """
        Compute misfit loss between current-model fields and reference fields.

        Bug 1 fix (corrected objective):
            L(mu) = (1/T) Σ_t (1/N) ||u_fwd(t; mu) - u_ref(t)||²

        The reference fields must be set before calling this method
        (via set_reference_fields or compute_reference_fields).

        If no reference fields are set, falls back to reconstruction loss
        with a loud warning so the user is alerted.

        Args:
            fields:            Forward-simulation wavefields for current mu.
            use_deterministic: If True, skip shot noise in quantum recon.

        Returns:
            Scalar misfit loss.
        """
        if self._reference_fields is None:
            # Fallback: quantum reconstruction error against self (old buggy
            # behaviour).  Warn loudly so the user notices.
            import warnings
            warnings.warn(
                "SeismicObjective: no reference fields set. "
                "Falling back to reconstruction loss (original buggy behaviour). "
                "Call set_reference_fields() or compute_reference_fields() "
                "before running optimisation.",
                UserWarning,
                stacklevel=2,
            )
            return self._compute_reconstruction_loss(fields, use_deterministic)

        return self._compute_misfit_loss(fields)

    def _compute_misfit_loss(self, fields: list) -> float:
        """
        Misfit loss: ||u_fwd(t; mu) - u_ref(t)||²  averaged over time.

        This is the scientifically correct objective for seismic inversion.
        """
        ref = self._reference_fields
        n_steps = min(len(fields), len(ref))
        loss_arr = []

        for i in range(1, n_steps):
            u_fwd = fields[i][1:-1]    # strip ghost nodes
            u_ref = ref[i][1:-1]       # strip ghost nodes
            loss_val = np.mean((u_fwd - u_ref) ** 2)
            loss_arr.append(loss_val)

        if not loss_arr:
            return 0.0
        return float(np.mean(loss_arr))

    def _compute_reconstruction_loss(
        self,
        fields: list,
        use_deterministic: bool = False,
    ) -> float:
        """
        Quantum reconstruction error (original code, kept for diagnostics).

        NOTE: This is NOT a suitable inversion objective because it measures
        how well the quantum circuit reconstructs its own input, not how close
        the current model is to the true model.  Use only for diagnostic
        comparison with the true misfit.
        """
        loss_arr = []
        for i in range(1, len(fields)):
            qr = quantum_reconstruct(
                fields[i], shots=None if use_deterministic else self.shots
            )
            u_classical = fields[i][1:-1]
            u_quantum = qr[1:-1]
            loss_val = np.mean((u_classical - u_quantum) ** 2)
            loss_arr.append(loss_val)
        if not loss_arr:
            return 0.0
        return float(np.mean(loss_arr))

    # ------------------------------------------------------------------ #
    #  Convenience wrappers                                               #
    # ------------------------------------------------------------------ #

    def compute_loss_and_fields(
        self,
        mu_arr: np.ndarray,
        rho_arr: np.ndarray,
        u0: np.ndarray,
        v0: np.ndarray,
        use_deterministic: bool = False,
    ) -> tuple:
        """
        Run forward simulation and compute misfit loss in one call.

        Returns:
            Tuple of (loss: float, fields: list)
        """
        fields = self.forward_simulate(mu_arr, rho_arr, u0, v0)
        loss = self.compute_loss(fields, use_deterministic)
        return loss, fields

    def compute_time_series_loss(
        self,
        fields: list,
    ) -> np.ndarray:
        """
        Per-timestep misfit loss (not averaged over time).

        Useful for debugging and visualisation.

        Returns:
            1-D array of per-timestep losses (length = len(fields)-1).
        """
        if self._reference_fields is None:
            # Fallback to reconstruction loss for diagnostics
            loss_arr = []
            for i in range(1, len(fields)):
                qr = quantum_reconstruct(fields[i], shots=None)
                u_c = fields[i][1:-1]
                u_q = qr[1:-1]
                loss_arr.append(np.mean((u_c - u_q) ** 2))
            return np.array(loss_arr)

        ref = self._reference_fields
        n_steps = min(len(fields), len(ref))
        loss_arr = []
        for i in range(1, n_steps):
            u_fwd = fields[i][1:-1]
            u_ref = ref[i][1:-1]
            loss_arr.append(np.mean((u_fwd - u_ref) ** 2))
        return np.array(loss_arr)

    def compute_overlap_with_reference(self, fields: list) -> list:
        """
        Compute quantum-state overlap between current-model fields and
        reference fields at every measured timestep.

        Overlap ∈ [0, 1]:  1 = perfect agreement, 0 = orthogonal.

        Returns:
            List of (time, overlap) tuples.
        """
        from src.encoding import amplitude_encode

        ref = self._reference_fields
        if ref is None:
            return []

        overlaps = []
        n_steps = min(len(fields), len(ref))
        for t_idx in range(1, n_steps):
            if t_idx % self.measure_every != 0:
                continue
            sv_fwd, _, _ = amplitude_encode(fields[t_idx])
            sv_ref, _, _ = amplitude_encode(ref[t_idx])
            if sv_fwd is None or sv_ref is None:
                overlaps.append((t_idx * self.dt, 0.0))
                continue
            L = max(len(sv_fwd), len(sv_ref))
            a = np.zeros(L, dtype=complex)
            b = np.zeros(L, dtype=complex)
            a[:len(sv_fwd)] = sv_fwd
            b[:len(sv_ref)] = sv_ref
            ov = abs(np.vdot(a, b)) ** 2
            overlaps.append((t_idx * self.dt, float(ov)))
        return overlaps
