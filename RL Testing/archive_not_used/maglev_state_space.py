"""
Full Linearized State-Space Model for the Guadaloop Maglev Pod
==============================================================

Combines three dynamic layers into a single LTI system  ẋ = Ax + Bu,  y = Cx:

  Layer 1 — Coil RL dynamics (electrical):
      di/dt = (V·pwm − R·i) / L
      This is already linear.  A first-order lag from PWM command to current.

  Layer 2 — Electromagnetic force/torque map (from Ansys polynomial):
      (F, τ) = f(iL, iR, roll, gap)
      Nonlinear, but the MaglevLinearizer gives us the Jacobian at any
      operating point, making it locally linear.

  Layer 3 — Rigid-body mechanics (Newton/Euler):
      m·z̈  = F_front + F_back − m·g       (heave)
      Iy·θ̈  = L_arm·(F_front − F_back)     (pitch from force differential)
      Ix·φ̈  = τ_front + τ_back              (roll from magnetic torque)
      These are linear once the force/torque are linearized.

The key coupling: the pod is rigid, so front and back yoke gaps are NOT
independent.  They are related to the average gap and pitch angle:

      gap_front = gap_avg − L_arm · pitch
      gap_back  = gap_avg + L_arm · pitch

This means a pitch perturbation changes both yoke gaps, which changes both
yoke forces, which feeds back into the heave and pitch dynamics.  The
electromagnetic Jacobian captures how force/torque respond to these gap
changes, creating the destabilizing "magnetic stiffness" that makes maglev
inherently open-loop unstable.

State vector (10 states):
    x = [gap_avg, gap_vel, pitch, pitch_rate, roll, roll_rate,
         i_FL, i_FR, i_BL, i_BR]

    - gap_avg [m]:    average air gap (track-to-yoke distance)
    - gap_vel [m/s]:  d(gap_avg)/dt
    - pitch [rad]:    rotation about Y axis (positive = back hangs lower)
    - pitch_rate [rad/s]
    - roll [rad]:     rotation about X axis
    - roll_rate [rad/s]
    - i_FL..BR [A]:   the four coil currents

Input vector (4 inputs):
    u = [pwm_FL, pwm_FR, pwm_BL, pwm_BR]   (duty cycles, dimensionless)

Output vector (3 outputs):
    y = [gap_avg, pitch, roll]
"""

import numpy as np
import os
from maglev_linearizer import MaglevLinearizer

# ---------------------------------------------------------------------------
# Physical constants and unit conversions
# ---------------------------------------------------------------------------
GRAVITY = 9.81          # m/s²
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# State indices (for readability)
GAP, GAPV, PITCH, PITCHV, ROLL, ROLLV, I_FL, I_FR, I_BL, I_BR = range(10)


# ===================================================================
# StateSpaceResult — the output container
# ===================================================================
class StateSpaceResult:
    """
    Holds the A, B, C, D matrices of the linearized plant plus
    operating-point metadata and stability analysis.
    """

    STATE_LABELS = [
        'gap_avg [m]', 'gap_vel [m/s]',
        'pitch [rad]', 'pitch_rate [rad/s]',
        'roll [rad]', 'roll_rate [rad/s]',
        'i_FL [A]', 'i_FR [A]', 'i_BL [A]', 'i_BR [A]',
    ]
    INPUT_LABELS = ['pwm_FL', 'pwm_FR', 'pwm_BL', 'pwm_BR']
    OUTPUT_LABELS = ['gap_avg [m]', 'pitch [rad]', 'roll [rad]']

    def __init__(self, A, B, C, D, operating_point,
                 equilibrium_force_error, plant_front, plant_back):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.operating_point = operating_point
        self.equilibrium_force_error = equilibrium_force_error
        self.plant_front = plant_front    # LinearizedPlant for front yoke
        self.plant_back = plant_back      # LinearizedPlant for back yoke

    @property
    def eigenvalues(self):
        """Eigenvalues of A, sorted by decreasing real part."""
        eigs = np.linalg.eigvals(self.A)
        return eigs[np.argsort(-np.real(eigs))]

    @property
    def is_open_loop_stable(self):
        return bool(np.all(np.real(self.eigenvalues) < 0))

    @property
    def unstable_eigenvalues(self):
        eigs = self.eigenvalues
        return eigs[np.real(eigs) > 1e-8]

    def to_scipy(self):
        """Convert to scipy.signal.StateSpace for frequency-domain analysis."""
        from scipy.signal import StateSpace
        return StateSpace(self.A, self.B, self.C, self.D)

    def print_A_structure(self):
        """Print the A matrix with row/column labels for physical insight."""
        labels_short = ['gap', 'ġap', 'θ', 'θ̇', 'φ', 'φ̇',
                        'iFL', 'iFR', 'iBL', 'iBR']
        print("\nA matrix (non-zero entries):")
        print("-" * 65)
        for i in range(10):
            for j in range(10):
                if abs(self.A[i, j]) > 1e-10:
                    print(f"  A[{labels_short[i]:>3}, {labels_short[j]:>3}] "
                          f"= {self.A[i,j]:+12.4f}")
        print("-" * 65)

    def print_B_structure(self):
        """Print the B matrix with labels."""
        labels_short = ['gap', 'ġap', 'θ', 'θ̇', 'φ', 'φ̇',
                        'iFL', 'iFR', 'iBL', 'iBR']
        u_labels = ['uFL', 'uFR', 'uBL', 'uBR']
        print("\nB matrix (non-zero entries):")
        print("-" * 50)
        for i in range(10):
            for j in range(4):
                if abs(self.B[i, j]) > 1e-10:
                    print(f"  B[{labels_short[i]:>3}, {u_labels[j]:>3}] "
                          f"= {self.B[i,j]:+12.4f}")
        print("-" * 50)

    def __repr__(self):
        op = self.operating_point
        eigs = self.eigenvalues

        at_eq = abs(self.equilibrium_force_error) < 0.5
        eq_str = ('AT EQUILIBRIUM' if at_eq
                  else f'NOT AT EQUILIBRIUM — {self.equilibrium_force_error:+.2f} N residual')

        lines = [
            "=" * 70,
            "LINEARIZED MAGLEV STATE-SPACE  (ẋ = Ax + Bu,  y = Cx)",
            "=" * 70,
            f"Operating point:",
            f"  gap = {op['gap_height']:.2f} mm,  "
            f"currL = {op['currL']:.2f} A,  "
            f"currR = {op['currR']:.2f} A,  "
            f"roll = {op['roll']:.1f}°,  "
            f"pitch = {op['pitch']:.1f}°",
            f"  F_front = {self.plant_front.f0:.3f} N,  "
            f"F_back = {self.plant_back.f0:.3f} N,  "
            f"F_total = {self.plant_front.f0 + self.plant_back.f0:.3f} N,  "
            f"Weight = {op['mass'] * GRAVITY:.3f} N",
            f"  >> {eq_str}",
            "",
            f"System: {self.A.shape[0]} states × "
            f"{self.B.shape[1]} inputs × "
            f"{self.C.shape[0]} outputs",
            f"Open-loop stable: {self.is_open_loop_stable}",
            "",
            "Eigenvalues of A:",
        ]

        # Group complex conjugate pairs
        printed = set()
        for i, ev in enumerate(eigs):
            if i in printed:
                continue
            re_part = np.real(ev)
            im_part = np.imag(ev)
            stability = "UNSTABLE" if re_part > 1e-8 else "stable"

            if abs(im_part) < 1e-6:
                lines.append(
                    f"  λ = {re_part:+12.4f}              "
                    f"  τ = {abs(1/re_part)*1000 if abs(re_part) > 1e-8 else float('inf'):.2f} ms"
                    f"  ({stability})"
                )
            else:
                # Find conjugate pair
                for j in range(i + 1, len(eigs)):
                    if j not in printed and abs(eigs[j] - np.conj(ev)) < 1e-6:
                        printed.add(j)
                        break
                omega_n = abs(ev)
                lines.append(
                    f"  λ = {re_part:+12.4f} ± {abs(im_part):.4f}j"
                    f"  ω_n = {omega_n:.1f} rad/s"
                    f"  ({stability})"
                )

        lines.extend(["", "=" * 70])
        return '\n'.join(lines)


# ===================================================================
# MaglevStateSpace — the builder
# ===================================================================
class MaglevStateSpace:
    """
    Assembles the full 10-state linearized state-space from the
    electromagnetic Jacobian + rigid body dynamics + coil dynamics.

    Physical parameters come from the URDF (pod.xml) and MagLevCoil.
    """

    def __init__(self, linearizer,
                 mass=5.8,
                 I_roll=0.0192942414,     # Ixx from pod.xml [kg·m²]
                 I_pitch=0.130582305,     # Iyy from pod.xml [kg·m²]
                 coil_R=1.1,              # from MagLevCoil in lev_pod_env.py
                 coil_L=0.0025,           # 2.5 mH
                 V_supply=12.0,           # supply voltage [V]
                 L_arm=0.1259):           # front/back yoke X-offset [m]
        self.linearizer = linearizer
        self.mass = mass
        self.I_roll = I_roll
        self.I_pitch = I_pitch
        self.coil_R = coil_R
        self.coil_L = coil_L
        self.V_supply = V_supply
        self.L_arm = L_arm

    @staticmethod
    def _convert_jacobian_to_si(jac):
        """
        Convert a linearizer Jacobian from mixed units to pure SI.

        The linearizer returns:
            Row 0: Force  [N]     per  [A, A, deg, mm]
            Row 1: Torque [mN·m]  per  [A, A, deg, mm]

        We need:
            Row 0: Force  [N]     per  [A, A, rad, m]
            Row 1: Torque [N·m]   per  [A, A, rad, m]

        Conversion factors:
            col 0,1 (current): ×1 for force,  ×(1/1000) for torque
            col 2   (roll):    ×(180/π) for force,  ×(180/π)/1000 for torque
            col 3   (gap):     ×1000 for force,  ×(1000/1000)=×1 for torque
        """
        si = np.zeros((2, 4))

        # Force row — already in N
        si[0, 0] = jac[0, 0]                     # N/A → N/A
        si[0, 1] = jac[0, 1]                     # N/A → N/A
        si[0, 2] = jac[0, 2] * RAD2DEG           # N/deg → N/rad
        si[0, 3] = jac[0, 3] * 1000.0            # N/mm → N/m

        # Torque row — from mN·m to N·m
        si[1, 0] = jac[1, 0] / 1000.0            # mN·m/A → N·m/A
        si[1, 1] = jac[1, 1] / 1000.0            # mN·m/A → N·m/A
        si[1, 2] = jac[1, 2] / 1000.0 * RAD2DEG  # mN·m/deg → N·m/rad
        si[1, 3] = jac[1, 3]                      # mN·m/mm → N·m/m (factors cancel)

        return si

    def build(self, gap_height, currL, currR, roll=0.0, pitch=0.0):
        """
        Build the A, B, C, D matrices at a given operating point.

        Parameters
        ----------
        gap_height : float  Average gap [mm]
        currL : float       Equilibrium left coil current [A] (same front & back)
        currR : float       Equilibrium right coil current [A]
        roll : float        Equilibrium roll angle [deg], default 0
        pitch : float       Equilibrium pitch angle [deg], default 0
                            Non-zero pitch means front/back gaps differ.

        Returns
        -------
        StateSpaceResult
        """
        m = self.mass
        Ix = self.I_roll
        Iy = self.I_pitch
        R = self.coil_R
        Lc = self.coil_L
        V = self.V_supply
        La = self.L_arm

        # ------------------------------------------------------------------
        # Step 1: Compute individual yoke gaps from average gap + pitch
        #
        # The pod is rigid.  If it pitches, the front and back yoke ends
        # are at different distances from the track:
        #   gap_front = gap_avg − L_arm · sin(pitch) ≈ gap_avg − L_arm · pitch
        #   gap_back  = gap_avg + L_arm · sin(pitch) ≈ gap_avg + L_arm · pitch
        #
        # Sign convention (from lev_pod_env.py lines 230-232):
        #   positive pitch = back gap > front gap  (back hangs lower)
        # ------------------------------------------------------------------
        pitch_rad = pitch * DEG2RAD
        # L_arm [m] * sin(pitch) [rad] → meters; convert to mm for linearizer
        gap_front_mm = gap_height - La * np.sin(pitch_rad) * 1000.0
        gap_back_mm = gap_height + La * np.sin(pitch_rad) * 1000.0

        # ------------------------------------------------------------------
        # Step 2: Linearize each yoke independently
        #
        # Each U-yoke has its own (iL, iR) pair and sees its own gap.
        # Both yokes see the same roll angle (the pod is rigid).
        # The linearizer returns the Jacobian in mixed units.
        # ------------------------------------------------------------------
        plant_f = self.linearizer.linearize(currL, currR, roll, gap_front_mm)
        plant_b = self.linearizer.linearize(currL, currR, roll, gap_back_mm)

        # ------------------------------------------------------------------
        # Step 3: Convert Jacobians to SI
        #
        # After this, all gains are in [N or N·m] per [A, A, rad, m].
        # Columns: [currL, currR, roll, gap_height]
        # ------------------------------------------------------------------
        Jf = self._convert_jacobian_to_si(plant_f.jacobian)
        Jb = self._convert_jacobian_to_si(plant_b.jacobian)

        # Unpack for clarity — subscript _f = front yoke, _b = back yoke
        # Force gains
        kFiL_f, kFiR_f, kFr_f, kFg_f = Jf[0]
        kFiL_b, kFiR_b, kFr_b, kFg_b = Jb[0]
        # Torque gains
        kTiL_f, kTiR_f, kTr_f, kTg_f = Jf[1]
        kTiL_b, kTiR_b, kTr_b, kTg_b = Jb[1]

        # ------------------------------------------------------------------
        # Step 4: Assemble the A matrix (10 × 10)
        #
        # The A matrix encodes three kinds of coupling:
        #
        #  (a) Kinematic identities:  gap_vel = d(gap)/dt, etc.
        #      These are always 1.0 on the super-diagonal of the
        #      position/velocity pairs.
        #
        #  (b) Electromagnetic coupling through current states:
        #      Coil currents produce forces/torques.  The linearized
        #      gains (∂F/∂i, ∂T/∂i) appear in the acceleration rows.
        #      This is the path from current states to mechanical
        #      acceleration — the "plant gain" that PID acts through.
        #
        #  (c) Electromagnetic coupling through mechanical states:
        #      Gap and roll perturbations change the force/torque.
        #      This creates feedback loops:
        #
        #      - ∂F/∂gap < 0  → gap perturbation changes force in a
        #        direction that AMPLIFIES the gap error  → UNSTABLE
        #        (magnetic stiffness is "negative spring")
        #
        #      - ∂T/∂roll     → roll perturbation changes torque;
        #        sign determines whether roll is self-correcting or not
        #
        #      - Pitch couples through gap_front/gap_back dependence
        #        on pitch angle, creating pitch instability too
        # ------------------------------------------------------------------
        A = np.zeros((10, 10))

        # (a) Kinematic identities
        A[GAP, GAPV] = 1.0
        A[PITCH, PITCHV] = 1.0
        A[ROLL, ROLLV] = 1.0

        # ------------------------------------------------------------------
        # HEAVE:  m · Δgap̈ = −(ΔF_front + ΔF_back)
        #
        # The negative sign is because force is upward (+Z) but gap
        # is measured downward (gap shrinks when pod moves up).
        # At equilibrium F₀ = mg; perturbation ΔF pushes pod up → gap shrinks.
        #
        # Expanding ΔF using the rigid-body gap coupling:
        #   ΔF_front = kFg_f·(Δgap − La·Δpitch) + kFr_f·Δroll + kFiL_f·ΔiFL + kFiR_f·ΔiFR
        #   ΔF_back  = kFg_b·(Δgap + La·Δpitch) + kFr_b·Δroll + kFiL_b·ΔiBL + kFiR_b·ΔiBR
        # ------------------------------------------------------------------
        # Gap → gap acceleration  (magnetic stiffness, UNSTABLE)
        A[GAPV, GAP] = -(kFg_f + kFg_b) / m
        # Pitch → gap acceleration  (cross-coupling through differential gap)
        A[GAPV, PITCH] = -(-kFg_f + kFg_b) * La / m
        # Roll → gap acceleration
        A[GAPV, ROLL] = -(kFr_f + kFr_b) / m
        # Current → gap acceleration  (the control path!)
        A[GAPV, I_FL] = -kFiL_f / m
        A[GAPV, I_FR] = -kFiR_f / m
        A[GAPV, I_BL] = -kFiL_b / m
        A[GAPV, I_BR] = -kFiR_b / m

        # ------------------------------------------------------------------
        # PITCH:  Iy · Δpitcḧ = La · (ΔF_front − ΔF_back)
        #
        # Pitch torque comes from DIFFERENTIAL FORCE, not from the
        # electromagnetic torque (which acts on roll).  This is because
        # the front yoke is at x = +La and the back at x = −La:
        #   τ_pitch = F_front·La − F_back·La = La·(F_front − F_back)
        #
        # At symmetric equilibrium, F_front = F_back → zero pitch torque. ✓
        # A pitch perturbation breaks this symmetry through the gap coupling.
        # ------------------------------------------------------------------
        # Gap → pitch acceleration  (zero at symmetric equilibrium)
        A[PITCHV, GAP] = La * (kFg_f - kFg_b) / Iy
        # Pitch → pitch acceleration  (pitch instability — UNSTABLE)
        # = −La²·(kFg_f + kFg_b)/Iy.  Since kFg < 0 → positive → unstable.
        A[PITCHV, PITCH] = -La**2 * (kFg_f + kFg_b) / Iy
        # Roll → pitch acceleration
        A[PITCHV, ROLL] = La * (kFr_f - kFr_b) / Iy
        # Current → pitch acceleration
        A[PITCHV, I_FL] = La * kFiL_f / Iy
        A[PITCHV, I_FR] = La * kFiR_f / Iy
        A[PITCHV, I_BL] = -La * kFiL_b / Iy
        A[PITCHV, I_BR] = -La * kFiR_b / Iy

        # ------------------------------------------------------------------
        # ROLL:  Ix · Δroll̈ = Δτ_front + Δτ_back
        #
        # Unlike pitch (driven by force differential), roll is driven by
        # the electromagnetic TORQUE directly.  In the Ansys model, torque
        # is the moment about the X axis produced by the asymmetric flux
        # in the left vs right legs of each U-yoke.
        #
        # The torque Jacobian entries determine stability:
        #   - ∂T/∂roll:  if this causes torque that amplifies roll → unstable
        #   - ∂T/∂iL, ∂T/∂iR:  how current asymmetry controls roll
        # ------------------------------------------------------------------
        # Gap → roll acceleration
        A[ROLLV, GAP] = (kTg_f + kTg_b) / Ix
        # Pitch → roll acceleration  (cross-coupling)
        A[ROLLV, PITCH] = (-kTg_f + kTg_b) * La / Ix
        # Roll → roll acceleration  (roll stiffness)
        A[ROLLV, ROLL] = (kTr_f + kTr_b) / Ix
        # Current → roll acceleration
        A[ROLLV, I_FL] = kTiL_f / Ix
        A[ROLLV, I_FR] = kTiR_f / Ix
        A[ROLLV, I_BL] = kTiL_b / Ix
        A[ROLLV, I_BR] = kTiR_b / Ix

        # ------------------------------------------------------------------
        # COIL DYNAMICS:  L·di/dt = V·pwm − R·i
        #
        # Rearranged:  di/dt = −(R/L)·i + (V/L)·pwm
        #
        # This is a simple first-order lag with:
        #   - Time constant τ_coil = L/R = 2.5ms/1.1 = 2.27 ms
        #   - Eigenvalue = −R/L = −440  (very fast, well-damped)
        #
        # The coil dynamics act as a low-pass filter between the PWM
        # command and the actual current.  For PID frequencies below
        # ~100 Hz, this lag is small but not negligible.
        # ------------------------------------------------------------------
        for k in range(I_FL, I_BR + 1):
            A[k, k] = -R / Lc

        # ------------------------------------------------------------------
        # Step 5: B matrix (10 × 4)
        #
        # Only the coil states respond directly to the PWM inputs.
        # The mechanical states are affected INDIRECTLY: pwm → current
        # → force/torque → acceleration.  This indirect path shows up
        # as the product A_mech_curr × B_curr_pwm in the transfer function.
        #
        # B[coil_k, pwm_k] = V_supply / L_coil
        # ------------------------------------------------------------------
        B = np.zeros((10, 4))
        for k in range(4):
            B[I_FL + k, k] = V / Lc

        # ------------------------------------------------------------------
        # Step 6: C matrix (3 × 10)
        #
        # Default outputs are the three controlled DOFs:
        #   gap_avg, pitch, roll
        # These are directly the position states.
        # ------------------------------------------------------------------
        C = np.zeros((3, 10))
        C[0, GAP] = 1.0      # gap_avg
        C[1, PITCH] = 1.0    # pitch
        C[2, ROLL] = 1.0     # roll

        # D = 0 (no direct feedthrough from PWM to position)
        D = np.zeros((3, 4))

        # ------------------------------------------------------------------
        # Step 7: Equilibrium check
        #
        # At a valid operating point, the total magnetic force should
        # equal the pod weight.  A large residual means the linearization
        # is valid mathematically but not physically meaningful (the pod
        # wouldn't hover at this point without acceleration).
        # ------------------------------------------------------------------
        F_total = plant_f.f0 + plant_b.f0
        weight = m * GRAVITY
        eq_error = F_total - weight

        return StateSpaceResult(
            A=A, B=B, C=C, D=D,
            operating_point={
                'gap_height': gap_height,
                'currL': currL, 'currR': currR,
                'roll': roll, 'pitch': pitch,
                'mass': m,
            },
            equilibrium_force_error=eq_error,
            plant_front=plant_f,
            plant_back=plant_b,
        )

    def find_equilibrium_current(self, gap_height, roll=0.0, tol=0.01):
        """
        Find the symmetric current (currL = currR = I) that makes
        total force = weight at the given gap height.

        Uses bisection over the current range.  The search assumes
        negative currents produce attractive (upward) force, which
        matches the Ansys model convention.

        Parameters
        ----------
        gap_height : float  Target gap [mm]
        roll : float        Roll angle [deg], default 0
        tol : float         Force tolerance [N]

        Returns
        -------
        float : equilibrium current [A]
        """
        target_per_yoke = self.mass * GRAVITY / 2.0

        def force_residual(curr):
            f, _ = self.linearizer.evaluate(curr, curr, roll, gap_height)
            return f - target_per_yoke

        # Bisection search over negative current range
        # (More negative = stronger attraction)
        a, b = -20.0, 0.0
        fa, fb = force_residual(a), force_residual(b)

        if fa * fb > 0:
            # Try positive range too
            a, b = 0.0, 20.0
            fa, fb = force_residual(a), force_residual(b)
            if fa * fb > 0:
                raise ValueError(
                    f"Cannot find equilibrium current at gap={gap_height}mm. "
                    f"Force at I=−20A: {target_per_yoke + force_residual(-20):.1f}N, "
                    f"at I=0: {target_per_yoke + force_residual(0):.1f}N, "
                    f"at I=+20A: {target_per_yoke + force_residual(20):.1f}N, "
                    f"target per yoke: {target_per_yoke:.1f}N"
                )

        for _ in range(100):
            mid = (a + b) / 2.0
            fmid = force_residual(mid)
            if abs(fmid) < tol:
                return mid
            if fa * fmid < 0:
                b = mid
            else:
                a, fa = mid, fmid

        return (a + b) / 2.0


# ======================================================================
# Demo
# ======================================================================
if __name__ == '__main__':
    model_path = os.path.join(os.path.dirname(__file__), 'maglev_model.pkl')
    lin = MaglevLinearizer(model_path)
    ss = MaglevStateSpace(lin)

    # ------------------------------------------------------------------
    # Find the equilibrium current at the target gap
    # ------------------------------------------------------------------
    TARGET_GAP_MM = 16.491741  # from lev_pod_env.py
    print("=" * 70)
    print("FINDING EQUILIBRIUM CURRENT")
    print("=" * 70)
    I_eq = ss.find_equilibrium_current(TARGET_GAP_MM)
    F_eq, T_eq = lin.evaluate(I_eq, I_eq, 0.0, TARGET_GAP_MM)
    print(f"Target gap:    {TARGET_GAP_MM:.3f} mm")
    print(f"Pod weight:    {ss.mass * GRAVITY:.3f} N  ({ss.mass} kg)")
    print(f"Required per yoke: {ss.mass * GRAVITY / 2:.3f} N")
    print(f"Equilibrium current: {I_eq:.4f} A  (symmetric, currL = currR)")
    print(f"Force per yoke at equilibrium: {F_eq:.3f} N")
    print(f"Equilibrium PWM duty cycle: {I_eq * ss.coil_R / ss.V_supply:.4f}")
    print()

    # ------------------------------------------------------------------
    # Build the state-space at equilibrium
    # ------------------------------------------------------------------
    result = ss.build(
        gap_height=TARGET_GAP_MM,
        currL=I_eq,
        currR=I_eq,
        roll=0.0,
        pitch=0.0,
    )
    print(result)
    print()

    # ------------------------------------------------------------------
    # Show the coupling structure
    # ------------------------------------------------------------------
    result.print_A_structure()
    result.print_B_structure()

    # ------------------------------------------------------------------
    # Physical interpretation of key eigenvalues
    # ------------------------------------------------------------------
    eigs = result.eigenvalues
    unstable = result.unstable_eigenvalues
    print(f"\nUnstable modes: {len(unstable)}")
    for ev in unstable:
        # Time to double = ln(2) / real_part
        t_double = np.log(2) / np.real(ev) * 1000  # ms
        print(f"  λ = {np.real(ev):+.4f}  →  amplitude doubles in {t_double:.1f} ms")
    print()
    print("The PID loop must have bandwidth FASTER than these unstable modes")
    print("to stabilize the plant.")

    # ------------------------------------------------------------------
    # Gain schedule: how eigenvalues change with gap
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GAIN SCHEDULE: unstable eigenvalues vs gap height")
    print("=" * 70)
    gaps = [8, 10, 12, 14, TARGET_GAP_MM, 18, 20, 25]
    header = f"{'Gap [mm]':>10} {'I_eq [A]':>10} {'λ_heave':>12} {'t_dbl [ms]':>12} {'λ_pitch':>12} {'t_dbl [ms]':>12}"
    print(header)
    print("-" * len(header))
    for g in gaps:
        try:
            I = ss.find_equilibrium_current(g)
            r = ss.build(g, I, I, 0.0, 0.0)
            ue = r.unstable_eigenvalues
            real_ue = sorted(np.real(ue), reverse=True)
            # Typically: largest = heave, second = pitch
            lam_h = real_ue[0] if len(real_ue) > 0 else 0.0
            lam_p = real_ue[1] if len(real_ue) > 1 else 0.0
            t_h = np.log(2) / lam_h * 1000 if lam_h > 0 else float('inf')
            t_p = np.log(2) / lam_p * 1000 if lam_p > 0 else float('inf')
            print(f"{g:10.2f} {I:10.4f} {lam_h:+12.4f} {t_h:12.1f} "
                  f"{lam_p:+12.4f} {t_p:12.1f}")
        except ValueError as e:
            print(f"{g:10.2f}  (no equilibrium found)")
