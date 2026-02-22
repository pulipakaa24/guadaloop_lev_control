"""
Magnetic Levitation Jacobian Linearizer

Computes the local linear (Jacobian) approximation of the degree-6 polynomial
force/torque model at any operating point.  The result is an LTI gain matrix
that relates small perturbations in (currL, currR, roll, gap_height) to
perturbations in (Force, Torque):

    [ΔF  ]       [∂F/∂currL  ∂F/∂currR  ∂F/∂roll  ∂F/∂gap] [ΔcurrL  ]
    [ΔTau]  ≈  J [∂T/∂currL  ∂T/∂currR  ∂T/∂roll  ∂T/∂gap] [ΔcurrR  ]
                                                              [Δroll   ]
                                                              [Δgap    ]

Since the polynomial is analytic, all derivatives are computed exactly
(symbolic differentiation of the power-product terms), NOT by finite
differences.

The chain rule is applied automatically to convert the internal invGap
(= 1/gap_height) variable back to physical gap_height [mm].

Usage:
    lin = MaglevLinearizer("maglev_model.pkl")
    plant = lin.linearize(currL=-15, currR=-15, roll=0.0, gap_height=10.0)
    print(plant)
    print(plant.dF_dcurrL)         # single gain
    print(plant.control_jacobian)  # 2×2 matrix mapping ΔcurrL/R → ΔF/ΔT
    f, t = plant.predict(delta_currL=0.5)  # quick what-if
"""

import numpy as np
import joblib
import os


class LinearizedPlant:
    """
    Holds the Jacobian linearization of the force/torque model at one
    operating point.

    Attributes
    ----------
    operating_point : dict
        The (currL, currR, roll, gap_height) where linearization was computed.
    f0 : float
        Force [N] at the operating point.
    tau0 : float
        Torque [mN·m] at the operating point.
    jacobian : ndarray, shape (2, 4)
        Full Jacobian:
            Row 0 = Force derivatives,  Row 1 = Torque derivatives.
            Columns = [currL [A], currR [A], rollDeg [deg], gap_height [mm]]
    """

    INPUT_LABELS = ['currL [A]', 'currR [A]', 'rollDeg [deg]', 'gap_height [mm]']

    def __init__(self, operating_point, f0, tau0, jacobian):
        self.operating_point = operating_point
        self.f0 = f0
        self.tau0 = tau0
        self.jacobian = jacobian

    # ---- Individual gain accessors ----
    @property
    def dF_dcurrL(self):
        """∂Force/∂currL [N/A] at operating point."""
        return self.jacobian[0, 0]

    @property
    def dF_dcurrR(self):
        """∂Force/∂currR [N/A] at operating point."""
        return self.jacobian[0, 1]

    @property
    def dF_droll(self):
        """∂Force/∂roll [N/deg] at operating point."""
        return self.jacobian[0, 2]

    @property
    def dF_dgap(self):
        """∂Force/∂gap_height [N/mm] at operating point.
        Typically positive (unstable): force increases as gap shrinks.
        """
        return self.jacobian[0, 3]

    @property
    def dT_dcurrL(self):
        """∂Torque/∂currL [mN·m/A] at operating point."""
        return self.jacobian[1, 0]

    @property
    def dT_dcurrR(self):
        """∂Torque/∂currR [mN·m/A] at operating point."""
        return self.jacobian[1, 1]

    @property
    def dT_droll(self):
        """∂Torque/∂roll [mN·m/deg] at operating point."""
        return self.jacobian[1, 2]

    @property
    def dT_dgap(self):
        """∂Torque/∂gap_height [mN·m/mm] at operating point."""
        return self.jacobian[1, 3]

    @property
    def control_jacobian(self):
        """2×2 sub-matrix mapping control inputs [ΔcurrL, ΔcurrR] → [ΔF, ΔT].

        This is the "B" portion of the linearized plant that the PID
        controller acts through.
        """
        return self.jacobian[:, :2]

    @property
    def state_jacobian(self):
        """2×2 sub-matrix mapping state perturbations [Δroll, Δgap] → [ΔF, ΔT].

        Contains the magnetic stiffness (∂F/∂gap) and roll coupling.
        This feeds into the "A" matrix of the full mechanical state-space.
        """
        return self.jacobian[:, 2:]

    def predict(self, delta_currL=0.0, delta_currR=0.0,
                delta_roll=0.0, delta_gap=0.0):
        """
        Predict force and torque using the linear approximation.

        Returns (force, torque) including the nominal operating-point values.
        """
        delta = np.array([delta_currL, delta_currR, delta_roll, delta_gap])
        perturbation = self.jacobian @ delta
        return self.f0 + perturbation[0], self.tau0 + perturbation[1]

    def __repr__(self):
        op = self.operating_point
        lines = [
            f"LinearizedPlant @ currL={op['currL']:.1f}A, "
            f"currR={op['currR']:.1f}A, "
            f"roll={op['roll']:.2f}°, "
            f"gap={op['gap_height']:.2f}mm",
            f"  F₀ = {self.f0:+.4f} N    τ₀ = {self.tau0:+.4f} mN·m",
            f"  ∂F/∂currL = {self.dF_dcurrL:+.4f} N/A       "
            f"∂T/∂currL = {self.dT_dcurrL:+.4f} mN·m/A",
            f"  ∂F/∂currR = {self.dF_dcurrR:+.4f} N/A       "
            f"∂T/∂currR = {self.dT_dcurrR:+.4f} mN·m/A",
            f"  ∂F/∂roll  = {self.dF_droll:+.4f} N/deg     "
            f"∂T/∂roll  = {self.dT_droll:+.4f} mN·m/deg",
            f"  ∂F/∂gap   = {self.dF_dgap:+.4f} N/mm      "
            f"∂T/∂gap   = {self.dT_dgap:+.4f} mN·m/mm",
        ]
        return '\n'.join(lines)


class MaglevLinearizer:
    """
    Jacobian linearizer for the polynomial maglev force/torque model.

    Loads the same .pkl model as MaglevPredictor, but instead of just
    evaluating the polynomial, computes exact analytical partial derivatives
    at any operating point.
    """

    def __init__(self, model_path='maglev_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                "Train and save the model from Function Fitting.ipynb first."
            )

        data = joblib.load(model_path)
        poly_transformer = data['poly_features']
        linear_model = data['model']

        # powers_: (n_terms, n_inputs) — exponent matrix from sklearn
        # Transpose to (n_inputs, n_terms) for broadcasting with x[:, None]
        self.powers = poly_transformer.powers_.T.astype(np.float64)

        self.force_coef = linear_model.coef_[0]        # (n_terms,)
        self.torque_coef = linear_model.coef_[1]       # (n_terms,)
        self.force_intercept = linear_model.intercept_[0]
        self.torque_intercept = linear_model.intercept_[1]

        self.degree = data['degree']
        self.n_terms = self.powers.shape[1]

    def _to_internal(self, currL, currR, roll, gap_height):
        """Convert physical inputs to the polynomial's internal variables."""
        invGap = 1.0 / max(gap_height, 1e-6)
        return np.array([currL, currR, roll, invGap], dtype=np.float64)

    def evaluate(self, currL, currR, roll, gap_height):
        """
        Evaluate the full (nonlinear) polynomial at a single point.

        Returns
        -------
        force : float  [N]
        torque : float  [mN·m]
        """
        x = self._to_internal(currL, currR, roll, gap_height)
        poly_features = np.prod(x[:, None] ** self.powers, axis=0)
        force = np.dot(self.force_coef, poly_features) + self.force_intercept
        torque = np.dot(self.torque_coef, poly_features) + self.torque_intercept
        return float(force), float(torque)

    def _jacobian_internal(self, x):
        """
        Compute the 2×4 Jacobian w.r.t. the internal polynomial variables
        (currL, currR, rollDeg, invGap).

        For each variable x_k, the partial derivative of a polynomial term
            c · x₁^a₁ · x₂^a₂ · x₃^a₃ · x₄^a₄
        is:
            c · a_k · x_k^(a_k - 1) · ∏_{j≠k} x_j^a_j

        This is computed vectorised over all terms simultaneously.
        """
        jac = np.zeros((2, 4))
        for k in range(4):
            # a_k for every term — this becomes the multiplicative scale
            scale = self.powers[k, :]  # (n_terms,)

            # Reduce the k-th exponent by 1 (floored at 0; the scale
            # factor of 0 for constant-in-x_k terms zeros those out)
            deriv_powers = self.powers.copy()
            deriv_powers[k, :] = np.maximum(deriv_powers[k, :] - 1.0, 0.0)

            # Evaluate the derivative polynomial
            poly_terms = np.prod(x[:, None] ** deriv_powers, axis=0)
            deriv_features = scale * poly_terms  # (n_terms,)

            jac[0, k] = np.dot(self.force_coef, deriv_features)
            jac[1, k] = np.dot(self.torque_coef, deriv_features)

        return jac

    def linearize(self, currL, currR, roll, gap_height):
        """
        Compute the Jacobian linearization at the given operating point.

        Parameters
        ----------
        currL : float      Left coil current [A]
        currR : float      Right coil current [A]
        roll : float       Roll angle [deg]
        gap_height : float Air gap [mm]

        Returns
        -------
        LinearizedPlant
            Contains the operating-point values (F₀, τ₀) and the 2×4
            Jacobian with columns [currL, currR, roll, gap_height].
        """
        x = self._to_internal(currL, currR, roll, gap_height)
        f0, tau0 = self.evaluate(currL, currR, roll, gap_height)

        # Jacobian in internal coordinates (w.r.t. invGap in column 3)
        jac_internal = self._jacobian_internal(x)

        # Chain rule: ∂f/∂gap = ∂f/∂invGap · d(invGap)/d(gap)
        #                     = ∂f/∂invGap · (−1 / gap²)
        jac = jac_internal.copy()
        jac[:, 3] *= -1.0 / (gap_height ** 2)

        return LinearizedPlant(
            operating_point={
                'currL': currL, 'currR': currR,
                'roll': roll, 'gap_height': gap_height,
            },
            f0=f0,
            tau0=tau0,
            jacobian=jac,
        )

    def gain_schedule(self, gap_heights, currL, currR, roll=0.0):
        """
        Precompute linearizations across a range of gap heights at fixed
        current and roll.  Useful for visualising how plant gains vary
        and for designing a gain-scheduled PID.

        Parameters
        ----------
        gap_heights : array-like of float [mm]
        currL, currR : float [A]
        roll : float [deg], default 0

        Returns
        -------
        list of LinearizedPlant, one per gap height
        """
        return [
            self.linearize(currL, currR, roll, g)
            for g in gap_heights
        ]


# ==========================================================================
# Demo / quick validation
# ==========================================================================
if __name__ == '__main__':
    import sys

    model_path = os.path.join(os.path.dirname(__file__), 'maglev_model.pkl')
    lin = MaglevLinearizer(model_path)

    # --- Single-point linearization ---
    print("=" * 70)
    print("SINGLE-POINT LINEARIZATION")
    print("=" * 70)
    plant = lin.linearize(currL=-15, currR=-15, roll=0.0, gap_height=10.0)
    print(plant)
    print()

    # Quick sanity check: compare linear prediction vs full polynomial
    dc = 0.5  # small current perturbation
    f_lin, t_lin = plant.predict(delta_currL=dc)
    f_act, t_act = lin.evaluate(-15 + dc, -15, 0.0, 10.0)
    print(f"Linearised  vs  Actual  (ΔcurrL = {dc:+.1f} A):")
    print(f"  Force:  {f_lin:.4f}  vs  {f_act:.4f}  (err {abs(f_lin-f_act):.6f} N)")
    print(f"  Torque: {t_lin:.4f}  vs  {t_act:.4f}  (err {abs(t_lin-t_act):.6f} mN·m)")
    print()

    # --- Gain schedule across gap heights ---
    print("=" * 70)
    print("GAIN SCHEDULE  (currL = currR = -15 A, roll = 0°)")
    print("=" * 70)
    gaps = [6, 8, 10, 12, 15, 20, 25]
    plants = lin.gain_schedule(gaps, currL=-15, currR=-15, roll=0.0)

    header = f"{'Gap [mm]':>10} {'F₀ [N]':>10} {'∂F/∂iL':>10} {'∂F/∂iR':>10} {'∂F/∂gap':>10} {'∂T/∂iL':>12} {'∂T/∂iR':>12}"
    print(header)
    print("-" * len(header))
    for p in plants:
        g = p.operating_point['gap_height']
        print(
            f"{g:10.1f} {p.f0:10.3f} {p.dF_dcurrL:10.4f} "
            f"{p.dF_dcurrR:10.4f} {p.dF_dgap:10.4f} "
            f"{p.dT_dcurrL:12.4f} {p.dT_dcurrR:12.4f}"
        )
    print()

    # --- Note on PID usage ---
    print("=" * 70)
    print("NOTES FOR PID DESIGN")
    print("=" * 70)
    print("""
At each operating point, the linearized electromagnetic plant is:

    [ΔF  ]   [∂F/∂iL  ∂F/∂iR] [ΔiL]   [∂F/∂roll  ∂F/∂gap] [Δroll]
    [ΔTau] = [∂T/∂iL  ∂T/∂iR] [ΔiR] + [∂T/∂roll  ∂T/∂gap] [Δgap ]
              ^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^
              control_jacobian             state_jacobian

The full mechanical dynamics (linearized) are:
    m  · Δg̈ap  = ΔF  - m·g    (vertical — note ∂F/∂gap > 0 means unstable)
    Iz · Δroll̈ = ΔTau          (roll)

So the PID loop sees:
    control_jacobian  →  the gain from current commands to force/torque
    state_jacobian    →  the coupling from state perturbations (acts like
                         a destabilising spring for gap, restoring for roll)
""")
