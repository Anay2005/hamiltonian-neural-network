"""

This script loads a trained Hamiltonian Neural Network and subjects it to six
rigorous physics tests. Unlike simple point-checks ("is the force correct at
one state?"), these tests probe the *continuous structure* of the learned
Hamiltonian energy surface.

WHAT IS BEING TESTED
The network was trained to predict a single scalar — the total energy H(q, p)
of a chain of masses connected by springs. From that scalar, all dynamics
(forces, velocities, trajectories) are derived via automatic differentiation.
These tests check whether that learned energy surface has the correct shape,
curvature, symmetries, and long-term stability.

THE SIX TESTS:
  1. Force-Displacement Curve  — Is the learned restoring force linear (Hooke's Law)?
  2. Energy Landscape          — Does the learned H match the true H across phase space?
  3. Hessian / Stiffness       — Does d²H/dq² recover the spring coupling matrix?
  4. Normal Mode Frequencies   — Does the model predict the correct oscillation frequencies?
  5. Long-Horizon Energy Drift — How well is energy conserved over 100 seconds?
  6. Generalization Sweep      — Does the model work for chain lengths it never saw?
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Physics simulations accumulate tiny errors at every timestep. With 32-bit floats
# (the ML default), these rounding errors snowball over thousands of ODE steps and
# corrupt energy conservation. 64-bit floats give ~15 decimal digits of precision
# vs ~7 for 32-bit, which is critical for the long-horizon energy drift test.
jax.config.update("jax_enable_x64", True)



# MODEL ARCHITECTURE — Must match training EXACTLY

"""
WHY THIS MUST MATCH
When we saved the trained weights to 'model_weights.eqx', they were stored as a
flat list of arrays. To load them back, we need a  model with the
exact same layer shapes. If we change anything here (hidden channels, kernel size,
activation), the weight shapes won't match and loading will crash 

THE ACTIVATION FUNCTION: Growing Cosine Unit (GCU)
    f(z) = z * cos(z)

Standard ML activations like ReLU have a second derivative of zero almost everywhere.
Physics requires smooth, non-zero higher-order derivatives because:
  - Forces are the 1st derivative of energy:     F = -dH/dq
  - Stiffness is the 2nd derivative of energy:   K = d²H/dq²
  - The Hessian test (Test 3) literally computes d²H/dq² and checks if it
    recovers the spring coupling matrix.

If we used ReLU, d²H/dq² would be zero everywhere, and the Hessian test
would return garbage. GCU's oscillatory nature also helps the network represent
the periodic energy landscape of oscillating systems.
"""

def gcu(z):
    return z * jnp.cos(z)


@dataclass
class TrainConfig:
    """Stores the network shape so we can rebuild the exact same model."""
    hidden_channels: int = 32  # Number of feature maps in the hidden conv layers
    kernel_size: int = 3       # Spatial width of the first conv filter


class ConvHNN(eqx.Module):
    """
    A Convolutional Hamiltonian Neural Network.

    WHAT IT DOES
    Takes the full state of the system [q₀, q₁, ..., q₄, p₀, p₁, ..., p₄]
    and outputs a single scalar: the predicted total energy H.

    WHY CONVOLUTIONS
    A chain of masses has spatial structure: mass 3 only interacts with masses
    2 and 4. A 1D convolution with kernel_size=3 naturally captures exactly
    this look at your two neighbors pattern. A fully-connected layer would
    connect every mass to every other mass, which is physically wrong for a
    nearest-neighbor spring system.

    ARCHITECTURE DIAGRAM
    Input: flat vector of length 2N = 10
        [q₀, q₁, q₂, q₃, q₄, p₀, p₁, p₂, p₃, p₄]
                            ↓ reshape
        Tensor shape: (2 channels, 5 positions)
        Channel 0: [q₀, q₁, q₂, q₃, q₄]   ← positions
        Channel 1: [p₀, p₁, p₂, p₃, p₄]   ← momenta
                            ↓ Conv1d(2→32, kernel=3)
        Shape: (32 channels, 5 positions)   ← 32 learned features per mass
                            ↓ GCU activation
                            ↓ Conv1d(32→32, kernel=1)
        Shape: (32 channels, 5 positions)   ← pointwise mixing of features
                            ↓ GCU activation
                            ↓ Conv1d(32→1, kernel=1)
        Shape: (1 channel, 5 positions)     ← local energy per mass
        [e₀, e₁, e₂, e₃, e₄]
                            ↓ sum
        Scalar: H = e₀ + e₁ + e₂ + e₃ + e₄  ← total energy
    """
    layers: list

    def __init__(self, key, config: TrainConfig = TrainConfig()):
        keys = jax.random.split(key, 3)
        hc, ks = config.hidden_channels, config.kernel_size
        self.layers = [
            # Layer 1: Spatial convolution. kernel_size=3 means each output
            # looks at 3 adjacent masses. padding=ks//2=1 keeps the length at N.
            #   Weight shape: (32, 2, 3) → 32 output channels, 2 input channels, width 3
            #   Bias shape:   (32, 1)
            eqx.nn.Conv1d(2, hc, ks, padding=ks // 2, key=keys[0]),

            # Layer 2: Pointwise (1×1) convolution. kernel_size=1 means each mass
            # is processed independently — no spatial mixing. This is equivalent to
            # a Dense layer applied per-position.
            #   Weight shape: (32, 32, 1)
            eqx.nn.Conv1d(hc, hc, 1, key=keys[1]),

            # Layer 3: Collapse 32 features down to 1 number per mass = local energy.
            #   Weight shape: (1, 32, 1)
            #   Output shape: (1, N) → one energy value per mass
            eqx.nn.Conv1d(hc, 1, 1, key=keys[2]),
        ]

    def __call__(self, x):
        """
        Forward pass: state vector → scalar energy.

        INPUT
            x: shape (2N,) = [q₀...q_{N-1}, p₀...p_{N-1}]
            For N=5: x = [q₀, q₁, q₂, q₃, q₄, p₀, p₁, p₂, p₃, p₄]

        INTERNAL RESHAPING
            We split x in half and stack into a (2, N) tensor:
            Row 0 = positions: [q₀, q₁, q₂, q₃, q₄]
            Row 1 = momenta:   [p₀, p₁, p₂, p₃, p₄]

            This is like a 2-channel image of length N, ready for Conv1d.

        OUTPUT
            Scalar: the total predicted Hamiltonian H
        """
        N = x.shape[0] // 2
        h = jnp.stack([x[:N], x[N:]], axis=0)  # shape: (2, N)

        # Apply first two layers with GCU activation
        # After layer 0: shape (32, N) — 32 learned features at each mass position
        # After layer 1: shape (32, N) — features mixed per-position
        for layer in self.layers[:-1]:
            h = gcu(layer(h))

        # Final layer: (32, N) → (1, N) → sum over all masses → scalar
        return jnp.sum(self.layers[-1](h))


# TRUE PHYSICS HELPERS
"""
These functions compute the exact analytical physics. They serve as the answer key
that every test compares the neural network against.

Each mass m has:
  - Position qᵢ: how far it is from its resting point (positive = right)
  - Momentum pᵢ: mass × velocity (pᵢ = m × q̇ᵢ)
  - Two springs connecting it to its neighbors (or to a wall at the ends)

The springs obey Hooke's Law: F = -k × (stretch), where k is the spring constant.
"""


def make_A(N, boundary="fixed"):
    """
    Builds the coupling matrix A that encodes which masses are connected.

    WHAT A REPRESENTS PHYSICALLY
    A is a tridiagonal matrix where:
      - A[i,i] = -2: mass i is pulled by TWO springs (left and right)
      - A[i,i+1] = +1: mass i is pulled toward mass i+1
      - A[i,i-1] = +1: mass i is pulled toward mass i-1

    For N=5 with fixed boundaries (walls at both ends):

        A = [ -2,  1,  0,  0,  0 ]    ← Mass 0: wall on left, spring to mass 1
            [  1, -2,  1,  0,  0 ]    ← Mass 1: springs to mass 0 and mass 2
            [  0,  1, -2,  1,  0 ]    ← Mass 2: springs to mass 1 and mass 3
            [  0,  0,  1, -2,  1 ]    ← Mass 3: springs to mass 2 and mass 4
            [  0,  0,  0,  1, -2 ]    ← Mass 4: spring to mass 3, wall on right

    WHY THE FORCE IS k·A·q
    If mass 2 is displaced by q₂ = 1.0 and all others are at rest:

        q = [0, 0, 1, 0, 0]

        Force on each mass = k × A × q:
        F₀ = k × ( 0×(-2) + 0×1  + 1×0  + 0×0  + 0×0 ) = 0      ← unaffected
        F₁ = k × ( 0×1  + 0×(-2) + 1×1  + 0×0  + 0×0 ) = +k     ← pulled RIGHT toward mass 2
        F₂ = k × ( 0×0  + 0×1  + 1×(-2) + 0×1  + 0×0 ) = -2k    ← pulled LEFT by both springs
        F₃ = k × ( 0×0  + 0×0  + 1×1  + 0×(-2) + 0×0 ) = +k     ← pulled LEFT toward mass 2
        F₄ = k × ( 0×0  + 0×0  + 1×0  + 0×1  + 0×(-2)) = 0      ← unaffected

    This makes physical sense: the displaced mass is pushed back toward rest,
    and its immediate neighbors are pulled toward it.

    PERIODIC BOUNDARIES
    For a ring of masses (mass N-1 connects back to mass 0), we add:
        A[0, N-1] = 1   and   A[N-1, 0] = 1
    This turns the chain into a closed loop with no walls.
    """
    A = jnp.diag(-2 * jnp.ones(N)) + jnp.diag(jnp.ones(N - 1), 1) + jnp.diag(jnp.ones(N - 1), -1)
    if boundary == "periodic":
        A = A.at[0, -1].set(1).at[-1, 0].set(1)
    return A


def true_H(q, p, m, k, A):
    """
    Computes the exact total energy (Hamiltonian) of the system.

    THE HAMILTONIAN: H = T + V
    where T is kinetic energy and V is potential energy.

    Kinetic Energy:
        T = Σᵢ pᵢ² / (2m)

        Each mass stores kinetic energy proportional to the square of its momentum.
        For our system with m=1.0:
            If p = [0, 0, 3, 0, 0], then T = 9/(2×1) = 4.5

    Potential Energy:
        V = (1/2) × k × qᵀ(-A)q

        This is the energy stored in ALL the springs simultaneously.
        The matrix (-A) is positive semi-definite, so V ≥ 0 always.

        For q = [0, 0, 1, 0, 0] and k=5:
            qᵀ(-A)q = q × [0, -1, 2, -1, 0] = 1×2 = 2
            V = 0.5 × 5 × 2 = 5.0

        This means pulling mass 2 by 1 unit stores 5.0 units of potential energy
        in the two springs attached to it.

    INPUT SHAPES
        q: (N,) — position of each mass
        p: (N,) — momentum of each mass
        m: scalar — mass of each particle (all equal)
        k: scalar — spring constant (all springs equal)
        A: (N, N) — coupling matrix from make_A()

    OUTPUT
        Scalar: total energy H = T + V
    """
    return jnp.sum(p ** 2) / (2 * m) + 0.5 * k * jnp.dot(q, jnp.dot(-A, q))


def true_force(q, k, A):
    """
    Computes the exact force on each mass from Hooke's Law.

    Force = k × A × q

    INPUT
        q: (N,) — positions
        k: scalar — spring constant
        A: (N, N) — coupling matrix

    OUTPUT
        (N,) — force on each mass. Negative means "pushed left", positive means "pushed right".

    EXAMPLE (N=5, k=5, mass 2 displaced by 1.0):
        q = [0, 0, 1, 0, 0]
        F = 5 × A × q = [0, 5, -10, 5, 0]

        Mass 2 feels a restoring force of -10 (back toward center).
        Masses 1 and 3 feel +5 (pulled toward mass 2).
    """
    return k * jnp.dot(A, q)


def model_force(model, q, p):
    """
    Extracts force from the neural network's learned energy surface.

    THE KEY IDEA: FORCE FROM ENERGY VIA AUTODIFF
    In Hamiltonian mechanics, force is the negative gradient of energy with
    respect to position:
        F = -dH/dq

    We don't code this derivative by hand. JAX's jax.grad() traces through
    every operation in the neural network (convolutions, GCU activations, sum)
    and computes the exact mathematical derivative using the chain rule.

    STEP BY STEP:
        1. Concatenate q and p into one flat state vector:
           state = [q₀, q₁, q₂, q₃, q₄, p₀, p₁, p₂, p₃, p₄]  shape: (10,)

        2. jax.grad(model)(state) returns:
           grads = [dH/dq₀, dH/dq₁, ..., dH/dq₄, dH/dp₀, ..., dH/dp₄]  shape: (10,)

        3. The force is the NEGATIVE of the first N entries:
           F = -grads[:5] = [-dH/dq₀, -dH/dq₁, ..., -dH/dq₄]

    INPUT
        model: the trained ConvHNN
        q: (N,) — positions
        p: (N,) — momenta

    OUTPUT
        (N,) — predicted force on each mass
    """
    state = jnp.concatenate([q, p])
    g = jax.grad(model)(state)
    return -g[: len(q)]


# ODE VECTOR FIELDS
"""
An ODE solver needs a function that says: "given the current state at time t,
what is the rate of change of that state?" These two functions provide that
for the true physics and for the neural network respectively.

Both return (dq/dt, dp/dt) — how fast each position and momentum is changing.
"""


def vf_true(t, y, args):
    """
    True physics vector field: Hamilton's equations with known m, k, A.

    Hamilton's Equations:
        dqᵢ/dt =  dH/dpᵢ = pᵢ / m      (velocity = momentum / mass)
        dpᵢ/dt = -dH/dqᵢ = k × (A·q)ᵢ   (force = spring restoring force)

    INPUT
        t: scalar — current time (unused, but required by diffrax API)
        y: tuple (q, p) where q and p are each shape (N,)
        args: tuple (m, k, A)

    OUTPUT
        tuple (dq/dt, dp/dt), each shape (N,)

    EXAMPLE (N=5, q = [0,0,1,0,0], p = [0,0,0,0,0]):
        dq/dt = p / m = [0, 0, 0, 0, 0]        ← all masses stationary
        dp/dt = k·A·q = [0, 5, -10, 5, 0]       ← springs start accelerating masses
    """
    q, p = y
    m, k, A = args
    return p / m, k * jnp.dot(A, q)


def vf_hnn(t, y, args):
    """
    Neural network vector field: Hamilton's equations using LEARNED energy.

    Instead of using the known formula H = T + V, we use the neural network's
    prediction of H. The ODE solver calls this function thousands of times,
    and at each call, JAX computes the gradient of the network's output.

    Hamilton's Equations (same math, different H):
        dqᵢ/dt =  dH/dpᵢ   (gradient of learned H w.r.t. momentum)
        dpᵢ/dt = -dH/dqᵢ   (negative gradient of learned H w.r.t. position)

    CRITICAL INSIGHT
    Notice there is NO mass m, NO spring constant k, and NO matrix A here.
    The network must have learned all of that physics implicitly from data.
    If it learned correctly, the gradients of its energy surface will produce
    exactly the same dq/dt and dp/dt as the true physics.

    INPUT
        t: scalar — current time (unused)
        y: tuple (q, p), each shape (N,)
        args: tuple (model, N)

    OUTPUT
        tuple (dq/dt, dp/dt), each shape (N,)

    AUTODIFF DETAIL
        state = [q₀...q₄, p₀...p₄]   shape: (10,)
        grads = jax.grad(model)(state) shape: (10,)

        grads[:5] = [dH/dq₀, ..., dH/dq₄]   → used for dp/dt (with minus sign)
        grads[5:] = [dH/dp₀, ..., dH/dp₄]   → used for dq/dt (directly)
    """
    model, N = args
    q, p = y
    g = jax.grad(model)(jnp.concatenate([q, p]))
    return g[N:], -g[:N]


def solve_ode(vf, y0, args, t_end=20.0, dt=0.05):
    """
    Integrates an ODE forward in time using the Tsitouras 5(4) method.

    This is a high-order Runge-Kutta solver. At each timestep, it evaluates
    the vector field 6 times at carefully chosen sub-steps, combines them to
    get a 5th-order accurate estimate, and uses a 4th-order estimate to
    control the step size.

    INPUT
        vf: vector field function (either vf_true or vf_hnn)
        y0: initial state, tuple (q₀, p₀) each shape (N,)
        args: passed through to vf
        t_end: final time in seconds
        dt: output save interval (the internal solver may use smaller steps)

    OUTPUT
        ts: (num_steps,) — times at which the solution was saved
        ys: tuple (q_history, p_history)
            q_history: (num_steps, N) — position of every mass at every saved time
            p_history: (num_steps, N) — momentum of every mass at every saved time

    EXAMPLE OUTPUT SHAPES (N=5, t_end=20.0, dt=0.05):
        ts:        (400,)
        q_history: (400, 5)  ← 400 snapshots × 5 masses
        p_history: (400, 5)
    """
    ts = jnp.arange(0.0, t_end, dt)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Tsit5(),
        t0=0.0, t1=t_end, dt0=dt, y0=y0,
        args=args, saveat=diffrax.SaveAt(ts=ts), max_steps=100_000,
    )
    return ts, sol.ys



# TEST 1 — Force-Displacement Curve
"""
WHAT THIS TESTS
Hooke's Law says force is proportional to displacement: F = -k_eff × q.
If the network truly learned the correct quadratic potential energy
V = (1/2)k_eff × q², then its gradient dV/dq = k_eff × q is a straight line.

Any curvature, saturation, or kink in the predicted force curve means the
network's energy surface is NOT quadratic — it learned the wrong physics.

HOW IT WORKS
We sweep the displacement of one mass from -3.0 to +3.0, keeping all other
masses at rest (q=0, p=0). At each displacement we compute:
  - True force from F = k·A·q (straight line)
  - Predicted force from -dH_learned/dq (should also be a straight line)

The R² score measures how close the predicted curve is to a perfect fit of
the true curve. R² = 1.0 means identical; R² < 0.99 means trouble.

ARRAY EXAMPLE (at displacement d = 1.5 for mass 2):
    q = [0.0, 0.0, 1.5, 0.0, 0.0]   ← only mass 2 moved
    p = [0.0, 0.0, 0.0, 0.0, 0.0]   ← all at rest

    True force on mass 2:  k × (A·q)[2] = 5 × (-2×1.5) = -15.0
    Model force on mass 2: -dH/dq₂ evaluated at this state
"""


def test_force_curve(model, ax, N=5, k=5.0, mass_idx=2):
    """Sweeps displacement and plots predicted vs true restoring force."""
    A = make_A(N)
    displacements = jnp.linspace(-3.0, 3.0, 200)  # 200 test points
    f_true_list, f_pred_list = [], []

    for d in displacements:
        # Build a state where only mass_idx is displaced
        q = jnp.zeros(N).at[mass_idx].set(d)   # shape: (5,), e.g. [0, 0, d, 0, 0]
        p = jnp.zeros(N)                         # shape: (5,), all zeros

        f_true_list.append(true_force(q, k, A)[mass_idx])   # scalar: force on mass 2
        f_pred_list.append(model_force(model, q, p)[mass_idx])

    f_true_arr = jnp.array(f_true_list)  # shape: (200,) — true force at each displacement
    f_pred_arr = jnp.array(f_pred_list)  # shape: (200,) — predicted force at each displacement

    # Plot: should see two overlapping straight lines
    ax.plot(displacements, f_true_arr, "r--", lw=2, label="True (Hooke's Law)")
    ax.plot(displacements, f_pred_arr, "b-", lw=1.5, label="HNN Learned")
    ax.set_xlabel(f"Displacement $q_{{{mass_idx}}}$")
    ax.set_ylabel("Force")
    ax.set_title("1 · Force vs Displacement\n(should be linear)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² = 1 - (sum of squared residuals) / (total variance of true values)
    # Perfect prediction → R² = 1.0
    r2 = 1 - jnp.sum((f_true_arr - f_pred_arr) ** 2) / jnp.sum((f_true_arr - jnp.mean(f_true_arr)) ** 2)
    ax.text(0.05, 0.92, f"$R^2 = {float(r2):.6f}$", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    return float(r2)



# TEST 2 — Energy Landscape
"""
WHAT THIS TESTS
The true Hamiltonian H(q, p) = p²/(2m) + (1/2)k·qᵀ(-A)q is a smooth
paraboloid in phase space — it looks like a bowl. We evaluate both the true
and learned H on a dense 2D grid of (q₂, p₂) values and plot the absolute
error |H_true - H_learned|.

WHY A 2D SLICE
The full phase space is 10-dimensional (5 positions + 5 momenta). We can't
visualize that. Instead, we vary only one mass's position and momentum while
keeping all others at zero. This gives a 2D slice through the 10D surface.

WHAT TO LOOK FOR
  - The error heatmap should be uniformly dark (low error everywhere).
  - Bright spots indicate regions where the network's energy surface deviates
    from the true parabola. These are typically at the corners (high |q| AND
    high |p| simultaneously) where training data was sparse.

GRID CONSTRUCTION
    qs = 80 values from -1.5 to 1.5   (position range)
    ps = 80 values from -3.0 to 3.0   (momentum range)

    At each grid point (q_val, p_val), we build:
        q = [0, 0, q_val, 0, 0]   ← only mass 2 has position
        p = [0, 0, p_val, 0, 0]   ← only mass 2 has momentum

    Then compute:
        H_true = p_val²/2 + (1/2)×5×q_val²×2 = p_val²/2 + 5×q_val²
        H_pred = model([0, 0, q_val, 0, 0, 0, 0, p_val, 0, 0])
"""


def test_energy_landscape(model, ax, N=5, m=1.0, k=5.0, mass_idx=2):
    """Evaluates learned vs true H on a (q, p) grid for one mass."""
    A = make_A(N)
    qs = jnp.linspace(-1.5, 1.5, 80)    # shape: (80,)
    ps = jnp.linspace(-3.0, 3.0, 80)    # shape: (80,)
    Q, P = jnp.meshgrid(qs, ps)          # each shape: (80, 80)

    H_true = np.zeros_like(Q)  # shape: (80, 80) — will hold true energy at each grid point
    H_pred = np.zeros_like(Q)  # shape: (80, 80) — will hold predicted energy

    for i in range(len(qs)):
        for j in range(len(ps)):
            # Build state with only mass_idx having nonzero q and p
            q = jnp.zeros(N).at[mass_idx].set(Q[j, i])  # shape: (5,)
            p = jnp.zeros(N).at[mass_idx].set(P[j, i])  # shape: (5,)

            H_true[j, i] = true_H(q, p, m, k, A)                     # scalar
            H_pred[j, i] = model(jnp.concatenate([q, p]))             # scalar

    # Absolute error at each grid point
    err = np.abs(H_true - H_pred)  # shape: (80, 80)

    # Plot: dark = accurate, bright = error
    im = ax.pcolormesh(Q, P, err, cmap="inferno", shading="auto")
    plt.colorbar(im, ax=ax, label="|H_true − H_pred|")
    ax.set_xlabel(f"$q_{{{mass_idx}}}$")
    ax.set_ylabel(f"$p_{{{mass_idx}}}$")
    ax.set_title("2 · Energy Landscape Error\n(darker = more accurate)")

    return float(jnp.mean(err))



# TEST 3 — Hessian / Stiffness Matrix 
"""
WHAT THIS TESTS
The second derivative of the Hamiltonian with respect to positions gives the
stiffness matrix — the spring constant map of the system:

    K_ij = ∂²H / ∂qᵢ∂qⱼ

For our system, the true stiffness is K = k × (-A):

    K_true = [  10,  -5,   0,   0,   0 ]
             [  -5,  10,  -5,   0,   0 ]
             [   0,  -5,  10,  -5,   0 ]
             [   0,   0,  -5,  10,  -5 ]
             [   0,   0,   0,  -5,  10 ]

    Diagonal entries (10): each mass is held by two springs each with k=5 → 2×5 = 10.
    Off-diagonal entries (-5): adjacent masses are coupled with strength -k = -5.
    Zero entries: non-adjacent masses have no direct coupling.

If the network's Hessian matches this matrix, it has learned the EXACT spring
topology and strength — not just the right forces, but the right *structure*.

HOW JAX COMPUTES THE HESSIAN
jax.hessian(model) returns a function that computes the full matrix of second
derivatives. For input dimension 2N=10:

    full_hess = jax.hessian(model)(state₀)   shape: (10, 10)

    The full Hessian has four blocks:
    [ ∂²H/∂q∂q    ∂²H/∂q∂p ]     [ K   M⁻¹ ]
    [ ∂²H/∂p∂q    ∂²H/∂p∂p ]  ≈  [ M⁻¹  0  ]

    We extract the top-left N×N block: ∂²H/∂q∂q = learned stiffness.

We evaluate this at the equilibrium point (all zeros) because:
  - At equilibrium, the kinetic cross-terms vanish
  - The stiffness matrix is constant for a linear system
  - Any deviations from the true K reveal structural errors in the learned potential

ERROR METRIC: Relative Frobenius Norm
    err = ||K_true - K_learned||_F / ||K_true||_F

    This gives a single number: 0.0 = perfect match, 1.0 = completely wrong.
    The Frobenius norm treats the matrix as a long vector and computes its length.
"""


def test_hessian(model, ax, N=5, k=5.0):
    """Checks if d²H/dq² at equilibrium recovers the true stiffness matrix."""
    A = make_A(N)
    true_stiffness = k * (-A)   # shape: (5, 5) — see matrix above

    # Evaluate Hessian at the all-zeros equilibrium state
    state0 = jnp.zeros(2 * N)   # shape: (10,) = [0,0,0,0,0, 0,0,0,0,0]

    # jax.hessian computes all second derivatives automatically
    hessian_fn = jax.hessian(model)
    full_hess = hessian_fn(state0)  # shape: (10, 10)

    # Extract the position-position block (top-left 5×5)
    learned_stiffness = full_hess[:N, :N]  # shape: (5, 5)

    # Visualize side by side with numerical values in each cell
    vmin = min(float(jnp.min(true_stiffness)), float(jnp.min(learned_stiffness)))
    vmax = max(float(jnp.max(true_stiffness)), float(jnp.max(learned_stiffness)))

    ax[0].imshow(true_stiffness, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax[0].set_title("True $\\partial^2 H / \\partial q^2$")
    for i in range(N):
        for j in range(N):
            ax[0].text(j, i, f"{float(true_stiffness[i,j]):.1f}", ha="center", va="center", fontsize=8)

    im = ax[1].imshow(learned_stiffness, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax[1].set_title("Learned $\\partial^2 H / \\partial q^2$")
    for i in range(N):
        for j in range(N):
            ax[1].text(j, i, f"{float(learned_stiffness[i,j]):.1f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax[1], shrink=0.8)

    frob_err = float(jnp.linalg.norm(true_stiffness - learned_stiffness) / jnp.linalg.norm(true_stiffness))
    ax[0].set_xlabel(f"Relative Frobenius Error: {frob_err:.4f}")
    return frob_err



# TEST 4 — Normal Mode Frequencies
"""
WHAT THIS TESTS
A chain of N masses has N independent vibration patterns called "normal modes".
Each mode oscillates at a specific frequency. For N=5 with fixed boundaries:

    Mode 1 (slowest):  All masses sway together     → lowest frequency
    Mode 2:            Two groups sway opposite      → higher frequency
    Mode 3:            Three groups                  → higher still
    Mode 4:            Four groups                   → ...
    Mode 5 (fastest):  Adjacent masses alternate     → highest frequency

The analytical frequencies are:
    ωₙ = 2√(k/m) × sin(nπ / (2(N+1)))    for n = 1, 2, ..., N
    fₙ = ωₙ / (2π)                         in Hz

When we pull one mass and let go, the resulting motion is a SUPERPOSITION of
all 5 modes mixed together. The FFT (Fast Fourier Transform) decomposes this
mixture back into its constituent frequencies.

HOW THE TEST WORKS
  1. Set initial condition: pull mass 2 to q₂ = 0.5, all others at rest.
  2. Roll out for 60 seconds (long enough for good frequency resolution).
  3. Record the displacement of mass 2 over time: q₂(t).
  4. FFT this time series to get amplitude vs frequency.
  5. Do steps 1-4 for BOTH the true physics AND the HNN.
  6. Compare: the spectral peaks should align.

If a peak is missing or shifted, the network has failed to capture that mode.
"""


def test_frequencies(model, ax, N=5, m=1.0, k=5.0):
    """FFTs for both true and HNN trajectories; compares peak frequencies."""
    A = make_A(N)
    mid = N // 2   # = 2 for N=5

    # Initial condition: pull the middle mass, everything else at rest
    q0 = jnp.zeros(N).at[mid].set(0.5)   # shape: (5,) = [0, 0, 0.5, 0, 0]
    p0 = jnp.zeros(N)                     # shape: (5,) = [0, 0, 0, 0, 0]
    y0 = (q0, p0)

    # Use a small timestep (0.02s) and long duration (60s) for sharp frequency peaks
    # Frequency resolution = 1/t_end = 1/60 ≈ 0.017 Hz
    # Nyquist frequency = 1/(2×dt) = 1/0.04 = 25 Hz (well above our ~0.5-1.5 Hz modes)
    dt = 0.02
    t_end = 60.0

    # Roll out both trajectories
    ts_true, ys_true = solve_ode(vf_true, y0, (m, k, A), t_end=t_end, dt=dt)
    ts_pred, ys_pred = solve_ode(vf_hnn, y0, (model, N), t_end=t_end, dt=dt)

    # Extract the middle mass position over time
    q_true_mid = ys_true[0][:, mid]   # shape: (3000,) — q₂ at every timestep
    q_pred_mid = ys_pred[0][:, mid]   # shape: (3000,)

    # FFT: time domain → frequency domain
    n = len(q_true_mid)               # number of samples
    freqs = jnp.fft.rfftfreq(n, d=dt) # shape: (1501,) — frequency axis in Hz

    # rfft returns complex amplitudes; we take absolute value for magnitude
    fft_true = jnp.abs(jnp.fft.rfft(q_true_mid - jnp.mean(q_true_mid)))  # shape: (1501,)
    fft_pred = jnp.abs(jnp.fft.rfft(q_pred_mid - jnp.mean(q_pred_mid)))

    # Normalise so the tallest peak = 1.0 (makes visual comparison easier)
    fft_true /= jnp.max(fft_true)
    fft_pred /= jnp.max(fft_pred)

    # Only plot frequencies below 2 Hz (our modes are around 0.3-1.4 Hz)
    mask = freqs < 2.0
    ax.plot(freqs[mask], fft_true[mask], "r--", lw=2, label="True")
    ax.plot(freqs[mask], fft_pred[mask], "b-", lw=1.5, label="HNN")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised Amplitude")
    ax.set_title("4 · Frequency Spectrum\n(peaks should align)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Report the dominant (tallest) peak frequency for both
    f_dom_true = float(freqs[jnp.argmax(fft_true)])
    f_dom_pred = float(freqs[jnp.argmax(fft_pred)])
    ax.text(0.55, 0.88, f"True peak: {f_dom_true:.3f} Hz\nHNN peak:  {f_dom_pred:.3f} Hz",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    return abs(f_dom_true - f_dom_pred)



# Test 5: Energy Drift
"""
WHAT THIS TESTS
In a real Hamiltonian system, total energy H is EXACTLY constant for all time.
The ODE solver introduces tiny numerical errors at each step. For the true
physics these errors stay bounded, but for the HNN they compound because the
learned energy surface isn't perfectly quadratic.

This test runs the HNN for 100 seconds (5× longer than the standard evaluation)
and tracks how much the true energy drifts from its initial value.

WHY MEASURE WITH TRUE H, NOT LEARNED H
The HNN conserves its OWN learned energy almost perfectly (by construction —
the ODE solver follows the gradients of the learned H). But we want to know:
does the HNN trajectory stay on the REAL energy surface?

We compute:
    E(t) = true_H(q_pred(t), p_pred(t))     ← true energy of the HNN's trajectory
    drift(t) = 100 × (E(t) - E(0)) / |E(0)|  ← percentage deviation from initial

WHAT TO EXPECT
  - For the true physics solver: drift ≈ 0.0% (limited by float64 precision)
  - For a good HNN: drift < 5% over 100 seconds
  - For a bad HNN: drift grows steadily, often exponentially
  - Periodic oscillations in drift are normal — they indicate the HNN's energy
    surface is slightly tilted, causing the trajectory to oscillate around
    the true energy level.
"""


def test_energy_drift(model, ax, N=5, m=1.0, k=5.0):
    """test for 100 time units and measures energy conservation."""
    A = make_A(N)
    mid = N // 2

    # Same initial condition as the standard evaluation
    q0 = jnp.zeros(N).at[mid].set(1.0)   # shape: (5,) = [0, 0, 1, 0, 0]
    p0 = jnp.zeros(N)
    y0 = (q0, p0)

    t_end = 100.0   # 5× the standard evaluation window
    dt = 0.05
    ts, ys = solve_ode(vf_hnn, y0, (model, N), t_end=t_end, dt=dt)

    q_pred, p_pred = ys
    # q_pred shape: (2000, 5) — position of all 5 masses at 2000 time points
    # p_pred shape: (2000, 5)

    # Compute true energy at every time step of the HNN trajectory
    # jax.vmap applies true_H to each row simultaneously (no python loop)
    E = jax.vmap(lambda q, p: true_H(q, p, m, k, A))(q_pred, p_pred)
    # E shape: (2000,) — true energy at each time step

    E0 = E[0]  # scalar — initial energy (should be 5.0 for our setup)
    relative_drift = (E - E0) / jnp.abs(E0) * 100  # shape: (2000,) — percent drift

    ax.plot(ts, relative_drift, "b-", lw=1)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy Drift (%)")
    ax.set_title("Energy Drift (100s)\n")
    ax.grid(True, alpha=0.3)

    max_drift = float(jnp.max(jnp.abs(relative_drift)))
    ax.text(0.05, 0.92, f"Max drift: {max_drift:.2f}%", transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    return max_drift



# TEST 6 — Generalization Sweep (N = 3 → 20)
"""
WHAT THIS TESTS
The model was trained ONLY on N=5 masses. Can it handle different chain lengths?

WHY THIS MIGHT WORK
Conv1d kernels are local: each kernel only looks at 3 adjacent masses. The
learned weights encode how a mass interacts with its two neighbors — a rule
that is independent of how many masses exist in total. If the model truly
learned local physics, it should work for ANY N.

WHY THIS MIGHT FAIL
  - The model may have memorized global patterns specific to N=5 (like the
    total energy scale or the boundary effects at mass 0 and mass 4).
  - For N > 5, the input tensor is longer, and edge effects might differ.
  - The model never saw data from larger systems during training.

HOW IT WORKS
For each N from 3 to 20:
  1. Build the true coupling matrix A of size N×N.
  2. Create a state with the middle mass displaced by 1.0.
  3. Compute true force = k·A·q and predicted force = -dH/dq.
  4. Report the Mean Absolute Error across all N masses.

EXAMPLE (N=10, middle mass displaced):
    q = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   shape: (10,)
    p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   shape: (10,)
    state = [q, p] concatenated             shape: (20,)

    The Conv1d layers receive a (2, 10) tensor instead of (2, 5).
    This works because convolutions slide across any length — the weights
    don't change, only the number of positions they're applied to.
"""


def test_generalization(model, ax, k=5.0):
    """Tests force accuracy as a function of system size N."""
    sizes = list(range(3, 21))  # N = 3, 4, 5, ..., 20
    errors = []

    for N in sizes:
        A = make_A(N)                                     # shape: (N, N)
        mid = N // 2
        q = jnp.zeros(N).at[mid].set(1.0)                # shape: (N,)
        p = jnp.zeros(N)                                   # shape: (N,)

        ft = true_force(q, k, A)                           # shape: (N,) — true force
        fp = model_force(model, q, p)                      # shape: (N,) — predicted force
        errors.append(float(jnp.mean(jnp.abs(ft - fp))))  # scalar: MAE across all masses

    ax.bar(sizes, errors, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(0.5, color="r", ls="--", label="Threshold")
    ax.set_xlabel("System Size N")
    ax.set_ylabel("Mean Absolute Force Error")
    ax.set_title("6 · Generalization vs System Size\n(trained on N=5)")
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    return {n: e for n, e in zip(sizes, errors)}



# MAIN 
"""
This function loads the trained model, runs all six tests, arranges the plots
into a 3×3 grid, and saves the result as a single PNG dashboard.

"""


def main():
    print("Loading model...")
    key = jax.random.PRNGKey(0)
    model = ConvHNN(key)

    # Load trained weight.
    model = eqx.tree_deserialise_leaves("model_weights.eqx", model)
    print("Model loaded.\n")

 
    # FIGURE 1: Dynamics & Energy Conservation
    fig1 = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig1.suptitle("HNN Evaluation: Dynamics & Energy", fontsize=22, fontweight="bold")
    
    # 2x2 grid for the four continuous trajectory/energy tests
    gs1 = fig1.add_gridspec(2, 2)
    ax1 = fig1.add_subplot(gs1[0, 0])  # Force curve
    ax2 = fig1.add_subplot(gs1[0, 1])  # Energy landscape
    ax4 = fig1.add_subplot(gs1[1, 0])  # Frequencies
    ax5 = fig1.add_subplot(gs1[1, 1])  # Energy drift

    print("--- Running Part 1 Tests ---")
    print("Test 1: Force-Displacement Curve...")
    r2 = test_force_curve(model, ax1)
    
    print("Test 2: Energy Landscape Error...")
    mean_E_err = test_energy_landscape(model, ax2)
    
    print("Test 4: Frequency Spectrum...")
    freq_err = test_frequencies(model, ax4)
    
    print("Test 5: Energy Drift (100s)...")
    drift = test_energy_drift(model, ax5)

    out_path1 = "hnn_eval_part1_dynamics.png"
    fig1.savefig(out_path1, dpi=150, bbox_inches="tight", facecolor='white')
    print(f"-> Saved {out_path1}\n")


    
    # FIGURE 2: Structure, Generalization & Summary
    fig2 = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig2.suptitle("HNN Evaluation: Structure & Generalization", fontsize=22, fontweight="bold")
    
    # 3 rows, 2 columns. 
    # height_ratios ensure the square matrices get more room than the text box.
    gs2 = fig2.add_gridspec(3, 2, height_ratios=[1.8, 1.2, 0.8])
    
    # Row 0: Hessians (1 column each to keep them square)
    ax_h1 = fig2.add_subplot(gs2[0, 0])
    ax_h2 = fig2.add_subplot(gs2[0, 1])
    
    # Row 1: Generalization (spans both columns for a wide bar chart)
    ax6 = fig2.add_subplot(gs2[1, :])
    
    # Row 2: Summary (spans both columns)
    ax_summary = fig2.add_subplot(gs2[2, :])

    print("--- Running Part 2 Tests ---")
    print("Test 3: Hessian / Stiffness Recovery...")
    frob = test_hessian(model, [ax_h1, ax_h2])
    
    print("Test 6: Generalization Sweep (N=3→20)...")
    gen_errors = test_generalization(model, ax6)
    n5_err = gen_errors.get(5, 0)
    n10_err = gen_errors.get(10, 0)
    n20_err = gen_errors.get(20, 0)

    # Summary box formatting
    ax_summary.axis("off")
    summary_text = (
        "━━━ Overall Evaluation Summary ━━━\n\n"
        f"Force Linearity (R²):      {r2:.6f}   |   Force MAE @ N=5:  {n5_err:.4f}\n"
        f"Energy Surface Error:      {mean_E_err:.4f}   |   Force MAE @ N=10: {n10_err:.4f}\n"
        f"Stiffness Recovery (Frob): {frob:.4f}   |   Force MAE @ N=20: {n20_err:.4f}\n"
        f"Frequency Error:           {freq_err:.4f} Hz\n"
        f"Energy Drift (100s):       {drift:.2f}%\n"
    )
    
    # Adjusted to a wider, two-column text layout for better space utilization
    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                    fontsize=14, family="monospace", 
                    horizontalalignment="center", verticalalignment="center",
                    bbox=dict(boxstyle="round,pad=1.2", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.5))

    out_path2 = "hnn_eval_part2_structure.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor='white')
    print(f"-> Saved {out_path2}\n")

    # Display both figures
    plt.show()


if __name__ == "__main__":
    main()