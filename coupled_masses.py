"""
Coupled Oscillator Data Generator
===================================
This script generates the training data for the Hamiltonian Neural Network.
It simulates many different initial conditions of a chain of masses connected
by springs, and records the resulting positions and momenta at every timestep.

THE PHYSICAL SYSTEM
    
    - Wall 
    - 5 identical masses (m = 1.0 kg each)
    - Connected by identical springs (k = 5.0 N/m each)
    - Fixed walls at both ends (Dirichlet boundary conditions)

WHAT THE OUTPUT DATA LOOKS LIKE
    We run 100 different simulations, each with a random starting position.
    Each simulation produces 2000 timesteps (20 seconds at dt=0.01).

    Total samples = 100 trajectories × 2000 timesteps = 200,000 rows.

    Saved arrays:
        q: shape (200000, 5)  — position of each mass at every sample
        p: shape (200000, 5)  — momentum of each mass at every sample
        m: scalar (1.0)       — mass value
        k: scalar (5.0)       — spring constant
        A: shape (5, 5)       — coupling matrix

WHY THE HNN NEEDS THIS DATA
    The HNN learns by comparing its predicted dynamics against the true dynamics.
    At each training step, it:
        1. Takes a batch of states [q, p] from this dataset
        2. Predicts dq/dt and dp/dt using gradients of its learned energy
        3. Compares against the true dq/dt = p/m and dp/dt = k·A·q
        4. Updates its weights to reduce the mismatch

    Different set of initial conditions (100 different random pulls) ensures
    the network sees the energy landscape from many angles, not just one trajectory.
"""

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# We generate training data in 64-bit precision to minimize numerical errors
# in the ODE integration. The HNN will learn from these trajectories, so any
# integration error in the data becomes noise in the training signal.
# 32-bit floats accumulate ~1e-7 error per step × 2000 steps = ~1e-4 total.
# 64-bit floats accumulate ~1e-16 error per step × 2000 steps = ~1e-13 total.
jax.config.update("jax_enable_x64", True)



# CONFIGURATION
"""
All simulation parameters in one place. Changing these values changes the
physics of the system and the resolution of the output data.

KEY RELATIONSHIPS:
    - dt controls both the ODE solver step and the save interval.
      Smaller dt = more accurate but more data points.
    - t_end / dt = number of timesteps per trajectory.
      20.0 / 0.01 = 2000 timesteps per simulation.
    - N determines the size of all arrays. Changing N from 5 to 10
      doubles the width of q and p arrays and quadruples the size of A.
"""

@dataclass
class SimConfig:
    N: int = 5            # Number of masses in the chain
    m: float = 1.0        # Mass of each particle (kg). All masses are identical.
    k: float = 5.0        # Spring constant (N/m). All springs are identical.
    t_start: float = 0.0  # Simulation start time (seconds)
    t_end: float = 20.0   # Simulation end time (seconds)
    dt: float = 0.01      # Timestep for saving snapshots (seconds)
    boundary: str = 'fixed'  # 'fixed' = walls at both ends, 'periodic' = ring



# PHYSICS ENGINE

def get_interaction_matrix(N, boundary):
    """
    Builds the coupling matrix A that encodes which masses are connected by springs.

    WHAT A REPRESENTS PHYSICALLY
    Each entry A[i,j] describes how mass j's displacement affects the force on mass i.

    CONSTRUCTION — THREE DIAGONAL ARRAYS
    We build A from three 1D arrays stacked as diagonals:

        main_diag = [-2, -2, -2, -2, -2]     length: N
            → Each mass is pulled by TWO springs (one on each side).
            → A[i,i] = -2 for all i.

        off_diag  = [1, 1, 1, 1]              length: N-1
            → Each mass is connected to its immediate neighbor.
            → A[i,i+1] = 1 (upper diagonal: mass i pulled toward mass i+1)
            → A[i,i-1] = 1 (lower diagonal: mass i pulled toward mass i-1)

    RESULTING MATRIX (N=5, fixed boundaries):

        A = [ -2,  1,  0,  0,  0 ]    Row 0: wall─spring─[m₀]─spring─[m₁]
            [  1, -2,  1,  0,  0 ]    Row 1: [m₀]─spring─[m₁]─spring─[m₂]
            [  0,  1, -2,  1,  0 ]    Row 2: [m₁]─spring─[m₂]─spring─[m₃]
            [  0,  0,  1, -2,  1 ]    Row 3: [m₂]─spring─[m₃]─spring─[m₄]
            [  0,  0,  0,  1, -2 ]    Row 4: [m₃]─spring─[m₄]─spring─wall

    PERIODIC BOUNDARIES (ring of masses, no walls):
        We add connections between the first and last mass:
            A[0, N-1] = 1   (mass 0 pulled toward mass N-1)
            A[N-1, 0] = 1   (mass N-1 pulled toward mass 0)

        A_periodic = [ -2,  1,  0,  0,  1 ]    ← mass 0 now connects to mass 4
                     [  1, -2,  1,  0,  0 ]
                     [  0,  1, -2,  1,  0 ]
                     [  0,  0,  1, -2,  1 ]
                     [  1,  0,  0,  1, -2 ]    ← mass 4 now connects to mass 0

    INPUT
        N: int — number of masses
        boundary: str — 'fixed' or 'periodic'

    OUTPUT
        A: shape (N, N) — the coupling matrix
    """
    # Main diagonal: each mass has -2 because it sits between two springs
    main_diag = -2 * jnp.ones(N)      # shape: (5,) = [-2, -2, -2, -2, -2]

    # Off-diagonals: coupling strength +1 between adjacent masses
    off_diag = jnp.ones(N - 1)        # shape: (4,) = [1, 1, 1, 1]

    # Assemble: jnp.diag(v, k) places vector v on the k-th diagonal
    #   k=0  → main diagonal
    #   k=1  → one above main (upper)
    #   k=-1 → one below main (lower)
    A = jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    # A shape: (5, 5)

    if boundary == 'periodic':
        # Close the chain into a ring
        A = A.at[0, -1].set(1)    # top-right corner: mass 0 ↔ mass N-1
        A = A.at[-1, 0].set(1)    # bottom-left corner: mass N-1 ↔ mass 0

    return A


def coupled_oscillator(t, state, args):
    """
    The equations of motion for the coupled spring-mass system.
    This is the function the ODE solver calls at every sub-step to ask:
    "Given the current positions and momenta, how fast is each one changing?"

    HAMILTON'S EQUATIONS OF MOTION:

        dqᵢ/dt =  ∂H/∂pᵢ = pᵢ / m        (velocity = momentum / mass)
        dpᵢ/dt = -∂H/∂qᵢ = k × (A·q)ᵢ    (force = Hooke's Law via coupling matrix)

    The first equation is trivial: velocity is momentum divided by mass.
    The second encodes all the spring physics through the matrix-vector product A·q.

    WORKED EXAMPLE (N=5, mass 2 displaced by 1.0, all others at rest):

        q = [0.0, 0.0, 1.0, 0.0, 0.0]
        p = [0.0, 0.0, 0.0, 0.0, 0.0]

        dq/dt = p / m = [0, 0, 0, 0, 0]           ← everything stationary

        dp/dt = k × A·q:
            A·q = [-2  1  0  0  0] [0]     [0]
                  [ 1 -2  1  0  0] [0]  =  [1]     ← mass 1 pulled right
                  [ 0  1 -2  1  0] [1]     [-2]    ← mass 2 pushed back (restoring force)
                  [ 0  0  1 -2  1] [0]     [1]     ← mass 3 pulled left
                  [ 0  0  0  1 -2] [0]     [0]

            dp/dt = 5 × [0, 1, -2, 1, 0] = [0, 5, -10, 5, 0]

        Physical meaning:
            Mass 2 feels -10 N (strong restoring force back toward equilibrium)
            Masses 1 and 3 feel +5 N (pulled toward the displaced mass 2)
            Masses 0 and 4 feel nothing (too far away, no direct spring connection)

    INPUT
        t: scalar — current time (unused but required by diffrax API)
        state: tuple (q, p)
            q: shape (N,) — position of each mass relative to its rest position
            p: shape (N,) — momentum of each mass (= mass × velocity)
        args: tuple (m, k, A)
            m: scalar — mass of each particle
            k: scalar — spring constant
            A: shape (N, N) — coupling matrix

    OUTPUT
        tuple (dq/dt, dp/dt), each shape (N,)
    """
    q, p = state
    m, k, A = args

    # Velocity: dqᵢ/dt = pᵢ / m
    # For p = [0, 0, 5, 0, 0] and m = 1.0:
    #   dq/dt = [0, 0, 5, 0, 0]  ← only mass 2 is moving
    dq_dt = p / m            # shape: (N,)

    # Force: dpᵢ/dt = k × Σⱼ Aᵢⱼ qⱼ
    # This is a matrix-vector product: each row of A dotted with q gives
    # the net spring force on that mass.
    dp_dt = k * jnp.dot(A, q)  # shape: (N,)

    return dq_dt, dp_dt

# PARALLEL ODE SOLVER
"""
WHY PARALLELISE?
We need 100 independent simulations with different initial conditions.
A naive Python loop would run them one at a time: 100 × (compile + solve).

jax.vmap ("vectorised map") transforms a function that solves ONE simulation
into a function that solves MANY simultaneously. On a GPU, all 100 simulations
execute in parallel. On a CPU, JAX still benefits from fused operations and
avoids Python loop overhead.

THE DECORATOR: @jax.jit
jit = "just-in-time compilation". The first call traces the function and
compiles it to optimised XLA machine code. Subsequent calls with the same
input shapes reuse the compiled code, which is 10-100× faster than
interpreted Python+NumPy.
"""


@jax.jit
def solve_batch(batch_y0, args, save_times, dt0):
    """
    Solves multiple ODE simulations in parallel using jax.vmap.

    WHAT HAPPENS INSIDE:
        1. solve_single() wraps the ODE solver for ONE initial condition.
        2. jax.vmap(solve_single) creates a BATCHED version that maps over
           the first axis of the input arrays.

    INPUT
        batch_y0: tuple (batch_q0, batch_p0)
            batch_q0: shape (B, N) — B different initial position vectors
            batch_p0: shape (B, N) — B different initial momentum vectors

            For B=100, N=5:
                batch_q0[0]  = [0, 0, 3.7, 0, 0]    ← trajectory 0: pull mass 2
                batch_q0[1]  = [0, -8.2, 0, 0, 0]   ← trajectory 1: pull mass 1
                batch_q0[2]  = [0, 0, 0, 0, 5.1]    ← trajectory 2: pull mass 4
                ...
                batch_q0[99] = [0, 0, 0, -1.3, 0]   ← trajectory 99: pull mass 3

        args: tuple (m, k, A) — shared physics parameters (same for all trajectories)
        save_times: shape (T,) — times at which to record the state
            For t_end=20.0, dt=0.01: save_times has T=2000 entries
        dt0: scalar — initial ODE solver step size

    OUTPUT
        batch_sol: a diffrax Solution object where .ys contains:
            batch_sol.ys[0]: shape (B, T, N) — positions of all masses at all times
            batch_sol.ys[1]: shape (B, T, N) — momenta of all masses at all times

        For B=100, T=2000, N=5:
            batch_sol.ys[0] has shape (100, 2000, 5)
            batch_sol.ys[0][42, 1500, 3] = position of mass 3 in trajectory 42 at timestep 1500
    """
    def solve_single(y0):
        """Solves the coupled oscillator ODE for a single initial condition."""
        # ODETerm wraps our physics function so diffrax understands it
        term = diffrax.ODETerm(coupled_oscillator)

        # Tsit5 = Tsitouras 5(4) method, a modern Runge-Kutta solver.
        # At each step it evaluates the vector field 6 times at carefully
        # chosen sub-points, combines them for a 5th-order accurate estimate,
        # and uses a 4th-order estimate for error control.
        solver = diffrax.Tsit5()

        # SaveAt tells the solver which times to record.
        # The solver may take MANY internal sub-steps between these save points.
        saveat = diffrax.SaveAt(ts=save_times)

        return diffrax.diffeqsolve(
            term, solver,
            t0=save_times[0], t1=save_times[-1], dt0=dt0,
            y0=y0,          # y0 = (q0, p0), each shape (N,)
            args=args,       # (m, k, A) — the physics parameters
            saveat=saveat,
            max_steps=10000  # safety limit to prevent infinite loops
        )

    # vmap maps solve_single over axis 0 of batch_y0.
    # Input:  batch_y0 = (batch_q0, batch_p0), each shape (B, N)
    # vmap calls solve_single B times, once per row:
    #   solve_single((batch_q0[0], batch_p0[0]))
    #   solve_single((batch_q0[1], batch_p0[1]))
    #   ...
    #   solve_single((batch_q0[B-1], batch_p0[B-1]))
    # But all B calls execute simultaneously, not sequentially.
    return jax.vmap(solve_single)(batch_y0)



# DATA GENERATION 


def run_simulation_parallel(config: SimConfig, num_trajectories=20):
    """
    Generates training data by running many simulations with random initial conditions.

    THE DATA GENERATION STRATEGY
    Each trajectory starts with:
        - ONE randomly chosen mass displaced to a random position in [-10, 10]
        - All other masses at their rest position (q = 0)
        - All momenta at zero (p = 0) — the system starts from rest

    This means the training data contains examples of:
        - Small displacements (0.1) and large displacements (10.0)
        - Every mass being the one that's pulled (mass 0, 1, 2, 3, or 4)
        - The full transient response as energy propagates through the chain

    WHY RANDOM SINGLE-MASS PULLS?
    A single displaced mass excites ALL normal modes of the system simultaneously.
    Different masses and different amplitudes sample different regions of phase space.
    This gives the HNN a diverse view of the energy landscape without needing
    multi-mass initial conditions (which would be redundant by superposition in
    a linear system).

    INPUT
        config: SimConfig — physical parameters and resolution
        num_trajectories: int — how many independent simulations to run (default 20)

    OUTPUT
        q_flat: shape (B×T, N) — ALL positions from ALL trajectories, stacked vertically
        p_flat: shape (B×T, N) — ALL momenta from ALL trajectories, stacked vertically
        q_batch: shape (B, T, N) — positions with trajectory structure preserved
        save_times: shape (T,) — time axis
        args: tuple (m, k, A) — physics parameters for the HNN training script

    ARRAY SHAPES EXAMPLE (100 trajectories, t_end=20.0, dt=0.01, N=5):
        B = 100, T = 2000, N = 5

        q_batch:    (100, 2000, 5)  — 100 trajectories × 2000 timesteps × 5 masses
        q_flat:     (200000, 5)     — all 200,000 snapshots stacked into one big table
        save_times: (2000,)         — [0.00, 0.01, 0.02, ..., 19.99]

    WHAT ONE ROW OF q_flat LOOKS LIKE:
        q_flat[0]     = [0.0, 0.0, 3.7, 0.0, 0.0]    ← t=0.00 of trajectory 0
        q_flat[1]     = [0.0, 0.001, 3.699, 0.001, 0.0] ← t=0.01 of trajectory 0
        ...
        q_flat[1999]  = [-0.2, 0.5, -0.8, 0.3, -0.1]  ← t=19.99 of trajectory 0
        q_flat[2000]  = [0.0, -8.2, 0.0, 0.0, 0.0]   ← t=0.00 of trajectory 1
        ...
    """
    # Build the coupling matrix
    A = get_interaction_matrix(config.N, config.boundary)  # shape: (N, N) = (5, 5)
    args = (config.m, config.k, A)

    # Create the time axis at which we'll save snapshots
    save_times = jnp.arange(config.t_start, config.t_end, config.dt)
    # shape: (2000,) = [0.00, 0.01, 0.02, ..., 19.99]

    print(f"Preparing {num_trajectories} initial conditions...")

    # ── Generate Random Initial Conditions ──
    # Each trajectory has a different random starting state.
    # We pick ONE mass at random and displace it by a random amount.
    batch_q0 = []
    batch_p0 = []

    for _ in range(num_trajectories):
        # Pick which mass to pull: random integer in [0, N-1]
        target_idx = np.random.randint(0, config.N)

        # Pick how far to pull it: random float in [-10.0, +10.0]
        # Negative = pulled left, positive = pulled right
        displacement = np.random.uniform(-10.0, 10.0)

        # Build initial position vector: all zeros except the target mass
        q = jnp.zeros(config.N).at[target_idx].set(displacement)
        # Example: target_idx=2, displacement=3.7
        #   q = [0.0, 0.0, 3.7, 0.0, 0.0]   shape: (5,)

        # All masses start at rest (zero momentum = zero velocity)
        p = jnp.zeros(config.N)
        # p = [0.0, 0.0, 0.0, 0.0, 0.0]   shape: (5,)

        batch_q0.append(q)
        batch_p0.append(p)

    # Stack individual vectors into batch arrays
    batch_q0 = jnp.stack(batch_q0)  # shape: (B, N) = (100, 5)
    batch_p0 = jnp.stack(batch_p0)  # shape: (B, N) = (100, 5)
    batch_y0 = (batch_q0, batch_p0)
    # batch_y0 is a tuple of two (100, 5) arrays

    """
    BATCH INITIAL CONDITIONS VISUALIZATION (first few rows of batch_q0):

        batch_q0 = [[ 0.0,  0.0,  3.7,  0.0,  0.0],   ← traj 0: mass 2 pulled right
                    [ 0.0, -8.2,  0.0,  0.0,  0.0],   ← traj 1: mass 1 pulled left
                    [ 0.0,  0.0,  0.0,  0.0,  5.1],   ← traj 2: mass 4 pulled right
                    [ 6.3,  0.0,  0.0,  0.0,  0.0],   ← traj 3: mass 0 pulled right
                    ...
                    [ 0.0,  0.0,  0.0, -1.3,  0.0]]   ← traj 99: mass 3 pulled left

    Each row is one initial condition. The ODE solver will evolve ALL of
    these forward in time simultaneously.
    """

    # ── Run All Simulations in Parallel ──
    print("Compiling and Running Simulation (Parallel)...")
    batch_sol = solve_batch(batch_y0, args, save_times, config.dt)

    # Extract Results 
    # batch_sol.ys is a tuple: (all_positions, all_momenta)
    q_batch = batch_sol.ys[0]  # shape: (B, T, N) = (100, 2000, 5)
    p_batch = batch_sol.ys[1]  # shape: (B, T, N) = (100, 2000, 5)

    """
    OUTPUT ARRAY STRUCTURE (q_batch):

        Axis 0 = trajectory index (which simulation)
        Axis 1 = time index (which timestep)
        Axis 2 = mass index (which mass)

        q_batch[42, 1500, 3] = position of mass 3
                                in trajectory 42
                                at timestep 1500 (t = 15.00 s)

        q_batch[0, 0, :]  = [0.0, 0.0, 3.7, 0.0, 0.0]   ← initial state of traj 0
        q_batch[0, -1, :] = [0.1, -0.3, 0.8, -0.5, 0.2]  ← final state of traj 0
    """

    # ── Flatten for Training ──
    # The HNN training loop doesn't care which trajectory or timestep a sample
    # came from. It just needs a big table of (q, p) pairs to learn from.
    # We reshape (B, T, N) → (B×T, N) by collapsing the first two axes.
    q_flat = q_batch.reshape(-1, config.N)  # shape: (200000, 5)
    p_flat = p_batch.reshape(-1, config.N)  # shape: (200000, 5)

    """
    FLATTENING VISUALIZATION:

        BEFORE (q_batch, shape 100×2000×5):
            Trajectory 0: [[q₀₀, q₀₁, q₀₂, q₀₃, q₀₄],   ← t=0.00
                           [q₁₀, q₁₁, q₁₂, q₁₃, q₁₄],   ← t=0.01
                           ...                              ← 2000 rows
                          ]
            Trajectory 1: [[...], [...], ...]               ← another 2000 rows
            ...
            Trajectory 99: [[...], [...], ...]

        AFTER (q_flat, shape 200000×5):
            Row 0:      [q₀₀, q₀₁, q₀₂, q₀₃, q₀₄]   ← traj 0, t=0.00
            Row 1:      [q₁₀, q₁₁, q₁₂, q₁₃, q₁₄]   ← traj 0, t=0.01
            ...
            Row 1999:   [...]                           ← traj 0, t=19.99
            Row 2000:   [...]                           ← traj 1, t=0.00
            ...
            Row 199999: [...]                           ← traj 99, t=19.99

    The training script will shuffle these rows and sample random mini-batches.
    """

    print(f"Done! Generated {len(q_flat)} samples.")
    return q_flat, p_flat, q_batch, save_times, args



# VISUALIZATION


def plot_sample_trajectory(q_batch, times, config):
    """
    Plots the displacement of all masses in ONE trajectory to sanity-check
    the physics before saving the data.

    WHAT TO LOOK FOR:
        - The displaced mass should oscillate back and forth.
        - Energy should propagate to neighboring masses over time.
        - The oscillation should NOT grow in amplitude (energy is conserved).
        - The motion should look smooth, not jagged (jagged = dt too large).

    PHYSICAL BEHAVIOUR:
        If mass 2 starts displaced and all others are at rest, you should see:
            t = 0s:    Mass 2 at max displacement, others at 0.
            t = 0.5s:  Mass 2 swinging back, masses 1 and 3 starting to move.
            t = 2s:    Energy has spread to all 5 masses.
            t = 10s:   Complex beat pattern as normal modes interfere.

    INPUT
        q_batch: shape (B, T, N) — positions from all trajectories
        times: shape (T,) — time axis
        config: SimConfig — for reading N
    """
    # Take the FIRST trajectory (index 0) from the batch
    q_sample = q_batch[0]   # shape: (T, N) = (2000, 5)

    """
    q_sample ARRAY STRUCTURE:

        Each ROW is a snapshot in time. Each COLUMN is a mass.

                    Mass 0   Mass 1   Mass 2   Mass 3   Mass 4
        t=0.00s  [  0.000,   0.000,   3.700,   0.000,   0.000 ]
        t=0.01s  [  0.000,   0.002,   3.697,   0.002,   0.000 ]
        t=0.02s  [  0.000,   0.007,   3.688,   0.007,   0.000 ]
        ...
        t=19.99s [ -0.213,   0.518,  -0.847,   0.312,  -0.104 ]

        q_sample[:, 2] gives the full time history of mass 2: shape (2000,)
        q_sample[100, :] gives the state of all masses at t=1.00s: shape (5,)
    """

    plt.figure(figsize=(10, 6))

    # Plot the position of each mass as a separate coloured line
    for i in range(config.N):
        # q_sample[:, i] extracts column i = time history of mass i
        # shape: (2000,) — one value per timestep
        plt.plot(times, q_sample[:, i], label=f'Mass {i}')

    plt.title(f"Sample Trajectory (1 of Batch)\nDisplacement propagation")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (q)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# MAIN EXECUTION
"""
PIPELINE SUMMARY:
    1. Define physics parameters (N=5, m=1.0, k=5.0, fixed walls)
    2. Generate 100 random initial conditions (one mass pulled each)
    3. Solve all 100 ODEs in parallel using vmap + Tsit5
    4. Flatten the results into training-ready arrays
    5. Plot one trajectory to verify the physics looks correct
    6. Save everything to a .npz file for the HNN training script

THE SAVED FILE STRUCTURE (coupled_oscillator_data.npz):
    'q': shape (200000, 5)  — positions   (the "input" for training)
    'p': shape (200000, 5)  — momenta     (the "input" for training)
    'm': scalar (1.0)       — mass        (needed to compute true dq/dt = p/m)
    'k': scalar (5.0)       — spring k    (needed to compute true dp/dt = k·A·q)
    'A': shape (5, 5)       — coupling    (needed to compute true dp/dt = k·A·q)

    The training script loads this file, splits into train/test sets,
    and uses (m, k, A) to compute the ground truth derivatives that
    the HNN learns to match.
"""

if __name__ == "__main__":
    # Define the physical system
    config = SimConfig(N=5, t_end=20.0, boundary='fixed')

    # Generate data: 100 trajectories × 2000 timesteps = 200,000 samples
    q_flat, p_flat, q_batch, times, args = run_simulation_parallel(
        config, num_trajectories=100
    )
    # q_flat shape: (200000, 5)
    # p_flat shape: (200000, 5)
    # q_batch shape: (100, 2000, 5)
    # times shape: (2000,)
    # args = (1.0, 5.0, A) where A is shape (5, 5)

    # Visual sanity check: plot one trajectory
    print("Plotting sample trajectory...")
    plot_sample_trajectory(q_batch, times, config)

    # Save to disk for the training script
    m, k, A = args
    np.savez(
        'coupled_oscillator_data.npz',
        q=q_flat,   # shape: (200000, 5)
        p=p_flat,   # shape: (200000, 5)
        m=m,        # scalar: 1.0
        k=k,        # scalar: 5.0
        A=A         # shape: (5, 5)
    )
    print("Data Saved.")