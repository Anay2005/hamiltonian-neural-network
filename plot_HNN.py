# ==========================================
# IMPORTS: THE PHYSICS-ML STACK
# ==========================================
# JAX is the core engine. It acts like NumPy but runs on GPUs/TPUs and, 
# crucially, can automatically calculate exact mathematical derivatives of any function.
import jax
import jax.numpy as jnp 

# Equinox is a library built on top of JAX specifically for Neural Networks.
# It treats neural networks as standard math functions, which plays nicely with JAX.
import equinox as eqx

# Diffrax is the ODE (Ordinary Differential Equation) solver built for JAX.
# It takes a function describing derivatives and steps it forward in time.
import diffrax

# Standard libraries for data handling and plotting
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------------------------
# PRECISION SETTING
# ------------------------------------------
# Physics relies on tiny, accumulating changes. Standard 32-bit AI floats (float32) 
# cause rounding errors that snowball during ODE integration. We force 64-bit (float64) 
# to ensure our energy conservation math stays mathematically rigorous.
jax.config.update("jax_enable_x64", True)


# ==========================================
# 1. CONFIGURATION & ARCHITECTURE 
# ==========================================

# This is the Growing Cosine Unit (GCU) activation function: f(z) = z * cos(z).
# Standard activations like ReLU are piece-wise linear (their second derivative is zero).
# Physics requires smooth, non-zero higher-order derivatives, making GCU excellent for oscillatory physics.
def gcu(z):
    return z * jnp.cos(z)

# A simple data container to hold the network's shape. 
# We need this so we can build the exact same "blank" network before loading our saved weights.
@dataclass
class TrainConfig:
    hidden_channels: int = 32
    kernel_size: int = 3

# This defines the "Brain" of the Hamiltonian Neural Network (HNN).
# Instead of predicting exactly where the mass goes, this network looks at 
# the state (position, momentum) and predicts a single number: the Total Energy (Hamiltonian).
class ConvHNN(eqx.Module):
    # Equinox requires us to declare the layers we will use up front.
    layers: list

    # The initialization function sets up the weights.
    def __init__(self, key, config: TrainConfig):
        # We need 3 random number generator keys for 3 convolutional layers.
        keys = jax.random.split(key, 3)
        hc = config.hidden_channels
        ks = config.kernel_size
        
        # We use 1D Convolutions because a chain of masses on springs has spatial structure.
        # Mass 3 only interacts with Mass 2 and Mass 4. A Conv1D layer with a kernel size of 3 
        # naturally learns this "local neighbor" interaction perfectly!
        self.layers = [
            # Layer 1: Takes 2 channels (Position q, Momentum p) and expands to 'hc' channels.
            eqx.nn.Conv1d(in_channels=2, out_channels=hc, kernel_size=ks, padding=ks//2, key=keys[0]),
            # Layer 2: A 1x1 convolution. This acts like a standard Dense layer applied to each mass individually.
            eqx.nn.Conv1d(in_channels=hc, out_channels=hc, kernel_size=1, key=keys[1]),
            # Layer 3: Compresses the 'hc' channels down to 1 channel (the local energy of each mass).
            eqx.nn.Conv1d(in_channels=hc, out_channels=1, kernel_size=1, key=keys[2])
        ]

    # The forward pass: what happens when data flows through the network.
    def __call__(self, x):
        # The input 'x' is a flat list of [q_0, q_1... q_N, p_0, p_1... p_N].
        # We split it in half to separate positions (q) and momenta (p).
        N = x.shape[0] // 2
        q = x[:N]
        p = x[N:]
        
        # We stack them so the shape is (2 channels, N masses). 
        # Now it looks like an "image" of 2 rows and N columns, ready for 1D Convolution.
        x_reshaped = jnp.stack([q, p], axis=0) 
        
        # Pass the data through the first two layers, applying our GCU activation function each time.
        for layer in self.layers[:-1]:
            x_reshaped = gcu(layer(x_reshaped))
            
        # The final layer outputs the "local energy" for each individual mass.
        local_energies = self.layers[-1](x_reshaped)
        
        # The total Hamiltonian (Energy) of the system is just the sum of all local energies.
        return jnp.sum(local_energies)


# ==========================================
# 2. PHYSICS ENGINE (Ground Truth)
# ==========================================

# This calculates the actual, theoretical total energy of the system using textbook physics.
def get_true_energy(q, p, m, k, A):
    # Kinetic Energy: $T = \sum \frac{p^2}{2m}$
    kinetic = jnp.sum(p**2) / (2 * m)
    
    # Potential Energy: $V = \frac{1}{2} k x^2$. 
    # For a coupled system, the matrix 'A' handles the relative displacements between adjacent masses.
    potential = 0.5 * k * jnp.dot(q, jnp.dot(-A, q))
    
    # Total Energy $H = T + V$
    return kinetic + potential

# This represents the true laws of physics. The ODE solver will use this to generate the "Red" line.
def vector_field_ground_truth(t, state, args):
    q, p = state
    m, k, A = args
    
    # Velocity is momentum divided by mass: $\dot{q} = \frac{p}{m}$
    dq_dt = p / m
    
    # Force is mass times acceleration (or the rate of change of momentum). Hooke's Law: $F = -kx$
    dp_dt = k * jnp.dot(A, q) 
    
    return dq_dt, dp_dt


# ==========================================
# 3. HNN ENGINE (Neural Network Dynamics)
# ==========================================

# This is the "magic" function. The ODE solver will use THIS to generate the "Blue" line.
# Notice that there is NO mass 'm' or spring constant 'k' here. The network has to figure it out.
def vector_field_hnn(t, state, args):
    model, N = args
    q, p = state
    
    # The network expects a single flat 1D array of inputs, so we stick q and p back together.
    flat_state = jnp.concatenate([q, p])
    
    # AD (Automatic Differentiation) IN ACTION:
    # 'jax.grad(model)' creates a NEW function that calculates the exact mathematical derivative 
    # of the model's output (Energy) with respect to its input (flat_state).
    # We pass 'flat_state' into this new gradient function to get a list of derivatives.
    grads = jax.grad(model)(flat_state)
    
    # We split the gradients back into "change in Energy wrt position" and "change in Energy wrt momentum".
    dH_dq = grads[:N]
    dH_dp = grads[N:]
    
    # Hamilton's Equations of Motion:
    # 1. How does position change over time? It equals the gradient of energy wrt momentum.
    # 2. How does momentum change over time? It equals the NEGATIVE gradient of energy wrt position.
    return dH_dp, -dH_dq


# ==========================================
# 4. MAIN VISUALIZATION ROUTINE
# ==========================================

def main():
    print("Initializing Visualization...")
    
    # Define where the data and the trained "brain" are stored on your hard drive.
    data_path = 'coupled_oscillator_data.npz'
    model_path = 'model_weights.eqx'
    
    # Load the Numpy file. We don't need the trajectories, just the physical constants 
    # (m, k) and the interaction matrix (A) so we can run the "Ground Truth" simulation for comparison.
    data = np.load(data_path)
    m = float(data['m'])
    k = float(data['k'])
    A = jnp.array(data['A'])
    N = A.shape[0] # N is the total number of masses in the system
    
    # Build a "blank" neural network with random, untrained weights.
    cfg = TrainConfig()
    key = jax.random.PRNGKey(0)
    model = ConvHNN(key, cfg)
    
    # Read the .eqx file and overwrite the blank network's random weights with your trained weights.
    try:
        model = eqx.tree_deserialise_leaves(model_path, model)
        print("Model Weights Loaded Successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # --- Simulation Setup ---
    print("Generating Test Trajectory...")
    # We create a brand new starting scenario to test if the model actually generalized.
    # We place all masses at 0.0, EXCEPT the middle mass, which we pull to 1.0.
    q0 = jnp.zeros(N).at[N//2].set(1.0) 
    p0 = jnp.zeros(N) # All masses start at rest (zero momentum)
    
    # Pack the initial positions and momenta into a tuple called y0.
    y0 = (q0, p0)
    
    # We will simulate from 0 to 20 seconds, asking the solver to save the state every 0.05 seconds.
    t_start, t_end, dt = 0.0, 20.0, 0.05
    save_times = jnp.arange(t_start, t_end, dt)

    # --- Integrate Ground Truth (The baseline) ---
    # We wrap our exact physics math in an 'ODETerm' so the diffrax solver understands it.
    term_true = diffrax.ODETerm(vector_field_ground_truth)
    # We use Tsitouras 5/4, a highly efficient Runge-Kutta solver.
    solver = diffrax.Tsit5()
    
    # Run the simulation. The solver takes tiny steps, constantly consulting 'vector_field_ground_truth'
    # to find out how q and p should change next.
    sol_true = diffrax.diffeqsolve(
        term_true, solver, t0=t_start, t1=t_end, dt0=dt, y0=y0,
        args=(m, k, A), saveat=diffrax.SaveAt(ts=save_times), max_steps=40000
    )
    
    # --- Integrate HNN Prediction (The AI) ---
    # We do the EXACT same thing, but this time the solver consults the Neural Network's 
    # learned energy gradients ('vector_field_hnn') instead of the true physics math.
    term_hnn = diffrax.ODETerm(vector_field_hnn)
    sol_hnn = diffrax.diffeqsolve(
        term_hnn, solver, t0=t_start, t1=t_end, dt0=dt, y0=y0,
        args=(model, N), saveat=diffrax.SaveAt(ts=save_times), max_steps=40000
    )

    # --- Extract Results ---
    # 'sol_true.ys' and 'sol_hnn.ys' contain the entire history of the system.
    # q_true is a 2D array: (Number of time steps, Number of masses)
    q_true, p_true = sol_true.ys
    q_pred, p_pred = sol_hnn.ys
    times = save_times

    # Calculate the True, textbook Energy of the system at every single time step.
    # jax.vmap applies the 'get_true_energy' function to every row in the time history simultaneously 
    # without needing a slow 'for' loop. We do this for both the real trajectory and the AI's trajectory.
    E_true_traj = jax.vmap(lambda q, p: get_true_energy(q, p, m, k, A))(q_true, p_true)
    E_pred_traj = jax.vmap(lambda q, p: get_true_energy(q, p, m, k, A))(q_pred, p_pred)
    
    # Calculate Mean Squared Error (MSE) at every time step.
    # We square the difference between true and predicted values, then average across all N masses.
    mse_q = jnp.mean((q_true - q_pred)**2, axis=1)
    mse_p = jnp.mean((p_true - p_pred)**2, axis=1)
    total_mse = mse_q + mse_p


    # ==========================================
    # 5. PLOTTING
    # ==========================================
    # (Standard Matplotlib code to generate the 4-panel dashboard)
    print("Plotting Results...")
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Phase Space. Plots momentum on the Y axis and position on the X axis.
    # In an oscillator, this creates circles/ellipses. If the model is accurate, the blue line follows the red.
    ax1 = fig.add_subplot(2, 2, 1)
    mass_idx = N // 2 # We only plot the middle mass to keep the graph readable
    ax1.plot(q_true[:, mass_idx], p_true[:, mass_idx], 'r--', label='Ground Truth', alpha=0.6)
    ax1.plot(q_pred[:, mass_idx], p_pred[:, mass_idx], 'b-', label='HNN Prediction', alpha=0.6)
    ax1.set_title(f"Phase Space (Mass {mass_idx})\n(Cycles should overlap)")
    ax1.set_xlabel(f"Position $q_{{{mass_idx}}}$")
    ax1.set_ylabel(f"Momentum $p_{{{mass_idx}}}$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy Conservation. 
    # Both lines should be perfectly horizontal. If the blue line drifts, the AI is "leaking" energy.
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, E_true_traj, 'r--', label='True Trajectory Energy', linewidth=2)
    ax2.plot(times, E_pred_traj, 'b-', label='HNN Trajectory Energy', linewidth=1.5)
    
    # Set the Y-axis limits to zoom in tightly on the energy line, making small fluctuations visible.
    y_min = min(jnp.min(E_true_traj), jnp.min(E_pred_traj)) * 0.9
    y_max = max(jnp.max(E_true_traj), jnp.max(E_pred_traj)) * 1.1
    ax2.set_ylim(y_min, y_max)
    
    ax2.set_title("Conservation of Energy\n(HNN should stay constant)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Total Hamiltonian ($H$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean Squared Error over time.
    # Because errors in ODE integration compound exponentially, a logarithmic scale is used.
    # An upward slope is normal; we just want it to stay as low as possible for as long as possible.
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, total_mse, 'k-')
    ax3.set_title("Prediction Error (MSE) vs Time\n(Lower is better)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("MSE Loss")
    ax3.set_yscale('log') 
    ax3.grid(True, alpha=0.3)

    # Plot 4: Standard Position vs Time graph.
    # Shows the physical back-and-forth wobble of the middle mass.
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(times, q_true[:, mass_idx], 'r--', label='True', alpha=0.6)
    ax4.plot(times, q_pred[:, mass_idx], 'b-', label='Pred', alpha=0.6)
    ax4.set_title(f"Displacement vs Time (Mass {mass_idx})")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position (q)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Clean up layout, save the image to the hard drive, and display it on screen.
    plt.tight_layout()
    plt.savefig('hnn_evaluation_metrics.png', dpi=150)
    print("Plots saved to 'hnn_evaluation_metrics.png'")
    plt.show()

    # --- ADVANCED CHECK: The Model's Internal "Belief" about Energy ---
    
    # 1. How much energy does the HNN think is in its OWN generated trajectory?
    # Because the ODE solver strictly follows the gradients of this learned energy surface, 
    # the model should believe energy is perfectly constant here.
    hnn_internal_energy = []
    for i in range(len(times)):
        state = jnp.concatenate([q_pred[i], p_pred[i]])
        hnn_internal_energy.append(model(state))
    hnn_internal_energy = jnp.array(hnn_internal_energy)

    # 2. How much energy does the HNN think is in the TRUE physics trajectory?
    # The network's learned energy surface isn't perfect. As the true trajectory moves through 
    # phase space, the network's evaluation of that energy will show small bumps and errors.
    true_data_internal_energy = []
    for i in range(len(times)):
        state = jnp.concatenate([q_true[i], p_true[i]])
        true_data_internal_energy.append(model(state))
    true_data_internal_energy = jnp.array(true_data_internal_energy)

    # Plotting this internal belief check.
    plt.figure()
    plt.plot(times, hnn_internal_energy, 'b-', label='HNN Trajectory (Model Output)')
    plt.plot(times, true_data_internal_energy, 'k--', label='True Trajectory (Model Output)')
    plt.title("HNN-Conserved Quantity\n(Blue should be flat, Black measures model error)")
    plt.legend()
    plt.show()

# Standard Python boilerplate to run the main function when the script is executed.
if __name__ == "__main__":
    main()