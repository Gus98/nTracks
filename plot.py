import matplotlib.pyplot as plt

import ode
import pde
import stochastic

plt.rcParams.update({
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# Uses stochastic in time_plotting mode and compares to PDE
#
def transient_plot():
    #configure constants
    stoch = stochastic.StochasticSim()
    alpha = 0.7
    stoch.alpha = alpha
    beta = 0.8
    stoch.beta = beta
    sig = stoch.delta
    delta = stoch.delta # This is actually junk. PDE calls delta sig, delta is switching which I dont' use
    N = 4
    stoch.N = N
    omega_A = 0.5
    stoch.omega_A = omega_A
    omega_D = 0.1
    stoch.omega_D = omega_D
    num_time_buckets = 4
    stoch.num_time_buckets = num_time_buckets
    stoch.simulation_time_value = 40.0
    simulation_time = stoch.simulation_time_value * delta
    stoch.num_runs = 2048
    num_processes = 4
    compare_pde = True
    stoch.left_track_entry = False
    stoch.left_reservoir_entry = True
    pde_boundary_condition = "neumann_left_dirichlet_right" # or "periodic"
    stoch.initial_state = "sinstate"

    # get the stochastic solution
    results = stochastic.solve_snapshots_parallel(stoch, num_processes)
    x_stochastic, track_stochastic, off_stochastic, track_time_density, off_time_density, stochastic_times = results

    # get the ODE solution
    initial_params = pde.make_params(
        alpha_value=alpha,
        beta_value=beta,
        delta_value=delta,
        sig_value=sig,
        N_value=N,
        omega_A_value=omega_A,
        omega_D_value=omega_D,
    )

    initial_x = initial_params.xs
    initial_track = pde.get_initial_state("sinstate", initial_params)
    initial_off = pde.get_off_track_density(initial_track, initial_params)

    pde_results = pde.solve_pde_by_time(
        alpha_value=alpha,
        beta_value=beta,
        delta_value=delta,
        N_value=N,
        omega_A_value=omega_A,
        omega_D_value=omega_D,
        initial_state="sinstate",
        simulation_time_value=simulation_time,
        num_time_buckets_value=num_time_buckets,
        boundary_condition=pde_boundary_condition,
        boundary_update=pde.dynamic_boundary,
    )
    x_pde, track_pde, off_pde, track_pde_time, off_pde_time, pde_time_weights = pde_results

    # plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("on-track")
    axes[1].set_title("off-track")

    axes[0].plot(initial_x, initial_track, color="black", linewidth=2.0, label="initial")
    axes[1].plot(initial_x, initial_off, color="black", linewidth=2.0, linestyle="--", label="initial")
    axes[0].plot(x_stochastic, track_time_density[0], color="0.35", linestyle=":", label="stochastic initial")
    axes[1].plot(x_stochastic, off_time_density[0], color="0.35", linestyle=":", label="stochastic initial")

    colors = plt.cm.viridis([i / max(num_time_buckets - 1, 1) for i in range(num_time_buckets)])
    for time_bucket, color in enumerate(colors):
        label = f"bucket {time_bucket + 1}"
        stochastic_index = time_bucket + 1
        if compare_pde and pde_time_weights[time_bucket] != 0:
            axes[0].plot(x_pde, track_pde_time[time_bucket], color=color, label=f"PDE {label}")
            axes[1].plot(x_pde, off_pde_time[time_bucket], color=color, label=f"PDE {label}")
        axes[0].plot(x_stochastic, track_time_density[stochastic_index], color=color, linestyle=":", label=f"stochastic {label}")
        axes[1].plot(x_stochastic, off_time_density[stochastic_index], color=color, linestyle=":", label=f"stochastic {label}")

    for ax in axes:
        ax.set_xlim(0, stoch.L)
        ax.set_ylim(bottom=0)
        #ax.legend()
    axes[1].set_ylim(0,0.18)
    
    fig.tight_layout()

    plt.savefig('transient_sol', format='png', bbox_inches = 'tight')
    plt.show()

# Uses stochastic in stationary mode and compares to ODE solution
def stationary_plot():
    stoch = stochastic.StochasticSim()
    alpha = 0.7
    stoch.alpha = alpha
    beta = 0.8
    stoch.beta = beta
    delta = stoch.delta
    N = 5
    stoch.N = N
    omega_A = 0.5
    stoch.omega_A = omega_A
    omega_D = 0.1
    stoch.omega_D = omega_D
    stoch.initial_state = "even"
    stoch.num_steps = 200_000
    stoch.burn_in_steps = 100_000
    stoch.left_track_entry = True
    stoch.left_reservoir_entry = False
    stoch.num_runs = 16
    stoch.return_time_buckets = False
    stoch.seed = 1
    stoch.rng = stochastic.np.random.default_rng(stoch.seed)

    # Get the solution of the ODE, x values as well as on/off concentrations
    x_ode, track_ode, off_ode = ode.solve_steady_state(
        alpha=alpha,
        beta=beta,
        delta=delta,
        N=N,
        omega_A=omega_A,
        omega_D=omega_D,
    )

    # Get the solution of the stochastic simulation, x values as well as on/off concentrations
    x_stochastic, track_stochastic, off_stochastic = stoch.solve_stochastic()

    plt.plot(x_ode, track_ode, color="red", label="ODE")
    plt.plot(x_stochastic, track_stochastic, color="red", linestyle=":", label="stochastic")
    plt.plot(x_ode, off_ode, color="green", label="ODE")
    plt.plot(x_stochastic, off_stochastic, color="green", linestyle=":", label="stochastic")
    plt.xlim(0, stoch.L)
    plt.ylim(0,1)
    plt.tight_layout()

    plt.savefig('MC.eps', format='eps', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    transient_plot()
