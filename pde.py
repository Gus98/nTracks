from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# default parameters
alpha = 0.7
beta = 0.1
omega_A = 0.5
omega_D = 0.1
N = 5
delta = 0.1
dur = 80
dt = 0.001
domain_length = 1.0
num_points = 64
sig = 0.1
v = 1.0


@dataclass
class PDEParams:
    alpha: float = alpha
    beta: float = beta
    delta: float = delta
    N: float = N
    omega_A: float = omega_A
    omega_D: float = omega_D
    dt: float = dt
    domain_length: float = domain_length
    num_points: int = num_points
    sig: float = sig
    v: float = v

    @property
    def T(self):
        return self.N * self.omega_A + self.omega_D

    @property
    def dx(self):
        return self.domain_length / self.num_points

    @property
    def xs(self):
        return np.arange(0.0, self.domain_length, self.dx)


def d1(state,l2,l1,r1,r2,params):
    return (-1*r2 + 8*r1 - 8*l1 + l2)/(12 * params.dx)


def d2(state,l2,l1,r1,r2,params):
    return (-1*l2 + 16*l1 - 30*state + 16*r1 -r2)/(12 * params.dx**2)


def d3(state,l2,l1,r1,r2,params):
    return (-1*l2 + 2*l1 - 2*r1 + r2)/(2 * params.dx**3)


def d4(state,l2,l1,r1,r2,params):
    return (l2 - 4*l1 + 6*state - 4*r1 + r2)/(params.dx**4)


def get_stencil_states(state, boundary_condition="periodic"):
    # periodic keeps the old np.roll behavior
    if boundary_condition == "periodic":
        return np.roll(state, 2), np.roll(state, 1), np.roll(state,-1), np.roll(state,-2)

    if boundary_condition in ("constant", "constant_boundary", "clamped"):
        # Ghost/stencil values outside the domain are held equal to the nearest
        # physical boundary value. This is the mode to pair with dynamic_boundary.
        l1 = np.empty_like(state)
        l2 = np.empty_like(state)
        r1 = np.empty_like(state)
        r2 = np.empty_like(state)

        l1[1:] = state[:-1]
        l1[0] = state[0]
        l2[2:] = state[:-2]
        l2[:2] = state[0]

        r1[:-1] = state[1:]
        r1[-1] = state[-1]
        r2[:-2] = state[2:]
        r2[-2:] = state[-1]
        return l2,l1,r1,r2

    if boundary_condition == "neumann_left_dirichlet_right":
        # left Neumann uses mirrored ghost values; right ghost values equal state[-1]
        l1 = np.empty_like(state)
        l2 = np.empty_like(state)
        r1 = np.empty_like(state)
        r2 = np.empty_like(state)

        l1[1:] = state[:-1]
        l1[0] = state[1] if state.size > 1 else state[0]
        l2[2:] = state[:-2]
        l2[1] = state[1] if state.size > 1 else state[0]
        l2[0] = state[2] if state.size > 2 else l1[0]

        r1[:-1] = state[1:]
        r1[-1] = state[-1]
        r2[:-2] = state[2:]
        r2[-2:] = state[-1]
        return l2,l1,r1,r2

    raise ValueError("boundary_condition must be 'periodic', 'constant', or 'neumann_left_dirichlet_right'")


def apply_boundary_values(state, boundary_values):
    # boundary_update functions can return {"left": value, "right": value},
    # {0: value, -1: value}, or (left_value, right_value).
    if boundary_values is None:
        return state
    if isinstance(boundary_values, dict):
        for key, value in boundary_values.items():
            if key == "left":
                state[0] = value
            elif key == "right":
                state[-1] = value
            else:
                state[key] = value
        return state

    left_value, right_value = boundary_values
    if left_value is not None:
        state[0] = left_value
    if right_value is not None:
        state[-1] = right_value
    return state


def apply_fixed_boundary(state, params, boundary_condition, right_dirichlet_value):
    if boundary_condition == "neumann_left_dirichlet_right":
        if right_dirichlet_value is None:
            right_dirichlet_value = 1 - params.beta
        state[-1] = right_dirichlet_value
    return state


def call_boundary_update(boundary_update, state, step_duration, params):
    if boundary_update is None:
        return None
    try:
        return boundary_update(state.copy(), step_duration, params)
    except TypeError:
        return boundary_update(state.copy(), step_duration)


def oneDimTrackPDE(state, d1_val, d2_val, d3_val, d4_val, params=None):
    if params is None:
        params = PDEParams()
    denominator = 1 + (
        (params.omega_D / params.omega_A)
        * (1 / (1 - state)**2)
        * (1 / params.N)
    )
    return (-params.v*(1-2*state)*d1_val + params.sig/2 * d2_val) / denominator


def dynamic_boundary(state, step_duration, params=None):
    if params is None:
        params = PDEParams()
    u = state[0]
    u_1 = state[1]
    v_right = state[-1]
    v_right_1 = state[-2]
    left_scale = params.delta * (
        params.N + (params.omega_D / params.omega_A) * (1 / (1 - u)**2)
    )
    right_scale = params.delta * (
        params.N + (params.omega_D / params.omega_A) * (1 / (1 - v_right)**2)
    )
    next_u = u + step_duration * (params.alpha - params.N * u * (1 - u_1)) / left_scale
    next_v = v_right + step_duration * (params.N * v_right_1 * (1 - v_right) - params.N * params.beta * v_right) / right_scale
    return {"left": next_u, "right": next_v}

def track_boundary(state, step_duration, params=None):
    if params is None:
        params = PDEParams()
    u = state[0]
    u_1 = state[1]
    v_right = state[-1]
    v_right_1 = state[-2]
    left_scale = params.delta * (
        params.N + (params.omega_D / params.omega_A) * (1 / (1 - u)**2)
    )
    right_scale = params.delta * (
        params.N + (params.omega_D / params.omega_A) * (1 / (1 - v_right)**2)
    )
    next_u = u + step_duration * (params.alpha * params.N - params.N * u * (1 - u_1)) / left_scale
    next_v = v_right + step_duration * (params.N * v_right_1 * (1 - v_right) - params.N * params.beta * v_right) / right_scale
    return {"left": next_u, "right": next_v}

def get_initial_state(initial_state, params):
    xs = params.xs
    if initial_state == "sinstate":
        return 0.25 + 0.15 * np.sin(2 * np.pi * xs / params.domain_length)
    if initial_state == "bump":
        state = np.zeros(xs.size)
        state[: xs.size // 5] = 1.0
        return state
    if initial_state == "positivebump":
        state = np.full(xs.size,0.1)
        state[: xs.size // 5] = 1.0
        return state
    if initial_state == "zero":
        state = np.full(xs.size, 0.0001)
        return state
    return np.full(xs.size,0.5)


def get_off_track_density(track_density, params):
    if params.omega_A == 0:
        return np.zeros_like(track_density)
    denominator = params.omega_A * (1 - track_density)
    return np.divide(
        params.omega_D * track_density,
        denominator,
        out=np.full_like(track_density, np.nan, dtype=float),
        where=denominator != 0,
    )


def make_params(
    alpha_value=None,
    beta_value=None,
    delta_value=None,
    N_value=None,
    omega_A_value=None,
    omega_D_value=None,
    dt_value=None,
    domain_length_value=None,
    num_points_value=None,
    sig_value=None,
    v_value=None,
):
    return PDEParams(
        alpha=alpha if alpha_value is None else alpha_value,
        beta=beta if beta_value is None else beta_value,
        delta=delta if delta_value is None else delta_value,
        N=N if N_value is None else N_value,
        omega_A=omega_A if omega_A_value is None else omega_A_value,
        omega_D=omega_D if omega_D_value is None else omega_D_value,
        dt=dt if dt_value is None else dt_value,
        domain_length=domain_length if domain_length_value is None else domain_length_value,
        num_points=num_points if num_points_value is None else int(num_points_value),
        sig=sig if sig_value is None else sig_value,
        v=v if v_value is None else v_value,
    )


def solve_pde_by_time(
    alpha_value=None,
    beta_value=None,
    delta_value=None,
    N_value=None,
    omega_A_value=None,
    omega_D_value=None,
    simulation_time_value=dur,
    num_time_buckets_value=10,
    initial_state="sinstate",
    PDE=oneDimTrackPDE,
    boundary_condition="periodic",
    right_dirichlet_value=None,
    boundary_update=None,
    dt_value=None,
    num_points_value=None,
):
    params = make_params(
        alpha_value=alpha_value,
        beta_value=beta_value,
        delta_value=delta_value,
        N_value=N_value,
        omega_A_value=omega_A_value,
        omega_D_value=omega_D_value,
        dt_value=dt_value,
        num_points_value=num_points_value,
    )
    simulation_time_value = float(simulation_time_value)
    num_time_buckets_value = int(num_time_buckets_value)
    if simulation_time_value <= 0:
        raise ValueError("simulation_time_value must be positive")
    if num_time_buckets_value <= 0:
        raise ValueError("num_time_buckets_value must be positive")

    state = get_initial_state(initial_state, params)
    state = apply_fixed_boundary(state, params, boundary_condition, right_dirichlet_value)
    boundary_values = call_boundary_update(boundary_update, state, 0, params)
    state = apply_boundary_values(state, boundary_values)

    track_time_accum = np.zeros((num_time_buckets_value, state.size), dtype=float)
    off_time_accum = np.zeros((num_time_buckets_value, state.size), dtype=float)
    time_bucket_durations = np.zeros(num_time_buckets_value, dtype=float)
    time_edges = np.linspace(0, simulation_time_value, num_time_buckets_value + 1)

    current_time = 0.0
    while current_time < simulation_time_value:
        step_end = min(current_time + params.dt, simulation_time_value)
        step_duration = step_end - current_time
        boundary_values = call_boundary_update(boundary_update, state, step_duration, params)
        off_state = get_off_track_density(state, params)

        for time_bucket in range(num_time_buckets_value):
            overlap_start = max(current_time, time_edges[time_bucket])
            overlap_end = min(step_end, time_edges[time_bucket + 1])
            duration = max(0, overlap_end - overlap_start)
            if duration == 0:
                continue
            track_time_accum[time_bucket] += state * duration
            off_time_accum[time_bucket] += off_state * duration
            time_bucket_durations[time_bucket] += duration

        l2,l1,r1,r2 = get_stencil_states(state, boundary_condition)
        d1_val = d1(state,l2,l1,r1,r2,params)
        d2_val = d2(state,l2,l1,r1,r2,params)
        d3_val = d3(state,l2,l1,r1,r2,params)
        d4_val = d4(state,l2,l1,r1,r2,params)
        try:
            state = state + step_duration * PDE(state, d1_val, d2_val, d3_val, d4_val, params)
        except TypeError:
            state = state + step_duration * PDE(state, d1_val, d2_val, d3_val, d4_val)
        state = apply_fixed_boundary(state, params, boundary_condition, right_dirichlet_value)
        state = apply_boundary_values(state, boundary_values)
        current_time = step_end

    active_buckets = time_bucket_durations > 0
    track_time_density = np.zeros_like(track_time_accum)
    off_time_density = np.zeros_like(off_time_accum)
    track_time_density[active_buckets] = (
        track_time_accum[active_buckets] / time_bucket_durations[active_buckets, np.newaxis]
    )
    off_time_density[active_buckets] = (
        off_time_accum[active_buckets] / time_bucket_durations[active_buckets, np.newaxis]
    )

    total_duration = time_bucket_durations.sum()
    track_density = track_time_accum.sum(axis=0) / total_duration
    off_density = off_time_accum.sum(axis=0) / total_duration
    return params.xs, track_density, off_density, track_time_density, off_time_density, time_bucket_durations


def main():
    pde_xs, track_density, off_density, track_time_density, off_time_density, time_bucket_durations = solve_pde_by_time(
        simulation_time_value=dur,
        num_time_buckets_value=10,
        initial_state="zero",
        boundary_condition="constant",
        boundary_update=dynamic_boundary,
    )

    colors = plt.cm.viridis(np.linspace(0, 1, track_time_density.shape[0]))
    for time_bucket, color in enumerate(colors):
        if time_bucket_durations[time_bucket] == 0:
            continue
        label = f"bucket {time_bucket + 1}"
        plt.plot(pde_xs, track_time_density[time_bucket], color=color, label=f"track {label}")
        plt.plot(pde_xs, off_time_density[time_bucket], color=color, linestyle="--", label=f"off-track {label}")

    plt.xlim(0, domain_length)
    plt.ylim(bottom=0)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Transient PDE TASEP")
    plt.legend()
    plt.show()

def TASEP_plot():
    pde_xs, track_density, off_density, track_time_density, off_time_density, time_bucket_durations = solve_pde_by_time(
        simulation_time_value=dur,
        num_time_buckets_value=1,
        initial_state="zero",
        boundary_condition="constant", # used for definig ghost point values
        boundary_update=track_boundary, # used for actually updating the value at the boundaries
    )

    colors = plt.cm.viridis(np.linspace(0, 1, track_time_density.shape[0]))
    for time_bucket, color in enumerate(colors):
        if time_bucket_durations[time_bucket] == 0:
            continue
        label = f"bucket {time_bucket + 1}"
        plt.plot(pde_xs, track_time_density[time_bucket], color=color, label=f"track {label}")
        plt.plot(pde_xs, off_time_density[time_bucket], color=color, linestyle="--", label=f"off-track {label}")

    plt.xlim(0, domain_length)
    plt.ylim(bottom=0)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Transient PDE TASEP")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    TASEP_plot()
