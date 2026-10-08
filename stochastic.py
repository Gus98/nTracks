import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

"""
Purpose is to create a function which plots an averaged trajectory given the constants
This is a true stochastic particle simulation
idea:
    represent the state as a nice array. Iterate it with some numpy stuff
    average over some set region sizes and print that graph at the time intervals
"""
# A class for containing all methods required for running and plotting a simulation
# Class data is ONLY simulation parameters
# The state of the simulation is generated and manipulated by class methods, but not actually stored in the object
class StochasticSim:
    # Initializes with default parameter values
    def __init__(self):
        # attaching and detaching rates for either end of the boundary
        self.alpha = 0.8
        self.beta = 0.9

        # falling on/off rates for each track
        self.omega_A = 0.9
        self.omega_D = 0.3

        # track constants
        self.N = 4 # number of tracks
        self.L = 1 # length of the track
        self.M = 32 # number of lattice points per track
        self.delta = self.L/(self.M+1) # the gap is smaller, because these points are fit evenly between 0,L

        # simulation configs
        self.num_runs = 16
        self.bucket = 1
        self.num_steps = 60_000
        self.burn_in_steps = self.num_steps-30_000
        self.num_time_buckets = 5
        self.simulation_time_value = 1.0
        self.max_iterations_value = None
        self.left_track_entry = False
        self.left_reservoir_entry = False
        self.rng = np.random.default_rng()
        self.initial_state="random"
        self.seed=None
        self.return_time_buckets=True


    # generates a state with a 50/50 to have a particle in each spot
    # % should probably be more clever about the distribution off track
    def initialize_random_track(self, fill_ratio):
        distribution = [1-fill_ratio, fill_ratio]
        return self.rng.choice([0,1], size=(self.N+1,self.M,self.num_runs), p=distribution)

    # helpful for basic TASEP boundaries, initializes to proper bulk distribution
    def initialize_even_track(self):
        if self.alpha > 1/2:
            if self.beta > 1/2:
                fill_ratio = 0.5
            else:
                fill_ratio = 1-self.beta
        else:
            if self.alpha > self.beta:
                fill_ratio = 1-self.beta
            else:
                fill_ratio = self.alpha

        fill_ratio = 0.276
        distribution = [1-fill_ratio, fill_ratio]
        tracks = self.rng.choice([0,1], size=(self.N+1,self.M,self.num_runs), p=distribution)
        if self.omega_A != 0 :
            if fill_ratio != 1:
                steady = (self.omega_D/self.omega_A) * (fill_ratio/(1-fill_ratio))
            else:
                steady = (self.omega_D/self.omega_A) * (0.99/0.01)
        else:
            steady = 0
        tracks[0,:,:] = self.rng.poisson(steady, size=(self.M,self.num_runs))
        return tracks

    def initialize_sinstate(self, midline, amplitude):
        # generates a sinusoidal density wave on the tracks
        # track probability is 0.5 + 0.35*sin(2*pi*x), which stays between 0.15 and 0.85
        x = (np.arange(self.M) + 1) * self.delta
        fill_ratio = midline + amplitude * np.sin(2 * np.pi * x)
        tracks = np.zeros((self.N+1,self.M,self.num_runs), dtype=int)
        tracks[1:,:,:] = self.rng.binomial(1, fill_ratio[np.newaxis, :, np.newaxis], size=(self.N,self.M,self.num_runs))

        if self.omega_A != 0:
            steady = (self.omega_D/self.omega_A) * (fill_ratio/(1-fill_ratio))
        else:
            steady = np.zeros(self.M)
        tracks[0,:,:] = self.rng.poisson(steady[:, np.newaxis], size=(self.M,self.num_runs))
        return tracks

    # calls the proper initialization function from the passed in string
    def initialize_state(self, initial_state):
        if initial_state == "empty":
            return self.initialize_empty_track()
        if initial_state == "even":
            return self.initialize_even_track()
        if initial_state == "sinstate":
            return self.initialize_sinstate(0.25, 0.15)
        return self.initialize_random_track(0.5)

    ### HELPERS FOR EXECUTING THE SIMULATION ###

    # iterates the state as a numpy array
    # gets move masks first, computes rates, and picks one
    def iterate_state(self, state, t, active_runs=None):
        moves = self.get_moves(state)
        hop_rate             = moves["hop_count"]
        attach_rate          = self.omega_A * moves["attach_count"]
        detach_rate          = self.omega_D * moves["detach_count"]
        track_entry_rate     = self.alpha * moves["enter_count"] if self.left_track_entry else np.zeros(self.num_runs)
        reservoir_entry_rate = np.full(self.num_runs, self.alpha) if self.left_reservoir_entry else np.zeros(self.num_runs)
        track_exit_rate      = self.beta * moves["exit_count"]
        total_rate           = (
            hop_rate
            + attach_rate
            + detach_rate
            + track_entry_rate
            + reservoir_entry_rate
            + track_exit_rate
        )
        dt                   = np.zeros(self.num_runs)
        if active_runs is None:
            active_runs = np.ones(self.num_runs, dtype=bool)
        else:
            active_runs = np.asarray(active_runs, dtype=bool)
        total_rate[~active_runs] = 0

        active_runs = total_rate > 0
        dt[active_runs] = self.rng.exponential(1 / total_rate[active_runs])
        roll = self.rng.random(self.num_runs) * total_rate

        for k in range(self.num_runs):
            if total_rate[k] == 0:
                continue

            threshold = hop_rate[k]
            if roll[k] < threshold:
                i,j = self.pick_move(moves["hop_mask"][:, :, k])
                state[i,j,k] = 0
                state[i,j+1,k] = 1
                continue

            threshold += attach_rate[k]
            if roll[k] < threshold:
                i,j = self.pick_weighted_move(moves["attach_weights"][:, :, k])
                state[i,j,k] = 1
                state[0,j,k] -= 1
                continue

            threshold += detach_rate[k]
            if roll[k] < threshold:
                i,j = self.pick_move(moves["detach_mask"][:, :, k])
                state[i,j,k] = 0
                state[0,j,k] += 1
                continue

            threshold += track_entry_rate[k]
            if roll[k] < threshold:
                i,j = self.pick_move(moves["enter_mask"][:, :, k])
                state[i,j,k] = 1
                continue

            threshold += reservoir_entry_rate[k]
            if roll[k] < threshold:
                state[0,0,k] += 1
                continue

            threshold += track_exit_rate[k]
            if roll[k] < threshold:
                i,j = self.pick_move(moves["exit_mask"][:, :, k])
                state[i,j,k] = 0
        t += dt
        return state, t

    def pick_move(self,mask):
        # pick one True entry without building a Python list of (i,j) tuples
        flat_moves = np.flatnonzero(mask)
        flat_choice = self.rng.choice(flat_moves)
        i,j = np.unravel_index(flat_choice, mask.shape)
        return i + 1,j

    def pick_weighted_move(self,weights):
        # attachment has one possible event per off-track particle, so empty sites
        # are sampled with probability proportional to their off-track count
        flat_weights = weights.ravel()
        cumulative_weights = np.cumsum(flat_weights)
        flat_choice = np.searchsorted(cumulative_weights, self.rng.random() * cumulative_weights[-1])
        i,j = np.unravel_index(flat_choice, weights.shape)
        return i + 1,j

    # gets the possible moves (as masks and counts) -- technically unrelated to simulation params
    # state is shape i,j,k
    #   i - track number <= N (0 indicates off track)
    #   j - lattice point number < M
    #   k - simulation id < num_runs
    def get_moves(self, state):
        # compute attachment events
        # on_track has shape track,lattice point,simulation id with the off-track row removed
        on_track = state[1:, :, :]
        off_track = state[0, :, :]
        empty = on_track == 0
        occupied = on_track == 1

        # loop through all particles on the tracks
        # These masks replace the old nested loop.  A True entry means the event is allowed
        # at that site.  Counts are summed by simulation id, so iterate_state can choose
        # the event type before converting a mask into one sampled move.
        hop_mask = np.zeros_like(occupied, dtype=bool)
        hop_mask[:, :-1, :] = occupied[:, :-1, :] & empty[:, 1:, :]

        # check for attachements if empty
        # attach_weights stores how many off-track particles can attach to each empty site.
        # This keeps the correct rate without duplicating the same (i,j) move many times.
        attach_weights = empty * off_track[np.newaxis, :, :]
        enter_mask = np.zeros_like(empty, dtype=bool)
        enter_mask[:, 0, :] = empty[:, 0, :] # can enter if on the left

        # if there's a particle, add detachments, exit_events, and hops
        detach_mask = occupied # anything can detach
        exit_mask = np.zeros_like(occupied, dtype=bool)
        exit_mask[:, -1, :] = occupied[:, -1, :] # things on the right exit

        return {
            "hop_mask": hop_mask,
            "attach_weights": attach_weights,
            "detach_mask": detach_mask,
            "exit_mask": exit_mask,
            "enter_mask": enter_mask,
            "hop_count": np.count_nonzero(hop_mask, axis=(0, 1)),
            "attach_count": attach_weights.sum(axis=(0, 1)),
            "detach_count": np.count_nonzero(detach_mask, axis=(0, 1)),
            "exit_count": np.count_nonzero(exit_mask, axis=(0, 1)),
            "enter_count": np.count_nonzero(enter_mask, axis=(0, 1)),
        }

    ### HELPERS FOR OUTPUT STATISTICS ###

    def get_bucket_densities(self, state, bucket):
        # average nearby lattice points into buckets, then average over all simulations
        N_trim = (state.shape[1] // bucket) * bucket
        num_buckets = N_trim // bucket

        off_local_density = (
            state[0, :N_trim, :]
            .reshape(num_buckets, bucket, state.shape[2])
            .mean(axis=(1, 2))
        )

        track_local_density = (
            state[1:, :N_trim, :]
            .reshape(state.shape[0] - 1, num_buckets, bucket, state.shape[2])
            .mean(axis=(0, 2, 3))
        )

        return off_local_density, track_local_density

    def get_bucket_densities_by_run(self, state, bucket):
        # average nearby lattice points into buckets, but keep each simulation separate
        N_trim = (state.shape[1] // bucket) * bucket
        num_buckets = N_trim // bucket

        off_local_density = (
            state[0, :N_trim, :]
            .reshape(num_buckets, bucket, state.shape[2])
            .mean(axis=1)
        )

        track_local_density = (
            state[1:, :N_trim, :]
            .reshape(state.shape[0] - 1, num_buckets, bucket, state.shape[2])
            .mean(axis=(0, 2))
        )

        return off_local_density, track_local_density

    ### FUNCTIONS FOR PERFORMING THE ENTIRE SIMULATION ###

    # solves the stochastic system, assuming it goes to steady-state
    def solve_stochastic(self):
        
        x = self.initialize_state(self.initial_state)
        t = np.zeros(self.num_runs)

        # The final state is one noisy snapshot.  For a steady-state profile, keep a
        # running average of the bucket densities instead of storing every timestep.
        # Increase burn_in_steps to ignore early transient states before averaging.
        # num_time_buckets splits the averaged time window into separate plots; for
        # example, 2 compares the first half of the window against the second half.
        off_shape, track_shape = self.get_bucket_densities(x, self.bucket)
        off_local_density = np.zeros_like(off_shape, dtype=float)
        track_local_density = np.zeros_like(track_shape, dtype=float)
        off_time_density = np.zeros((self.num_time_buckets, *off_shape.shape), dtype=float)
        track_time_density = np.zeros((self.num_time_buckets, *track_shape.shape), dtype=float)
        time_bucket_counts = np.zeros(self.num_time_buckets, dtype=int)
        num_averaged_steps = 0
        averaging_steps = self.num_steps - self.burn_in_steps
        if averaging_steps <= 0:
            raise ValueError("burn_in_steps must be less than num_steps")
        for i in range(self.num_steps):
            x,t = self.iterate_state(x,t)
            if i >= self.burn_in_steps:
                off_step_density, track_step_density = self.get_bucket_densities(x, self.bucket)
                num_averaged_steps += 1
                off_local_density += (off_step_density - off_local_density) / num_averaged_steps
                track_local_density += (track_step_density - track_local_density) / num_averaged_steps

                time_bucket = min(
                    self.num_time_buckets - 1,
                    ((i - self.burn_in_steps) * self.num_time_buckets) // averaging_steps)
                
                time_bucket_counts[time_bucket] += 1
                off_time_density[time_bucket] += (
                    off_step_density - off_time_density[time_bucket]
                ) / time_bucket_counts[time_bucket]
                track_time_density[time_bucket] += (
                    track_step_density - track_time_density[time_bucket]
                ) / time_bucket_counts[time_bucket]

        num_buckets = len(track_local_density)
        xs = (np.arange(num_buckets) + 0.5) * self.bucket * self.delta

        if self.return_time_buckets:
            return xs, track_local_density, off_local_density, track_time_density, off_time_density, time_bucket_counts
        return xs, track_local_density, off_local_density

    # Solves the stochastic system - keeping track of actual time values
    def solve_stochastic_by_time(self):
        # x stores all runs at once; t stores the current physical/Gillespie
        # time for each run separately, since runs do not take events in sync.
        x = self.initialize_state(self.initial_state)
        t = np.zeros(self.num_runs)
        simulation_time_value = float(self.simulation_time_value)
        if simulation_time_value <= 0:
            raise ValueError("simulation_time_value must be positive")

        # These arrays accumulate density*time. They are divided by the total
        # time duration later, giving a real time average instead of an
        # event-count average.
        off_shape, track_shape = self.get_bucket_densities(x, self.bucket)
        off_time_accum = np.zeros((self.num_time_buckets, *off_shape.shape), dtype=float)
        track_time_accum = np.zeros((self.num_time_buckets, *track_shape.shape), dtype=float)

        # Tracks how much total run-time has contributed to each time bucket.
        # With num_runs runs, each full bucket should receive roughly
        # num_runs * bucket_width total duration.
        time_bucket_durations = np.zeros(self.num_time_buckets, dtype=float)
        time_edges = np.linspace(0, simulation_time_value, self.num_time_buckets + 1)

        iterations = 0
        while np.any(t < simulation_time_value):
            if self.max_iterations_value is not None and iterations >= self.max_iterations_value:
                raise RuntimeError("Maximum iterations reached before simulation_time_value")

            old_t = t.copy()
            # Keep densities per run here. Each run may contribute a different
            # duration to the current time bucket, so averaging over runs too
            # early would give the wrong real-time weighting.
            off_by_run, track_by_run = self.get_bucket_densities_by_run(x, self.bucket)
            active_runs = old_t < simulation_time_value

            # Advance every active run by one stochastic event. Completed runs
            # are masked off so they no longer change.
            x,t = self.iterate_state(x,t,active_runs=active_runs)

            # If a run has no possible moves, its current state persists to the final time.
            stopped_runs = active_runs & (t == old_t)
            t[stopped_runs] = simulation_time_value

            # Clamp event end times to the requested final time. If an event
            # would jump past the end, only the part before the final time
            # contributes to the averages.
            end_t = np.minimum(t, simulation_time_value)
            for time_bucket in range(self.num_time_buckets):
                # Compute how much of each run's interval [old_t, end_t]
                # overlaps this bucket's real-time interval.
                overlap_start = np.maximum(old_t, time_edges[time_bucket])
                overlap_end = np.minimum(end_t, time_edges[time_bucket + 1])
                durations = np.maximum(0, overlap_end - overlap_start)
                total_duration = durations.sum()
                if total_duration == 0:
                    continue

                # Matrix-vector product sums over runs:
                #   density_by_bucket_by_run @ duration_by_run
                # leaving one accumulated density*time value per spatial bucket.
                off_time_accum[time_bucket] += off_by_run @ durations
                track_time_accum[time_bucket] += track_by_run @ durations
                time_bucket_durations[time_bucket] += total_duration

            iterations += 1

        # Convert accumulated density*time into average density in each time bucket.
        off_time_density = np.zeros_like(off_time_accum)
        track_time_density = np.zeros_like(track_time_accum)
        active_buckets = time_bucket_durations > 0
        off_time_density[active_buckets] = (
            off_time_accum[active_buckets] / time_bucket_durations[active_buckets, np.newaxis]
        )
        track_time_density[active_buckets] = (
            track_time_accum[active_buckets] / time_bucket_durations[active_buckets, np.newaxis]
        )

        # Also return one average over the full requested time interval.
        total_duration = time_bucket_durations.sum()
        if total_duration > 0:
            off_local_density = off_time_accum.sum(axis=0) / total_duration
            track_local_density = track_time_accum.sum(axis=0) / total_duration
        else:
            off_local_density = np.zeros_like(off_shape, dtype=float)
            track_local_density = np.zeros_like(track_shape, dtype=float)

        num_buckets = len(track_local_density)
        xs = (np.arange(num_buckets) + 0.5) * self.bucket * self.delta

        if self.return_time_buckets:
            return xs, track_local_density, off_local_density, track_time_density, off_time_density, time_bucket_durations
        return xs, track_local_density, off_local_density

    # each trace is not averaged in time, just a snapshot of the state at that time
    def solve_stochastic_by_snapshots(self):

        # x stores all runs at once; t stores the current physical/Gillespie
        # time for each run separately, since runs do not take events in sync.
        x = self.initialize_state(self.initial_state)
        t = np.zeros(self.num_runs)
        simulation_time_value = float(self.simulation_time_value)
        if simulation_time_value <= 0:
            raise ValueError("simulation_time_value must be positive")

        # snapshot_times are the marked times where we want the stochastic
        # profile.  With simulation_time_value=10 and num_time_buckets=10,
        # this gives 0, 1, 2, ..., 10.
        snapshot_times = np.linspace(0, simulation_time_value, self.num_time_buckets + 1)

        # These arrays accumulate the density from each run at each marked time.
        # They are divided by snapshot_counts later, giving an ensemble average
        # at each time without averaging across a time interval.
        off_shape, track_shape = self.get_bucket_densities(x, self.bucket)
        off_snapshot_accum = np.zeros((snapshot_times.size, *off_shape.shape), dtype=float)
        track_snapshot_accum = np.zeros((snapshot_times.size, *track_shape.shape), dtype=float)
        snapshot_counts = np.zeros(snapshot_times.size, dtype=int)

        # Every run starts exactly at t=0, so record the initial state first.
        off_by_run, track_by_run = self.get_bucket_densities_by_run(x, self.bucket)
        off_snapshot_accum[0] = off_by_run.sum(axis=1)
        track_snapshot_accum[0] = track_by_run.sum(axis=1)
        snapshot_counts[0] = self.num_runs

        iterations = 0
        while np.any(snapshot_counts < self.num_runs):
            if self.max_iterations_value is not None and iterations >= self.max_iterations_value:
                raise RuntimeError("Maximum iterations reached before all snapshots were recorded")

            old_t = t.copy()
            active_runs = old_t < simulation_time_value

            # Store the pre-event densities.  The state is constant between
            # events, so a time marker crossed during this event interval uses
            # this pre-event state.
            off_by_run, track_by_run = self.get_bucket_densities_by_run(x, self.bucket)

            # Advance every active run by one stochastic event. Completed runs
            # are masked off so they no longer change.
            x,t = self.iterate_state(x,t,active_runs=active_runs)

            # If a run has no possible moves, its current state persists to the final time.
            stopped_runs = active_runs & (t == old_t)
            t[stopped_runs] = simulation_time_value

            # Clamp to the requested final time so a long final event does not
            # create snapshots beyond the requested time window.
            end_t = np.minimum(t, simulation_time_value)

            for snapshot_index in range(1, snapshot_times.size):
                if snapshot_counts[snapshot_index] == self.num_runs:
                    continue

                marked_time = snapshot_times[snapshot_index]
                crossed_mark = (old_t < marked_time) & (marked_time <= end_t)
                if not np.any(crossed_mark):
                    continue

                # Add only the runs that crossed this particular time marker.
                # Each run contributes exactly once to each snapshot.
                off_snapshot_accum[snapshot_index] += off_by_run[:, crossed_mark].sum(axis=1)
                track_snapshot_accum[snapshot_index] += track_by_run[:, crossed_mark].sum(axis=1)
                snapshot_counts[snapshot_index] += np.count_nonzero(crossed_mark)

            iterations += 1

        off_snapshot_density = off_snapshot_accum / snapshot_counts[:, np.newaxis]
        track_snapshot_density = track_snapshot_accum / snapshot_counts[:, np.newaxis]

        off_local_density = off_snapshot_density[-1]
        track_local_density = track_snapshot_density[-1]

        num_buckets = len(track_local_density)
        xs = (np.arange(num_buckets) + 0.5) * self.bucket * self.delta

        if self.return_time_buckets:
            return xs, track_local_density, off_local_density, track_snapshot_density, off_snapshot_density, snapshot_times
        return xs, track_local_density, off_local_density


def get_simulation_config(simulation):
    # Process workers receive a plain config dictionary instead of sharing the
    # simulation object. This avoids accidentally reusing the same RNG state in
    # every process.
    return {
        "alpha": simulation.alpha,
        "beta": simulation.beta,
        "omega_A": simulation.omega_A,
        "omega_D": simulation.omega_D,
        "N": simulation.N,
        "L": simulation.L,
        "M": simulation.M,
        "delta": simulation.delta,
        "num_runs": simulation.num_runs,
        "bucket": simulation.bucket,
        "num_steps": simulation.num_steps,
        "burn_in_steps": simulation.burn_in_steps,
        "num_time_buckets": simulation.num_time_buckets,
        "simulation_time_value": simulation.simulation_time_value,
        "max_iterations_value": simulation.max_iterations_value,
        "left_track_entry": simulation.left_track_entry,
        "left_reservoir_entry": simulation.left_reservoir_entry,
        "initial_state": simulation.initial_state,
        "return_time_buckets": True,
    }


def simulation_wrapper(config):
    simulation = StochasticSim()
    seed = config.pop("seed")
    for name, value in config.items():
        setattr(simulation, name, value)
    simulation.seed = seed
    simulation.rng = np.random.default_rng(seed)
    return simulation.solve_stochastic_by_snapshots()


def average_snapshot_results(results):
    # All workers use the same grid and snapshot times, so only the density
    # arrays need to be averaged.
    xs, _, _, _, _, snapshot_times = results[0]
    track_snapshots = np.mean([result[3] for result in results], axis=0)
    off_snapshots = np.mean([result[4] for result in results], axis=0)
    return (
        xs,
        track_snapshots[-1],
        off_snapshots[-1],
        track_snapshots,
        off_snapshots,
        snapshot_times,
    )


def solve_snapshots_parallel(simulation, num_processes):
    if num_processes <= 1:
        old_return_time_buckets = simulation.return_time_buckets
        simulation.return_time_buckets = True
        result = simulation.solve_stochastic_by_snapshots()
        simulation.return_time_buckets = old_return_time_buckets
        return result

    seed_sequence = np.random.SeedSequence(simulation.seed)
    worker_seeds = seed_sequence.generate_state(num_processes)
    base_config = get_simulation_config(simulation)
    worker_configs = []
    for seed in worker_seeds:
        worker_config = base_config.copy()
        worker_config["seed"] = int(seed)
        worker_configs.append(worker_config)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(simulation_wrapper, worker_configs))
    return average_snapshot_results(results)


def stoch_buildup_density_profile():
    simulation = StochasticSim()
    simulation.alpha = 0.8
    simulation.beta = 0.2
    simulation.N = 4
    simulation.left_track_entry = False
    simulation.left_reservoir_entry = True
    simulation.return_time_buckets = True
    simulation.initial_state = "even"
    simulation.simulation_time_value = 1000.0
    simulation.num_time_buckets = 7
    simulation.num_runs = 16
    xs, on_track, off_track, track_snapshot_density, off_snapshot_density, snapshot_times = simulation.solve_stochastic_by_snapshots()
    print(off_snapshot_density.shape)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, snapshot_time in enumerate(snapshot_times):
        axes[0].plot(xs, track_snapshot_density[i])
        axes[1].plot(xs, off_snapshot_density[i])
    #plt.plot(xs,on_track)
    for ax in axes:
        ax.set_xlim(0, simulation.L)
        #ax.set_ylim(bottom=0)
        ax.set_ylim(0,1)
    plt.show()

def buildup_heatmap():
    simulation = StochasticSim()
    simulation.alpha = 0.8
    simulation.beta = 0.2
    simulation.N = 4
    simulation.left_track_entry = False
    simulation.left_reservoir_entry = True
    simulation.return_time_buckets = True
    simulation.initial_state = "even"
    simulation.simulation_time_value = 1000.0
    simulation.num_time_buckets = 32
    simulation.num_runs = 16
    simulation.seed = None
    num_processes = 8
    xs, on_track, off_track, track_snapshot_density, off_snapshot_density, snapshot_times = solve_snapshots_parallel(
        simulation,
        num_processes,
    )

    off_snapshot_density = np.log(np.maximum(off_snapshot_density,0.01))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    heatmap_extent = [xs[0], xs[-1], snapshot_times[0], snapshot_times[-1]]

    track_plot = axes[0].imshow(
        track_snapshot_density,
        aspect="auto",
        origin="lower",
        extent=heatmap_extent,
        vmin=0,
        vmax=1,
    )
    off_plot = axes[1].imshow(
        off_snapshot_density,
        aspect="auto",
        origin="lower",
        extent=heatmap_extent,
    )

    fig.colorbar(track_plot, ax=axes[0])
    fig.colorbar(off_plot, ax=axes[1])
    fig.tight_layout()

    plt.savefig('buildup_heatmap.svg', format='svg', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    buildup_heatmap()
