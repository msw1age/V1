import numpy as np
import numba as nb

# Set plotting_mode to False for pure performance testing
plotting_mode = False

### Constants
# These values are typically not used, but are listed for reference or if needed
V_E = 14/3
V_I = -2/3
num_E = 300
num_PV = 66
num_SOM = 34
pv_prop = 2/3
som_prop = 1/3
S_EE = 0.0255
S_EPV = 0.052
S_ESOM = 0.026
S_PVE = 0.01
S_PVPV = 0.03
S_PVSOM = 0.01
S_SOME = 0.004
S_SOMPV = 0.04
S_SOMSOM = 0.001
S_amb = 0.003
tau_E = 2/1000
tau_I = 4/1000
h = 1/10000
T = 1
c = 10
lam_E = 0.288*c
lam_PV = 0.972*c
lam_amb = 0.433 * 1000
lam_lat = 0.5
amb_update = S_amb / tau_E
df_EE = 0.8 # E --> E
df_EPV = 0.5 # PV --> E
df_ESOM = -0.3 # SOM --> E
df_PVE = 0.8 # E --> PV
df_PVPV = 0.5 # PV --> PV
df_PVSOM = -0.3 # SOM --> PV
df_SOME = -0.3 # E --> SOM
df_SOMPV = 0.5 # PV --> SOM
df_SOMSOM = 0.3 # SOM --> SOM
td_EE = 0.03
td_EPV = 0.007
td_ESOM = 0.001
td_PVE = 0.03
td_PVPV = 0.007
td_PVSOM = 0.001
td_SOME = 0.001
td_SOMPV = 0.007
td_SOMSOM = 0.007
N = num_E + num_PV + num_SOM
tau_leak = np.random.normal(loc=20, scale=2, size=N) / 1000.0
max_delay_steps = 30

@nb.njit(cache=True)
def create_external_stimuli_fast(lams, N, T, h):
    """
    Numba-compiled function to randomly generate Poisson spike data.
    lams: vector of rate constants of shape (N)
    N: Number of neurons
    T: Length of simulation
    h: Length of one time step
    Returns N x (T/h) matrix of spike times
    """
    num_steps = int(T / h)
    spike_probs = lams * h
    rand_vals = np.random.rand(N, num_steps)
    return (rand_vals < spike_probs.reshape(-1, 1)).astype(np.int8)

def create_neurons_fast(params):
    """
    Optimized function to create neurons and their connectivity.
    Takes in all parameters of the simulation.
    Returns dictionary object representing the set of neurons.
    """
    S_EE, S_EPV, S_ESOM, S_PVE, S_PVPV, S_PVSOM, S_SOME, S_SOMPV, S_SOMSOM, \
    lam_E, lam_PV, lam_lat, \
    df_EE, df_EPV, df_ESOM, df_PVE, df_PVPV, df_PVSOM, df_SOME, df_SOMPV, df_SOMSOM, \
    td_EE, td_EPV, td_ESOM, td_PVE, td_PVPV, td_PVSOM, td_SOME, td_SOMPV, td_SOMSOM = params

    def create_split(a,b,c): return np.concatenate((a*np.ones(num_E), b*np.ones(num_PV), c*np.ones(num_SOM)))
    def create_num_presynaptics(mu, sd, size): return np.clip(np.random.normal(mu, sd, size), mu-sd, mu+sd).astype(np.int64)

    # Generates the number of presynaptic neurons to each postsynaptic neuron, differentiated by neuron type
    e_to_e_presynaptic_counts = create_num_presynaptics(mu = 80, sd = 15, size = num_E)
    pv_to_e_presynaptic_counts = create_num_presynaptics(mu = 50*(pv_prop), sd = 7.5*(pv_prop), size = num_E)
    som_to_e_presynaptic_counts = create_num_presynaptics(mu = 50*(som_prop), sd = 7.5*(som_prop), size = num_E)
    e_to_pv_presynaptic_counts = create_num_presynaptics(mu = 240, sd = 37.5, size = num_PV)
    pv_to_pv_presynaptic_counts = create_num_presynaptics(mu = 50*(pv_prop), sd = 7.5*(pv_prop), size = num_PV)
    som_to_pv_presynaptic_counts = create_num_presynaptics(mu = 50*(som_prop), sd = 7.5*(som_prop), size = num_PV)
    e_to_som_presynaptic_counts = create_num_presynaptics(mu = 240*(som_prop), sd = 37.5*(som_prop), size = num_SOM)
    pv_to_som_presynaptic_counts = create_num_presynaptics(mu = 50*(pv_prop), sd = 7.5*(pv_prop), size = num_SOM)
    som_to_som_presynaptic_counts = create_num_presynaptics(mu = 50*(som_prop), sd = 7.5*(som_prop), size = num_SOM)

    postsynaptics_list = [[] for _ in range(N)]
    postsynaptic_delays_list = [[] for _ in range(N)]
    postsynaptic_dfs_list = [[] for _ in range(N)]
    postsynaptic_tds_list = [[] for _ in range(N)]

    e_indices = np.arange(num_E, dtype=np.int64)
    pv_indices = np.arange(num_E, num_E + num_PV, dtype=np.int64)
    som_indices = np.arange(num_E + num_PV, N, dtype=np.int64)

    # Loop assigns presynaptic neurons to their postsynaptic targets along with the synaptic delay and the depression/facilitation factor
    for i in range(N):
        # i represents index of postsynaptic neuron
        if i < num_E:
            e_count, pv_count, som_count = e_to_e_presynaptic_counts[i], pv_to_e_presynaptic_counts[i], som_to_e_presynaptic_counts[i]
            e_delays = np.random.uniform(1, 2.3, e_count) / 1000.0
            depression_factors = np.concatenate((df_EE*np.ones(e_count), df_EPV*np.ones(pv_count), df_ESOM*np.ones(som_count)))
            tau_depressions = np.concatenate((td_EE*np.ones(e_count), td_EPV*np.ones(pv_count), td_ESOM*np.ones(som_count)))
        elif i < num_E + num_PV:
            e_count, pv_count, som_count = e_to_pv_presynaptic_counts[i - num_E], pv_to_pv_presynaptic_counts[i - num_E], som_to_pv_presynaptic_counts[i - num_E]
            e_delays = np.ones(e_count) / 1000.0
            depression_factors = np.concatenate((df_PVE*np.ones(e_count), df_PVPV*np.ones(pv_count), df_PVSOM*np.ones(som_count)))
            tau_depressions = np.concatenate((td_PVE*np.ones(e_count), td_PVPV*np.ones(pv_count), td_PVSOM*np.ones(som_count)))
        else:
            e_count, pv_count, som_count = e_to_som_presynaptic_counts[i - num_E - num_PV], pv_to_som_presynaptic_counts[i - num_E - num_PV], som_to_som_presynaptic_counts[i - num_E - num_PV]
            e_delays = np.ones(e_count) / 1000.0
            depression_factors = np.concatenate((df_SOME*np.ones(e_count), df_SOMPV*np.ones(pv_count), df_SOMSOM*np.ones(som_count)))
            tau_depressions = np.concatenate((td_PVE*np.ones(e_count), td_SOMPV*np.ones(pv_count), td_SOMSOM*np.ones(som_count)))
        
        pv_delays, som_delays = np.random.uniform(0.8, 1.5, pv_count) / 1000.0, np.random.uniform(0.8, 1.5, som_count) / 1000.0
        e_presynaptic, pv_presynaptic, som_presynaptic = np.random.choice(np.setdiff1d(e_indices, i), e_count, replace=False), np.random.choice(np.setdiff1d(pv_indices, i), pv_count, replace=False), np.random.choice(np.setdiff1d(som_indices, i), som_count, replace=False)
        presynaptic_neurons, delays = np.concatenate((e_presynaptic, pv_presynaptic, som_presynaptic)), np.concatenate((e_delays, pv_delays, som_delays))

        for idx, presynaptic_neuron in enumerate(presynaptic_neurons):
            postsynaptics_list[presynaptic_neuron].append(i)
            postsynaptic_delays_list[presynaptic_neuron].append(delays[idx])
            postsynaptic_dfs_list[presynaptic_neuron].append(depression_factors[idx])
            postsynaptic_tds_list[presynaptic_neuron].append(tau_depressions[idx])

    neurons = {
        'potential': np.random.rand(N), 
        'gE': np.concatenate((np.random.uniform(0, 90, num_E), np.random.uniform(0, 60, num_PV + num_SOM))),
        'gI': np.concatenate((np.random.uniform(0, 330, num_E), np.random.uniform(0, 400, num_PV + num_SOM))),
        'refractory_end_time': np.random.uniform(0, 0.002, N),#np.zeros(N), 
        'type': create_split(1, 0, -1),
        'E_Update': create_split(S_EE, S_PVE, S_SOME) / tau_E, 
        'PV_Update': create_split(S_EPV, S_PVPV, S_SOMPV) / tau_I,
        'SOM_Update': create_split(S_ESOM, S_PVSOM, S_SOMSOM) / tau_I,
        'spike_times': [[] for _ in range(N)],#np.zeros((N, int(T/h))),
        'synapse_values': np.ones((N, N)), 
        'synapse_last_update_times': np.zeros((N, N))
    }
    
    lams_ext = create_split(lam_E * 1000, lam_PV * 1000, 0)
    lams_lat = create_split(0, lam_lat * 1000, lam_lat * 1000)
    neurons['feedforward_spike_times'] = create_external_stimuli_fast(lams_ext, N, T, h)
    neurons['ambient_spike_times'] = create_external_stimuli_fast(np.full(N, lam_amb), N, T, h)
    neurons['lateral_spike_times'] = create_external_stimuli_fast(np.full(N, lams_lat), N, T, h)
    neurons['postsynaptics'] = np.concatenate(postsynaptics_list).astype(np.int64)
    neurons['postsynaptic_delays'] = np.concatenate(postsynaptic_delays_list)
    neurons['postsynaptic_dfs'] = np.concatenate(postsynaptic_dfs_list)
    neurons['tau_depression'] = np.concatenate(postsynaptic_tds_list)
    neurons['postsynaptic_ids'] = np.cumsum(np.array([0] + [len(arr) for arr in postsynaptics_list])).astype(np.int64)
    return neurons

@nb.njit(cache=True)
def _calculate_cv_from_bins(binned_counts):
    """Calculates CV from a numpy array of binned spike counts."""
    mean_counts = np.mean(binned_counts)
    if mean_counts < 1e-9: # Avoid division by zero if the population is silent
        return 0.0
    
    std_dev_counts = np.std(binned_counts)
    return std_dev_counts / mean_counts

@nb.njit(cache=True)
def _run_simulation_numba(
    potential, gE, gI, refractory_end_time, neuron_type, E_Update, PV_Update, SOM_Update,
    tau_depression, synapse_values, synapse_last_update_times,
    postsynaptics, postsynaptic_ids, 
    postsynaptic_delays, postsynaptic_dfs,
    feedforward_spike_times, ambient_spike_times, lateral_spike_times, T, h, tau_leak, V_E, V_I, 
    tau_E, tau_I, amb_update
):
    """
    Numba-compiled function to run the simulation given the neurons. Neuron parameters have to be
    passed in individually due to numba not accepting dictionaries.
    Returns the number of spikes across neuron types, how depressed/facilitated the synapses are,
    the last time each synapse updated, and the timings of all the spikes (if in plotting mode).
    """
    total_E_spikes, total_PV_spikes, total_SOM_spikes = 0, 0, 0
    if plotting_mode: all_spikes = []
    else: all_spikes = 0

    # --- NEW: Setup for Online CV Calculation ---
    # Calculate binning parameters
    bin_size_ms = 5.0
    bin_size_steps = int(round((bin_size_ms / 1000.0) / h))
    if bin_size_steps == 0: bin_size_steps = 1 # Ensure bin size is at least one time step
    num_bins = int(round(T / h)) // bin_size_steps

    # Create arrays to store the binned spike counts for each population
    e_binned_counts = np.zeros(num_bins, dtype=np.float64)
    pv_binned_counts = np.zeros(num_bins, dtype=np.float64)
    som_binned_counts = np.zeros(num_bins, dtype=np.float64)
    # --- End of New Setup ---

    # We maintain a buffer for gE and gI for fast, numba-compatible integration of delayed spikes
    gE_buffer, gI_buffer = np.zeros((N, max_delay_steps)), np.zeros((N, max_delay_steps))
    t = 0.0
    while t < T:
        t_int = int(np.round(t / h))
        buffer_idx = t_int % max_delay_steps
        
        gE += gE_buffer[:, buffer_idx]; gI += gI_buffer[:, buffer_idx]
        gE_buffer[:, buffer_idx] = 0.0; gI_buffer[:, buffer_idx] = 0.0

        # Update based on external spiking
        gE[feedforward_spike_times[:, t_int] == 1] += E_Update[feedforward_spike_times[:, t_int] == 1]
        gE[ambient_spike_times[:, t_int] == 1] += amb_update
        gE[lateral_spike_times[:, t_int] == 1] += E_Update[lateral_spike_times[:, t_int] == 1]
        # Exponential decay
        gE -= h * gE / tau_E; gI -= h * gI / tau_I

        # Updated potential of neurons not in refractory period
        not_in_refractory = (t >= refractory_end_time)
        potential[not_in_refractory] += h * (
            -potential[not_in_refractory]/tau_leak[not_in_refractory] 
            - (potential[not_in_refractory] - V_E)*gE[not_in_refractory] 
            - (potential[not_in_refractory] - V_I)*gI[not_in_refractory]
        )

        spiking_indices = np.where((potential >= 1.0) & not_in_refractory)[0]
        if spiking_indices.size > 0:
            # --- NEW: Populate Bins with Spike Counts ---
            current_bin = t_int // bin_size_steps
            if current_bin < num_bins:
                # Efficiently count spikes per type using the neuron_type array
                types_of_spikers = neuron_type[spiking_indices]
                e_binned_counts[current_bin] += np.sum(types_of_spikers == 1)
                pv_binned_counts[current_bin] += np.sum(types_of_spikers == 0)
                som_binned_counts[current_bin] += np.sum(types_of_spikers == -1)

            for spiking_neuron in spiking_indices:
                neuron_class = neuron_type[spiking_neuron]
                if t > 0.2:
                    if neuron_class == 1: total_E_spikes += 1
                    elif neuron_class == 0: total_PV_spikes += 1
                    else: total_SOM_spikes += 1

                potential[spiking_neuron] = 0.0
                refractory_end_time[spiking_neuron] = t + 2.0/1000

                # Postsynaptic IDs downstream of spiking neuron lie between these two indices
                post_start, post_end = postsynaptic_ids[spiking_neuron], postsynaptic_ids[spiking_neuron + 1]
                
                for i in range(post_start, post_end):
                    downstream_neuron = postsynaptics[i]
                    
                    last_val = synapse_values[spiking_neuron, downstream_neuron]
                    last_t = synapse_last_update_times[spiking_neuron, downstream_neuron]
                    
                    time_elapsed = t - last_t
                    tau = tau_depression[i]
                    transmitter_val = 1.0 + (last_val - 1.0) * np.exp(-time_elapsed / tau)

                    df = postsynaptic_dfs[i]
                    new_val = transmitter_val * (1 - df)
                    new_val = min(new_val, 10.0)
                    
                    synapse_values[spiking_neuron, downstream_neuron] = new_val
                    synapse_last_update_times[spiking_neuron, downstream_neuron] = t

                    delay = postsynaptic_delays[i]
                    delay_steps = int(round(delay / h))
                    if delay_steps < max_delay_steps:
                        target_idx = (buffer_idx + delay_steps) % max_delay_steps
                        if neuron_type[spiking_neuron] == 1:
                            gE_buffer[downstream_neuron, target_idx] += E_Update[downstream_neuron] * transmitter_val
                        elif neuron_type[spiking_neuron] == 0:
                            gI_buffer[downstream_neuron, target_idx] += PV_Update[downstream_neuron] * transmitter_val
                        else:
                            gI_buffer[downstream_neuron, target_idx] += SOM_Update[downstream_neuron] * transmitter_val

        if plotting_mode and len(spiking_indices) > 0: all_spikes.append((spiking_indices, t_int))
        t += h
    # 40 onwards ignores first 200ms
    cv_E = _calculate_cv_from_bins(e_binned_counts[40:])
    cv_PV = _calculate_cv_from_bins(pv_binned_counts[40:])
    cv_SOM = _calculate_cv_from_bins(som_binned_counts[40:])

    return total_E_spikes, total_PV_spikes, total_SOM_spikes, synapse_values, synapse_last_update_times, all_spikes, cv_E, cv_PV, cv_SOM

def run_simulation(params = np.array([S_EE, S_EPV, S_ESOM, S_PVE, S_PVPV, S_PVSOM, S_SOME, S_SOMPV, S_SOMSOM, 
    lam_E, lam_PV, lam_lat, df_EE, df_EPV, df_ESOM, df_PVE, df_PVPV, df_PVSOM, df_SOME, df_SOMPV, df_SOMSOM,
    td_EE, td_EPV, td_ESOM, td_PVE, td_PVPV, td_PVSOM, td_SOME, td_SOMPV, td_SOMSOM])):
    neurons = create_neurons_fast(params)

    total_E_spikes, total_PV_spikes, total_SOM_spikes, final_vals, final_times, all_spikes, cv_E, cv_PV, cv_SOM = _run_simulation_numba(
        neurons['potential'], neurons['gE'], neurons['gI'], neurons['refractory_end_time'],
        neurons['type'], neurons['E_Update'], neurons['PV_Update'], neurons['SOM_Update'],
        neurons['tau_depression'], neurons['synapse_values'], neurons['synapse_last_update_times'],
        neurons['postsynaptics'], neurons['postsynaptic_ids'],
        neurons['postsynaptic_delays'], neurons['postsynaptic_dfs'],
        neurons['feedforward_spike_times'], neurons['ambient_spike_times'], neurons['lateral_spike_times'],
        T, h, tau_leak, V_E, V_I, tau_E, tau_I, amb_update
    )

    E_rate, PV_rate, SOM_rate = total_E_spikes / (T * num_E), total_PV_spikes / (T * num_PV), total_SOM_spikes / (T * num_SOM)
    
    return E_rate, PV_rate, SOM_rate, cv_E, cv_PV, cv_SOM
    
def log_uniform(low, high):
    log_low = np.log(low)
    log_high = np.log(high)
    uniform_log_samples = np.random.uniform(log_low, log_high)
    return np.exp(uniform_log_samples)

def run_one_simulation(_):
    """
    S_EE = np.random.uniform(0.02, 0.026)
    S_EPV = np.random.uniform(0.045, 0.055)
    S_ESOM = np.random.uniform(0.02, 0.04)
    S_PVE = np.random.uniform(0.0085, 0.02)
    S_PVPV = np.random.uniform(0.02, 0.03)
    S_PVSOM = np.random.uniform(0.01, 0.02)
    S_SOME = np.random.uniform(0.005, 0.02)
    S_SOMPV = np.random.uniform(0.01, 0.05)
    S_SOMSOM = np.random.uniform(0.0005, 0.004)
    lam_E = log_uniform(0.25, 2)
    lam_PV = log_uniform(0.9, 7)
    lam_lat = np.random.uniform(0, 1)
    df_EE = np.random.uniform(0.3, 0.7)
    df_EPV = np.random.uniform(0.3, 0.7)
    df_ESOM = np.random.uniform(-0.6, -0.1)
    df_PVE = np.random.uniform(0.2, 0.9)
    df_PVPV = np.random.uniform(0.1, 0.7)
    df_PVSOM = np.random.uniform(-0.8, -0.2)
    df_SOME = np.random.uniform(-0.7, -0.3)
    df_SOMPV = np.random.uniform(0.3, 0.7)
    df_SOMSOM = np.random.uniform(0.3, 0.7)
    tau_dE = np.random.uniform(0.02, 0.1)
    tau_dPV = np.random.uniform(0.006, 0.015)
    tau_dSOM = np.random.uniform(0.006, 0.12)
    """
    S_EE = np.random.uniform(0.018, 0.024)
    # Weaken feedback inhibition slightly to soften the "veto"
    S_EPV = np.random.uniform(0.035, 0.055)  # PV -> E
    S_ESOM = np.random.uniform(0.01, 0.035)  # SOM -> E

    # CRITICAL: This is the sweet spot for E->I drive.
    # Stronger than your original set, but weaker than the last attempt.
    S_PVE = np.random.uniform(0.010, 0.028)   # E -> PV
    S_SOME = np.random.uniform(0.008, 0.028)  # E -> SOM

    # Keep other inhibitory connections in a reasonable range
    S_PVPV = np.random.uniform(0.022, 0.036)
    S_PVSOM = np.random.uniform(0.005, 0.015)
    S_SOMPV = np.random.uniform(0.015, 0.045)
    S_SOMSOM = np.random.uniform(0.0005, 0.003)

    # --- EXTERNAL DRIVES (lam) ---
    # Keep these wide to allow the optimizer to find the right drive level
    lam_E = log_uniform(0.2, 3.0)
    lam_PV = log_uniform(0.5, 9.0)
    lam_lat = np.random.uniform(0.1, 1.5)

    # --- SYNAPTIC DYNAMICS (df) ---
    # Your original ranges for these are generally fine
    df_EE = np.random.uniform(0.2, 0.8)
    df_EPV = np.random.uniform(0.2, 0.7)
    df_ESOM = np.random.uniform(-0.8, -0.1)
    df_PVE = np.random.uniform(0.3, 0.95)
    df_PVPV = np.random.uniform(0.2, 0.7)
    df_PVSOM = np.random.uniform(-0.8, -0.2)
    df_SOME = np.random.uniform(-0.8, 0.0) 
    df_SOMPV = np.random.uniform(0.2, 0.7)
    df_SOMSOM = np.random.uniform(0.2, 0.7)

    # --- SYNAPTIC TIME CONSTANTS (tau) ---
    # Keeping the longer time constants is still important for integration
    td_EE = np.random.uniform(0.02, 0.08)
    td_EPV = np.random.uniform(0.005, 0.015)
    td_ESOM = np.random.uniform(0.0005, 0.003)
    td_PVE
    tau_dSOM = np.random.uniform(0.025, 0.120)
    params = np.array([S_EE, S_EPV, S_ESOM, S_PVE, S_PVPV, S_PVSOM, S_SOME, S_SOMPV, S_SOMSOM, 
    lam_E, lam_PV, lam_lat, df_EE, df_EPV, df_ESOM, df_PVE, df_PVPV, df_PVSOM, df_SOME, df_SOMPV, df_SOMSOM,
    tau_dE, tau_dPV, tau_dSOM])
    E_rate, PV_rate, SOM_rate, cv_E, cv_PV, cv_SOM = run_simulation(params)
    return params, np.array([E_rate, PV_rate, SOM_rate, cv_E, cv_PV, cv_SOM])

def worker_function_for_optuna(params_dict):
    """
    This function implements a hierarchical objective:
    1. It heavily penalizes trials that are not asynchronous or have high SOM rates.
    2. Within the "good" trials, it seeks to maximize the E-rate.
    """
    param_order = [
        "S_EE", "S_EPV", "S_ESOM", "S_PVE", "S_PVPV", "S_PVSOM", 
        "S_SOME", "S_SOMPV", "S_SOMSOM", "lam_E", "lam_PV", "lam_lat", 
        "df_EE", "df_EPV", "df_ESOM", "df_PVE", "df_PVPV", "df_PVSOM", 
        "df_SOME", "df_SOMPV", "df_SOMSOM", 
        "td_EE", "td_EPV", "td_ESOM", "td_PVE", "td_PVPV", "td_PVSOM",
        "td_SOME", "td_SOMPV", "td_SOMSOM"
    ]
    params = np.array([params_dict[key] for key in param_order])
    
    outputs = run_simulation(params)
    rate_E, rate_PV, rate_SOM, cv_E, cv_PV, cv_SOM = outputs
    
    # Guardrail for dead networks
    if rate_E < 1.0 or rate_PV < 1.0:
        return 1000.0

    # Constraint A: Asynchrony.
    if cv_SOM == 0:
        target_cvs = np.array([1.0, 1.0, 0])
    else:
        target_cvs = np.array([0.8, 0.8, 1.5])
    actual_cvs = np.array([cv_E, cv_PV, cv_SOM])
    cv_error = np.sum((actual_cvs - target_cvs)**2)
    if cv_error > 1.5: 
        return 100.0 + cv_error 

    # Constraint B: SOM rate. Must be below the ceiling.
    som_rate_ceiling = 30.0
    if rate_SOM > som_rate_ceiling:
        som_rate_penalty = ((rate_SOM - som_rate_ceiling) / som_rate_ceiling)**2
        return 50.0 + som_rate_penalty

    # Constraint C: PV/E ratio. Must be around 3.
    PV_E_ratio = 3
    PV_E_ratio_error = np.abs(rate_PV / rate_E - 3)
    if PV_E_ratio_error > 2:
        return 25.0 + PV_E_ratio_error
        
    return -rate_E + cv_error