#%% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from tqdm import tqdm


#%% ========= DATA LOADING (this part stays the same) =========

def reading_csv(file_path):
    df = pd.read_csv(file_path)
    mean_used = df['S2_val'].tolist()
    result    = df['Decision (S1>S2)'].tolist()
    data_tuple = (result, mean_used)
    return data_tuple

def data_var2(file_path):
    df = pd.read_csv(file_path)
    var2 = df['S2_std'].to_numpy()
    var2 = pd.unique(var2)
    return list(var2)

def psycometrique(data_tuple, var2, block_size=1500, big_blocs=11):
    result, mean_used = data_tuple
    true_array_mean_used = []
    true_array_result    = []

    size_one_block = block_size * big_blocs

    for j in range(len(var2)):
        array_mean_result = []
        array_mean_used   = []
        for i in range(0, size_one_block, block_size):
            block_r = result[size_one_block*j + i : size_one_block*j + i + block_size]
            mean_r  = sum(block_r) / len(block_r)
            array_mean_result.append(mean_r)

            block_m = mean_used[size_one_block*j + i : size_one_block*j + i + block_size]
            mean_m  = sum(block_m) / len(block_m)
            array_mean_used.append(mean_m)

        true_array_result.append(array_mean_result)
        true_array_mean_used.append(array_mean_used)

    return true_array_result, true_array_mean_used

def build_targets_from_csv(file_path, block_size=1500, big_blocs=11):
    data_tuple = reading_csv(file_path)
    var2_list  = data_var2(file_path)
    array_mean_result, array_mean_used = psycometrique(
        data_tuple, var2_list,
        block_size=block_size,
        big_blocs=big_blocs
    )

    target = {}
    mus_test = array_mean_used[0]
    vars_test = var2_list

    for j, std2 in enumerate(var2_list):
        for k, mu2 in enumerate(array_mean_used[j]):
            p_emp = array_mean_result[j][k]
            target[(mu2, std2)] = p_emp

    return target, mus_test, vars_test


#%% ========= LOAD DATA =========

target, mus_test, vars_test = build_targets_from_csv('experiment_results_1500.csv')
print("Target dictionary loaded!")
print(f"mus_test = {mus_test}")
print(f"vars_test = {vars_test}")

mus_test  = [-10,-8,-6,-4,-2,0,2,4,6,8,10]
vars_test = [0,2,4,6,8]


#%% ========= CORRECTED MODEL =========
# 
# KEY CHANGE: The prior is on STIMULUS VALUE, not on the difference!
#
# OLD (WRONG): 
#   - dobs = x1 - x2
#   - posterior on the difference
#   - decide if posterior mean > 0
#
# NEW (CORRECT):
#   - observe x1, compute posterior for S1
#   - observe x2, compute posterior for S2  
#   - compare: if posterior_S1 > posterior_S2, respond 1
#

def posterior_mean_for_one_stimulus(x_obs, std_obs, mu0, std0):
    """
    Compute the posterior MEAN for ONE stimulus.
    
    The agent has prior N(mu0, std0^2) and observes x_obs with noise std_obs.
    
    Parameters:
    -----------
    x_obs : float or array
        The noisy observation(s) of the stimulus
    std_obs : float
        The standard deviation of the observation noise
    mu0 : float
        Prior mean (what the agent expects stimuli to be)
    std0 : float
        Prior standard deviation (how confident the agent is)
    
    Returns:
    --------
    mu_post : float or array
        The posterior mean estimate of the stimulus
    """
    # Handle the special case: perfect observation (no noise)
    if std_obs == 0:
        return x_obs   # Posterior = exactly what you observed
    
    var0 = std0**2
    var_obs = std_obs**2
    
    var_post = 1.0 / (1.0/var0 + 1.0/var_obs)
    mu_post = var_post * (mu0/var0 + x_obs/var_obs)
    
    return mu_post


def simulate_agent_correct(S1_val, S1_std, S2_val, S2_std, mu0, std0, n_trials=2000):
    """
    Simulate the CORRECT Bayesian agent for many trials.
    
    On each trial:
    1. Agent observes x1 ~ N(S1_val, S1_std^2)
    2. Agent observes x2 ~ N(S2_val, S2_std^2)
    3. Agent computes posterior mean for S1: mu_post1 = weighted_avg(mu0, x1)
    4. Agent computes posterior mean for S2: mu_post2 = weighted_avg(mu0, x2)
    5. Agent decides: if mu_post1 > mu_post2, respond 1, else 0
    
    Returns:
    --------
    P(respond 1) : float
        Proportion of trials where agent responded 1
    """
    # Generate noisy observations for all trials at once (vectorized)
    x1_observations = np.random.normal(S1_val, S1_std, size=n_trials)
    x2_observations = np.random.normal(S2_val, S2_std, size=n_trials)
    
    # Compute posterior means for each stimulus
    mu_post1 = posterior_mean_for_one_stimulus(x1_observations, S1_std, mu0, std0)
    mu_post2 = posterior_mean_for_one_stimulus(x2_observations, S2_std, mu0, std0)
    
    # Decision: respond 1 if posterior for S1 > posterior for S2
    responses = (mu_post1 > mu_post2).astype(int)
    
    return responses.mean()


def simulate_for_params_correct(mu0, std0, S1_val, S1_std, n_trials=2000):
    """
    Simulate the agent for ALL (S2_val, S2_std) conditions.
    
    Parameters:
    -----------
    mu0, std0 : float
        Prior parameters to test
    S1_val, S1_std : float
        Fixed S1 parameters (from the experiment)
    n_trials : int
        Number of trials per condition
    
    Returns:
    --------
    results : dict
        results[(S2_val, S2_std)] = P(respond 1) for that condition
    """
    results = {}
    
    for S2_val in mus_test:
        for S2_std in vars_test:
            p_respond_1 = simulate_agent_correct(
                S1_val, S1_std, 
                S2_val, S2_std, 
                mu0, std0, 
                n_trials
            )
            results[(S2_val, S2_std)] = p_respond_1
    
    return results


def mse_for_params_correct(mu0, std0, target, S1_val, S1_std, n_trials=2000):
    """
    Compute MSE between simulated agent and target data.
    """
    sim = simulate_for_params_correct(mu0, std0, S1_val, S1_std, n_trials)
    
    err = 0.0
    for key, target_val in target.items():
        err += (sim[key] - target_val)**2
    
    return err / len(target), sim


#%% ========= GRID SEARCH WITH CORRECT MODEL =========

# S1 is fixed in the experiment (you need to check what values were used!)
# Looking at the CSV format: Trial, S1_val, S1_std, S2_val, S2_std
# You should check what S1_val and S1_std were in your experiment

S1_val = 0.0    # <-- CHECK THIS! What was S1_val in your experiment?
S1_std = 2.0    # <-- CHECK THIS! What was S1_std in your experiment?

# Grid search parameters
mu0_grid  = np.linspace(-5, 5, 100)
std0_grid = np.linspace(0.5, 10.0, 100)

best_mu0  = None
best_std0 = None
best_mse  = np.inf
best_sim  = None

print("Grid search with CORRECT model...")
print(f"Using S1_val = {S1_val}, S1_std = {S1_std}")
print(f"Testing {len(mu0_grid)} x {len(std0_grid)} = {len(mu0_grid)*len(std0_grid)} combinations")

for mu0 in tqdm(mu0_grid):
    for std0 in std0_grid:
        mse, sim = mse_for_params_correct(mu0, std0, target, S1_val, S1_std, n_trials=1000)
        if mse < best_mse:
            best_mse  = mse
            best_mu0  = mu0
            best_std0 = std0
            best_sim  = sim

print("\n" + "="*50)
print("BEST PARAMETERS (CORRECT MODEL):")
print(f"mu0  = {best_mu0:.3f}  (prior mean)")
print(f"std0 = {best_std0:.3f}  (prior std)")
print(f"MSE  = {best_mse:.6f}")
print("="*50)


#%% ========= PLOT RESULTS =========

plt.figure(figsize=(10, 6))
colors = {
    0: "tab:purple",
    2: "tab:blue",
    4: "tab:orange",
    6: "tab:green",
    8: "tab:red"
}

for std2 in vars_test:
    xs = mus_test
    ys_model  = [best_sim[(mu2, std2)] for mu2 in mus_test]
    ys_target = [target[(mu2, std2)] for mu2 in mus_test]

    plt.plot(xs, ys_model, "-o", label=f"model (S2_std={std2})", color=colors[std2])
    plt.plot(xs, ys_target, "--", alpha=0.5, color=colors[std2])

plt.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Chance (0.5)")
plt.xlabel("S2_val (mean of S2)", fontsize=12)
plt.ylabel("P(respond '1' = 'S1 > S2')", fontsize=12)
plt.title(f"CORRECT Model: Psychometric curves\nBest fit: μ₀={best_mu0:.2f}, σ₀={best_std0:.2f}", fontsize=14)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('psychometric_correct_model.png', dpi=150)
plt.show()

print("\nPlot saved as 'psychometric_correct_model.png'")