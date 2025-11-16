import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def sample_dobs(mu1, std1, mu2, std2):
    A = np.random.normal(mu1, std1)
    B = np.random.normal(mu2, std2)
    return A - B

def posterior_params(dobs, std1, std2, mu0, std0):
    var_lik = std1**2 + std2**2      # v
    var0    = std0**2               # sigma0^2

    sigma_post2 = 1.0 / (1.0/var0 + 1.0/var_lik)
    mu_post     = sigma_post2 * (mu0/var0 + dobs/var_lik)
    return mu_post, np.sqrt(sigma_post2)

def Phi(z):
    """
    CDF de la normale standard : Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    Version utilisant scipy.special.erf.
    """
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def prob_response_1(mu1, std1, mu2, std2, mu0, std0):
    """
    Tire un dobs, calcule le posterior, renvoie P(répondre 1) = P(x>0|dobs).
    """
    dobs = sample_dobs(mu1, std1, mu2, std2)
    mu_post, sigma_post = posterior_params(dobs, std1, std2, mu0, std0)
    return Phi(mu_post / sigma_post)

def agent_response(mu1, std1, mu2, std2, mu0, std0):
    """
    Agent stochastique : tire une Bernoulli avec proba = P(répondre 1).
    """
    p1 = prob_response_1(mu1, std1, mu2, std2, mu0, std0)
    return 1 if np.random.rand() < p1 else 0


relation = 0.15  # si tu veux garder mu0 = relation * sigma0^2

candidates = []
for std0 in np.arange(0.1, 10, 0.2):
    var0 = std0**2
    mu0 = relation * var0
    candidates.append((mu0, std0))

mus_test  = [-10,-8,-6,-4,-2,0,2,4,6,8,10]
vars_test = [2,4,6,8]
mu1  = 0.0
std1 = 0.2

total_responses = {}
n_trials = 1000

for mu2 in mus_test:
    for std2 in vars_test:
        # on moyenne sur les candidats (ici tous ont le même lambda)
        responses = []
        for mu0, std0 in candidates:
            moy = 0
            for _ in range(n_trials):
                moy += agent_response(mu1, std1, mu2, std2, mu0, std0)
            responses.append(moy / n_trials)
        total_responses[(mu2, std2)] = np.mean(responses)

# ----- PLOT -----
plt.figure(figsize=(8,5))
colors = {2:"tab:blue", 4:"tab:orange", 6:"tab:green", 8:"tab:red"}

for std2 in vars_test:
    xs = mus_test
    ys = [total_responses[(mu2, std2)] for mu2 in mus_test]
    plt.plot(xs, ys, marker="o", label=f"var = {std2}", color=colors[std2])

plt.axhline(0.5, linestyle="--", color="red", alpha=0.5, label="Chance (0.5)")
plt.xlabel("Mean S2 value (mu2)")
plt.ylabel("Mean decision P(1)")
plt.title("Psychometric Functions (agent bayésien, P(x>0|posterior))")
plt.legend()
plt.grid(True)
plt.show()
