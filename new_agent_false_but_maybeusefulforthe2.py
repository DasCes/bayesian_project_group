import numpy as np
import matplotlib.pyplot as plt

def likelihood_ratio(mu1, mu2, std1, std2):
    """
    Partie 'données' du score postérieur : diff / var
    """
    var = std1**2 + std2**2
    A = np.random.normal(mu1, std1)
    B = np.random.normal(mu2, std2)
    dobs = A - B
    return dobs / var

def prior(mu0, std0):
    """
    Partie 'a priori' du score : mu0 / sigma0**2
    """
    return mu0 / (std0**2)

def posterior_score(mu1, std1, mu2, std2, mu0, std0):
    """
    Score proportionnel à mu_post (signe du posterior).
    Agent dit 1 si ce score > 0.
    """
    return prior(mu0, std0) + likelihood_ratio(mu1, mu2, std1, std2)

def agent_response(mu1, std1, mu2, std2, mu0, std0):
    """
    Retourne 1 ou 0 selon la règle bayésienne (seuil 0.5).
    """
    score = posterior_score(mu1, std1, mu2, std2, mu0, std0)
    return 1 if score > 0 else 0

relation = 0.15  # mu0 / sigma0**2

candidates = []
for std0 in np.arange(0.1, 10, 0.2):
    var0 = std0**2
    mu0 = relation * var0
    candidates.append((mu0, std0))

mus_test = [-10,-8,-6,-4,-2,0,2,4,6,8,10]
vars_test = [2,4,6,8]
mu1 = 0.0
std1 = 0.2

total_responses = {}
for mu2 in mus_test:
    for std2 in vars_test:
        responses = []
        for mu0,std0 in candidates:
            moy = 0
            for i in range(1000):
                response = agent_response(mu1, std1, mu2, std2, mu0, std0)
                moy += response
            responses.append(moy / 1000)
        total_responses[(mu2, std2)] = np.mean(responses)

for mu2 in mus_test:
    for std2 in vars_test:
        print("pour", (mu2, std2), ":", total_responses[(mu2, std2)])
        
# ----- PLOT des psychometric functions -----
plt.figure(figsize=(8,5))

colors = {2:"tab:blue", 4:"tab:orange", 6:"tab:green", 8:"tab:red"}

for std2 in vars_test:
    xs = mus_test
    ys = [total_responses[(mu2, std2)] for mu2 in mus_test]
    plt.plot(xs, ys, marker="o", label=f"var = {std2}", color=colors[std2])

plt.axhline(0.5, linestyle="--", color="red", alpha=0.5, label="Chance (0.5)")
plt.xlabel("Mean S2 value (mu2)")
plt.ylabel("Mean decision P(1)")
plt.title("Psychometric Functions (agent bayésien)")
plt.legend()
plt.grid(True)
plt.show()