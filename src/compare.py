import json
import matplotlib.pyplot as plt

#load histories
ga_hist  = json.load(open('../models/ga_history.json'))
pso_hist = json.load(open('../models/pso_history.json'))
#baseline is constant-- i use the first GA value (or recalc)
baseline_rmse = ga_hist[0]

#start the plot
plt.figure(figsize=(8, 5))
gens = list(range(1, len(ga_hist) + 1))
plt.plot(gens, ga_hist, marker='o', label='GA')
plt.plot(gens, pso_hist, marker='s', label='PSO')
plt.hlines(baseline_rmse, gens[0], gens[-1], linestyles='--', label='Baseline RMSE')
plt.xticks(gens)
plt.xlabel('Generation')
plt.ylabel('RMSE')
plt.title('Hyperparameter Search Convergence: GA vs. PSO')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/convergence_comparison.png')
plt.show()
