import numpy as np
import matplotlib.pyplot as plt
import tudatpy.util as util
import pickle

# Analye reszults of multiobjective optimization
fitness_list, individuals_list = pickle.load(open("results.pkl", "rb"))

# Plot Pareto for different Generations
fig_paret = plt.figure(figsize=(4,3))
ax_paret = plt.axes()
ax_paret.grid()
ax_paret.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
ax_paret.set_ylabel(r'$t_{flight}$ [$days$]')
ax_paret.set_xlabel(r'$\Delta_V$ [$m/s$]')

all_fitness = np.vstack(fitness_list)
all_individuals = np.vstack(individuals_list)

cs = ax_paret.scatter(all_fitness[:, 0],
                all_fitness[:, 1],
                s=100,
                c="blue",
                marker='.',
                alpha=0.65)

# Add the Pareto from itself in green
optimum_mask = util.pareto_optimums(np.array([all_fitness[:, 0], all_fitness[:, 1]]).T)
ax_paret.step(
    sorted(all_fitness[:, 0][optimum_mask], reverse=True),
    sorted(all_fitness[:, 1][optimum_mask], reverse=False),
    color="blue",
    linewidth=2,
    alpha=0.75)

plt.show()
