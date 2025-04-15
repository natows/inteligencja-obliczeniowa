import math, numpy as np, pygad, matplotlib.pyplot as plt

def endurance(x, y, z, u, v, w): 
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w) 

gene_range = np.linspace(0, 1, 100)[:-1]  

num_genes = 6
sol_per_pop = 10

num_parents_mating = 5
num_generations = 50
keep_parents = 2

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20

def fitness_func(model, solution, solution_idx):
    x,y,z,u,v,w = solution
    return endurance(x, y, z, u, v, w)

fitness_function = fitness_func

ga_instance = pygad.GA(gene_space = gene_range,
                       num_genes=num_genes,
                       num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       num_parents_mating=num_parents_mating,
                       keep_parents=keep_parents,
                       parent_selection_type=parent_selection_type,
                       crossover_type= crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       fitness_func=fitness_function)

ga_instance.run()


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution)
print(solution_fitness)


# ga_instance.plot_fitness()
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, label='Najlepszy fitness')
plt.xlabel("Pokolenie")
plt.ylabel("Fitness")
plt.title("Zmiana fitness w kolejnych pokoleniach")
plt.legend()
plt.grid(True)

# plt.show()
plt.savefig("fitness_plot.png", dpi=300, bbox_inches="tight")


# #najlepsze wyniki
# [0.23232323 0.23232323 0.97979798 0.98989899 0.08080808 0.05050505]
# 2.824812745390774



