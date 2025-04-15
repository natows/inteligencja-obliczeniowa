import numpy, pygad, random, matplotlib.pyplot as plt, time
# seed_value = 15


items = [{"name": 'zegar', "value": 100, "weight": 7},
         {"name": 'obraz-pejzaz', "value": 300, "weight": 7},
         {"name": 'obraz-portret', "value": 200, "weight": 6},
         {"name": 'radio', "value": 40, "weight": 2},
         {"name": 'laptop', "value": 500, "weight": 5},
         {"name": 'lampka nocna', "value": 70, "weight": 6},
         {"name": 'srebrne sztucce', "value": 100, "weight": 1},
         {"name": 'porcelana', "value": 250, "weight": 3},
         {"name": 'figura z brazu', "value": 300, "weight": 10},
         {"name": 'skorzana torebka', "value": 280, "weight": 3},
         {"name": 'odkurzacz', "value": 300, "weight": 15}] 
n = 25

gene_range = [0, 1]

sol_per_pop = 10 
num_genes = len(items)

num_parents_mating = 5
num_generations = 30
keep_parents = 2

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

def fitness_func(model, solution, solution_idx):
    total_value = numpy.sum(solution * [item["value"] for item in items])
    total_weight = numpy.sum(solution * [item["weight"] for item in items])
    
    if total_weight > n:
        return 0  
    
    return total_value

def on_generation(ga_instance):
    if ga_instance.best_solution()[1] >= 1630:
        return "stop"

fitness_function = fitness_func
ga_instance = pygad.GA(gene_space=gene_range,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=on_generation,                      
                       )
start = time.time()
ga_instance.run()
end = time.time()

print(f"Time taken: {(end - start):.5f} seconds")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
solution_items = [items[i] for i in range(len(solution)) if solution[i] == 1]
print(solution)
print(solution_items)
print(f"Best fitness value: {solution_fitness}")
print(f"Predicted value: {numpy.sum(solution * [item['value'] for item in items])}")
print(f"Predicted weight: {numpy.sum(solution * [item['weight'] for item in items])}")


# [0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 0.]
# [{'name': 'obraz-pejzaz', 'value': 300, 'weight': 7}, 
# {'name': 'obraz-portret', 'value': 200, 'weight': 6},
# {'name': 'laptop', 'value': 500, 'weight': 5},
# {'name': 'srebrne sztucce', 'value': 100, 'weight': 1},
# {'name': 'porcelana', 'value': 250, 'weight': 3},
# {'name': 'skorzana torebka', 'value': 280, 'weight': 3}]
# Best fitness value: 1630.0
# Predicted value: 1630.0
# Predicted weight: 25.0

ga_instance.plot_fitness()
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, label='Najlepszy fitness')
plt.xlabel("Pokolenie")
plt.ylabel("Fitness")
plt.title("Zmiana fitness w kolejnych pokoleniach")
plt.legend()
plt.grid(True)

# # Zapisanie wykresu do pliku zamiast wyświetlania
# plt.show()
plt.savefig("fitness_plot.png", dpi=300, bbox_inches="tight")



        
