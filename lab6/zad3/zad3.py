import pygad, time

labirynt = [['c' for _ in range(12)],
            ['c','s','b','b','c','b','b','b','c','b','b','c'],
            ['c','c','c','b','b','b','c','b','c','c','b','c'],
            ['c','b','b','b','c','b','c','b','b','b','b','c'],
            ['c','b','c','b','c','c','b','b','c','c','b','c'],
            ['c','b','b','c','c','b','b','b','c','b','b','c'],
            ['c','b','b','b','b','b','c','b','b','b','c','c'],
            ['c','b','c','b','b','c','c','b','c','b','b','c'],
            ['c','b','c','c','c','b','b','b','c','c','b','c'],
            ['c','b','c','b','c','c','b','c','b','c','b','c'],
            ['c','b','c','b','b','b','b','b','b','b','e','c'],
            ['c' for _ in range(12)]]

max_steps = 30

gene_space =[1,2,3,4] #1-gora 2-prawo 3-lewo 4-dol
num_genes = max_steps

stop = False

def fitness_func(model, solution, solution_idx):
    x,y=1,1
    visited = set()
    visited.add((x,y))
    steps = 0
    solution = [int(g) for g in solution]
        
    for move in solution:
        if move == 1:
            new_x,new_y = x-1,y
        elif move == 2:
            new_x, new_y = x, y+1
        elif move == 3:
            new_x, new_y = x, y-1
        else: 
            new_x, new_y = x+1,y

        if (new_x, new_y) in visited:
            steps += 2  
        else:
            visited.add((new_x, new_y))  
        if not (0 <= new_x < 12 and 0 <= new_y < 12):
            steps += 50
        elif labirynt[new_x][new_y] != 'c':
            x,y = new_x,new_y
            visited.add((x,y))
            steps +=1
        else:
            steps += 50
        
        if labirynt[x][y] == 'e':
            global stop 
            stop = True
            return 1000 - steps

        
    dist = abs(x - 10) + abs(y - 10)  
    return 800 -steps - dist


sol_per_pop = 300
num_parent_mating = 10
num_generations = 800
keep_parents = 10
parent_selection_type = 'sss'
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 8
fitness_function = fitness_func

def on_generation(ga_instance):
     if stop:
          return "stop"
     
on_generation = on_generation
     
ga_instance = pygad.GA(gene_space=gene_space,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       num_parents_mating=num_parent_mating,
                       num_generations=num_generations,
                       keep_parents=keep_parents,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       fitness_func=fitness_function,
                       on_generation=on_generation)

start = time.time()
ga_instance.run()
end = time.time()

print(f"Time taken: {(end - start):.5f} seconds")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution)
print(solution_fitness)


def check_path(solution):
    x, y = 1, 1
    visited = set() 
    visited.add((x, y))  

    for i in solution:
        if i == 1:
            new_x, new_y = x-1, y
        elif i == 2:
            new_x, new_y = x, y+1
        elif i == 3:
            new_x, new_y = x, y-1
        else:
            new_x, new_y = x+1, y

        if new_x not in range(0, 12) or new_y not in range(0, 12):
            return "wylazl poza labirynt", visited
        elif labirynt[new_x][new_y] == 'c':
            visited.add((new_x, new_y))  
            return "zla sciezka", visited
        elif labirynt[new_x][new_y] == 'e':
            visited.add((new_x, new_y))  
            return "dobra sciezka", visited

        x, y = new_x, new_y
        visited.add((x, y))  

    return "nie dotarl do celu", visited

result, visited = check_path(solution)
print(result)

import matplotlib.pyplot as plt

def draw_path_full(visited):
    lab_copy = [row[:] for row in labirynt]

    for x, y in visited:
        if 0 <= x < len(labirynt) and 0 <= y < len(labirynt[0]):
            if lab_copy[x][y] == 'c':  
                lab_copy[x][y] = 'x' 
            elif lab_copy[x][y] not in ['s', 'e']:
                lab_copy[x][y] = '*'

    plt.figure(figsize=(8, 8))
    for i in range(len(lab_copy)):
        for j in range(len(lab_copy[i])):
            if lab_copy[i][j] == 'c': 
                plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'black')
            elif lab_copy[i][j] == 's':  
                plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'green')
            elif lab_copy[i][j] == 'e':  
                plt.fill([j, j+1, j+1, j], [i, i, i+1, i+1], 'red')
            elif lab_copy[i][j] == '*': 
                plt.fill([j+0.25, j+0.75, j+0.75, j+0.25], [i+0.25, i+0.25, i+0.75, i+0.75], 'blue')
            elif lab_copy[i][j] == 'x':  # Wejście na ścianę
                plt.fill([j+0.25, j+0.75, j+0.75, j+0.25], [i+0.25, i+0.25, i+0.75, i+0.75], 'red')

    plt.gca().invert_yaxis() 
    plt.grid(True)
    plt.title("Ścieżka w labiryncie (pełna)")
    plt.savefig("labirynt_full.png", dpi=300, bbox_inches="tight")


draw_path_full(visited)


# [2. 2. 4. 4. 1. 4. 3. 3. 4. 4. 2. 4. 2. 4. 2. 1. 2. 1. 4. 1. 2. 2. 4. 2.
#  2. 4. 2. 4. 4. 4.]
# 962
# dobra sciezka
#zrobil 26 krokow

#screen dobre2
# [2. 2. 4. 2. 2. 3. 2. 4. 1. 3. 2. 1. 2. 2. 4. 4. 2. 3. 4. 4. 4. 4. 1. 2.
#  2. 4. 2. 4. 4. 4.]
# 956
# dobra sciezka
#zrobil 30 krokow


# [2. 2. 4. 4. 1. 4. 4. 1. 3. 3. 4. 4. 2. 4. 2. 2. 2. 1. 2. 2. 4. 2. 2. 4.
#  2. 4. 4. 4. 1. 4.]
# 966
# dobra sciezka
#zrobil 28 krokow


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

    
# mean time 4.199656