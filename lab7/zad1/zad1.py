import pyswarms as ps
import numpy as np
import math
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt


max_bound = np.ones(6)
min_bound = np.zeros(6)
bounds = (min_bound, max_bound)
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
#c1 - swoja najlepsza
#c2 - najlepsza ogolem
#w - swoja pozycja
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, 
options=options, bounds=bounds)

#przy minimalizacji funkcji sphere oraz min bound [1,1] i max_bound [2,2] best cost: 2.0239804903895173, best pos: [1.00585072 1.00610378]

def endurance(array):
    return math.exp(-2*(array[1]-math.sin(array[0]))**2)+math.sin(array[2]*array[3])+math.cos(array[4]*array[5])
#endurance osiaga maximum w punkcie (0,0,0,0,0,0) i wynosi 2.0
# ale pso bedzie szukal takich punktow gdzie endurance jest najmniejsze
def f(x):
    n_particles = x.shape[0] # bo x to tablica n_particles tablic kazda o dlugosci tutaj 6
    counted = [endurance(x[i]) for i in range (n_particles)]
    return -np.array(counted)

cost, pos = optimizer.optimize(f, iters=500)

# 2025-04-13 13:08:24,012 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 1.076995013377783, best pos: [0.06489214 0.89517268 0.44940541 0.14282607 0.91336569 0.77291579] 
#tu dazy do minimum 

#po dodaniu minusa do funkcji celu, pso szuka minimum funkcji endurance, czyli przeciwienstwomaksimum funkcji endurance
# 2025-04-13 13:11:53,622 - pyswarms.single.global_best - INFO - Optimization finished | best cost: -2.8343712558958205, best pos: [0.81141756 0.74654199 0.9918864  0.99980015 0.2021871  0.28628051] 

plot_cost_history(optimizer.cost_history)
plt.savefig('pso-endurance.png')
plt.show()




