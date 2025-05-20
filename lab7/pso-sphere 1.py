# Import modules
import numpy as np
from matplotlib import pyplot as plt
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history


#PSO - particle swarm optimization, optymalizacja rojem czastek
#pso najczesciej uzywa sie do znajdowania minimum funkcji

#hiperparametry - parametry sterujace ruchem czastek
# c1 - wspolczynnik kognitywny - jak bardzo czastka ufa swojej najlepszej pozycji
#c2 - wspolczynnik spoleczny - jak bardzo czaska ufa najlepszej pozycji w calym roju
# w - wspolczynnik bezwladnosci - jak bardzo czastka ufa swojemu aktualnemu kierunkowi ruchu

# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.5}

# Call instance of PSO optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
#n_particles - liczba czastek w roju
#dimetions - wymiar przestrzeni, w ktorej czastki sie poruszaja
#tu przestrzen 2d wiec sa dwa parametry do zoptymalizowania
#options - slownik z parametrami sterujacymi ruchem czastek

#dzialanie
#na poczatki kazda czastka ma losowa pozycje i predkosc
#w kazdej iteracji kazda czastka roju wykonuje 3 glowne kroki:
# 1 ocena aktualnej pozycji - kazda czastka ma pozycje x w przestrzeni, predkosc v, wlasna najlepsza dotychczasowa pozycja pbest i najlepsza pozycja calego roju gbest
#ocena odbywa sie przez funkcje celu
# 2 aktualizacja predkosci - wg wzoru: v = w*v + c1*rand()*(pbest - x) + c2*rand()*(gbest - x)
# 3 aktualizacja pozycji - wg wzoru: x = x + v

# Perform optimization
cost, pos = optimizer.optimize(fx.sphere, iters=200)
#uzywamy funckji celu fx.sphere, ktora jest funkcja sferyczna
# f(x) (x to wektor jakcos) = sigma(od i=1 do n) x_i ^2, jej minimum to f(0,0) = 0
#cost - wartosc funkcji celu w minimum
#pos - pozycja czastki w minimum

# Obtain cost history from optimizer instance
cost_history = optimizer.cost_history
print(pos)

# Plot!
plot_cost_history(cost_history)
plt.savefig('pso-sphere.png')
plt.show()
