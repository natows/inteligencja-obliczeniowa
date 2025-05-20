import matplotlib.pyplot as plt
import random
import time
from aco import AntColony

#ACO - ant colony optimization

plt.style.use("dark_background")


COORDS = ( 
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (50, 26),
    (84, 39),
    (60, 10),
    (11, 15),
    (12, 55),
    (40, 15),
    (90, 20),
    (80, 50),
    (90, 70),
    (10, 80),
    (20, 10),
    (30, 30),
    (40, 40),
    (50, 50),
    (60, 60),
    (70, 70),
    (80, 80),
    (90, 90),
)



# def plot_nodes(w=12, h=8):
#     for x, y in COORDS:
#         plt.plot(x, y, "g.", markersize=15)
#     plt.axis("off")
#     fig = plt.gcf()
#     fig.set_size_inches([w, h])


# def plot_all_edges():
#     paths = ((a, b) for a in COORDS for b in COORDS)

#     for a, b in paths:
#         plt.plot((a[0], b[0]), (a[1], b[1]))


# plot_nodes()


start = time.time()
colony = AntColony(COORDS, ant_count=500, alpha=0.8, beta=1.5, 
                    pheromone_evaporation_rate=0.5, pheromone_constant=1000.0,
                    iterations=300)
# 1 lista punktow do odwiedzenia
#ant_count=300 - liczba mrówek
#alpha=0.5 - waga feromonu
#beta=1.2 - waga odległości
#pheromone_evaporation_rate=0.40 - tempo parowania feromonu
#pheromone_constant=1000.0 - poczatkowa ilosc feromonu na kazdej krawedzi
#iterations=300 - liczba iteracji

optimal_nodes = colony.get_path()
end = time.time()
print("Czas wykonania:", end - start, "s")

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

plt.savefig("aco_tsp_modified7.png", dpi=300, bbox_inches="tight")
plt.show()


# dla wiekszej ilosci wezlow z tymi samymi parametrami czas wykonania
#  23.15952444076538 s

# ze zweikszonymm antcount do 500
# 38.135321617126465 s czas sie wydluzyl ale rozwiazanie jest epsze

#400 mrowek
#  33.1800479888916 s


#bez wykresow
# 400 mrowek i alpha 0.7, evap rate 0.3
# zwiekszenie alpha zmniejsza czas szukania
#36.253971576690674 s

# 400 mrowek i alpha 0.5, evap rate 0.4, beta 1.5
# 36.574344635009766 s


# 400 mrowek alpha 0.8 beta 1.5, evap rate 0.5, 
# 29.935840368270874 s


# ponowne zwiekszenie liczby wierzcholkow z default paametrami
# 72.01710343360901 s

# 400 mrowek alpha 0.8 beta 1.5, evap rate 0.5,
#  93.30224204063416 s

# 500 mrowek alpha 0.5 beta 1.3, evap rate 0.35,
# takie samo rozwiazanie jak poprzednio w dluzszym czasie
#116.22728133201599 s

# 500 mrowek alpha 0.8 beta 1.5, evap rate 0.5,
# 112.42681789398193 s najlepsze rozwiazanie