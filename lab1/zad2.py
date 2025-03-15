import random, math, matplotlib.pyplot as plt, numpy as np

height = 100
speed = 50


def draw_plot(angle, aim):
    t_max = (speed * math.sin(angle) + math.sqrt(speed**2 * math.sin(angle)**2 + 2 * height * 9.81)) / 9.81

    t_values = np.linspace(0, t_max, num=500)

    x_values = speed * np.cos(angle) * t_values
    y_values = height + speed * np.sin(angle) * t_values - 0.5 * 9.81 * t_values**2

    plt.plot(x_values, y_values, label="Trajectory")
    plt.axvline(x=aim, color='r', linestyle='--', label="Target Distance")
    plt.title("Projectile Trajectory")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.grid(True)
    plt.legend()
    plt.savefig("trajektoria.png")
    plt.show()
    


def count_distance(angle):
    g = 9.81

    distance = (speed * math.sin(angle) + math.sqrt(speed**2 * math.sin(angle)**2 + 2 * height * g)) * speed * math.cos(angle) / g

    return distance

def main():
    tries = 0
    while True:
        aim = random.randint(50,340)
        print(f"Aim is in the distance of: {aim} meters")
        angle = int(input("Input angle of launch: "))
        tries +=1
        angle = math.radians(float(angle))
        distance = count_distance(angle)
        if aim-5 <= distance <= aim+5: 
            print(f"You hit the aim after {tries} tries!")
            draw_plot(angle, aim)
            break
        else:
            print(f"You hit the point int the distance of {distance} meters. Try again!")


if __name__ == "__main__":
    main()


