from anneal import SimAnneal
import matplotlib.pyplot as plt
import random
import time


def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


def generate_random_coords(num_nodes):
    return [[random.uniform(-1000, 1000), random.uniform(-1000, 1000)] for i in range(num_nodes)]


if __name__ == "__main__":
    coords = read_coords("cities400.txt")  # generate_random_coords(100)
    # coords = generate_random_coords(400)

    # with open("cities400.txt", 'w') as fp:
    #     for item in coords:
    #         # write each item on a new line
    #         fp.write(f"{int(item[0])} {int(item[1])}\n")
    #     print('Done')


    initial_temp = 100
    final_temp=0.0000001
    cooling_rate = 0.9995
    iterations = 10000000


    sa = SimAnneal(coords, initial_temp,cooling_rate,final_temp,iterations)

    tic = time.perf_counter()
    sa.anneal()

    tac = time.perf_counter()

    Total_time = int(tac-tic)

    print(f"{sa.N}\t{sa.T_save}\t{cooling_rate}\t{sa.alpha}\t{sa.stopping_temperature}\t{sa.iteration}\t{int(sa.fitness_list[0])}\t{int(sa.fitness_list[-1])}\t{Total_time}")



    print(f"####### The time took to compute the Simulated Annealing Algorithm: {Total_time}")

    # sa.plot_learning(Total_time)

    # plt.ylim(min(sa.alpha_list)*1.1, max(sa.alpha_list)*1.1)

    plt.plot([i for i in range(1,sa.iteration)],sa.alpha_list,'b')
    plt.title(f"Dynamic Cooling rate value over iterations",size=18, fontweight="bold")
    
    plt.show()

