import math
import random
import visualize_tsp
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.alpha_list =[]

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()

        init_alpha_percent = 0.005
        temp_dec_percent_prev = self.T_save
        change=False

        print("Starting annealing.")
        print(f"### Started Initial Temperature: {self.T_save}")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            self.accept(candidate)

            
            
            #Logic for Dynamic Colling rate updation with a limit 
            temp_dec = ((temp_dec_percent_prev-self.T)/temp_dec_percent_prev)*100

            if temp_dec >=90.0 and (1-self.alpha)>0.000005:

                if self.alpha*(1+init_alpha_percent)<1.0:
                    self.alpha = self.alpha*(1+init_alpha_percent)
                else:
                    if self.alpha*(1+init_alpha_percent/100.0) < 1.0:
                        self.alpha = self.alpha*(1+init_alpha_percent/100.0)

                if self.alpha > 0.999995:
                    self.alpha = 0.999995
                temp_dec_percent_prev=self.T

                # print(f"### Alpha after: {self.alpha}")

            self.T *= self.alpha
            self.alpha_list.append(self.alpha)
            
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)
            print(f"## Current Temperate: {self.T}",end='\r')

        print(end='\x1b[2K')

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def plot_learning(self,Total_time):
        """
        Plot the fitness through iterations.
        """

        x = []; y = []
        for i in self.best_solution:
            x.append(self.coords[i][0])
            y.append(self.coords[i][1])
        x.append(self.coords[self.best_solution[0]][0])
        y.append(self.coords[self.best_solution[0]][1])

        plotting_list = list(zip(x,y))



        fig,(ax1,ax2) = plt.subplots(1,2)

    
        fig.suptitle(f"Optimizing the TSP using Simulated Annealing \n No. of Cities: {self.N}, Initial Temp: {self.T_save}, Cooling Rate: {self.alpha}, Final Temp: {self.stopping_temperature}, No. Iterations: {self.iteration}",size=18, fontweight="bold")
        fig.set_size_inches(16, 12)

        ax1.plot(*zip(*plotting_list),marker="o")
        ax1.set_xlabel(" X-Coord")
        ax1.set_ylabel(" Y-Coord")

        ax2.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        ax2.annotate(f"Final: {int(self.fitness_list[-1])}",(len(self.fitness_list)-1,self.fitness_list[-1]),ha="left")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel(" Fitness")

        text_x_coord = int(int(self.fitness_list[-1])/3)
        text_y_coord = int(max(self.fitness_list))


        plt.text(text_x_coord,text_y_coord,f"The Process took: {Total_time} Seconds",fontsize = 22)


        plt.savefig(f"NoCiti_{self.N}_InTemp_{self.T_save}_Col_{self.alpha}_FinTemp_{self.stopping_temperature}_Iter_{self.iteration}.png",dpi=100)


        plt.show()

        

