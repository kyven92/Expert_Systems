from os import lseek
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import time
import csv


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"



class Fitness:
    """
    Each Fitness class is used for storing and calculating the route details and total distance of the route
    """
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    
    def routeDistance(self):
        """
        Using simple distance calculation formula between two points in cartesian coordinate system
        """
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        """
        Route Fitness is the inverse of the distance of the route
        """


        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness



def initialPopulation(popSize, cityList):
    population = []
    citylength = len(cityList)

    for i in range(0, popSize):

        route = random.sample(cityList, citylength)
        
        population.append(route)
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)



def selection(popRanked, eliteSize):
    """
    Fitness proportionate selection is used to select the majority of the next generation
    population apart from using elitism as a selection criteria

    """
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool

# Create a crossover function for two parents to create one child

def crossover(parent1, parent2):
    """
    Using the Two Point crossover method to create a child
    """
    child = []

    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    children = [None] * len(parent1)


    
    for i in range(startGene, endGene + 1, 1):
        children[i] = parent1[i]
    pointer = 0
    for i in range(len(parent1)):
        if children[i] is None:
            while parent2[pointer] in children:
                pointer += 1
            children[i] = parent2[pointer]
    child = children


    return child

def get_weights_according_to_ranks(matingpool):

    """
    Assigning weights according to route rank to increase the probability of 
    fittest population get selected more than the worst children
    """

    length = len(matingpool)
    weights = [1]*length

    sorted_pop = rankRoutes(matingpool)


    for i in range(length):

        index,fitness = sorted_pop[i]

        weights[index]=length-i


    return weights



# Create function to run crossover over full mating pool

def breedPopulation(matingpool, eliteSize):
    """
    Choosing parents according to the fitness rank for crossover
    """

    children = []
    length = len(matingpool) - eliteSize



    for i in range(0,eliteSize):
        children.append(matingpool[i])

    rank_weights = []



    rank_weights = get_weights_according_to_ranks(matingpool)

    

    for i in range(0, length):


        parent1,parent2 = random.choices(matingpool,rank_weights,k=2)
        child = crossover(parent1,parent2)

        children.append(child)
    return children



def mutate(individual, mutationRate):
    """
    Considering mutation as a rare case with the given probability, if gets selected for mutation,
    mutation happens by swaping randomly choosing two points and swaping the positions
    """


    if(random.random() < mutationRate):

        first_index,second_index = random.choices(range(len(individual)), k=2)
        individual[first_index],individual[second_index]=individual[second_index],individual[first_index]
    return individual



def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration




def createCities(noOfCities=100,file=""):
    """
    Cities are either loaded from a sample cities file or
    generate randomly 
    """

    cityList = []

    if file !="":
        ## Loading the cities from sample file

        with open(file,'r') as cities_file:

            tmp_cities = pd.read_csv(cities_file,delimiter=",",usecols =[1,2]).values.tolist()

            for city in tmp_cities:

                # print(f"###### City from city: {city}")

                cityList.append(City(x=int(city[0]), y=int(city[1])))

   
    else:
        ### Generating cities using random generators


        for i in range(0,noOfCities):
            cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    return cityList





def geneticAlgorithm(noOfCities, popSize, elitismPercent, mutationPercent, generations,file=""):

    """
    For each generation the best child is picked based on fitness and the next generations is stopped
    if the past 100 generations are choosing the same child as best option.

    Here, algorithm uses both fixed no of generations and stagnation of fitness as stoping criteria


    Also, the preference is given more for larger population than more generations according to the
    research
    """


    tic = time.perf_counter()


    population = createCities(noOfCities,file)
    eliteSize=int(noOfCities*elitismPercent/100)
    mutationRate=mutationPercent/100.0

    pop = initialPopulation(popSize, population)
  
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    best_route=[]
    
    breaking_condition = 100

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

        sorted_pop = rankRoutes(pop)

        best_fitness_index,best_fitness = sorted_pop[0]
        best_route=pop[best_fitness_index]

        progress.append(1 / best_fitness)
      
        if int(i*100/generations)>50 and len(set(progress[-breaking_condition:]))==1:
            break

    tac = time.perf_counter()

    Total_time = int(tac-tic)

    print(f"####### The time took to compute the genetic Algorithm: {Total_time}")

    best_routes_list=[]

    for ele in best_route:

        best_routes_list.append((ele.x,ele.y))


    fig,(ax1,ax2) = plt.subplots(1,2)
    
    fig.suptitle("Optimizing the TSP using Genetic Algorithm")
    fig.set_size_inches(16, 12)

    ax1.plot(*zip(*best_routes_list),marker="o")
    ax1.set_xlabel(" No. of Generations")
    ax1.set_ylabel(" Distance of the best route at given generation")

    ax2.plot(progress)
    ax2.annotate(f"Final: {int(progress[-1])}",(len(progress)-1,progress[-1]),ha="left")
    ax2.set_xlabel(" No. of Generations")
    ax2.set_ylabel(" Distance of the best route at given generation")

    text_x_coord = int(generations/3)
    text_y_coord = int(max(progress))

    plt.text(text_x_coord,text_y_coord,f"The Process took: {Total_time} Seconds",fontsize = 22)

    plt.savefig("test_plot.png",dpi=100)


    plt.show()


### Below Function call uses a static cities of data for 100 which is helpful in comparing the results

geneticAlgorithm(noOfCities=100, popSize=300, elitismPercent=10, mutationPercent=4, generations=300,file="cities100.csv")


##### Uncomment the below line incase if you want to use randomly generated cities data

# geneticAlgorithm(noOfCities=100, popSize=300, elitismPercent=10, mutationPercent=4, generations=300)






