from AntColonyOptimizer import AntColonyOptimizer

import math
import sys,getopt
import numpy as np,random


def distance(city1: dict, city2: dict):
    return math.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)


def populate_distance_matrix(file_path="",noOfCities=0):
    cities = []
    points = []
    if noOfCities==0:
        with open(file_path) as f:
            for line in f.readlines():
                city = line.split(' ')
                cities.append(dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
                points.append((int(city[1]), int(city[2])))
    else:
        for i in range(0,noOfCities):
                cities.append(dict(index=i,x=int(random.random() * noOfCities), y=int(random.random() * noOfCities)))
    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(cities[i], cities[j]))
        cost_matrix.append(row)

    return np.array(cost_matrix)



def main(argv):

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    # print('Input file is "', inputfile)
    # print('Output file is "', outputfile)

    distance_matrix = populate_distance_matrix(file_path="sample_400.txt")

    optimizer = AntColonyOptimizer(ants=30, evaporation_rate=0.1, intensification=1, alpha=1, beta=2,
                                beta_evaporation_rate=0, choose_best=.1)
    
    best = optimizer.fit(distance_matrix, 400,verbose=False)
    optimizer.plot()

if __name__ == '__main__':
    main(sys.argv[1:])

    