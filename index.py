import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def get_data(file_name):
    return pd.read_csv(file_name)

class Chromosome:
    def __init__(self, chromosome_size):
        self.chromosome_size = chromosome_size
        self.chromosome = []
        self.generate_chromosome()
        self.generate_fitness()

    def generate_chromosome(self):
        for i in range(self.chromosome_size):
            self.chromosome.append(randint(0, 39))

    def generate_fitness(self):
        self.fitness = getFitness(self.chromosome)

    def get_chromosome(self):
        return self.chromosome


def generatePopulation(population_size, chromosome_size):
    population = []
    for i in range(population_size):
        population.append(Chromosome(chromosome_size))

    return population

def getMeanFitness(population):
    sum = 0
    for individual in population:
        sum += individual.fitness

    return sum / len(population)

def selection(population):
    population.sort(key=lambda x: x.fitness, reverse=True)

    return population[:int(len(population) * 0.5)]

def crossover(population):
    new_population = []
    for i in range(0, len(population) - 1, 2):
        chromosome1 = population[i]
        chromosome2 = population[i + 1]
        new_chromosome = []
        for i in range(len(chromosome1.get_chromosome())):
            if randint(0, 1):
                new_chromosome.append(chromosome1.get_chromosome()[i])
            else:
                new_chromosome.append(chromosome2.get_chromosome()[i])

            new_population.append(Chromosome(len(new_chromosome)))
            new_population[-1].chromosome = new_chromosome
    return new_population

def mutation(population):
    for individual in population:
        if randint(0, 100) < 5:
            individual.chromosome[randint(0, CHROMOSOME_SIZE - 1)] = randint(0, 39)
            individual.generate_fitness()
    return population

def getBestIndividual(population):
    if (population.__len__() > 0):
        best = population[0]
        for individual in population:
            if individual.fitness > best.fitness:
                best = individual
        return best
    else:
        print("No individual in population")

def knnFit(x_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=3)
    knn = knn.fit(x_train, y_train.values.ravel())

    return knn

def getFitness(chromosome):
    data = get_data('data/AG.csv')

    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]

    df = x_data.iloc[:, chromosome]
    x_train, x_test, y_train, y_test = train_test_split(df, y_data, test_size=0.3, random_state=42)

    knn = knnFit(x_train, y_train)

    y_predict = knn.predict(x_test)

    return r2_score(y_test, y_predict)

def printPopulation(population):
    print("Population: ====================================")
    for individual in population:
        print(individual.get_chromosome(), individual.fitness)

    print("===============================================")

#---------------------------- Genetic Algorithm ------------------------------------#
POPULATION_SIZE = 10
CHROMOSOME_SIZE = 20

population = generatePopulation(POPULATION_SIZE, CHROMOSOME_SIZE)

generation = 0
flag = False

print("population size " + str(len(population)))

while(generation < 100 or flag):

    #Avaliate individuals
    bestIndividuals = selection(population)

    if (len(bestIndividuals) > 0):
        # printPopulation(bestIndividuals)
        print("best individuals size " + str(len(bestIndividuals)))
    else:
        print("No individual in best population")


    #Cruzamentos entre os individuos escolhidos. E no mesmo método faço a mutação do filho gerado
    #No método crossover eu gero um lista de novos indivíduos.
    nextPopulation = crossover(bestIndividuals)

    if (len(nextPopulation) > 0):
        # printPopulation(nextPopulation)
        print("next population size " + str(len(nextPopulation)))
    else:
        print("No individual in next population")

    mutatedPopulation = mutation(nextPopulation)

    if (len(mutatedPopulation) > 0):
        # printPopulation(mutatedPopulation)
        print("mutated population size " + str(len(mutatedPopulation)))
    else:
        print("No individual in mutated population")

    bestIndividual = getBestIndividual(mutatedPopulation)

    if bestIndividual.fitness > 0.95:
        # print("Generation: " + str(generation) + " | " + str(bestIndividual.get_chromosome()) + " | " + str(bestIndividual.fitness))
        flag = True
        break

    #Faço a avaliação do novo indivíduo e se valer a pena adiciono ele na população.
    generation += 1
    population = mutatedPopulation

print("Generation: " + str(generation) + " | " + str(bestIndividual.get_chromosome()) + " | " + str(bestIndividual.fitness))
