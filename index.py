import pandas as pd
from random import randint
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from matplotlib import pyplot


def get_data(file_name):
    return pd.read_csv(file_name)


data = get_data('data/AG.csv')
x_data = data.iloc[:, :-1]
y_data = data.iloc[:, -1]


class Chromosome:
    def __init__(self, chromosome_size):
        self.chromosome_size = chromosome_size
        self.chromosome = []
        self.generate_chromosome()
        self.fitness = 0
        self.numberVariables = None
        self.y_test = None
        self.y_predict = None
        self.generate_fitness()

    def generate_chromosome(self):
        for i in range(self.chromosome_size):
            self.chromosome.append(randint(0, 1))

    def generate_fitness(self):
        true_indexes_list = [index for index, state in enumerate(self.chromosome) if state == 1]
        self.fitness, self.y_test, self.y_predict = getFitness(true_indexes_list)
        self.numberVariables = len(true_indexes_list)

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


# def crossover(population):
#     new_population = []
#     random.shuffle(population)
#     for i in range(0, len(population) - 1, 2):
#         chromosome1 = population[i]
#         chromo1_len = len(chromosome1.get_chromosome())
#         chromosome2 = population[i + 1]
#         new_chromosome = []
#         new_chromosome2 = []
#
#         for i in range(chromo1_len):
#             if randint(0, 1):
#                 new_chromosome.append(chromosome1.get_chromosome()[i])
#                 new_chromosome2.append(chromosome2.get_chromosome()[i])
#             else:
#                 new_chromosome.append(chromosome2.get_chromosome()[i])
#                 new_chromosome2.append(chromosome1.get_chromosome()[i])
#
#         new_population.append(Chromosome(len(new_chromosome)))
#         new_population[-1].chromosome = new_chromosome
#         new_population[-1].generate_fitness()
#
#         new_population.append(Chromosome(len(new_chromosome2)))
#         new_population[-1].chromosome = new_chromosome2
#         new_population[-1].generate_fitness()
#
#     return new_population

def crossover(best_individuals):
    new_population = []
    best_individuals_len = len(best_individuals)

    random.shuffle(best_individuals)

    # one point crossover
    it = 0
    slice_point_1 = 34
    while it < best_individuals_len:
        chromosome_1 = best_individuals[it].get_chromosome()
        chromosome_2 = best_individuals[it + 1].get_chromosome()

        new_chromosome_1 = Chromosome(40)
        new_chromosome_2 = Chromosome(40)

        new_chromosome_1.chromosome = chromosome_1[0:slice_point_1] + chromosome_2[slice_point_1:40]
        new_chromosome_1.generate_fitness()
        new_chromosome_2.chromosome = chromosome_2[0:slice_point_1] + chromosome_1[slice_point_1:40]
        new_chromosome_2.generate_fitness()

        new_population.append(new_chromosome_1)
        new_population.append(new_chromosome_2)

        it += 2

    it = 0

    # two points crossover
    slice_point_1 = 16
    slice_point_2 = 25
    while it < best_individuals_len:
        chromosome_1 = best_individuals[it].get_chromosome()
        chromosome_2 = best_individuals[it + 1].get_chromosome()

        new_chromosome_1 = Chromosome(40)
        new_chromosome_2 = Chromosome(40)

        new_chromosome_1.chromosome = chromosome_1[0:slice_point_1] + chromosome_2[
                                                                      slice_point_1:slice_point_2] + chromosome_1[
                                                                                                     slice_point_2:40]
        new_chromosome_1.generate_fitness()
        new_chromosome_2.chromosome = chromosome_2[0:slice_point_1] + chromosome_1[
                                                                      slice_point_1:slice_point_2] + chromosome_2[
                                                                                                     slice_point_2:40]
        new_chromosome_2.generate_fitness()

        new_population.append(new_chromosome_1)
        new_population.append(new_chromosome_2)

        it += 2

    return new_population


def mutation(population):
    for individual in population:
        for index, bit in enumerate(individual.get_chromosome()):
            if randint(0, 99) < 1:
                individual.chromosome[index] = 1 if bit == 0 else 0
                individual.generate_fitness()

    return population


def getBestIndividual(population):
    best = population[0]

    for individual in population:
        if individual.fitness > best.fitness:
            best = individual

    return best


def knnFit(x_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=3)
    knn = knn.fit(x_train, y_train.values.ravel())

    return knn


def getFitness(chromosome):
    df = x_data.iloc[:, chromosome]
    x_train, x_test, y_train, y_test = train_test_split(df, y_data, test_size=0.3, random_state=42)
    knn = knnFit(x_train, y_train)
    y_predict = knn.predict(x_test)

    return r2_score(y_test, y_predict), y_test, y_predict


def printPopulation(population):
    print("\nPopulation: \n====================================")
    for individual in population:
        print(individual.get_chromosome(), individual.fitness)

    print("===============================================")


def plotScatterGraphic(bestIndividual):
    pyplot.scatter(bestIndividual.y_test, bestIndividual.y_predict)
    pyplot.title("Gráfico de Dispersão entre Y Test e Y Predict")
    pyplot.xlabel("Y Test")
    pyplot.ylabel("Y Predict")
    pyplot.show()


# ---------------------------- Genetic Algorithm ------------------------------------#
def geneticAlgorithm():
    POPULATION_SIZE = 20
    CHROMOSOME_SIZE = 40

    population = generatePopulation(POPULATION_SIZE, CHROMOSOME_SIZE)

    generation = 1

    print("population size " + str(len(population)))

    while (True):

        # Avaliate individuals
        bestIndividuals = selection(population)

        # print("Best individuals size: " + str(len(bestIndividuals)))

        # newPopulation = generatePopulation(POPULATION_SIZE // 2, CHROMOSOME_SIZE) + bestIndividuals

        # print("New POpulation size: " + str(len(newPopulation)))

        # nextPopulation = crossover(newPopulation)
        nextPopulation = crossover(bestIndividuals)

        # print("Next population size: " + str(len(nextPopulation)))

        mutatedPopulation = mutation(nextPopulation)

        # print("Mutated population size: " + str(len(mutatedPopulation)))

        bestIndividual = getBestIndividual(mutatedPopulation)

        print("\n\nGeneration: " + str(generation) + "\nbest individual: "
              + str(bestIndividual.get_chromosome())
              + "\nfitness: "
              + str(bestIndividual.fitness)
              + "\nNumber of variables: " + str(bestIndividual.numberVariables)
              )

        population = mutatedPopulation

        if bestIndividual.fitness > 0.95:
            break

        generation += 1

    print("\n=================== END =====================\n")
    print("Generation: " + str(generation) + " | " + str(bestIndividual.get_chromosome()) + " | " + str(
        bestIndividual.fitness))

    plotScatterGraphic(bestIndividual)


geneticAlgorithm()
