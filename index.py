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

def selection():
    return 0
    # TO DO
    # select the best individuals among population

def crossover():
    return 0
    # TO DO
    # receives a couple return a new individual

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

#---------------------------- Genetic Algorithm ------------------------------------#
POPULATION_SIZE = 100
CHROMOSOME_SIZE = 20

population = generatePopulation(POPULATION_SIZE, CHROMOSOME_SIZE)

generation = 0

# while(melhorSolucao):

#     #Avaliate individuals
#     bestIndividuals = selection(population)

#     #Cruzamentos entre os individuos escolhidos. E no mesmo método faço a mutação do filho gerado
#     #No método crossover eu gero um lista de novos indivíduos.
#     newPopulation = crossover(bestIndividuals)

#     #Faço a avaliação do novo indivíduo e se valer a pena adiciono ele na população.
#     population = updatePopulation(population, newPopulation)
