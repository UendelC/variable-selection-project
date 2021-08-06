import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def get_data(file_name):
    return pd.read_csv(file_name)

def generatePopulation():
    return 0
    # TO DO
    # generate initial population

def selection():
    return 0
    # TO DO
    # select the best individuals among population

def crossover():
    return 0
    # TO DO
    # receives a couple return a new individual

def fitness():
    return 0
    # TO DO
    # receives a n indidividual and give it a fitness score

#---------------------------- Genetic Algorithm ------------------------------------#

#Initialize population
population = generatePopulation()

while(melhorSolucao):

    #Avaliate individuals
    bestIndividuals = selection(population)

    #Cruzamentos entre os individuos escolhidos. E no mesmo método faço a mutação do filho gerado
    #No método crossover eu gero um lista de novos indivíduos.
    newPopulation = crossover(bestIndividuals)

    #Faço a avaliação do novo indivíduo e se valer a pena adiciono ele na população.
    population = updatePopulation(population, newPopulation)
