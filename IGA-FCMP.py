# imports
import numpy as np
import random
from copy import deepcopy
from sklearn import preprocessing
from sklearn.datasets import load_iris


def binarize_data():
    dataIris = load_iris()
    x = dataIris.data
    y = dataIris.target

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    normalize = (x - mean) / std
    BinaryIris = np.zeros((150, 20))
    bins = np.linspace(-1, 1, num=5)
    c = 0
    for i in range(4):
        for j in range(5):
            BinaryIris[:, c] = normalize[:, i] < bins[j]
            c += 1
    return BinaryIris, y


X, Y = binarize_data()


def gen_random_pop(size, chhrom_size):
    population = list()
    for i in range(size):
        population.append((np.random.permutation(20)[0:chhrom_size], np.random.randint(2, size=chhrom_size)))
    return np.array(population)


# fitness function
def fitness(chromosome, classId):
    M = 20

    p = Y == classId  # we assume that 0 is positive others are negative
    n = Y != classId
    chosen_p = np.squeeze(X[np.ix_(p, chromosome[0])])
    chosen_n = np.squeeze(X[np.ix_(n, chromosome[0])])
    num_p_matches = np.sum(np.sum(chosen_p == chromosome[1], axis=1) == len(chromosome[0]))
    num_n_matches = np.sum(np.sum(chosen_n == chromosome[1], axis=1) == len(chromosome[0]))
    Rp = num_p_matches / len(chosen_p)
    Rn = num_n_matches / len(chosen_n)
    if Rp == 0 and Rn == 0:
        return 1000
    elif Rp == 0 and Rn != 0:
        return 1000
    elif Rp != 0 and Rn == 0:
        return 1.0 / Rp
    return 1.0 / Rp + M * Rn


# # selection
# def selection(roulette, n):
#     probabilities = list()
#     for j in range(n):
#         rand = random.random()
#         for i in range(len(population)):
#             if rand <= roulette[i]:
#                 probabilities.append(population[i])
#                 break
#     return probabilities


def twoPointsCrossover(parent1, parent2):
    rand = random.random()
    if rand <= 0.9:
        size = len(parent2[0])
        # Randomly select the two crossover points
        crossover_first_point = random.randint(1, int(size / 2))  # 3
        crossover_second_point = random.randint(int(size / 2 + 1), size)  # 4
        # crossover_first_point.append(crossover_second_point)
        child1 = np.concatenate((parent1[0:crossover_first_point], parent2[crossover_first_point:]))
        child2 = np.concatenate((parent2[0:crossover_second_point], parent1[crossover_second_point:]))
        return child1, child2
    else:
        return parent1, parent2


# mutation less than 0.01 random
def mutation(a):
    # rand = random.randint(0, 20)
    # for i in range(len(a)):
    #     if a[0][i] == rand:
    #         a[0][i] = random.randint(0, 20)
    #         a[1][i] = random.randint(0, 2)
    pass


def survival_selection(pars, pars_fit, chils, chils_fit):
    h = int(len(pars) / 2)
    chils = np.array(chils)

    p_order = np.argsort(np.squeeze(pars_fit))
    c_order = np.argsort(np.squeeze(chils_fit))
    fiftyParents = np.squeeze(pars[p_order[0:h], :, :])
    fiftyChilds = np.squeeze(chils[c_order[0:h], :, :])
    chromes = np.vstack((fiftyParents,fiftyChilds))
    fits = np.vstack((pars_fit[p_order[0:h]], chils_fit[c_order[0:h]]))
    return chromes, fits


def GA(iter, classId, chromosomeSize):
    pop_size = 100
    # initialization
    parent_gen = gen_random_pop(pop_size, chromosomeSize)
    parent_fitness = np.zeros((pop_size, 1))
    for i in range(pop_size):
        parent_fitness[i] = fitness(parent_gen[i], classId)
    bestChromosome = []
    for i in range(iter):
        offsprings = []
        # crossover in algorithm
        for j in range(50):
            # mating selection
            parents = np.random.choice(range(0, len(parent_gen)), 2)
            child1, child2 = twoPointsCrossover(parent_gen[parents[0]], parent_gen[parents[1]])
            offsprings.append(child1)
            offsprings.append(child2)
        # print(f"{i} : we generated {len(offsprings)} children")
        newOff = deepcopy(offsprings)

        # mutation in algorithm
        for child in offsprings:
            if random.random() < 0.01:
                mutation(child)

        # compute fitness
        children_fitness = np.zeros((len(offsprings), 1))
        for g in range(len(offsprings)):
            children_fitness[g] = fitness(offsprings[g], classId)

        # survival selection
        next_gen, next_gen_fit = survival_selection(parent_gen, parent_fitness, offsprings, children_fitness)
        parent_gen = next_gen
        parent_fitness = next_gen_fit
        print(f"{i} : mean of fitness = {next_gen_fit.mean()}")
        bestChromosome = parent_gen[parent_fitness.argsort()][0]
    return np.squeeze(bestChromosome)


def CheckAccuracy(bestChromosome, chSize, classId):
    dataByClass = Y[np.squeeze(np.ix_(np.sum(np.squeeze(X[:, np.ix_(bestChromosome[0])]) == bestChromosome[1], axis=1) == chSize))]
    countX = len(dataByClass)
    countTrue = np.sum(dataByClass == classId)
    countFalse = countX - countTrue
    print(f"Accuracy X{classId} = {(countTrue + (100 - countFalse)) / 150}\n")


if __name__ == "__main__":
    chromosomeSize = 5
    best1 = GA(100, 0, chromosomeSize)
    best2 = GA(100, 1, chromosomeSize)
    best3 = GA(100, 2, chromosomeSize)
    CheckAccuracy(best1, chromosomeSize, 0)
    CheckAccuracy(best2, chromosomeSize, 1)
    CheckAccuracy(best3, chromosomeSize, 2)
    # data1 = X[np.ix_(Y == 0)]
    # count1 = np.sum(np.sum(np.squeeze(data1[:, np.ix_(best1[0])]) == best1[1], axis=1) == chromosomeSize) / 50
    # data2 = X[np.ix_(Y == 1)]
    # count2 = np.sum(np.sum(np.squeeze(data2[:, np.ix_(best2[0])]) == best2[1], axis=1) == chromosomeSize) / 50
    # data3 = X[np.ix_(Y == 2)]
    # count3 = np.sum(np.sum(np.squeeze(data3[:, np.ix_(best3[0])]) == best3[1], axis=1) == chromosomeSize) / 50
    #
    #

    # print(f"best1 = {count1}\nbest2 = {count2}\nbest3 = {count3}")
