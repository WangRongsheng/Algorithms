import numpy as np
from CSAIndividual import CSAIndividual
import random
import copy
import matplotlib.pyplot as plt


class CloneSelectionAlgorithm:

    '''
    the class for clone selection algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[beta, pm, alpha_max, alpha_min]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros(self.sizepop)
        self.trace = np.zeros((self.MAXGEN, 2))

    def initialize(self):
        '''
        initialize the population of ba
        '''
        for i in range(0, self.sizepop):
            ind = CSAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self):
        '''
        evaluation the fitness of the population
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def solve(self):
        '''
        the evolution process of the clone selection algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluation()
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        while self.t < self.MAXGEN:
            self.t += 1
            tmpPop = self.reproduction()
            tmpPop = self.mutation(tmpPop)
            self.selection(tmpPop)
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])

            self.avefitness = np.mean(self.fitness)
            self.trace[self.t - 1, 0] = \
                (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t - 1, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t - 1, 0], self.trace[self.t - 1, 1]))
        print("Optimal function value is: %f; " % self.trace[self.t - 1, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

    def reproduction(self):
        '''
        reproduction
        '''
        tmpPop = []
        for i in range(0, self.sizepop):
            nc = int(self.params[1] * self.sizepop)
            for j in range(0, nc):
                ind = copy.deepcopy(self.population[i])
                tmpPop.append(ind)
        return tmpPop

    def mutation(self, tmpPop):
        '''
        hypermutation
        '''
        for i in range(0, self.sizepop):
            nc = int(self.params[1] * self.sizepop)
            for j in range(1, nc):
                rnd = np.random.random(1)
                if rnd < self.params[0]:
                    # alpha = self.params[
                    #     2] + self.t * (self.params[3] - self.params[2]) / self.MAXGEN
                    delta = self.params[2] + self.t * \
                        (self.params[3] - self.params[3]) / self.MAXGEN
                    tmpPop[i * nc + j].chrom += np.random.normal(0.0, delta, self.vardim)
                    # tmpPop[i * nc + j].chrom += alpha * np.random.random(
                    # self.vardim) * (self.best.chrom - tmpPop[i * nc +
                    # j].chrom)
                    for k in range(0, self.vardim):
                        if tmpPop[i * nc + j].chrom[k] < self.bound[0, k]:
                            tmpPop[i * nc + j].chrom[k] = self.bound[0, k]
                        if tmpPop[i * nc + j].chrom[k] > self.bound[1, k]:
                            tmpPop[i * nc + j].chrom[k] = self.bound[1, k]
                    tmpPop[i * nc + j].calculateFitness()
        return tmpPop

    def selection(self, tmpPop):
        '''
        re-selection
        '''
        for i in range(0, self.sizepop):
            nc = int(self.params[1] * self.sizepop)
            best = 0.0
            bestIndex = -1
            for j in range(0, nc):
                if tmpPop[i * nc + j].fitness > best:
                    best = tmpPop[i * nc + j].fitness
                    bestIndex = i * nc + j
            if self.fitness[i] < best:
                self.population[i] = copy.deepcopy(tmpPop[bestIndex])
                self.fitness[i] = best

    def printResult(self):
        '''
        plot the result of clone selection algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Clone selection algorithm for function optimization")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
     bound = np.tile([[-600], [600]], 25)
     csa = CloneSelectionAlgorithm(50, 25, bound, 500, [0.3, 0.4, 5, 0.1])
     csa.solve()