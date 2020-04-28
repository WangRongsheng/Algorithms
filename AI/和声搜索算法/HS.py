import numpy as np
from HSIndividual import HSIndividual
import random
import copy
import math
import matplotlib.pyplot as plt


class HarmonySearch:

    '''
    the class for harmony search algorithm
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[HMCR, PAR]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.MAXGEN = MAXGEN
        self.params = params
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))

    def initialize(self):
        '''
        initialize the population of hs
        '''
        for i in range(0, self.sizepop):
            ind = HSIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluation(self):
        '''
        evaluation the fitness of the population
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def improvise(self):
        '''
        improvise a new harmony
        '''
        ind = HSIndividual(self.vardim, self.bound)
        ind.chrom = np.zeros(self.vardim)
        for i in range(0, self.vardim):
            if random.random() < self.params[0]:
                if random.random() < self.params[1]:
                    ind.chrom[i] += self.best.chrom[i]
                else:
                    worstIdx = np.argmin(self.fitness)
                    xr = 2 * self.best.chrom[i] - \
                        self.population[worstIdx].chrom[i]
                    if xr < self.bound[0, i]:
                        xr = self.bound[0, i]
                    if xr > self.bound[1, i]:
                        xr = self.bound[1, i]
                    ind.chrom[i] = self.population[worstIdx].chrom[
                        i] + (xr - self.population[worstIdx].chrom[i]) * random.random()
            else:
                ind.chrom[i] = self.bound[
                    0, i] + (self.bound[1, i] - self.bound[0, i]) * random.random()
        ind.calculateFitness()
        return ind

    def update(self, ind):
        '''
        update harmony memory
        '''
        minIdx = np.argmin(self.fitness)
        if ind.fitness > self.population[minIdx].fitness:
            self.population[minIdx] = ind
            self.fitness[minIdx] = ind.fitness

    def solve(self):
        '''
        the evolution process of the hs algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluation()
        best = np.max(self.fitness)
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)
        self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
        self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
        print("Generation %d: optimal function value is: %f; average function value is %f" % (
            self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        while self.t < self.MAXGEN - 1:
            self.t += 1
            ind = self.improvise()
            self.update(ind)
            best = np.max(self.fitness)
            bestIndex = np.argmax(self.fitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
        print("Optimal function value is: %f; " % self.trace[self.t, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

    def printResult(self):
        '''
        plot the result of abs algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Harmony search algorithm for function optimization")
        plt.legend()
        plt.show()
        
        
if __name__ == "__main__":
     bound = np.tile([[-600], [600]], 25)
     hs = HarmonySearch(60, 25, bound, 5000, [0.9950, 0.4])
     hs.solve()