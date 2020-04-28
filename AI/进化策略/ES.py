import numpy as np
from ESIndividual import ESIndividual
import random
import copy
import matplotlib.pyplot as plt


class EvolutionaryStrategy:

    '''
    the class for evolutionary strategy
    '''

    def __init__(self, sizepop, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of[delta_max, delta_min]
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
        initialize the population of es
        '''
        for i in range(0, self.sizepop):
            ind = ESIndividual(self.vardim, self.bound)
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
        the evolution process of the evolutionary strategy
        '''
        self.t = 0
        self.initialize()
        self.evaluation()
        bestIndex = np.argmax(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])
        while self.t < self.MAXGEN:
            self.t += 1
            tmpPop = self.mutation()
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

    def mutation(self):
        '''
        mutate the population by a random normal distribution
        '''
        tmpPop = []
        for i in range(0, self.sizepop):
            ind = copy.deepcopy(self.population[i])
            delta = self.params[0] + self.t * \
                (self.params[1] - self.params[0]) / self.MAXGEN
            ind.chrom += np.random.normal(0.0, delta, self.vardim)
            for k in range(0, self.vardim):
                if ind.chrom[k] < self.bound[0, k]:
                    ind.chrom[k] = self.bound[0, k]
                if ind.chrom[k] > self.bound[1, k]:
                    ind.chrom[k] = self.bound[1, k]
            ind.calculateFitness()
            tmpPop.append(ind)
        return tmpPop

    def selection(self, tmpPop):
        '''
        update the population
        '''
        for i in range(0, self.sizepop):
            if self.fitness[i] < tmpPop[i].fitness:
                self.population[i] = tmpPop[i]
                self.fitness[i] = tmpPop[i].fitness

    def printResult(self):
        '''
        plot the result of evolutionary strategy
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Evolutionary strategy for function optimization")
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
     bound = np.tile([[-600], [600]], 25)    
     es = EvolutionaryStrategy(60, 25, bound, 1000, [10, 1])
     es.solve()