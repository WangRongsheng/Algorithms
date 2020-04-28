import numpy as np
import ObjFunction


class BFOIndividual:

    '''
    individual of baterial clony foraging algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.trials = 0

    def generate(self):
        '''
        generate a random chromsome for baterial clony foraging algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        # self.fitness = ObjFunction.GrieFunc(
        #     self.vardim, self.chrom, self.bound)
        s1 = 0.
        s2 = 1.
        for i in range(1, self.vardim + 1):
            s1 = s1 + self.chrom[i - 1] ** 2
            s2 = s2 * np.cos(self.chrom[i - 1] / np.sqrt(i))
        y = (1. / 4000.) * s1 - s2 + 1
        self.fitness = y