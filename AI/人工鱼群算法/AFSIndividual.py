import numpy as np
import ObjFunction
import copy


class AFSIndividual:

    """class for AFSIndividual"""

    def __init__(self, vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound

    def generate(self):
        '''
        generate a rondom chromsome
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        self.velocity = np.random.random(size=len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]
        self.bestPosition = np.zeros(len)
        self.bestFitness = 0.

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = ObjFunction.GrieFunc(
            self.vardim, self.chrom, self.bound)