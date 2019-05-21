#%%
import os
import math
import numpy as np
import random

#%%
class GeneticOptimiser():

    def __init__(self, Dimension, FitnessFunction, SearchSpace, StepSize, *args, **kwargs):
        self.FitnessFunction = FitnessFunction
        self.SearchSpace = SearchSpace
        self.StepSize = StepSize
        self.number, self.length = Dimension[0], Dimension[1]

        ''' initiallising the cromosomes '''
        self.cromosomes = np.random.rand(self.number, self.length)


    def mutation(self, maxnumber=0.45, m_number=1, alpha=1):
        ''' 
        alpha from 0 to 1 controles how much of the population
        will get mutated 
        '''

        for i in range(math.ceil(alpha*self.number)):
            ''' initial cromosomes will get mutated '''

            a = random.choice(range(1, math.ceil(maxnumber*self.length)+1))
            ''' 
            maxnumber is for maximum how much gene is to be mutated 
            '''

            gene_to_mutate = random.sample(range(self.length), a)
            '''
            gene_to_mutate is for choosing the which gene is to be mutated for a 
            single cromosome
            '''
            
            for j in gene_to_mutate:
                self.cromosomes[i][j] = m_number * random.random()


    def crossover(self, type='uniform', beta=0.45, gamma=0.45):
        ''' 
        gamma defines how much top percentile of cromosomes will cross over 
        and beta is for much genes will cross over
        '''
        n = math.ceil(gamma*self.number)
        self.cromosomes = np.vstack((self.cromosomes, self.cromosomes))
        
        def parents_gen():
            p1 =  random.sample(range(1, self.number+1), n)
            p2 =  random.sample(range(1, self.number+1), n)
            return p1, p2

        for p1, p2 in parents_gen():
            to_cross =  random.sample(range(self.length), math.ceil(beta*self.length))

            for gene in to_cross:
                self.cromosomes[self.length+p1][gene] = self.cromosomes[p2][gene]


    def evolute(self):
        self.cromosomes = sorted(self.cromosomes, key=self.FitnessFunction, reverse=True)
        self.cromosomes = self.cromosomes[:self.number]

#%%
if __name__ == "__main__":
    
    s = GeneticOptimiser(Dimension=(4, 4), FitnessFunction=(lambda x: x[0]-x[1]+x[2]-x[3]), SearchSpace=5, StepSize=1)
    print('initial')
    print(s.cromosomes)
    
    s.mutation()
    print('')
    print('mutated')
    print(s.cromosomes)
    
    s.crossover()
    print('')
    print('crossovered')
    print(s.cromosomes)

    s.evolute()
    print('')
    print('evolved')
    print(s.cromosomes)


#%%

