#%%
import math
import numpy as np
import random

#%%
class GeneticOptimiser():

    def __init__(self, number_of_cromosomes, length_of_cromosomes,  fitness_function, lowest_gene_value=0, highest_gene_value=9, step_size=1, feed_cromosomes=False, *args, **kwargs):
        '''
        * number_of_cromosomes (integer), it corresponds to population.

        * length_of_cromosomes (integer).

        * fitness_function must be in lamba function form.

        * lowest_gene_value (integer).

        * highest_gene_value (integer).

        * step_size (float), it corresponds to the granuality in gene values.

        * feed_cromosoms:True to feed your own custom cromosomes values directly in initialise method, else it is False.
        '''
        self.FitnessFunction = fitness_function
        self.StepSize = step_size
        self.Low = lowest_gene_value
        self.High = highest_gene_value
        self.number = number_of_cromosomes
        self.length = length_of_cromosomes
        self.feed_cromosomes = feed_cromosomes


    def rand_with_step(self, size='tup'):
        '''
        Gives array of size=size with linespace = stepsize
        * size take a tuple
        '''
        n = 1 / self.StepSize

        if self.number > 1:
            val = np.random.randint(self.Low*n, self.High*n+1, size=size) * self.StepSize
        
        return val


    def initialise(self, manual_cromosomes='NaN'):

        if self.feed_cromosomes:
            ''' manually initiallising the cromosomes '''

            if manual_cromosomes.shape[0] < self.number:
                self.cromosomes = np.vstack((manual_cromosomes, 
                self.rand_with_step(size=(self.number - manual_cromosomes.shape[0], self.length))))
            
            elif manual_cromosomes.shape[0] >= self.number:
                self.cromosomes = manual_cromosomes[: self.number]

            else: 
                self.cromosomes = manual_cromosomes

        else:
            ''' randomly initiallising the cromosomes '''
            self.cromosomes = self.rand_with_step(size=(self.number, self.length))


    def mutation(self, maxnumber_mutation=0.9, alpha_mutation=.45):
        ''' 
        mutates lower alpha*100 % of the population(cromosomes) except best cromosome(first cromosome)

        * maxnumber_mutation from 0 to 1 controles how much of the Cromosomes will get mutated.

        * alpha_mutation is for maximum how much gene is to be mutated.
        '''

        for i in range(math.floor(maxnumber_mutation*self.number)-1, 0, -1):
            ''' initial cromosomes will get mutated '''

            a = random.choice(range(1, math.ceil(alpha_mutation*self.length)+1))
            gene_to_mutate = random.sample(range(self.length), a)
            '''
            gene_to_mutate is for choosing the which gene is to be mutated for a 
            single cromosome
            '''
            
            for j in gene_to_mutate:
                self.cromosomes[i][j] = self.rand_with_step(1)


    def crossover(self, type='uniform', maxnumber_cross=0.45, alpha_cross=0.45):
        ''' 
        it will do uniform crossover in the cromosomes

        * maxnumber_cross controles how much top cromosomes will cross over

        * alpha_cross is for how much genes will cross over
        '''
        n = math.ceil(maxnumber_cross*self.number)
        self.cromosomes = np.vstack((self.cromosomes, self.cromosomes))

        for i in range(n):
            to_cross =  random.sample(range(self.length), k=math.ceil(alpha_cross*self.length))

            for gene in to_cross:
                self.cromosomes[self.length+i][gene] = self.cromosomes[i+1][gene]


    def evolve(self, reverse=True):
        '''
        sorts the cromosome w.r.t. their fitness function and returns first best NumberOfCromosomes.

        * reverse is whether to put the genes in ascending(False) or descending(True) order.
        '''
        self.cromosomes = sorted(self.cromosomes, key=self.FitnessFunction, reverse=reverse)
        self.cromosomes = self.cromosomes[:self.number][:]


    def run(self, threshold, iterations=1000, epsilon=0, maxnumber_mutation=0.45, 
            alpha_mutation=1, type='uniform', maxnumber_cross=0.45, alpha_cross=0.45, reverse=True):

        '''
        * runs a simple optimiser. needs to give a threshold that is indicator of minimum fitness value we want
         and also when to halt the optimiser 

        * it can be customised .

        * if we want it we can define our own optimiser using mutation, crossover and evolve mathods over an instance.
        '''    

        thres = threshold
        iterations = iterations
        i = 0

        self.initialise()
        self.evolve(reverse)
            
        if reverse==True:  

            while epsilon < thres:
                self.crossover(type='uniform', maxnumber_cross=0.45, alpha_cross=0.45)
                    
                self.mutation(maxnumber_mutation=0.45, alpha_mutation=1)
                self.evolve(reverse)
                    
                best_crom = self.cromosomes[0]
                epsilon = self.FitnessFunction(best_crom)
                
                if i >= iterations:
                    break
                i += 1

        if reverse==False:  

            while epsilon > thres:

                self.crossover(type='uniform', maxnumber_cross=0.45, alpha_cross=0.45)
                    
                self.mutation(maxnumber_mutation=0.45, alpha_mutation=1)
                self.evolve(reverse)
                    
                best_crom = self.cromosomes[0]
                epsilon = self.FitnessFunction(best_crom)

                if i >= iterations:
                    break
                i += 1
        
        print(f'best cromosomes found is {best_crom},  with {epsilon} fitness value and in {i} iterations')
#%%
if __name__ == "__main__":
    own_croms = np.array([[1, 0], [0, 0]])

    # croms = GeneticOptimiser(number_of_cromosomes=4, length_of_cromosomes=3, lowest_gene_value=0,
    #     highest_gene_value=2, step_size=1, fitness_function=(lambda x:x [1]))

    constraint = (lambda x: x[0]-x[1]+x[2]-x[3]+x[4]-x[5]+x[6]-x[7]+x[8]-x[9])

    '''
    we define Genetic Optimiser as
    # '''
    croms = GeneticOptimiser(number_of_cromosomes=100, length_of_cromosomes=10, lowest_gene_value=0,
                    highest_gene_value=9, step_size=1, feed_cromosomes=False, fitness_function=constraint)

    '''now running the optimiser'''
    croms.run(threshold=45)
