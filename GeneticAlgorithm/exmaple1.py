''' 
1. An example of genetic algorithm can be finding the array of integers from 0 to 9 having
maximum sum under some constraints.

Lets define a constraint that, array's every secong element reduces the sum and every odd 
adds to the sum and say array is of size 10 (no of elements).
It has each element from 0 to 9 and say, they are intigers
'''
#%%
from geneticAlgorithm.geneticOptimiser import GeneticOptimiser
import numpy as np

#%%
# solution 1.
'''
under given constarint, fitness function can be given as
'''
constraint = (lambda x: x[0]-x[1]+x[2]-x[3]+x[4]-x[5]+x[6]-x[7]+x[8]-x[9])

'''
we define Genetic Optimiser as
'''
croms = GeneticOptimiser(number_of_cromosomes=50, length_of_cromosomes=10, lowest_gene_value=0,
                highest_gene_value=9, step_size=1, feed_cromosomes=False, fitness_function=constraint)

'''now running the optimiser'''
croms.run(threshold=45)


# output comes out to be
'''
best cromosomes found is [9 0 9 0 9 0 9 0 9 0],  with 45 fitness value and in 64 iterations
'''
#%%
''' 
for finding minimun value in that array
'''
croms.run(threshold=-44, reverse=False, epsilon=100)
