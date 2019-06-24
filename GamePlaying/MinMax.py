#%%
import numpy as np
import math


#%%
class MinMax():

    def __init__(self, leaf_values, tree_matrix, *args, **kwargs):
        '''
        * tree_matrix : no of rows corresponds to the depths of the game-tree and elements corresponds
        to the branching of the nodes at that depth (eg. 3 can be seen as at that node 3 branches are comming)

        * leaf_values : values (scores) at leaf
        '''
        self.leaf_values = leaf_values
        self.tree_matrix = tree_matrix
        self.depth = self.tree_matrix.shape[0] 


    @staticmethod
    def compute_row(turn, values, row_of_tree_matrix):

        '''
        computes values a level down in the tree towards root
        '''

        val = np.array([])
        # ab_tracker = np.array([])
        track = 0

        if turn == 'min':

            for branching in row_of_tree_matrix:
                if branching != 0: 

                    val = np.hstack((val, np.amin(values[track:track+branching])))
                    track += branching

                else:   continue


        if turn == 'max':
            for branching in row_of_tree_matrix:
                if branching != 0: 

                    val = np.hstack((val, np.amax(values[track:track+branching])))
                    track += branching       

                else:   continue

        return val


    def run(self):

        '''
        runs the function compute_row on all tree
        '''

        turn = 'min' if self.depth % 2 == 0  else 'max'
        value = self.leaf_values

        for i in range(self.depth):

            value = self.compute_row(turn=turn, values=value, row_of_tree_matrix=self.tree_matrix[self.depth - i - 1])
            turn = 'max' if turn == 'min' else 'min'

        print(value)
        

    
#%%
if __name__ == "__main__":
    
    leaf = np.array([2, 3, 4, 5, 6, 7])
    tree = np.array([[3, 0, 0],
            [2, 1, 3]])

    game = MinMax(leaf, tree)
    game.run()