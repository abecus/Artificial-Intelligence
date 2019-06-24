#%%
import os
import pandas as pd

#%%
class Searches():
    
    def __init__(self, paths_csv_file):
        # reads paths as data frame and initialising the extended node list
        columns=['from', 'to', 'length', 'adm_huristic']
        self.paths = pd.read_csv(paths_csv_file, delimiter=',', names=columns)
        self.extended_nodes_set = set()
        
    
    def path_to_goal(self):
        # for printing the searched path asthetically... (hope so....)
        path = []

        try:
            ans_len = len(self.ans_path)
            if self.ans_path[2] == 0:   self.cost = 'NaN'
            else:   self.cost = self.ans_path[2]       

            for i in range(3, ans_len):
                if i == ans_len-1:
                    path.append(self.ans_path[i])
                else:
                    path.append(self.ans_path[i])
                    path.append('-->')
            return path
        except:    pass                
    
    
    def initialise(self, starting_node, end_node, verbose):
        self.start = starting_node
        self.goal = end_node

        # keeping nodes to discover
        self.queue = [[0,0,0]]        
        
        # checking, if start and goal is in the paths
        if self.start in self.paths['from'].values and self.goal in self.paths['to'].values:
                self.queue[0].append(self.start)

                if verbose > 0:    print('initiallising... ', self.start)
        else:
            raise KeyError('check start and end nodes')

    
    def extend(self, current_node, extended_nodes=True):
        # extendes the node given as current node after looking into the paths data frame
        temp_ext_list = self.paths.where(self.paths['from'] == current_node)
        temp_ext_list =  temp_ext_list.dropna(subset=['to'])
        temp_ext_list = temp_ext_list.drop_duplicates(subset=['to'])

        # checking if current node has been explored before and if so, removing them
        if extended_nodes:
            temp_ext_list = temp_ext_list[temp_ext_list != self.extended_nodes_set]

        return temp_ext_list


    def enqeueing(self, search_type, current_node, temp_ext_list, verbose):
        c = 0
        '''for hill climbing search creating new list for shorting the nodes 
        found after extending the parent node, sorting them and going in dfs manner'''
        if search_type == 'hill_climbing':  temp_queue = []

        if verbose > 0:
            print('extending... ', current_node)
            if verbose > 1:    
                print('traversing over',self.queue)
            

        # removing first node from queue for futher enqeueing
        popped = self.queue.pop(0)

        ''' making new lists and appending new nodes which has to be explored 
        depending upon the type of search '''
        def enq(search_type, current_node, temp_ext_list):

                y = popped[:]
                y.append(to)

                if not(search_type=='bfs' or search_type=='dfs'):
                    try:    
                        y[2] += (temp_ext_list['length'].where(temp_ext_list['to'] == to).dropna().values[0])         
                        y[0] = (temp_ext_list['adm_huristic'].where(temp_ext_list['to'] == to).dropna().values[0])
                    except:     pass

                    if search_type == 'A*':
                        try:    y[1] = y[0]+y[2]
                        except:     pass

                return y
        
        for to in temp_ext_list['to'].values:
            c += 1
            if verbose > 0:    print('updating...', to)

            if c == 1:
                y = enq(search_type, current_node, temp_ext_list)
                if search_type == 'dfs':    self.queue.insert(0, y)
                elif search_type == 'hill_climbing':    temp_queue.append(y)
                else:   self.queue.append(y)

            else:
                y = enq(search_type, current_node, temp_ext_list)
                if search_type == 'dfs':    self.queue.insert(0, y)
                elif search_type == 'hill_climbing':    temp_queue.append(y)
                else:   self.queue.append(y)
        
        if search_type == 'hill_climbing':
            temp_queue = sorted(temp_queue, key=(lambda x: x[0]), reverse=True)
            for i in temp_queue:
                self.queue.insert(0, i)
        
        print('')

        return self.queue


    def run(self, starting_node, end_node, search_type='bfs', extended_nodes=True, max_depth=50, beam=2, verbose=0):
        
        if verbose > 0:    print(f'starting search with {search_type}')
        self.initialise(starting_node, end_node, verbose)

        if extended_nodes == True:  
            if verbose > 1:    print('extended nodes being captured...')
        
        current = self.start
        found = False
        self.depth = 0

        while not found and self.depth < max_depth:

            # checking if condition for extended list is True
            if extended_nodes == True:
                
                '''checking if current node in extended nodes list, if so
                removing the node from queue list and not to be explored
                again and choosing the next node as the current node'''

                if current in self.extended_nodes_set:
                    self.queue.pop(0)
                    current = self.queue[0][-1]
                    continue

            if extended_nodes == True: 
                if verbose > 1:       
                    print('extended nodes', self.extended_nodes_set)

            # extending current node
            temp_ext_list = self.extend(current, extended_nodes)

            # updating enquing list depending on search type 
            if search_type == 'best_first':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)
                min_list = min(self.queue, key=(lambda x: x[0]))
                self.queue.remove(min_list);    
                self.queue.insert(0, min_list)

            elif search_type == 'hill_climbing':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)

            elif search_type == 'bfs':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)
            
            elif search_type == 'beam':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)
                try:    self.queue = sorted(self.queue, key=(lambda x: x[0]))[:beam]
                except:     pass

            elif search_type == 'dfs':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)

            elif search_type == 'A*':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)
                min_list = min(self.queue, key=(lambda x: x[1]))
                self.queue.remove(min_list)
                self.queue.insert(0, min_list)

            elif search_type == 'branch_and_bound':
                self.queue = self.enqeueing(search_type, current, temp_ext_list, verbose)
                min_list = min(self.queue, key=(lambda x: x[2]))
                self.queue.remove(min_list)
                self.queue.insert(0, min_list)

            ''' 
                checking if goal node is found
            '''
            if search_type== 'branch_and_bound' or search_type=='A*':
                if self.goal == self.queue[0][-1]:
                        self.ans_path = self.queue[0]
                        found = True
                        if verbose > 0:    print('Goal Has Been Found!')
                        return  

            else:
                for i in range(len(self.queue)):
                    if self.goal == self.queue[i][-1]:
                        self.ans_path = self.queue[i]
                        found = True
                        if verbose > 0:    print('Goal Has Been Found!')
                        return                                      

            # adding current node in the extended nodes list and updating it 
            self.extended_nodes_set.add(current) 
            current = self.queue[0][-1]
            self.depth = len(self.queue[0])-3

        else:
            print('No path to goal has been Found, check whether depths is sufficient and try after increasing it')

#%%
if __name__ == "__main__":
        # search_types = ['dfs', 'bfs', 'hill_climbing', 'beam', 'best_first','branch_and_bound', 'A*']
        # verbose = [0, 1, 2]
        my_path = os.getcwd()
        path = os.path.join(my_path, 'new_path.csv')
        s = Searches(path)
        # print(s.paths)

        s.run('a','g', search_type='A*', max_depth=10, extended_nodes=True, verbose=2)
        # s.search('e', 'g', search_type='bfs')
        print(s.path_to_goal())
        print(s.depth)
        print(s.cost)
        