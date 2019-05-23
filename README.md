# Artificial-Intelligence

AI Algorithms

--Installation

clone the this reporitory using,

`https://github.com/abecus/Artificial-Intelligence.git`

## 1. searches

seaerches is a python script use to apply searches on graphs, trees or search spaces.
It currently having Breath First Search and Depth First Search, Other type of searches will be updated soon.

--Installation
searches runs on python 3.x. You can install the python requirements with
`python3 -m pip install pandas`.

### Using Searches

    search(starting_node, end_node, search_type='bfs', extended_nodes=True, max_depth=50, beam=2, verbose=0)

starting_node: start node in graph.

end_node: end node in graph.

search_type: choose any from:-

* dfs (Depth First Search)
* bfs (Breadth First Search)
* hill_climbing (Hill Climbing Algorithm)
* beam (Beam Search)
* best_first (Best First Search)
* branch_and_bound (Branch and Bound Search Algorithm)
* A* (A* Search Algoritm).

extended_nodes: Whether to use extended nodes or not (True and False). If True, No nodes wich has been explored will be explored again. "must use in case of dfs".

max_depth: maximum depth that is allowed to explore in a graph (For Safty Reasons).if it exceeds, searches will be halted.

beam: set value of beam width 'W' in case of Beam Search.

verbose: choose from 0, 1 and 2 from no to every detail of search steps used in an particular Algorithm of a given Graph.

Try running the following:

    s =  Searches('...\searches\paths.csv')
    print(s.paths)
    s.search('s', 'g', search_type='A*', depth=10, extended_nodes=True, verbose=2)
    print(s.path_to_goal())
    print(s.depth)
    print(s.cost)

--Search Spaces Formate:--

search space formate must be in formate:

`From, To, Length or Cost, Admissible Huristic`

first, second, third and fourth columns consists the nodes 'From' at where an edge starts, 'To' at where an edge ends, cost or length of an edge and huristics(admissible cost of 'To' node to the 'Goal' node) respectively.

### Output

        from to  length  adm_huristic
        0    s  a       3           7.5
        1    s  b       5           6.0
        2    a  d       3           5.0
        3    a  b       4           6.0
        4    b  c       4           7.5
        5    d  g       5           0.0
        6    c  e       6           4.0

        starting search with A*
        initiallising...  a
        extended nodes being captured...
        extended nodes set()
        extending...  a
        traversing over [[0, 0, 0, 'a']]
        updating... b
        updating... c

        extended nodes {'a'}
        extending...  b
        traversing over [[1.0, 2.0, 1.0, 'a', 'b'], [9.0, 13.0, 4.0, 'a', 'c']]
        updating... a
        updating... c
        updating... d

        extended nodes {'a', 'b'}
        extending...  c
        traversing over [[9.0, 11.0, 2.0, 'a', 'b', 'c'], [9.0, 13.0, 4.0, 'a', 'c'], [10.0, 12.0, 2.0, 'a', 'b', 'a'], [7.0, 13.0, 6.0, 'a', 'b', 'd']]
        updating... a
        updating... b
        updating... d

        extended nodes {'a', 'c', 'b'}
        extending...  d
        traversing over [[7.0, 13.0, 6.0, 'a', 'b', 'd'], [10.0, 16.0, 6.0, 'a', 'b', 'c', 'a'], [7.0, 12.0, 5.0, 'a', 'b', 'c', 'd']]
        updating... b
        updating... c
        updating... e
        updating... f
        updating... g

        extended nodes {'d', 'a', 'c', 'b'}
        extending...  e
        traversing over [[1.5, 15.5, 14.0, 'a', 'b', 'd', 'e'], [4.5, 13.5, 9.0, 'a', 'b', 'd', 'f'], [0.0, 15.0, 15.0, 'a', 'b', 'd', 'g']]
        updating... d
        updating... g

        extended nodes {'c', 'a', 'b', 'd', 'e'}
        extending...  f
        traversing over [[4.5, 13.5, 9.0, 'a', 'b', 'd', 'f'], [0.0, 15.0, 15.0, 'a', 'b', 'd', 'g'], [7.0, 29.0, 22.0, 'a', 'b', 'd', 'e', 'd'], [0.0, 16.0, 16.0, 'a', 'b', 'd', 'e', 'g']]
        updating... d
        updating... g

        Goal Has Been Found!
        ['a', '-->', 'b', '-->', 'd', '-->', 'f', '-->', 'g']
        4
        14.0


## 2. GeneticOptimiser

Genetic Optimiser is python script which try to mimic the natural evolution (that, survival of the fittest).
It currently provides some customisable methods like mutation, crossover and evolution types.

--Installation
searches runs on python 3.x. You can install the python requirements with
`python3 -m pip install numpy`.

### Using GeneticOptimiser

    GeneticOptimiser(number_of_cromosomes=50, length_of_cromosomes=10,lowest_gene_value=0, highest_gene_value=9, step_size=1, feed_cromosomes=False, fitness_function=constraint)

number_of_cromosomes: (integer) it corresponds to initial minimum and maximum no of cromosomes (population).

length_of_cromosomes: (integer) length of crommosomes.

lowest_gene_value: (integer) a lowest value that any gene may have.

highest_gene_value: (integer) a highest value that any gene may have.

step_size: (float) it corresponds to the granuality in gene values or simply minimum difference in any two gene.

feed_cromosoms: True to feed your own custom cromosomes values directly in initialise method, else it is False.

fitness_function: A lamba function for applying costraint (since i has been proved that lamda-function is complete under general function, we can always find one lamda-function thatv corresponds to a general function).

Try running the following:

    croms = GeneticOptimiser(number_of_cromosomes=5, length_of_cromosomes=4, lowest_gene_value=0, highest_gene_value=1, step_size=1, fitness_function=(lambda x: x[0]-x[1]+x[2]-x[3]))

    croms.run(threshold=2)

### Output

iterations may vary (obviously).

        best cromosomes found is [1 0 1 0],  with 2 fitness value and in 2 iterations
        
To feed your own cromosomes:

        own_croms = np.array([[1, 0], [0, 0]])

        croms = GeneticOptimiser(number_of_cromosomes=2, length_of_cromosomes=2, lowest_gene_value=0, highest_gene_value=1, step_size=1, feed_cromosomes=True, fitness_function=(lambda x: x[0]-x[1]))

        croms.initialise(manual_cromosomes=own_croms)
        
        print(croms.cromosomes)

An example might be useful to get use to it, which can be found in 'example1.py' file (read that file).
