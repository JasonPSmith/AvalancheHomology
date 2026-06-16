import numpy as np
import itertools
import gudhi
import matplotlib.pyplot as plt


def out_degrees_matrix(M,sinks):
    """
    Compute the out-degree of each vertex in a graph.

    The out-degree is computed as the row sum of the adjacency matrix.
    Vertices listed in ``sinks`` are treated as having an additional
    outgoing edge to the sink vertex.

    :param M: Adjacency matrix of the graph as dense NumPy array.
    :type M: numpy.ndarray

    :param sinks: Indices of vertices connected to the sink.
    :type sinks: list[int]

    :returns: Array whose ``i``-th entry is the out-degree of vertex ``i``.
    :rtype: numpy.ndarray
    """
    out_degs = np.sum(M,axis=1)
    # if M is a sparse matrix, then np.sum returns a column as a np.matrix, so we must apply the following
    if isinstance(out_degs,np.matrix):
        out_degs = np.array(out_degs.transpose())[0]
    return (out_degs+[i in sinks for i in range(len(M))]).astype(int)



def Laplacian(M,out_deg):
    """
    Compute the out-degree Laplacian matrix of a graph.

    The Laplacian is defined as

    .. math::

        L = D - A

    where ``D`` is the diagonal matrix of out-degrees and ``A`` is the
    adjacency matrix.

    :param M: Adjacency matrix of the graph.
    :type M: numpy.ndarray

    :param out_deg: Array of vertex out-degrees.
    :type out_deg: numpy.ndarray

    :returns: Laplacian matrix of the graph.
    :rtype: numpy.ndarray
    """
    return (np.diag(out_deg)-np.array(M)).astype('int32')



def is_stable(out_deg,configuration):
    """
    Determine whether a sandpile configuration is stable.

    A configuration is stable if every vertex contains fewer grains than
    its out-degree.

    :param out_deg: Array of vertex out-degrees.
    :type out_deg: numpy.ndarray

    :param configuration: Current sandpile configuration.
    :type configuration: numpy.ndarray

    :returns: ``True`` if the configuration is stable, otherwise ``False``.
    :rtype: bool
    """
    return (configuration < out_deg).all()




def run_avalanche_sandpile(L,out_deg,configuration,avalanches,display=False):
    """
    Simulate the sandpile model using parallel toppling.

    The simulation proceeds by repeatedly toppling all unstable vertices
    simultaneously until either the requested number of avalanches has
    occurred or the configuration stabilises.

    :param L: Laplacian matrix of the graph.
    :type L: numpy.ndarray

    :param out_deg: Array of vertex out-degrees.
    :type out_deg: numpy.ndarray

    :param configuration: Initial sandpile configuration.
    :type configuration: numpy.ndarray

    :param avalanches: Maximum number of avalanche topplings to perform.
    :type avalanches: int

    :param display: If ``True``, print configurations after each toppling.
    :type display: bool

    :returns:
        A tuple ``(configs, avalanche_firing_sequence)`` where:

        - ``configs`` is the list of configurations encountered.
        - ``avalanche_firing_sequence`` records which vertices fired at
          each avalanche step as a binary vector of length |G|, 
          where position i is 1 iff vertex i topples.
    :rtype: tuple[list, list]
    """
    if display:
        print("Start configuration: ", configuration)

    fired = 0
    configs = [configuration]
    avalanche_firing_sequence = []

    while fired < avalanches:
        if is_stable(out_deg,configuration):
            if display:
                print("Stable configuration: ",configuration)
            return configs,avalanche_firing_sequence
        else:
            ready_to_fire = (configuration >= out_deg).astype('int32')
            configuration = configuration - np.matmul(np.transpose(L),ready_to_fire)
            avalanche_firing_sequence.append(tuple(ready_to_fire))
            configs.append(tuple(configuration))
            fired += 1
 
            if display:
                print(f"Configuration after {fired}. avalanche: ",configuration)
    
    return configs,avalanche_firing_sequence



def avalanche_complex(M, config, sinks=[], firings=1000, print_firings=False, persistence=False, return_simplices=False):
    """
    Construct the nerve complex of the avalanche complex.

    :param M: Adjacency matrix of the graph.
    :type M: numpy.ndarray

    :param config: Initial sandpile configuration.
    :type config: list[int] or numpy.ndarray

    :param sinks: Vertices connected to the sink.
    :type sinks: list[int]

    :param firings: Maximum number of firing steps to simulate.
    :type firings: int

    :param print_firings: If ``True``, print firing sets during simulation.
    :type print_firings: bool

    :param persistence:
        If ``True``, assign filtration values and compute persistent homology.
    :type persistence: bool

    :param return_simplices:
        If ``True``, also return a dictionary mapping simplex to firing sets.
    :type return_simplices: bool

    :returns:
        Either:

        - a ``gudhi.SimplexTree`` object, or
        - ``(SimplexTree, simplex_dictionary)`` if
          ``return_simplices=True``.
    :rtype: gudhi.SimplexTree or tuple
    """
    out_deg = out_degrees_matrix(M,sinks)  
    L = Laplacian(M,out_deg)

    seqs = set()
    confs = set()
    current_config = tuple(config)
    configs = []
    firing_sets = []
    simplices = []

    go = True
    while go:
        ready_to_fire = (current_config >= out_deg).astype('int32')
        current_config = current_config - np.matmul(np.transpose(L),ready_to_fire)
        current_firing_set = tuple(ready_to_fire)

        if len(ready_to_fire) == 0:
            break
        
        if tuple(current_config) in configs or is_stable(out_deg,current_config) or len(configs)>firings:
            go = False

        if print_firings:
            print("t =",len(configs)," : ",list(np.where(current_firing_set)[0]))

        configs.append(tuple(current_config))

        subset_existing_facet = False
        for f in firing_sets:
            if -1 not in np.subtract(f[0],current_firing_set):
                subset_existing_facet = True
                break
        
        num_simps = len(simplices)
        if not subset_existing_facet:
            # add the current firing set with filtration value current timestep
            firing_sets.append((current_firing_set,len(configs)-1))
            added_vertex = False
            for i in range(num_simps):
                # for each current simplex, check if add new vertex of current_firing_set makes a dim+1 simplex
                # if it does add the new firing set to that simplex
                # otherwise check is all subsets of that simplex
                if sum(np.logical_and.reduce(np.array(simplices[i]+[current_firing_set]), axis=0)) > 0:
                    simplices[i] = simplices[i]+[current_firing_set]
                    added_vertex = True
                else:
                    new_simplices = []
                    for r in range(len(simplices[i])-1,0,-1):
                        for idxs in itertools.combinations(range(len(simplices[i])), r):
                            if not any(set(idxs).issubset(t) for t in new_simplices):
                                if sum(np.logical_and.reduce(np.array([simplices[i][x] for x in idxs]+[current_firing_set]), axis=0)) > 0:
                                    new_simplices.append(set(idxs))
                    for s in new_simplices:
                        simplices.append([simplices[i][x] for x in s]+[current_firing_set])
                    if len(new_simplices) > 0:
                        added_vertex = True
            if not added_vertex:
                simplices.append([current_firing_set])

    #dictionary that convert the tuple of firing_sets to an int
    simplex_to_int = {firing_sets[i][0]:i for i in range(len(firing_sets))}
    int_to_simplex = {a:frozenset(np.where(b)[0]) for b,a in simplex_to_int.items()}
    #initialise the vertices
    S = gudhi.SimplexTree()
    for f in firing_sets:
        if persistence:
            _ = S.insert([simplex_to_int[f[0]]],f[1])
        else:
            _ = S.insert([simplex_to_int[f[0]]])


    #Add the simplices
    for f in simplices:
        if len(f) > 1:
            _ = S.insert([simplex_to_int[i] for i in f])

    if persistence:  
        for f in S.get_simplices():
            filt_value = max([firing_sets[i][1] for i in f[0]])
            _ = S.assign_filtration(f[0],filt_value)

    S.compute_persistence(persistence_dim_max=True)
    #return S.betti_numbers()
    if return_simplices:
        return S, int_to_simplex
    else:
        return S

# Computes the avalanche complex homology
#Input:
#    M: adjacency matrix, as a numpy array
#    config: initial configuration, as a list of the same length as number of vertices
#    sinks: list of vertices which have an out edge to the sink, can be left empty for graphs with no sink
#    steps: number of steps to run sandpile before checking if recurrent
#    firings: total number of topplings todo before terminating if not stabilised or recurrent
#    persistence: Boolean, if true adds the filtration value 
#Output:
#    A SimplexTree
def avalanche_complex_original(M, config, sinks=[], steps=100, firings=1000, print_firings=False, persistence=False):
    """
    Construct the avalanche complex directly (as opposed to the nerve complex used above)

    This approach is significantly slower than the nerve approach implemented in avalanche_complex above

    :param M: Adjacency matrix of the graph.
    :type M: numpy.ndarray

    :param config: Initial sandpile configuration.
    :type config: list[int] or numpy.ndarray

    :param sinks: Vertices connected to the sink.
    :type sinks: list[int]

    :param steps:
        Number of avalanche steps to simulate before recurrence checks.
    :type steps: int

    :param firings:
        Maximum total number of firings allowed.
    :type firings: int

    :param print_firings:
        If ``True``, print firing sets and timestamps.
    :type print_firings: bool

    :param persistence:
        If ``True``, assign filtration values for persistent homology.
    :type persistence: bool

    :returns: The resulting avalanche complex.
    :rtype: gudhi.SimplexTree
    """    
    out_deg = out_degrees_matrix(M,sinks) 
    L = Laplacian(M,out_deg)

    seqs = set()
    confs = set()
    total_firings = 0
    S = gudhi.SimplexTree()

    go = True
    while go:
        conf, seq = run_avalanche_sandpile(L,out_deg,config,steps,display=False)
        for i in range(len(seq)):
            if print_firings:
                print(list(np.where(seq[i])[0]),total_firings+i)
            if persistence:
                _ = S.insert(list(np.where(seq[i])[0]),filtration=total_firings+i)
            else:
                _ = S.insert(list(np.where(seq[i])[0]))

        topplings = len(conf)
        seq = set(seq)
        conf={tuple(i) for i in conf}
        total_firings += steps
        #conditions check if sandpile has stabilised by a configuration already occurring (bool 1 & 2)
        # or become recurrent (bool 3), or reached max firing (bool 4)
        if len(seq & seqs) > 0 or topplings > len(conf) or topplings < steps or total_firings > firings:
            go = False
        seqs.update(seq)
        conf.update(conf)

    S.compute_persistence(persistence_dim_max=True)
    #return S.betti_numbers()
    return S


def avalanche_complex_persistence_diagram(S, title=''):
    """ 
    Plot the persistence diagram of an avalanche complex. 

    :param S: Simplicial complex represented as a ``gudhi.SimplexTree``. 
    :type S: gudhi.SimplexTree 

    :param title: Title for the persistence diagram plot. 
    :type title: str 

    :returns: None 
    :rtype: None 
    """
    ax = gudhi.plot_persistence_diagram(S.persistence(persistence_dim_max=True))
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.show()