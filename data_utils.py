import pickle
import numpy as np
import networkx as nx

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

def get_desc(G):
    """
    Compute various network measures for a given networkx graph.

    Args:
        G (networkx.Graph): The input graph for which network measures will be computed.

    Returns:
        numpy.ndarray: An array containing the following network measures in order:
            1. Number of nodes
            2. Number of edges
            3. Algebraic connectivity
            4. Diameter
            5. Radius
            6. Average degree
            7. Average neighbor degree
            8. Network density
            9. Mean degree centrality
            10. Mean betweenness centrality
            11. Degree assortativity coefficient
    """
    x1      = nx.number_of_nodes(G)
    x2      = nx.number_of_edges(G)
    x3      = nx.algebraic_connectivity(G)
    x4      = nx.diameter(G)
    x5      = nx.radius(G)
    degrees = [degree for _, degree in G.degree()]
    x6      = sum(degrees) / len(G.nodes())
    x7      = np.mean(list(nx.average_neighbor_degree(G).values()))
    x8      = nx.density(G)
    x9      = np.mean(list(nx.degree_centrality(G).values()))
    x10     = np.mean(list(nx.betweenness_centrality(G).values()))
    x11     = nx.degree_assortativity_coefficient(G)
    
    return np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])

def load_data(data_dir, fold, n_fold=5, if_validation=False, verbose=False):
    """
    Load and preprocess data from the specified directory.

    Args:
        data_dir (str): Directory path where the data is stored.
        fold (int): Index of the fold to be used as the test set.
        n_fold (int, optional): Number of folds to split the data into. Default is 5.
        if_validation (bool, optional): Whether to include a validation set. Default is False.

    Returns:
        tuple: Tuple containing training, validation (optional), and test datasets,
               along with topo_class names, scaler, and label encoder.
    """
    
    with open(data_dir, 'rb') as handle:
        x, y, topo_desc, topo_class, poly_param, graph = [pickle.load(handle) for _ in range(6)]
    
    # x:            graph feature
    # y:            rg2 value
    # topo_desc:    topological descriptors
    # topo_class:   topology classes
    # poly_param:   polymer generation parameters
    # graph:        networkx objects
    
    # preprocessing
    y = y[..., 0]
    
    SCALER    = StandardScaler()
    topo_desc = SCALER.fit_transform(topo_desc)

    topo_class[topo_class == 'astar'] = 'star'
    topo_desc = np.where(np.isnan(topo_desc), -2, topo_desc) # only node assortativity has 0, should be [-1, 1]

    le         = LabelEncoder()
    topo_class = le.fit_transform(topo_class)
    NAMES      = le.classes_
    
    # random shuffle
    x          = np.random.RandomState(0).permutation(x)
    y          = np.random.RandomState(0).permutation(y)
    topo_class = np.random.RandomState(0).permutation(topo_class)
    topo_desc  = np.random.RandomState(0).permutation(topo_desc)
    poly_param = np.random.RandomState(0).permutation(poly_param)
    graph      = np.random.RandomState(0).permutation(graph)

    # use one fold for testing
    skf   = StratifiedKFold(n_splits=n_fold)
    count = -1
    for _, (train_idx, test_idx) in enumerate(skf.split(x, topo_class)):
        datasets = [x, y, topo_desc, topo_class, graph]
        train_data = [data[train_idx] for data in datasets]
        test_data = [data[test_idx] for data in datasets]
        
        x_train, y_train, l_train, c_train, graph_train = train_data
        x_test, y_test, l_test, c_test, graph_test = test_data

        if if_validation:
            skf2              = StratifiedKFold(n_splits=n_fold)
            train_idx2, valid_idx = next(iter(skf2.split(x_train, c_train)))
            datasets2         = [x_train, y_train, l_train, c_train, graph_train]
            
            x_valid, y_valid, l_valid, c_valid, graph_valid = ([data[valid_idx] for data in datasets2])
            x_train, y_train, l_train, c_train, graph_train = ([data[train_idx2] for data in datasets2])
                
        count += 1
        if count == fold:
            break

    if if_validation:
        if verbose:
            print(f'Train: {len(x_train)} Valid: {len(x_valid)} Test: {len(x_test)}')
        return ((x_train, y_train, c_train, l_train, graph_train),
                (x_valid, y_valid, c_valid, l_valid, graph_valid),
                (x_test, y_test, c_test, l_test, graph_test),
                NAMES, SCALER, le)
            
    else:
        if verbose:
            print(f'Train: {len(x_train)} Test: {len(x_test)}')
        return ((x_train, y_train, c_train, l_train, graph_train),
                (x_test, y_test, c_test, l_test, graph_test),
                NAMES, SCALER, le)
