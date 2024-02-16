import networkx as nx
import numpy as np

def remove_component(graph, component):
    """
    Remove a specified component from the graph.

    Args:
        graph (numpy.ndarray): A numpy array representing the adjacency matrix of the graph.
        component (list): A list of nodes representing the component to be removed.

    Returns:
        numpy.ndarray: The modified graph with the specified component removed.
    """
    for node in component:
        graph[node, :] = 0
        graph[:, node] = 0
    return graph


def dfs_component(graph, node, visited):
    """
    Perform a depth-first search to find all nodes in the connected component of a given node.

    Args:
        graph (numpy.ndarray): A numpy array representing the adjacency matrix of the graph.
        node (int): The node from which the DFS starts.
        visited (list of bool): A list indicating whether each node has been visited.

    Returns:
        list: A list of nodes that are in the same connected component as the starting node.
    """
    component = []
    stack = [node]
    
    while stack:
        current_node = stack.pop()
        if not visited[current_node]:
            visited[current_node] = True
            component.append(current_node)
            for neighbor in range(len(graph[current_node])):
                if graph[current_node][neighbor] == 1 and not visited[neighbor]:
                    stack.append(neighbor)
    return component


def graph_anneal(graph):
    """
    Modify the graph by removing small connected components and updating the adjacency matrix.

    Args:
        graph (numpy.ndarray): A numpy array representing the adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The modified graph adjacency matrix with smaller components removed.
    """
    num_nodes = len(graph)
    visited = [False] * num_nodes

    for i in range(num_nodes):
        if not visited[i]:
            component = dfs_component(graph, i, visited)
            if len(component) < 10:
                graph = remove_component(graph, component)

    # Update graph to include only nodes in larger components
    mask = np.any(graph, axis=1)
    graph = graph[mask][:, mask]

    return graph


def remove_edge(graph, v, u):
    """
    Remove an edge from the graph by setting its weight to zero.

    Args:
        graph (networkx.Graph): A NetworkX graph object.
        v (int): The first node in the edge.
        u (int): The second node in the edge.

    Returns:
        numpy.ndarray: The modified graph with the specified edge removed.
    """
    graph[v][u] = 0
    graph[u][v] = 0
    return graph


def break_edges_keep_largest_circle(graph, adj, max_cycle_length=80):
    """
    Break edges to keep the largest circle in the graph under the specified maximum length.

    Args:
        graph (networkx.Graph): A NetworkX graph object.
        adj (numpy.ndarray): A numpy array representing the adjacency matrix of the graph.
        max_cycle_length (int, optional): The maximum length of the cycle to consider. 
                                          Default is 80.

    Returns:
        networkx.Graph: A modified NetworkX graph with edges broken to ensure 
                        all cycles are below the specified maximum length.
    """
    while True:
        cycles = sorted([cycle for cycle in nx.cycle_basis(graph) if len(cycle) < max_cycle_length], key=len)
        
        if len(cycles) < 1:
            break
            
        cycle = cycles[0]
        min_weight = float('inf')
        edge_to_remove = None
        
        for i in range(len(cycle)):
            node1 = cycle[i]
            node2 = cycle[(i+1) % len(cycle)]
            
            edge_weight = adj[node1][node2]
            
            if edge_weight < min_weight:
                min_weight = edge_weight
                edge_to_remove = (node1, node2)
        
        if edge_to_remove:
            graph.remove_edge(*edge_to_remove)
            
    return graph


def keep_largest_connected_component(graph):
    """
    Retain only the largest connected component of the given graph.

    Args:
        graph (networkx.Graph): A NetworkX graph object.

    Returns:
        networkx.Graph: A NetworkX graph object representing the largest connected
                        component of the input graph.
    """
    connected_components = list(nx.connected_components(graph))

    if len(connected_components) == 0:
        return graph
    largest_component = max(connected_components, key=len)
    largest_component_graph = graph.subgraph(largest_component).copy()

    return largest_component_graph


def graph_anneal_break_largest_circle(adj):
    """
    Perform graph annealing and retain the largest circle in the graph.

    Args:
        adj (numpy.ndarray): A numpy array representing the adjacency matrix of the graph.

    Returns:
        networkx.Graph: A NetworkX graph object representing the largest circle 
                        retained after graph annealing and processing.
    """
    a = graph_anneal(np.round(adj))
    G = nx.from_numpy_array(a)
    G = break_edges_keep_largest_circle(G, adj)
    G = keep_largest_connected_component(G)
    return G
