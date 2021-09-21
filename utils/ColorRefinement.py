"""
    Author: Parker Burchett
"""


import numpy as np
import pandas as pd
import networkx as nx
import copy

"""

This is an implementation of the Color Refinement algorithm 
(also known as 1-dimensional Weisfeiler-Leman algorithm) that initializes node colors as a hash of node attributes.

Node coloring contains information about both graph structure and node attributes. It does not contain information about edge attributes.

Primarily based on this lecture:
       Stanford Online, Professor Jure Leskovec
       CS224W: Machine Learning with Graphs | 2021 | Lecture 2.3 - Traditional Feature-based Methods: Graph
       https://www.youtube.com/watch?v=buzsHTa4Hgs&t=701s
 
 
Embed a graph G into a bag of colors of size K based on node attributes and local neighborhoods.
There is a (I*N)/K probability of a spurious hash collision where I is the number of iterations, N is the number of nodes, and K is the number of buckets.

Pseudocode code:

    1. Each Node is assigned color_0 with hash(node.attributes) % K
        see create_inital_color_graph()
    2. For each iteration I each node is assigned the attribute color_I = hash(current_color, product of neighbor colors) % (modulo) K
        see reate_color_hash_function()
 
 
Interpretation:
 
   For color_0 a when two nodes, either within the same graph or between graphs, have the same color C, they are very likely the same node.
       Because hashing is deterministic there is no possibility of a False Negative, but a 1/K possibility of a False Positive due to random hash collisions.
 
   color_1: contains all the 1-hop information around the node.
       If two nodes have the same color it is very likely that they have the same 1 hop neighborhood.
       The probability of a false positive here is still very close to zero but I haven't worked it out yet.
 increase
 
Usages:
 
    Node Embedding: Nodes are embedded as an integer such that when two nodes share a color their local neighborhoods are very likely identical.
    
    Graph Embedding: Graphs can be embedded as a histogram of node colors. eg the more similar colors the more similar the graphs are.

    
    For example these graphs are converted into vectors and then there are lots of different ways to measure distance between them.
    
                    [Color: count of nodes]
    Graph_A.Color_3  = [1:2, 2:0, 3:3]
    Graph_B.Color_3  = [1:2, 2:1, 3:2]

"""
 

###################### Need to refactor before pushing to PIP ###########################

def create_color_hash_function(num_buckets: int):

    def get_next_node_color(node_color:int, neighbor_colors:list)-> int:
        pre_hash_color = str(node_color) + str(np.prod(neighbor_colors))
        new_color = hash(pre_hash_color) % num_buckets
        return new_color

    hash_function = get_next_node_color
    return hash_function


def convert_node_view_to_node_color_dict(node_data: nx.classes.reportviews.NodeDataView, num_buckets: int)-> dict:
    """
        Set the inital color of each of the nodes based on hashing the node attributes and a number of buckets. 

        Returns a dictionary of inital color to make the nodes
    """

    node_data_as_dict = dict(node_data)
    node_names = list(node_data_as_dict.keys())
    node_starting_colors = dict()

    for index, node_num in enumerate(node_names):
        node_attributes = tuple(sorted(list(node_data_as_dict[node_names[index]].items())))
        inital_node_color = hash(node_attributes) % num_buckets
        node_starting_colors[node_num] = inital_node_color

    return node_starting_colors


def create_inital_color_graph(graph: nx.classes.graph.Graph, num_buckets:int) ->nx.classes.graph.Graph:

    starting_colors = convert_node_view_to_node_color_dict(graph.nodes.data(True), num_buckets)
    color_graph = copy.deepcopy(graph)
    nx.set_node_attributes(color_graph, starting_colors, name="color_0")

    # remove unneeded tags
    for index, node in enumerate(color_graph.nodes.data(True)):
        node_attributes = list(color_graph.nodes[index].keys())
        attributes_to_remove = [a for a in node_attributes if a !='color_0']
        [color_graph.nodes[index].pop(a) for a in attributes_to_remove]

    return color_graph


def next_iteration_of_color_graph(color_graph, iter_num:int, color_hashing_function) -> None:
    prev_color_key = f'color_{iter_num-1}'

    # do an iteration
    updated_colors = dict()
    for index, current_node in enumerate(color_graph.nodes.data(True)):
        neighbor_nodes = [n for n in color_graph.neighbors(index)]
        current_color = color_graph.nodes[current_node[0]]
        neighbor_colors = ([color_graph.nodes[n][prev_color_key] for n in neighbor_nodes])
        updated_colors[index] = color_hashing_function(current_color,neighbor_colors)

    nx.set_node_attributes(color_graph, updated_colors, name=f'color_{iter_num}')

def compute_K_color_refinements(G: nx.classes.graph.Graph, K:int, num_buckets:int) ->nx.classes.graph.Graph:
    """
    Given a graph G and a hash_function f, compute K iterations of the color refinement algorithm (link to algo)
    """
    color_hashing_function = create_color_hash_function(num_buckets)
    color_graph = create_inital_color_graph(G, num_buckets)

    for iter_num in range(1,K):
        next_iteration_of_color_graph(color_graph, iter_num=iter_num, color_hashing_function=color_hashing_function)
    return color_graph


def extact_node_colors(node, node_num)-> tuple:
    df = pd.DataFrame.from_dict(node.items())
    df.columns = ['iteration_number', f'Node_{node_num}']
    df.set_index('iteration_number', inplace=True)
    return df

def convert_color_graph_to_DataFrame(color_graph: nx.classes.graph.Graph) -> pd.DataFrame:
    """
        Converts the node colors into a pandas data frame where the rows are the iterations and the columns are the node colors
    """
    color_node_view = color_graph.nodes.data(True)
    node_colors = [extact_node_colors(color_node_view[i], i) for i in range(0,len(color_node_view))]
    df = pd.concat(node_colors, axis=1)
    return df


def compute_graph_embeddings(G: nx.classes.graph.Graph, K:int, num_buckets:int) -> pd.DataFrame:
    color_graph = compute_K_color_refinements(G=G, K=K, num_buckets=num_buckets)
    embedding_df = convert_color_graph_to_DataFrame(color_graph=color_graph)
    return embedding_df


def compute_bag_of_colors_vector(iteration_colors: np.array, num_buckets:int):
    bag_of_colors = np.zeros(shape=num_buckets).astype(int)
    for color in iteration_colors:
        bag_of_colors[color] +=1
    return bag_of_colors



def compute_call_color_bag(embedding_df: pd.DataFrame, num_buckets:int) -> pd.DataFrame:
    bag_of_colors_df = pd.DataFrame(index= embedding_df.index, columns=[a for a in range(num_buckets)])
    for iteration_name in bag_of_colors_df.index:
        iteration_colors = embedding_df.loc[iteration_name].values
        bag_of_colors_vector = compute_bag_of_colors_vector(iteration_colors, num_buckets)
        bag_of_colors_df.loc[iteration_name, :] = bag_of_colors_vector

    return bag_of_colors_df


def embedd_graph_with_color_refinement(G: nx.classes.graph.Graph, K:int, num_buckets:int) -> pd.DataFrame:
    embedding_df = compute_graph_embeddings(G,K,num_buckets)
    bag_of_colors_df = compute_call_color_bag(embedding_df,num_buckets)
    return bag_of_colors_df


def get_n_dem_embedding(color_graphs:list, n:int) -> pd.DataFrame:
    """
        Givens a list of color graphs, and a int n. extact all the embeddings of the embedding N.

        Eg If you pass 10 color graphs and n=3

        number_of_colors = number of columns in each color graph. Must be the same for each graph.

        Returns a a DataFrame of (10,number_of_colors) where is row is a graph and embedded as a 'bag of colors'
    """
    embeddings = []
    for g in color_graphs: # is a dataframe
        embeddings.append(g.values[n,:])
    return pd.DataFrame(np.array(embeddings))


def embedd_graphs(graphs:list, num_hops:int, num_colors:int) -> list:
    """
        Given a list of graphs: graphs, num_hops and num_colors create a list of DataFrames where each Dataframe is a hop embedding of that graph.
        
        Each df where each row is a 'bag of colors' vector for each graph.

        The only method that should be called by the outside
    """

    color_graphs = [compute_graph_embeddings(G=g,K=num_hops, num_buckets=num_colors) for g in graphs]
    print('in embedd_graphs')
    # is correct
    print('color graph is type')
    print(type(color_graphs))
    print('one instance is ')
    print(type(color_graphs[0]))
    
    embeddings_dfs = []
    for hop_num in range(num_hops):
        df = get_n_dem_embedding(color_graphs=color_graphs,n=hop_num) 
        embeddings_dfs.append(df)
    return embeddings_dfs

