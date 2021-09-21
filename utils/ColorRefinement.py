"""
    Author: Parker Burchett
"""


import numpy as np
import pandas as pd
import networkx as nx
import copy

from pandas.core.frame import DataFrame

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

def _create_color_hash_function(num_buckets: int):

    def get_next_node_color(node_color:int, neighbor_colors:list)-> int:
        pre_hash_color = str(node_color) + str(np.prod(neighbor_colors))
        new_color = hash(pre_hash_color) % num_buckets
        return new_color

    hash_function = get_next_node_color
    return hash_function


def _convert_node_view_to_node_color_dict(node_data: nx.classes.reportviews.NodeDataView, num_buckets: int)-> dict:
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


def _create_inital_color_graph(graph: nx.classes.graph.Graph, num_buckets:int) ->nx.classes.graph.Graph:

    starting_colors = _convert_node_view_to_node_color_dict(graph.nodes.data(True), num_buckets)
    color_graph = copy.deepcopy(graph)
    nx.set_node_attributes(color_graph, starting_colors, name="color_0")

    # remove unneeded tags
    for index, node in enumerate(color_graph.nodes.data(True)):
        node_attributes = list(color_graph.nodes[index].keys())
        attributes_to_remove = [a for a in node_attributes if a !='color_0']
        [color_graph.nodes[index].pop(a) for a in attributes_to_remove]

    return color_graph


def _next_iteration_of_color_graph(color_graph, iter_num:int, color_hashing_function) -> None:
    prev_color_key = f'color_{iter_num-1}'

    # do an iteration
    updated_colors = dict()
    for index, current_node in enumerate(color_graph.nodes.data(True)):
        neighbor_nodes = [n for n in color_graph.neighbors(index)]
        current_color = color_graph.nodes[current_node[0]]
        neighbor_colors = ([color_graph.nodes[n][prev_color_key] for n in neighbor_nodes])
        updated_colors[index] = color_hashing_function(current_color,neighbor_colors)

    nx.set_node_attributes(color_graph, updated_colors, name=f'color_{iter_num}')

def _compute_K_color_refinements(G: nx.classes.graph.Graph, K:int, num_buckets:int) ->nx.classes.graph.Graph:
    """
    Given a graph G and a hash_function f, compute K iterations of the color refinement algorithm (link to algo)
    """
    color_hashing_function = _create_color_hash_function(num_buckets)
    color_graph = _create_inital_color_graph(G, num_buckets)

    for iter_num in range(1,K):
        _next_iteration_of_color_graph(color_graph, iter_num=iter_num, color_hashing_function=color_hashing_function)
    return color_graph


def _extact_node_colors(node, node_num)-> tuple:
    df = pd.DataFrame.from_dict(node.items())
    df.columns = ['iteration_number', f'Node_{node_num}']
    df.set_index('iteration_number', inplace=True)
    return df


def _convert_color_graph_to_DataFrame(color_graph: nx.classes.graph.Graph) -> pd.DataFrame:
    """
        Converts the node colors into a pandas data frame where the rows are the iterations and the columns are the node colors
    """
    color_node_view = color_graph.nodes.data(True)
    node_colors = [_extact_node_colors(color_node_view[i], i) for i in range(0,len(color_node_view))]
    df = pd.concat(node_colors, axis=1)
    return df


def _compute_bag_of_colors_vector(iteration_colors: np.array, num_buckets:int):
    bag_of_colors = np.zeros(shape=num_buckets).astype(int)
    for color in iteration_colors:
        bag_of_colors[color] +=1
    return bag_of_colors


def _compute_node_colors(G: nx.classes.graph.Graph, K:int, num_buckets:int) -> pd.DataFrame:
    color_graph = _compute_K_color_refinements(G=G, K=K, num_buckets=num_buckets)
    node_color_df = _convert_color_graph_to_DataFrame(color_graph=color_graph)
    return node_color_df


def _node_color_df_to_bag_of_colors(node_color_df: pd.DataFrame, num_colors:int) -> pd.DataFrame:
    bag_of_colors_df = pd.DataFrame(index= node_color_df.index, columns=[a for a in range(num_colors)])
    for iteration_name in bag_of_colors_df.index:
        iteration_colors = node_color_df.loc[iteration_name].values
        bag_of_colors_vector = _compute_bag_of_colors_vector(iteration_colors, num_colors)
        bag_of_colors_df.loc[iteration_name, :] = bag_of_colors_vector
    return bag_of_colors_df



def compute_graph_embedding(G:nx.graph.Graph, num_hops: int, num_colors:int)-> DataFrame:
    """
        Convert the Graph G into a Dataframe where each row is a 'bag of colors of the n hop color refinement algorithm' 
        and each column is the count of the colors in that graph at that hop. 

        This is a deterministic algorithm. 

        graph_embedding: pd.DataFrame with shape (num_hops, num_colors)

    """
    node_color_df = _compute_node_colors(G, num_hops, num_colors)
    graph_embedding = _node_color_df_to_bag_of_colors(node_color_df,num_colors)
    return graph_embedding


def extract_n_hop_features(list_of_graph_embeddings:list, hop_number:int) -> list:
    """
        Given a list of graph of embeddings, and a hop_number create a DataFrame where is row is the 'bag of colors at hop_number'
    """
    n_hop_features = []
    for df in list_of_graph_embeddings:
        n_hop_features.append(df.values[hop_number,:])
    return pd.DataFrame(np.array(n_hop_features))



def create_hop_feature_dfs(graphs:list, num_hops: int, num_colors:int) -> list:
    """
        Given a list of graphs, num_hops, num_colors calcuates a data frame of the bag of colors 

        returns num_colors DataFrames where each df is the N_hop embedding of the graph. 
        
        For example in 3rd DataFrame each row is a graph, and each column si the count of each color found after 3 hops of ColorRefinement/
    
    """
    list_of_graph_embeddings = [compute_graph_embedding(g, num_hops, num_colors) for g in graphs]
    feature_dfs = [extract_n_hop_features(list_of_graph_embeddings, hop_num) for hop_num in range(num_hops)]
    return feature_dfs





