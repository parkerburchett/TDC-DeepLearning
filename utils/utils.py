import networkx as nx
import pandas as pd
import numpy as np
from tdc.single_pred import ADME
from pysmiles import read_smiles
import networkx as nx #  Graph representation library
import pickle

def load_graph_repersentations():
    df = pickle.load(open('data/graph_df.pkl', 'rb'))
    return df


