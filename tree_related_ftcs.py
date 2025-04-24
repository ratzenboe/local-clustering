import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### make a leaf node dataframe ###
def extract_leaf_nodes(tree, df):
    leaf_data = []
    for i, node in enumerate(tree.hierarchy.traverse_top_down()):
        if node.is_leaf():
            print(node.node_id)
            print(i)
            indices = node.data_indices
            if len(indices) > 0:
                node_data = df.iloc[indices].copy()
                node_data['cluster'] = node.node_id  # Cluster-ID
                leaf_data.append(node_data)
    return pd.concat(leaf_data, ignore_index=False) 