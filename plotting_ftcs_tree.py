import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TreeSigMA import TreeSigMAWithHierarchy  
import os

### Plot the hierarchy of TreeSigMA ###
def plot_tree(node, tree, x=0, y=0, level=0, x_spacing=1.5, y_spacing=1.5, node_positions=None, parent_positions=None, parent=None):
    """
    Recursively plot tree structure
    
    Args:
        node (TreeNode): The current node to plot.
        x (float): X-coordinate of the current node.
        y (float): Y-coordinate of the current node.
        level (int): Current level in the tree.
        x_spacing (float): Horizontal spacing between nodes.
        y_spacing (float): Vertical spacing between levels.
        node_positions (dict): Dictionary to store node positions for plotting.
        parent_positions (dict): Dictionary to store parent positions for edges.
        parent (tuple): Position of the parent node.
    """
    if node_positions is None:
        node_positions = {}
    if parent_positions is None:
        parent_positions = []

    # Store the current node's position
    node_positions[node.node_id] = (x, y)
    if parent is not None:
        parent_positions.append((parent, (x, y)))

    # Compute positions for children
    n_children = len(node.children)
    for i, child in enumerate(node.children):
        child_x = x - x_spacing * (n_children - 1) / 2 + i * x_spacing
        child_y = y - y_spacing
        plot_tree(child, tree, x=child_x, y=child_y, level=level + 1,
                  x_spacing=x_spacing / 1.5, y_spacing=y_spacing,
                  node_positions=node_positions, parent_positions=parent_positions,
                  parent=(x, y))
    
    # Plot when at the root level
    if level == 0:
        fig, ax = plt.subplots(figsize=(40, 25))
        for (start, end) in parent_positions:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', zorder=1)
        for node_id, (nx, ny) in node_positions.items():
            current_node = tree.hierarchy.nodes[node_id]
            original_label = current_node.node_id.split('_')[0] # Get the original_label
            ax.scatter(nx, ny, s=800, c='lightblue', edgecolors='black', zorder=2)
            #ax.text(nx, ny, str(node_id), fontsize=10, ha='center', va='center', zorder=3)
            ax.text(nx, ny, f"{original_label}", fontsize=15, ha='center', va='center', zorder=3)
        ax.axis('off')
        plt.savefig('TreeSigMA.png')
        plt.show()



### Plot the data of each node of the tree ###
def plot_node_data(node, df, labels, output_dir, cols, alphas, zorders, x_col='x', y_col='y'):
    """
    Plot the data points contained in a tree node, colored by true labels.

    Args:
        node (TreeNode): Tree node to plot.
        df (pd.DataFrame): Dataframe containing the data points.
        output_dir (str): Directory to save the plots.
        x_col (str): Column name for x-axis data.
        y_col (str): Column name for y-axis data.
    """
    indices = node.data_indices
    if len(node.data_indices) == 0: #skip virtual node
        return
        
    node_data = df.iloc[indices]
    node_labels = labels[indices]

    plt.figure(figsize=(8, 6))
    for label, color in cols.items():
        mask = node_labels == label
        plt.scatter(node_data.loc[mask, x_col],
                    node_data.loc[mask, y_col], 
                    s=10,c=color,
                    alpha=alphas.get(label, 0.9), 
                    zorder=zorders.get(label, 1),
                    label=f'Label {label}'
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Node ID: {node.node_id}")
   # plt.show()

    # Save plotx
    plt.savefig(os.path.join(output_dir, f"Node_{node.node_id}.png"))
    plt.close()

### Plot tree with percentages ### 
def plot_tree_with_percentages(tree, node, df, labels, cols, x=0, y=0, level=0, 
                                                      x_spacing=1.5, y_spacing=1.5, 
                                                      node_positions=None, parent_positions=None, parent=None):
    if node_positions is None:
        node_positions = {}
    if parent_positions is None:
        parent_positions = []

    # Store the current node's position
    node_positions[node.node_id] = (x, y)
    if parent is not None:
        parent_positions.append((parent, (x, y)))

    # Compute percentages for background and clusters
    indices = node.data_indices
    label_counts = {label: 0 for label in np.unique(labels)}
    if len(indices) > 0:
        node_labels = labels[indices]
        unique_labels, counts = np.unique(node_labels, return_counts=True)
        label_counts.update(dict(zip(unique_labels, counts)))
        total_in_node = len(indices)

        # Compute percentages for background and each cluster
        total_background_points = np.sum(labels == 0)
        percentages = {}
        for label, count in label_counts.items():
            if label == 0:  # Background
                percentages[label] = (count / total_in_node) * 100  # Percentage in node
            else:  # Cluster points
                total_cluster_points = np.sum(labels == label)
                percentages[label] = (count / total_cluster_points) * 100  # Percentage relative to cluster
    else:
        percentages = None

    # Prepare the text for the node
    if percentages:
        text_lines = [
            f"{label}: {percent:.1f}%" for label, percent in percentages.items()
        ]
    else:
        text_lines = ["Virtual Node"]

    # Compute positions for children
    n_children = len(node.children)
    for i, child in enumerate(node.children):
        child_x = x - x_spacing * (n_children - 1) / 2 + i * x_spacing
        child_y = y - y_spacing
        plot_tree_with_percentages(tree,
            child, df, labels, cols, 
            x=child_x, y=child_y, level=level + 1,
            x_spacing=x_spacing / 1.5, y_spacing=y_spacing,
            node_positions=node_positions, 
            parent_positions=parent_positions, 
            parent=(x, y)
        )
    
    # Plot when at the root level
    if level == 0:
        fig, ax = plt.subplots(figsize=(20, 12))
        for (start, end) in parent_positions:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', zorder=1)
        for node_id, (nx, ny) in node_positions.items():
            current_node = tree.hierarchy.nodes[node_id]
            indices = current_node.data_indices
            indices
            label_counts = {label: 0 for label in np.unique(labels)}
            if len(indices) > 0:
                node_labels = labels[indices]
                unique_labels, counts = np.unique(node_labels, return_counts=True)
                label_counts.update(dict(zip(unique_labels, counts)))
                total_in_node = len(indices)

                # Compute percentages for background and each cluster
                percentages = {}
                for label, count in label_counts.items():
                    if label == 0:  # Background
                        percentages[label] = (count / total_in_node) * 100  # Percentage in node
                    else:  # Cluster points
                        total_cluster_points = np.sum(labels == label)
                        percentages[label] = (count / total_cluster_points) * 100  # Percentage relative to cluster
            else:
                percentages = None
            
            # Add text label for percentage breakdown
            if percentages:
                text_lines = [
                    f"{label}: {percent:.1f}%" for label, percent in percentages.items()
                ]
                colors = [cols[label] for label, percent in percentages.items()]
            else:
                text_lines = ["Virtual Node"]
                colors = ['black']
            
            ax.scatter(nx, ny, s=800, c='lightblue', edgecolors='black', zorder=2)
            for i, (line, color) in enumerate(zip(text_lines, colors)):
                ax.text(nx, ny - i * 0.2, line, fontsize=8, color=color, 
                        ha='center', va='center', zorder=3)
        ax.axis('off')
        plt.savefig('Tree_with_Percentages.png')
        plt.show()

