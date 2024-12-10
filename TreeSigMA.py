import numpy as np
import pandas as pd
from Distant_SigMA.SigMA.SigMA import SigMA
from sklearn.metrics.cluster import contingency_matrix
from itertools import product
from sklearn.metrics.cluster import contingency_matrix

class TreeSigMA: 
    """Class applying the initial partitioning with SigMA
        data: pandas data frame
        ---
        alpha_th : Upper alpha threshold for hierarchy
        ---
        sigma_kwargs : Parameters for the SigMA algorithm

        """
    
    def __init__(
        self,
        data: pd.DataFrame,
        cluster_features: list, 
        scale_factors: dict = None,
        nb_resampling: int = 0,
        max_knn_density: int = 100,
        beta: float = 0.99,
        knn_initcluster_graph: int = 35,
        knn : int = 20, 
        alpha_threshold: float = 0.2, 
        sigma_kwargs: dict = None,
    ):
       
    # Check if data is pandas dataframe or series
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("Data input needs to be pandas DataFrame!")
        
        self.data = data
        self.cluster_features = cluster_features
        self.alpha_threshold = alpha_threshold 
        self.knn = knn
        self.sigma_kwargs = sigma_kwargs or {}
        self.current_alpha_idx = 0

        # Initialize SigMA 
        self.clusterer = SigMA(
            data=self.data,
            cluster_features=self.cluster_features,
            scale_factors=None,
            **self.sigma_kwargs
        )
        self.clusterer.initialize_clustering(knn=self.knn) #produce initial clusters
        self.clusterer.initialize_mode_neighbor_dict()
        self.clusterer.resample_k_distances()
        
        self.tree = {} # Stores tree levels
        self.node_indices = {} # keys are cluster labels, values are lists of indices corresponding to the data points in that cluster at given alpha level.        
    
    def run(self):    

        """
        Build the hierarchy for the respective alpha threshold

        Returns:
        - labels_tree (list): List of cluster labels for the hierarchy levels
        """

        #run SigMA with pvalues
        labels, pvalues = self.clusterer.run_sigma(alpha=-np.infty,knn=self.knn,return_pvalues= True) 
        sorted_pvalues = sorted(pvalues)

        # pvalues up to alpha threshold
        lower_pvalues = [x for x in sorted_pvalues if x <= self.alpha_threshold]
        #make a list of alpha values that are the centers between all these pvalues
        alpha_values = [(lower_pvalues[i] + lower_pvalues[i+1]) / 2 for i in range(len(lower_pvalues) - 1)]
        
        # Run merge clustering for each alpha and store results
        for alpha in alpha_values:
            merged_labels, merged_pvalues = self.clusterer.merge_clusters(knn=20, alpha=alpha)

            self.tree[alpha] = merged_labels
            
            # Store the node indices for each cluster label
            node_indices = {}
            for idx, label in enumerate(merged_labels):
                if label not in node_indices:
                    node_indices[label] = []
                node_indices[label].append(idx)     
            self.node_indices[alpha] = node_indices
            
        self.alpha_values = alpha_values
        return alpha_values
       
    def next(self):
        """
        Get the labels for the next alpha value.

        Returns:
        - A tuple (alpha, labels) where:
          - alpha: The current alpha value.
          - labels: The cluster labels for this alpha.
        """
        if self.current_alpha_idx >= len(self.alpha_values):
            raise StopIteration("No more alpha values to iterate.")

        alpha = self.alpha_values[self.current_alpha_idx]
        self.current_alpha_idx += 1
        return alpha, self.tree[alpha], self.node_indices[alpha]
    
    def __iter__(self):
        """
        Initialize the iterator. Ensures that iteration starts from the first alpha value when using the object in an iterable.
        """
        self.current_alpha_idx = 0
        return self

    def __next__(self):
        """
        Advance the iterator. Gets called during each iteration (e.g. in a for loop). Returns the next tuple of (alpha, labels, node_indices).
        """
        return self.next()
        

class TreeNode:
    """A single node in the hierarchy tree."""
    def __init__(self, node_id, parent=None, children=None, data=None, alpha_levels=None):
        """
        Initialize a tree node.
        
        Args:
            node_id (int): Unique identifier for the node.
            parent (TreeNode, optional): The parent node. Defaults to None.
            children (list[TreeNode], optional): List of child nodes. Defaults to None.
            data (dict, optional): Data associated with this node (e.g., indices). Defaults to None.
        """
        self.node_id = node_id
        self.parent = parent
        self.children = children or []
        self.data = data or {}
        self.alpha_levels = alpha_levels or []

    def add_child(self, child_node):
        """Add a child node to this node."""
        child_node.parent = self
        self.children.append(child_node)

    def is_leaf(self):
        """Check if the node is a leaf."""
        return len(self.children) == 0

    def update_alpha_levels(self, new_alpha):
            """Update the alpha levels for this node."""
            self.alpha_levels.append(new_alpha)


class TreeStructure:
    """Manages the tree of nodes."""
    def __init__(self):
        """Initialize the tree structure."""
        self.nodes = {}  # Dictionary of node_id -> TreeNode
        self.root = None

    def add_node(self, node_id, parent_id=None, data=None):
        """
        Add a node to the tree.
        
        Args:
            node_id (int): Unique identifier for the node.
            parent_id (int, optional): Identifier of the parent node. Defaults to None (root node).
            data (dict, optional): Data associated with the node.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id {node_id} already exists!")

        # Create the new node
        new_node = TreeNode(node_id=node_id, data=data)

        if parent_id is None:
            # This is the root node
            if self.root is not None:
                raise ValueError("Tree already has a root!")
            self.root = new_node
        else:
            # Attach to the parent node
            if parent_id not in self.nodes:
                raise ValueError(f"Parent node with id {parent_id} does not exist!")
            parent_node = self.nodes[parent_id]
            parent_node.add_child(new_node)

        # Store the new node
        self.nodes[node_id] = new_node

    def traverse_top_down(self, start_node=None):
        """
        Generator for top-down traversal of the tree.
        
        Args:
            start_node (TreeNode, optional): Node to start traversal from. Defaults to the root.
        
        Yields:
            TreeNode: The next node in top-down order.
        """
        if start_node is None:
            start_node = self.root
        if start_node is None:
            return  # Empty tree
        yield start_node
        for child in start_node.children:
            yield from self.traverse_top_down(child)

    def traverse_bottom_up(self, start_node=None):
        """
        Generator for bottom-up traversal of the tree.
        
        Args:
            start_node (TreeNode, optional): Node to start traversal from. Defaults to the root.
        
        Yields:
            TreeNode: The next node in bottom-up order.
        """
        if start_node is None:
            start_node = self.root
        if start_node is None:
            return  # Empty tree
        for child in start_node.children:
            yield from self.traverse_bottom_up(child)
        yield start_node

    def labels(self, a_step):
            """
            Retrieve all labels containing the specified alpha level in their alpha_values list.
            
            Args:
                a_step (int): The alpha level to search for.
            
            Returns:
                list: A list of node IDs that have the specified alpha level.
            """
            matching_labels = []
            for node in self.traverse_top_down():
                if a_step in node.alpha_levels:
                    matching_labels.append(node.node_id)
            return matching_labels


# Integration with TreeSigMA
class TreeSigMAWithHierarchy(TreeSigMA):
    """Extends TreeSigMA to build and manage a hierarchy of clusters."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy = TreeStructure()
        self.survivability = {}  # Dictionary to store survivability info

    def build_hierarchy(self):
        """
        Build the hierarchical tree structure from the clustering results.
        """
        virtual_root_id = "virtual_root"
        self.hierarchy.add_node(
            node_id=virtual_root_id,
            parent_id=None,
            data={"info": "This is the virtual root"}
        )
    
        unique_id_map = {}  # Map to store unique IDs for each label
    
        for alpha_idx, alpha in enumerate(self.alpha_values):
            labels = self.tree[alpha]
            node_indices = self.node_indices[alpha]

            for label, indices in node_indices.items():
                # Generate a unique ID for the node
                unique_id = f"{label}_{alpha}"
                
                # Determine the parent based on JD and alpha levels
                if alpha_idx == 0:
                    parent_id = virtual_root_id
                    jd_value = None
                else:
                    # Call `find_parent` with current node details
                    parent_id, jd_value = self.find_parent(
                        label=label,
                        current_alpha=alpha,
                        unique_id_map=unique_id_map,
                        current_indices=indices,
                        current_alpha_idx=alpha_idx
                    )
                    
                # If a parent was found and JD=1, only update the alpha_level list
                if jd_value == 1:
                    parent_node = self.hierarchy.nodes[parent_id]
                    parent_node.update_alpha_levels(alpha_idx)
                    continue
                
                # create new node and add to tree
                self.hierarchy.add_node(
                    node_id=unique_id,
                    parent_id=parent_id,
                    data={"original_label": label, "indices": indices, "alpha": alpha}
                )
                
                # Initialize or append to the unique ID map
                if label not in unique_id_map:
                    unique_id_map[label] = []
                unique_id_map[label].append((alpha, unique_id))
                
                # Initialize alpha levels for the new node
                self.hierarchy.nodes[unique_id].update_alpha_levels(alpha_idx)

                print(f"Adding node {unique_id} (original label {label}) with parent {parent_id}, alpha {alpha}")
    
    

    def find_parent(self, label, current_alpha, unique_id_map, current_indices, current_alpha_idx):
        """
        Find the parent node's unique ID based on Jaccard distance.
    
        Args:
            label (int): The label of the current node.
            current_alpha (float): The alpha value of the current node.
            unique_id_map (dict): The mapping of labels to their unique ID history.
            current_indices (list): Indices of the current node.
            current_alpha_idx (int): current Alpha-Index.
    
        Returns:
            str or None: The unique ID of the parent node, or None if the node should not be added.
        """

        # parent cadidates from previous alpha level 
        previous_alpha_idx = current_alpha_idx - 1
        parent_candidates = [
            (node_id, node.data["indices"])
            for node_id, node in self.hierarchy.nodes.items()
            if previous_alpha_idx in node.alpha_levels
        ]
        
        # If no parent candidates, no parents 
        if not parent_candidates:
            return None, None
        
        # Compute JD-Matrix for all potential parents
        parent_ids, parent_indices = zip(*parent_candidates)
        jacc_matrix = self.compute_jaccard_matrix([current_indices], parent_indices)
        print(jacc_matrix)

        # Check for race condition: more than one non-zero entry in the Jaccard matrix should not occur
        non_zero_count = np.count_nonzero(jacc_matrix)
        if non_zero_count > 1:
            raise RaceError(
                f"Race condition detected: {non_zero_count} non-zero entries in Jaccard matrix for label {label} at alpha {current_alpha}."
            )

        # find best JD-Match 
        max_similarity = np.max(jacc_matrix)
        best_parent_idx = np.argmax(jacc_matrix)
        best_parent_id = parent_ids[best_parent_idx]
        
        if max_similarity == 1:
            # update alpha_levels of parental node
            return best_parent_id, 1
        elif 0 < max_similarity < 1:
            # add the node of the children
            return best_parent_id, max_similarity
        else:
            # no parent node
            return None, None


    def compute_jaccard_matrix(self, labels_1, labels_2):
            """
            Compute the Jaccard similarity matrix between two lists of index sets.
        
            Args:
                indices_1 (list of list): Each inner list contains the indices of a cluster in the first set.
                indices_2 (list of list): Each inner list contains the indices of a cluster in the second set.
        
            Returns:
                np.ndarray: 2D array containing the Jaccard similarity between each pair of clusters.
            """
            n = len(labels_1)
            m = len(labels_2)
            jacc_matrix = np.zeros((n, m), dtype=np.float16)
        
            for i, cluster_1 in enumerate(labels_1):
                set_1 = set(cluster_1)
                for j, cluster_2 in enumerate(labels_2):
                    set_2 = set(cluster_2)
                    intersection_size = len(set_1 & set_2)
                    union_size = len(set_1 | set_2)
                    if union_size > 0:
                        jacc_matrix[i, j] = intersection_size / union_size
        
            return jacc_matrix

    def compute_survivability(self):
        """
        Compute survivability for each cluster.
        Survivability is defined as the number of alpha levels a cluster persists.
        """
        for node_id, node in self.hierarchy.nodes.items():
            if node.parent:  # Exclude root node
                first_alpha = min(node.alpha_levels)
                last_alpha = max(node.alpha_levels)
                survivability = last_alpha - first_alpha + 1
                self.survivability[node_id] = survivability

    def get_survivability(self, cluster_id=None):
        """
        Get survivability information.
        
        Args:
            cluster_id: Cluster ID to query. If None, returns info for all clusters.
        
        Returns:
            Survivability info for the specified cluster or all clusters.
        """
        if cluster_id:
            return self.survivability.get(cluster_id, "Cluster not found")
        return self.survivability


