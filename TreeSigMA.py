import numpy as np
import pandas as pd
from Distant_SigMA.SigMA.SigMA import SigMA


#class ts =  treeSigMA(d,alpha threshold)
#ts.next() gets labels of highest alpha
#each time you call 
#e alpha = 0 
#alphath = (...)
#remove noise function

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
        self.alpha_threshold = alpha_threshold #sorted(alpha_thresholds, reverse=True)
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
        
        # Store tree levels
        self.tree = {}
        self.node_indices = {}  # store indices of data points in each cluster
    
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

        # dictionary to store results of merge_clusters for each alpha
        merged_results = {}

        # Run merge clustering for each alpha and store results
        for alpha in alpha_values:
            merged_labels, merged_pvalues = self.clusterer.merge_clusters(knn=20, alpha=alpha)
            merged_results[alpha] = {
                "labels": merged_labels,
                "pvalues": merged_pvalues
            }

            self.tree[alpha] = merged_labels
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
        return alpha, self.tree[alpha]
    
    def __iter__(self):
        """
        Initialize the iterator.
        """
        self.current_alpha_idx = 0
        return self

    def __next__(self):
        """
        Advance the iterator.
        """
        return self.next()
            
    
       


    