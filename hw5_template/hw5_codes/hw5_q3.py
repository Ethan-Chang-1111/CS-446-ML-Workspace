import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter):
        """Init the center of clusters.

        Parameters
        ----------
        n_clusters : number of clusters
        max_iter : number of interations
        

        Attribute
        -------
        cluster_centers_ : cluster centers
        max_iter: number of interations
        n_clusters: number of clusters
        """
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Init the center of clusters.

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        

        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers 
        """
        
        # return {key: X[np.random.choice(X.shape[0])] for key in range(self.n_clusters)}
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
    
        
        
        

    def fit(self, X: np.ndarray):
        """Fit with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.
        

        Returns
        -------
        Centers : np.ndarray [n_cluster,3]
            The cluster centers 
        """
        self.cluster_centers_ = self._init_centers(X)
        # for t in range(self.max_iter):
        #     new_clusters = {i: [] for i in range(self.cluster_centers_.shape[0])}
        #     for x in X:
        #         best_dist = np.inf
        #         best_center = None
        #         for c in range(self.cluster_centers_.shape[0]):
        #             diff = x - self.cluster_centers_[c]
        #             dist = np.dot(diff, diff)
        #             if dist < best_dist:
        #                 best_dist = dist
        #                 best_center = c
        #         new_clusters[best_center].append(x)

        #     for c in new_clusters:
        #         if len(new_clusters[c]) == 0:
        #             new_clusters[c] = [X[np.random.choice(X.shape[0])]]
            
        #     clusters_t = np.array([np.mean(np.stack(new_clusters[c], axis = 0), axis = 0) for c in new_clusters])
        #     no_diff = True
        #     for c in range(self.cluster_centers_.shape[0]):
        #         if not np.allclose(clusters_t[c], self.cluster_centers_[c]):
        #             no_diff = False
        #     if no_diff:
        #         self.cluster_centers_ = clusters_t
        #         return self.cluster_centers_
                
        #     self.cluster_centers_ = clusters_t
                        
        # return self.cluster_centers_

        for t in range(self.max_iter):
            # Calculates all (X - centroid) pairs upfront
            # Then take the norm of these pairs and find the minimum for each cluster
            diff = X[:, np.newaxis] - self.cluster_centers_
            labels = np.argmin(np.sum(diff * diff, axis=2), axis = 1)

            new_clusters = np.zeros_like(self.cluster_centers_)
            for c in range(self.n_clusters):
                cluster_points = X[labels == c]

                if len(cluster_points) == 0:
                    new_clusters[c] = np.random.choice(X)
                else:
                    new_clusters[c] = np.mean(cluster_points, axis=0)

            for c in range(self.cluster_centers_.shape[0]):
                if np.allclose(new_clusters, self.cluster_centers_):
                    self.cluster_centers_ = new_clusters
                    break

            self.cluster_centers_ = new_clusters

        self.labels_ = labels
        return self.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with Kmeans algorithm

        Parameters
        ----------
        X : np.ndarray
            The training input samples.

        Returns
        -------
        Labels : np.ndarray
                 Predicted labels
        """

        diff = X[:, np.newaxis] - self.cluster_centers_
        return np.argmin(np.sum(diff * diff, axis=2), axis = 1)
    

if __name__ == "__main__":
    model = KMeans(n_clusters=2, max_iter=100)
    x = [(-2, -1), (-5, -3), (-6, -5), (-1, -2), (-3, -4), (-4, -6)]
    data = np.array(x)
    # labels = np.array([1, 1, 1, 0, 0, 0])
    model.fit(data)
    model.predict(data)

    print(model.cluster_centers_)
    print(model.labels_)


    # a = np.array(((-2, -1), (-5, -3), (-6, -5), (-1, -2), (-3, -4), (-4, -6)))
    # cluster = np.array(((-1, 0.5), (-5.5, -4.5)))

    # b = a[:, np.newaxis] - cluster
    # print(b)
    # c = np.sum(b * b, axis=2)
    # print(c)
    # d = np.argmin(c, axis = 1)
    # print(d)

