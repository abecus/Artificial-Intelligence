import numpy as np

class PCA():

    @classmethod
    def fit(cls, data_matrix):
        """
        Features must span the column space or
        Each observation must be a column in "data matrix"
        """
        N = data_matrix.shape[1]

        # centering data_matrix
        M = np.mean(data_matrix, axis=1).reshape((data_matrix.shape[0], 1))

        A = (data_matrix - M)

        # computing coverient matrix by 1/(n-1) * AA^T
        C = (1 / (N-1)) * np.matmul(A, A.T)

        # computing SVD of coverirnt matrix
        U, S, V = np.linalg.svd(C, full_matrices=False)

        cls._U = -U
        cls._V = V
        cls._lambdas = S

    @classmethod
    def transform(cls, matrix):
        M = np.mean(matrix, axis=1).reshape((matrix.shape[0], 1))
        A = (matrix - M)

        return np.matmul(cls._U, A)
        

if __name__ == "__main__":
        
    # a = np.random.randint(low=0, high=100, size=(3, 6))

    # X = np.array([
    #             [7,4,6,8,8,7,5,9,7,8],
    #             [4,1,3,6,5,2,3,5,4,2],
    #             [3,8,5,1,7,9,3,8,5,2]
    #             ])

    A = np.array([[1, 2], [3, 4], [5, 6]]).T

    a = PCA
    a.fit(data_matrix = A)

    print(a._lambdas)
    print(a._U)
    print(a.transform(A))
