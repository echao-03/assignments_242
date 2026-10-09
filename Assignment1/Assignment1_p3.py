import numpy as np
import matplotlib.pyplot as plt



def generate_data(n_samples, seed, mean, covariance):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.
    
        Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
        (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
        """

    
    # Used default_rng to set the numpy generator to the specific seed that was set
    randomGen = np.random.default_rng(seed)

   
    # Used function from numpy website. 
    # The .T at the end represents transpose, which is how we are able to obtain the X and Y coordinate of each data point.
    samples = randomGen.multivariate_normal(mean, covariance, n_samples)
    x, y = samples.T
    
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Q2 2-Dimensional Dataset')
    # print(np.corrcoef(samples.T)[0, 1])
    plt.show()
    
    return samples
    
    
# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not."""
    
    retValue = {}
    
    transposeData = X.T
    
    # Logic for calculating the mean of the generated data
    mean = np.mean(X, axis = 0)
    # print(xMean)
    retValue["Mean"] = mean
    
    # Logic for centering the dataset
    centered = X - mean
    retValue["Centered"] =  centered
    
    # Logic for computing the covariance matrix
    covMatrix = np.cov(transposeData)
    retValue["Covariance"] = covMatrix
    
    
    
    # Logic for computing the eigenvalues and eigenvectors, using the covariance matrix
    values, vectors = np.linalg.eigh(covMatrix)
    
    eigenDescending = np.argsort(values)[::-1]
    retValue["Eigenvalues"] = values[eigenDescending]
    
    # Logic for obtaining the components with eigenvectors
    components = vectors[:, eigenDescending]
    retValue["Components"] = components
    
    # Logic for computing the score of the total rows
    scores = np.dot(centered, components)
    retValue["Scores"] = scores
    # print(mean)
    # print(covMatrix)
    # print(components)
    
    return retValue
    
    
    

def main():
    """ Using default input that was given from instructions. 
     Instead of non-default arguments ie. n_samples=500, had to change it
     to default arguments to avoid errors. """
    
    generated_sample = generate_data(500, 2, [2, -1], [[3, 1.2], [1.2, 1]])
    result = pca(generated_sample)
    
    plt.scatter(generated_sample[:, 0], generated_sample[:, 1], alpha=0.5)

    mean = result["Mean"]
    components = result["Components"]
    eigenvalues = result["Eigenvalues"]

    for i, color in [(0, "red"), (1, "green")]:
        direction = components[:, i] * np.sqrt(eigenvalues[i])

        plt.quiver(
            mean[0],
            mean[1],
            direction[0],
            direction[1],
            angles="xy",
            scale_units="xy",
            scale=0.5,
            color=color,
            label=f"PC{i + 1}"
        )

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Data with Principal Component Directions")
    plt.axis("equal")
    plt.legend()
     # plt.show()
    
    
    """
    From the generated dataset, we can see a positive correlationfrom both the X and Y coordinates. We can also
    see that the plotted points are bundled within (2, -1), which is where the mean is set. 
    When obtaining the correlation coefficient, we get 0.697 from the covariance matrix.
    """
    


if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    main()