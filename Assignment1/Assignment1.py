"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np


# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    """
    AB = [[(3 * 1) + (4 * 0), (3 * 2) + (4 * -1)], [(2 * 1) + (1 * 0), (2 * 2) + (1 * -1)]]
       = [[3, 2], [2, 3]]
    BA = [[(1 * 3) + (2 * 2), (1 * 4) + (2 * 1)], [(0 * 3) + (-1 * 2), (0 * 4) + (-1 * 1)]]
       = [[7, 6], [-2, -1]]
    
    (A + B)^T = ([[4, 6], [2, 0]])^T
              = [[4, 2], [6, 0]]
    
    AB^T = [[3, 2], [2, 3]]

    trace(A) = 3 + 1
             = 4

    trace(AB) = 3 + 3
              = 6
    
    """


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """

    """
        Using determinant to find eigenvalues: det(M - λI) = 0
        
            det([[2, -1], [-1, 2]] - λ[[1, 0], [0, 1]]) = 0
            det([[2 - λ , -1], [-1, 2 - λ]]) = 0
            (2 - λ)(2 - λ) - (-1)(-1) = 0
            4 + λ^2 + λ - 1 = 0
            λ^2 + λ + 3 = 0
            λ1 = 3, λ2 = 1

            To find the eigenvectors to prove that Mv = λv:
                let λ1 = 3
                ([[2 - λ , -1], [-1, 2 - λ]])
                ([[2 - 3 , -1], [-1, 2 - 3]])
                = ([[-1, -1], [-1, -1])[[x], [y]] = [[0], [0]]

                let v1 = [[1], [-1]]
                To verify Mv1 = λ1v1:
                Mv1 = [[2, -1], [-1, 2]] * [[1], [-1]] = [[3], [-3]] 
                    = 3[[1], [-1]] = λ1v1
                
                let λ2 = 1
                ([[2 - λ , -1], [-1, 2 - λ]])
                ([[2 - 1 , -1], [-1, 2 - 1]])
                = [[1, -1], [-1, 1]] [[x], [y]] = [[0], [0]]

                let v2 = [[1], [1]]
                To verify Mv2 = λ2v2:
                Mv1 = [[2, -1], [-1, 2]] * [[1], [1]] = [[3], [1]] 
                                    = 1[[1], [1]] = λ2v2

            
            Therefore, Mv = λv


    """
    


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

"""
    trace(AB) = 6
    (Obtained from Q1)
    trace(BA) = 
    [[7, 6], [-2, -1]]
     = 6

    Therefore, trace(AB) = trace(BA)    
    

"""

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    """
    To compute the inverse of a 2x2 matrix, A^(-1):
        1 / ad - bc * [[d, -b], [-c, a]]
         = 1 / (4)(6) - (7)(2) * [[6, -7], [-2, 4]]
         = 1 / 10 * [[6, -7], [-2, 4]]
         = [[0.6, -0.7], [-0.2, 0.4]]
    The given matrix follows the properties of an inverse matrix:
        1. It is a square
        2. It is non-singular
        3. Let A be any matrices, in which AA^(-1) = I, where I is the identity matrix
        4. A^(-1)A = I

    To prove AA^(-1) = I:
        [[4, 7], [2, 6]] * [[0.6, -0.7], [-0.2, 0.4]] = [[2.4 - 1.4, -2.8 + 2.8], [1.2 - 1.2, -1.4 + 2.4]]
        = [[1, 0], [0, 1]]
    
    To prove that A^(-1)A = I
        [[0.6, -0.7], [-0.2, 0.4]] * [[4, 7], [2, 6]]
        = [[1, 0], [0, 1]]
    
        
    Therefore, the inverse of C is [[0.6, -0.7], [-0.2, 0.4]], which is a valid inverse of C.
    """


# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    raise NotImplementedError()


# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    pass
