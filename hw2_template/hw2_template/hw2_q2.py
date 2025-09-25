import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split



def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MLE estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MLE estimation of mu in N(mu, sigma2)

    '''

    # Calculate sample mean, mu
    # Then use to calculate sigma2, sum squared differences, (divided by n-1)
    # Where X=1, Y=y

    # Outputs are 2xN tensors, for y=0, and y=1

    
    # print(X.shape)
    # print(y.unsqueeze(-1).shape)
    
    zeros = X[y == 0]
    print(zeros.shape)
    ones = X[y == 1]
    print(ones.shape)

    # output as (var, mean)
    a = torch.var_mean(zeros, dim=0)
    print(a)

    b = torch.var_mean(ones, dim=0)
    print(b)

    # Standard is to have 0 on top, then 1
    # Shouldn't actually matter 

    mu = torch.stack((a[1], b[1]))
    sigma2 = torch.stack((a[0], b[0]))

    print(mu.shape)
    print(sigma2.shape)

    return mu, sigma2

def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        # Scalar, as per instructions
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''

    # num Y=0 divided by number of samples?
    # Should be done

    numerator = y[y==0]
    denom = y.size
    print(numerator.shape)
    return numerator / denom

def gaussian_classify(mu,sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1
    
    '''

    # Calculate log-likelihood. Equation should be on the slides, if not, post to campuswire

    

    pass


def main():
    # For testing
    X, y = utils.gaussian_dataset("train", prefix="gaussian")
    mu, sigma2  = gaussian_theta(X, y)
    p = gaussian_p(y)
    

if __name__ == "__main__":
    main()
