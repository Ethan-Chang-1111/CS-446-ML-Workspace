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
    # print(zeros.shape)
    ones = X[y == 1]
    # print(ones.shape)

    # output as (var, mean)
    a = torch.var_mean(zeros, dim=0, unbiased=False)
    # print(a)

    b = torch.var_mean(ones, dim=0, unbiased=False)
    # print(b)

    # Standard is to have 0 on top, then 1
    # Shouldn't actually matter 

    mu = torch.stack((a[1], b[1]))
    sigma2 = torch.stack((a[0], b[0]))

    # print(mu.shape)
    # print(sigma2.shape)
    # print("gaussian_theta() \n")
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

    numerator = y[y==0].size(dim=0)
    denom = y.size(dim=0)
    # print(numerator, denom)
    # print("gaussian_p() \n")
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

    out = []
    log_p = torch.log(torch.tensor(p))
    for x in X:
        # TODO: torch.log and torch.sum(unbiased=False) is not a thing
        # class = 0
        pt1_0 = -1 * torch.log(torch.sqrt(2 * torch.pi * sigma2[0]))
        pt2_0 = -1 * (torch.square(x - mu[0])) / (2 * sigma2[0])
        yhat_0 = torch.sum(pt1_0) + torch.sum(pt2_0) + log_p
        
        # class = 1
        pt1_1 = -1 * torch.log(torch.sqrt(2 * torch.pi * sigma2[1]))
        pt2_1 = -1 * (torch.square(x - mu[1])) / (2 * sigma2[1])
        yhat_1 = torch.sum(pt1_1) + torch.sum(pt2_1) + log_p
        
        pred = 0 if yhat_0 > yhat_1 else 1
        out.append(pred)


    # x = X[0]
    # pt1_0 = -1 * torch.log(torch.sqrt(2 * torch.pi * sigma2[0]))
    # pt2_0 = -1 * (torch.square(x - mu[0])) / (2 * sigma2[0])
    # yhat_0 = torch.sum(pt1_0) + torch.sum(pt2_0) + torch.log(torch.tensor(p))

    # pt1_1 = -1 * torch.log(torch.sqrt(2 * torch.pi * sigma2[1]))
    # pt2_1 = -1 * (torch.square(x - mu[1])) / (2 * sigma2[1])
    # yhat_1 = torch.sum(pt1_1) + torch.sum(pt2_1) + torch.log(torch.tensor(p))
    
    # pred = 0 if yhat_0 > yhat_1 else 1
    # print(yhat_0, yhat_1, pred)

    
    
    # print("gaussian_classify() \n")
    # print(X.shape)
    # print(torch.tensor(out).shape)
    return torch.tensor(out)



def main():
    # For testing
    # X, y = utils.gaussian_dataset("train", prefix="gaussian")
    # mu, sigma2  = gaussian_theta(X, y)
    # p = gaussian_p(y)

    ypred, xtest = utils.gaussian_eval()
    print(ypred)
    

if __name__ == "__main__":
    main()
