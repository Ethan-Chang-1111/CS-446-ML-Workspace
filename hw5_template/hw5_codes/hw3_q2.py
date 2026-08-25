import hw5_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=None, c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''   
    N, d = x_train.shape
    alpha = torch.zeros(N, requires_grad=True, dtype=torch.float32)

    k = torch.empty(N,N)
    for i in range(N):
        for j in range(N):
            k[i][j] = kernel(x_train[i], x_train[j])
    
    Y = torch.outer(y_train, y_train)
    precompute = Y * k
    # print("Precompute:", precompute.shape)
    for i in range(num_iters):
        # print("Epoch: ", i, "\n", alpha.grad)
        if alpha.grad is not None:
            alpha.grad.zero_()
            # print("clear")

        # loss = 0.5 * (alpha.T @ (precompute @ alpha)) - alpha
        pt1 = precompute @ alpha
        pt2 = alpha.T @ pt1
        loss = 0.5 * pt2 - torch.sum(alpha)

        # print("loss:", loss)
        loss.backward()

        with torch.no_grad():
            alpha -= (lr * alpha.grad)
            torch.clamp_(alpha, min=0)
            if c is not None:
                torch.clamp_(alpha, max=c)
        alpha.requires_grad_()

    return alpha

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=None):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    # Support Vectors
    mask = alpha > 0

    alpha_sv = alpha[mask]
    alpha_sv = alpha_sv.to(torch.float64)
    x_train_sv = x_train[mask]
    y_train_sv = y_train[mask]

    # print(alpha_sv.shape, x_train_sv.shape, y_train_sv.shape)
    # print(alpha.dtype, x_train.dtype, y_train.dtype)
    # print(alpha_sv.dtype, x_train_sv.dtype, y_train_sv.dtype)

    min_index = torch.argmin(alpha_sv)
    y_train_min_sv = y_train_sv[min_index]
    x_train_min_sv = x_train_sv[min_index]
    # print("mins: ", x_train_min_sv, y_train_min_sv)

    # BS #
    N, d = x_train_sv.shape
    M = x_test.shape[0]
    k1 = torch.empty((N,M), dtype=torch.float64)
    for i in range(N):
        for j in range(M):
            k1[i][j] = kernel(x_train_sv[i], x_test[j])
    term1 = k1.T @ (alpha_sv * y_train_sv)
    # print("term1:", term1.shape)

    k2 = torch.empty((N,), dtype=torch.float64)
    for i in range(N):
        k2[i] = kernel(x_train_min_sv, x_train_sv[i])
    term2 = y_train_min_sv - k2.T @ (alpha_sv * y_train_sv)
    # print("term2:", term2.shape)

    y_pred = term1 + term2
    # print(y_pred.shape)
    return y_pred

if __name__ == "__main__":
    # test()
    X, y = hw5_utils.xor_data()
    x_test, y_test = hw5_utils.xor_data()
    
    iter = 10000
    kernel_test = hw5_utils.rbf(5)
    lr_space = [1, 0.1, 0.08, 0.05, 0.03, 0.01]
    c_space = [None, 0.01, 0.1, 1, 10]

    best_error = torch.inf
    best_lr = 0
    best_c = 0
    for lr in lr_space:
        for c_val in c_space:
            a = svm_solver(x_train=X, y_train=y, lr=lr, num_iters=iter, kernel=kernel_test, c = c_val)
            # print(a, lr, c_val)
            if torch.all(a > 0):
                y_pred = svm_predictor(alpha = a, x_train = X, y_train = y, x_test = x_test, kernel=kernel_test)
                error = torch.sum(torch.square(y_pred - y_test))
                if error < best_error:
                    best_error = error
                    best_lr = lr
                    best_c = c_val
                print(a, lr, c_val, error)

    a = svm_solver(x_train=X, y_train=y, lr=best_lr, num_iters=iter, kernel=kernel_test, c = best_c)
    hw5_utils.svm_contour(lambda x_test: svm_predictor(alpha = a, x_train = X, y_train = y, x_test = x_test, kernel=kernel_test))
    y_pred = svm_predictor(alpha = a, x_train = X, y_train = y, x_test = x_test, kernel=kernel_test)
    print("alphas:", a, "lr:", best_lr, c_val, "y_pred:", y_pred)


    # Poly3, lr = 0.05
    # rbf1, lr = 1
    # rbf3, lr = 0.1, c = 100
    # rbf5, lr = 0.1, c = 10