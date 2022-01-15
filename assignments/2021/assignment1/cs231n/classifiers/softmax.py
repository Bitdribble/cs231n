from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)               # Shape C
        
        # When computing the exponentials, the numbers become very large, 
        # and the floating point computation gives large error.
        # For numerical stability, per https://cs231n.github.io/linear-classify/#softma
        # subtract a large constant from all scores
        scores -= np.max(scores)
        
        # Note:
        # scores[j] = X[i, :].W[:, j] = \sum_k X[i, k]*W[k, j] = (matrix mult (X, W))[i, j]
        
        loss += (-scores[y[i]] + np.log(np.sum(np.exp(scores))))
        
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) / np.exp(scores).sum()) * X[i,:]
        dW[:, y[i]] -= X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X, W) # Shape (N, C)
    #print(f"scores.shape={scores.shape}")
    num_classes = W.shape[1] # C=10
    num_train = X.shape[0]   # N=500
    num_data = X.shape[1]    # D=3073
    
    print(f"num_train={num_train}, num_classes={num_classes}, num_data={num_data}")
    
    # Normalize to avoid fractions of large exponents
    scores -= np.max(scores,axis=1).reshape(-1,1) # Shape (N, 1)
    
    correct_class_score = scores[range(num_train),y].reshape(-1,1) # Shape (N, 1)
    
    loss = np.sum(np.log(np.sum(np.exp(scores), axis=1)))
    loss -= scores[range(num_train), y].sum() 

    coeff_X = np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1,1) # (1), shape (N, C)
    coeff_X[range(num_train),y] -= 1 # (2)

    # Dot product below is \sum_n X.T(:,n)*coeff_X(n,:)
    #
    # Contribution from (1): 
    # coeff_X[n, c] = (np.exp(scores[n, c]) / np.exp(scores[n]).sum())
    #
    # Contribution from (2): 
    # coeff_X[range(num_train),y] is zero except at coeff_X(n, y[n])
    # so we get dW[:, y[n]] -= X.T[:,n] for all n < N
    dW += np.dot(X.T, coeff_X) 
    
    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
