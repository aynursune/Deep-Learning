"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    M = X.dot(W)
 
    for i in range(0,X.shape[0]):
        sum_exp = 0
        maxi = np.max(M[i])
        softmax = np.zeros(W.shape[1])
        for k in range(0,W.shape[1]):
            sum_exp += np.exp(M[i][k]-maxi)
        for k in range(0,W.shape[1]):
            e = np.exp(M[i][k]-maxi)
            softmax[k] = (e)/(sum_exp)
        loss += -np.log(softmax[y[i]]) 
        for k in range(0,W.shape[1]):
            dW[:,k] += X[i]*softmax[k]
        dW[:,y[i]] -= X[i]
    
    loss = loss/X.shape[0]
    loss += reg*np.sum(W*W)
    dW = dW/X.shape[0]
    dW += 2*reg*W 
       
            

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    M = X.dot(W)
    maxi = np.max(M,axis=1)
    softmax = (np.exp(M-maxi[:,None]))/(np.exp(M-maxi[:,None]).sum(axis=1)[:,None])
    loss = -np.sum(np.log(softmax[np.arange(X.shape[0]),y]))/X.shape[0]
    loss += reg*np.sum(W*W)
    trues = np.zeros_like(softmax)
    trues[np.arange(X.shape[0]),y] = 1
    softmax = softmax - trues
    dW = ((np.transpose(X)).dot(softmax))/X.shape[0]
    dW += 2*reg*W 
              

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [2e-6,3e-6, 4e-6,5e-6,1e-7]
    regularization_strengths = [1e4,0.5e4, 1e-3,5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    
    combinations = np.array(np.meshgrid(learning_rates,regularization_strengths)).T.reshape(-1,2)
    
    for pair in combinations:
        model = SoftmaxClassifier()
        all_classifiers.append(model)
        model.train(X_train, y_train, learning_rate=pair[0], reg=pair[1], num_iters=1000) 
        y_train_pred = model.predict(X_train)
        train_acc = np.sum(y_train_pred == y_train)/y_train.shape[0]
        y_val_pred = model.predict(X_val)
        val_acc = np.sum(y_val_pred == y_val)/y_val.shape[0]
        if val_acc > best_val:
            best_val = val_acc
            best_softmax = model
        results[(pair[0],pair[1])] = (train_acc,val_acc)
        
 
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
