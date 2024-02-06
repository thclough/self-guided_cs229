import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)
    
    labels = pred > .5
    print("accuracy", (y_valid == labels).sum()/len(y_valid))
    
    # Plot decision boundary on validation set
    util.plot(x_valid, y_valid, model.theta, f"{save_path[:-4]}.png")
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, pred > .5, fmt="%d")
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        m, n = x.shape
        
        if self.theta is None:
            self.theta = np.zeros(n+1)
            
        # Find phi, mu_0, mu_1, and sigma
        phi = (1/m) * (y==1).sum()
        
        mu0 = x[np.where(y==0), :].mean(axis=1)
        
        mu1 = x[np.where(y==1), :].mean(axis=1)
        
        errors = np.where((y==0).reshape(-1,1), x-mu0, x)
        errors = np.where((y==1).reshape(-1,1), errors-mu1, errors)
        
        sigma = (1/m) * errors.T @ errors
        
        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        
        self.theta[0] = .5 * (mu0 + mu1) @ sigma_inv @ (mu0 - mu1).T - np.log((1-phi)/phi)
        # self.theta[0] =  
        self.theta[1:] = (mu1 - mu0) @ sigma_inv

        # *** END CODE HERE ***
        

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        preds = 1/(1 + np.exp(-1 * (self.theta[1:] @ x.T + self.theta[0])))
        return preds
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
