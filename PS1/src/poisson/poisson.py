import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    pois_glm = PoissonRegression(step_size=lr)
    pois_glm.fit(x_train, y_train)
    
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    pred = pois_glm.predict(x_valid)
    
    print(np.mean(y_valid - pred))
    np.savetxt(save_path, pred)
    
    plt.scatter(y_valid, pred)
    plt.savefig(f"{save_path[:-4]}.png")
    
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # fetch number of samples
        n, d = x.shape
        if self.theta is None:
            self.theta = np.zeros((d,))

        # instantiate iteration counter and dummy update norm
        iteration = 0
        update_norm = 1
    
        # fit data through Newton's method
        while iteration <= self.max_iter and update_norm >= self.eps:
            
            #print(self.theta)
            eta = self.theta @ x.T
            
            cur_loss = y @ eta
            
            if self.verbose and iteration % 1000 == 0:
                print(f"iteration:{iteration} loss:{cur_loss}")
            
            update = self.step_size * 1/n * (y - np.exp(eta)) @ x
            
            #print(self.theta + update)
            
            self.theta += update 
            
            update_norm = np.linalg.norm(update, ord=1)
            
            iteration += 1
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        eta = self.theta @ x.T
        
        return np.exp(eta)
        
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
