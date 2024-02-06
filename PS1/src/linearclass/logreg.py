import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    pred = clf.predict(x_valid)
    
    #labels = pred > .5
    #print("accuracy", (y_valid == labels).sum()/len(y_valid)) # 0.83, 0.86
    
    # Plot decision boundary on top of validation set
    util.plot(x_valid, y_valid, clf.theta, f"{save_path[:-4]}.png")
    
    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, pred > .5, fmt="%d")
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # fetch number of samples
        m, n = x.shape
        y = y.reshape(-1,1)
        if self.theta is None:
            self.theta = np.zeros((n,1))

        # instantiate iteration counter and dummy update
        iteration = 0
        update_norm = 1
    
        # fit data through Newton's method
        while iteration <= self.max_iter and update_norm >= self.eps:
            # calculate loss and print if verbose
            z = x @ self.theta
            h_x = self.sigmoid(z)
            
            cur_loss = (-1/m) * (y.T @ np.log(h_x+self.eps) + (1-y).T @ np.log(1-h_x+self.eps))
            
            if self.verbose:
                print(f"iteration {iteration}: {cur_loss}")

            # perform newton's method iteration
            gradient = (1/m) * x.T @ (h_x - y)
            hessian = (x * (1/m) * h_x * (1 - h_x)).T @ x
            
            update = np.linalg.inv(hessian) @ gradient
            update_norm = np.linalg.norm(update, ord=1)
            
            self.theta -= update
            
            # add 1 to iteration
            iteration += 1
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self.sigmoid(x @ self.theta).reshape(-1,)
        
        # *** END CODE HERE ***
        
    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
