# Important note: you do not have to modify this file for your homework.
#%%
import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad, probs


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad, probs = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return 

def logistic_regression_mod(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    # initialize for graphs
    marker = []
    obj = []
    l2 = []

    i = 0
    while True and i <= 40000:
        i += 1
        prev_theta = theta
        grad, probs = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 100 == 0:
            print('Finished %d iterations' % i)

            # record data
            marker.append(i)
            l2.append(np.linalg.norm(theta))
            obj.append(np.sum(Y.dot(np.log1p(probs)) + (1-Y).dot(np.log1p(1.-probs))))
            
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return marker, obj, l2


def main():
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)

    print('==== Training model on data set A ====')
    
    m_a, ob_a, l2_a = logistic_regression_mod(Xa, Ya)

    print('\n==== Training model on data set B ====')

    m_b, ob_b, l2_b = logistic_regression_mod(Xb, Yb)
    
    track_a = pd.DataFrame({"Iter": m_a, "obj": ob_a, "l2": l2_a})
    track_b = pd.DataFrame({"Iter": m_b, "obj": ob_b, "l2": l2_b})
    
    g = sns.lineplot(data=track_a.melt(id_vars="Iter"), x="Iter", y="value", hue="variable")
    g.set_title("Training Set A: Objective Function and L2 Norm")
    plt.figure()
    g2 = sns.lineplot(data=track_b.melt(id_vars="Iter"), x="Iter", y="value", hue="variable")
    g2.set_title("Training Set B: Objective Function and L2 Norm")

if __name__ == '__main__':
    main()