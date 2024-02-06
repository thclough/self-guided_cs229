import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # load data
    x_train, y_train_t = util.load_dataset(train_path, label_col="t", add_intercept=True)
    x_valid, y_valid_t = util.load_dataset(valid_path, label_col="t", add_intercept=True)
    x_test, y_test_t = util.load_dataset(test_path, label_col="t", add_intercept=True)
    _, y_train_y = util.load_dataset(train_path, label_col="y")
    _, y_valid_y = util.load_dataset(valid_path, label_col="y")
    _, y_test_y = util.load_dataset(test_path, label_col="y")
    
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    true_model = LogisticRegression()
    true_model.fit(x_train, y_train_t)
    true_pred = true_model.predict(x_test)
    true_pred_labels = true_pred > .5
    
    np.savetxt(output_path_true, true_pred_labels, fmt="%d")
    
    util.plot(x_test, y_test_t, true_model.theta, f"{output_path_true[:-4]}.png")
    
    #accuracy = ((true_pred > .5) == y_test_t).sum()/len(y_test_t)
    #print("accuracy" , accuracy)
    
    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    naive_model = LogisticRegression()
    naive_model.fit(x_train, y_train_y)
    naive_pred = naive_model.predict(x_test)
    naive_pred_labels = naive_pred > .5
    
    #accuracy = ((naive_pred > .5) == y_test_t).sum()/len(y_test_t)
    #print("accuracy" , accuracy)
    np.savetxt(output_path_naive, naive_pred_labels, fmt="%d")
    
    util.plot(x_test, y_test_t, naive_model.theta, f"{output_path_naive[:-4]}.png")
    
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    
    # caluclate alpha scaling factor
    naive_pred_val = naive_model.predict(x_valid)
    num_labeled = y_valid_y.sum()
    
    alpha = naive_pred_val[y_valid_y == 1].sum()/num_labeled
    
    # scale the predictions
    naive_pred_rescaled  = 1/alpha * naive_pred
    
    # save and plot
    np.savetxt(output_path_adjusted, naive_pred_rescaled, fmt="%d")
    
    util.plot(x_test, y_test_t, naive_model.theta, f"{output_path_adjusted[:-4]}.png", alpha)
    
    #accuracy = ((naive_pred_rescaled > .5) == y_test_t).sum() / len(y_test_t)
    #print("accuracy" , accuracy)
    
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
