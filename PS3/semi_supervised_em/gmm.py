import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n, d = x.shape
    group_label = np.random.choice(K, n)

    mu = np.array([np.mean(x[group == group_label, :], axis=0) for group in range(K)])
    sigma = np.array([np.cov(x[group == group_label, :].T) for group in range(K)])
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K,), fill_value=1/K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n,K), fill_value=1/K)
    # *** END CODE HERE ***
    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):

        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        w = e_step(x, w, phi, mu, sigma)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = m_step(x, w, mu, sigma)

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        prev_ll = ll
        ll = log_likelihood(x, phi, mu, sigma)

        if it % 100 == 0:
            print(f"iter:{it}, ll: {ll}")

        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        
        it += 1
    print(f"Regular converged in {it} iterations")
        # *** END CODE HERE ***

    return w

def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w = e_step(x, w, phi, mu, sigma)
        
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi, mu, sigma = m_step_semi(x, x_tilde, z_tilde, w, phi, mu, sigma, alpha)
       
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = log_likelihood(x, phi, mu, sigma)
        ll += alpha * log_likelihood(x_tilde, phi, mu, sigma, z_tilde)

        if it % 100 == 0:
            print(f"iter:{it}, ll: {ll}")
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        it += 1
    print(f"Semi-supervised em converged in {it} iterations")
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
def e_step(x, w, phi, mu, sigma):
    n, d = x.shape
    k = len(phi)
    raw_w = np.zeros((n,k))

    for j in range(K):
        sigma_j = sigma[j]
        mu_j = mu[j]
        phi_j = phi[j]

        det_j = np.linalg.det(sigma_j)
        sig_inv_j = np.linalg.inv(sigma_j)

        raw_w_j = np.sqrt(1/det_j) * np.exp(-.5 * np.sum(((x-mu_j).dot(sig_inv_j)) * (x-mu_j), axis=1)) * phi_j

        raw_w[:, j] = raw_w_j

    w = raw_w / np.sum(w, axis=1, keepdims=1)

    return w 

def m_step(x, w, mu, sigma):
    n, d = x.shape
    k = len(mu)

    sum_w = np.sum(w,axis=0)
        
    phi_next = (1/n) * sum_w
    mu_next = ((x.T.dot(w)) / sum_w).T

    sigma_next = np.array([((x-mu[j]) * (w[:,j]).reshape(-1,1)).T.dot((x-mu[j])) / sum_w[j] for j in range(k)])

    return phi_next, mu_next, sigma_next

def m_step_semi(x, x_tilde, z_tilde, w, phi, mu, sigma, alpha=20):
    n, d = x.shape
    n_tilde, _ = x_tilde.shape
    k = len(phi)

    sum_w = np.sum(w,axis=0)

    class_totals = np.array([np.sum(z_tilde == j) for j in range(k)])
    x_tilde_sums = np.array([np.sum(x_tilde[(z_tilde == j).reshape(-1), :], axis=0) for j in range(k)])

    phi_next = (sum_w + alpha * class_totals) / (n + alpha * n_tilde)
    mu_next = (((x.T.dot(w)).T + alpha * x_tilde_sums) / (sum_w + alpha * class_totals).reshape(-1,1))

    sigma_next = np.array([(((x-mu[j]) * (w[:,j]).reshape(-1,1)).T.dot((x-mu[j])) +
                           alpha * (x_tilde[(z_tilde == j).reshape(-1)]-mu[j]).T.dot((x_tilde[(z_tilde == j).reshape(-1)]-mu[j]))) 
                            / (sum_w[j] + alpha * class_totals[j])
                            for j in range(k)])

    return phi_next, mu_next, sigma_next

def log_likelihood(x, phi, mu, sigma, z=None):
    k = len(phi)
    if z is not None:
        a = z.astype(int).reshape(-1)
    ll = 0
    for i in range(len(x)):
        x_i = x[i]
        if z is not None:
            j = a[i]
            p_x = p_x_given_z(x_i, mu[j], sigma[j]) * phi[j]
        else:
            p_x = 0
            for j in range(k):
                p_x += p_x_given_z(x_i, mu[j], sigma[j]) * phi[j]
                
        ll += np.log(p_x)

    return ll

def p_x_given_z(x, mu, sigma):
    det_j = np.linalg.det(sigma)
    sig_inv_j = np.linalg.inv(sigma)

    p_xz = np.sqrt(1/(det_j * 2 * np.pi)) * np.exp(-.5 * (x-mu).dot(sig_inv_j).dot(x-mu))
    
    return p_xz
# *** END CODE HERE ***

def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)

def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z

#%%

if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
