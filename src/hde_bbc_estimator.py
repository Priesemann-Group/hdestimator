from sys import exit, stderr
import numpy as np
import mpmath as mp
from scipy.optimize import newton, minimize
import hde_utils as utl


def d_xi(beta, K):
    """
    First derivative of xi(beta).

    xi(beta) is the entropy of the system when no data has been observed.
    d_xi is the prior for the nsb estimator
    """

    return K * mp.psi(1, K * beta + 1.) - mp.psi(1, beta + 1.)

def d2_xi(beta, K):
    """
    Second derivative of xi(beta) (cf d_xi).
    """

    return K ** 2 * mp.psi(2, K * beta + 1) - mp.psi(2, beta + 1)

def d3_xi(beta, K):
    """
    Third derivative of xi(beta) (cf d_xi).
    """

    return K ** 3 * mp.psi(3, K * beta + 1) - mp.psi(3, beta + 1)


def rho(beta, mk, K, N):
    """
    rho(beta, data) is the Dirichlet multinomial likelihood.

    rho(beta, data) together with the d_xi(beta) make up
    the posterior for the nsb estimator
    """

    return np.prod([mp.power(mp.rf(beta, np.double(n)), mk[n]) for n in mk]) / mp.rf(K * beta,
                                                                                     np.double(N))

def unnormalized_posterior(beta, mk, K, N):
    """
    The (unnormalized) posterior in the nsb estimator.

    Product of the likelihood rho and the prior d_xi;
    the normalizing factor is given by the marginal likelihood
    """

    return rho(beta, mk, K, N) * d_xi(beta, K)


def d_log_rho(beta, mk, K, N):
    """
    First derivate of the logarithm of the Dirichlet multinomial likelihood.
    """

    return K * (mp.psi(0, K * beta) - mp.psi(0, K * beta + N)) - K * mp.psi(0, beta) \
        + np.sum((mk[n] * mp.psi(0, n + beta) for n in mk))

def d2_log_rho(beta, mk, K, N):
    """
    Second derivate of the logarithm of the Dirichlet multinomial likelihood.
    """

    return K ** 2 * (mp.psi(1, K * beta) - mp.psi(1, K * beta + N)) - K * mp.psi(1, beta) \
        + np.sum((mk[n] * mp.psi(1, n + beta) for n in mk))



def d_log_rho_xi(beta, mk, K, N):
    """
    First derivative of the logarithm of the nsb (unnormalized) posterior.
    """

    return d_log_rho(beta, mk, K, N) + d2_xi(beta, K) / d_xi(beta, K)

def d2_log_rho_xi(beta, mk, K, N):
    """
    Second derivative of the logarithm of the nsb (unnormalized) posterior.
    """

    return d2_log_rho(beta, mk, K, N) \
        + (d3_xi(beta, K) * d_xi(beta, K) - d2_xi(beta, K) ** 2) / d_xi(beta, K) ** 2


def log_likelihood_DP_alpha(a, K1, N):
    """
    Alpha-dependent terms of the log-likelihood of a Dirichlet Process.
    """

    return (K1 - 1.) * mp.log(a) - mp.log(mp.rf(a + 1., N - 1.))



def get_beta_MAP(mk, K, N):
    """
    Get the maximum a posteriori (MAP) value for beta.

    Provides the location of the peak, around which we integrate.

    beta_MAP is the value for beta for which the posterior of the 
    NSB estimator is maximised (or, equivalently, of the logarithm 
    thereof, as computed here).
    """
    
    K1 = K - mk[0]
    
    if d_log_rho(10**1, mk, K, N) > 0:
        print("Warning: No ML parameter was found.", file=stderr, flush=True)
        beta_MAP = np.float('nan')
    else:
        try:
            # first guess computed via posterior of Dirichlet process
            DP_est   = alpha_ML(mk, K1, N) / K
            beta_MAP = newton(lambda beta: float(d_log_rho_xi(beta, mk, K, N)), DP_est,
                              lambda beta: float(d2_log_rho_xi(beta, mk, K, N)),
                              tol=5e-08, maxiter=500)
        except:
            print("Warning: No ML parameter was found. (Exception caught.)", file=stderr, flush=True)
            beta_MAP = np.float('nan')
    return beta_MAP

def alpha_ML(mk, K1, N):
    """
    Compute first guess for the beta_MAP (cf get_beta_MAP) parameter 
    via the posterior of a Dirichlet process.
    """

    mk         = utl.remove_key(mk, 0)
    # rnsum      = np.array([_logvarrhoi_DP(n, mk[n]) for n in mk]).sum()
    estlist    = [N * (K1 - 1.) / r / (N - K1) for r in np.arange(6., 1.5, -0.5)]
    varrholist = {}
    for a in estlist:
        # varrholist[_logvarrho_DP(a, rnsum, K1, N)] = a
        varrholist[log_likelihood_DP_alpha(a, K1, N)] = a
    a_est      = varrholist[max(varrholist.keys())]
    res        = minimize(lambda a: -log_likelihood_DP_alpha(a[0], K1, N),
                          a_est, method='Nelder-Mead')
    return res.x[0]



def get_integration_bounds(mk, K, N):
    """
    Find the integration bounds for the estimator.

    Typically it is a delta-like distribution so it is sufficient
    to integrate around this peak. (If not this function is not
    called.)
    """

    beta_MAP = get_beta_MAP(mk, K, N)
    if np.isnan(beta_MAP):
        intbounds = np.float('nan')
    else:
        std       = np.sqrt(- d2_log_rho_xi(beta_MAP, mk, K, N) ** (-1))
        intbounds = [np.float(np.amax([10 ** (-50), beta_MAP - 8 * std])),
                     np.float(beta_MAP + 8 * std)]

    return intbounds
        
def H1(beta, mk, K, N):
    """
    Compute the first moment (expectation value) of the entropy H.

    H is the entropy one obtains with a symmetric Dirichlet prior 
    with concentration parameter beta and a multinomial likelihood.
    """

    norm = N + beta * K
    return mp.psi(0, norm + 1) - np.sum((mk[n] * (n + beta) *
                                         mp.psi(0, n + beta + 1) for n in mk)) / norm

        
def nsb_entropy(mk, K, N):
    """
    Estimate the entropy of a system using the NSB estimator.

    :param mk: multiplicities
    :param K:  number of possible symbols/ state space of the system
    :param N:  total number of observed symbols
    """

    mp.pretty = True

    # find the concentration parameter beta
    # for which the posterior is maximised
    # to integrate around this peak
    integration_bounds  = get_integration_bounds(mk, K, N)

    if np.any(np.isnan(integration_bounds)):
        # if no peak was found, integrate over the whole range
        # by reformulating beta into w so that the range goes from 0 to 1
        # instead of from 1 to infinity

        integration_bounds = [0, 1]
        
        def unnormalized_posterior_w(w, mk, K, N):
            sbeta = w / (1 - w)
            beta = sbeta * sbeta
            return unnormalized_posterior(beta, mk, K, N) * 2 * sbeta / (1 - w) / (1 - w)
        def H1_w(w, mk, K, N):
            sbeta = w / (1 - w)
            beta = sbeta * sbeta
            return H1(w, mk, K, N)
        marginal_likelihood = mp.quadgl(lambda w: unnormalized_posterior_w(w, mk, K, N),
                                        integration_bounds)
        H_nsb = mp.quadgl(lambda w: H1_w(w, mk, K, N) * unnormalized_posterior_w(w, mk, K, N),
                          integration_bounds) / marginal_likelihood

    else:
        # integrate over the possible entropies, weighted such that every entropy is equally likely
        # and normalize with the marginal likelihood
        marginal_likelihood = mp.quadgl(lambda beta: unnormalized_posterior(beta, mk, K, N),
                                        integration_bounds)
        H_nsb = mp.quadgl(lambda beta: H1(beta, mk, K, N) * unnormalized_posterior(beta, mk, K, N),
                          integration_bounds) / marginal_likelihood

    return H_nsb


def plugin_entropy(mk, N):
    """
    Estimate the entropy of a system using the Plugin estimator.

    (In principle this is the same function as utl.get_shannon_entropy,
    only here it is a function of the multiplicities, not the probabilities.)

    :param mk: multiplicities
    :param N:  total number of observed symbols
    """

    mk = utl.remove_key(mk, 0)
    return - sum((mk[n] * (n / N) * np.log(n / N) for n in mk))


def get_multiplicities(symbol_counts, alphabet_size):
    """
    Get the multiplicities of some given symbol counts.

    To estimate the entropy of a system, it is only important how
    often a symbol/ event occurs (the probability that it occurs), not
    what it represents. Therefore, computations can be simplified by
    summarizing symbols by their frequency, as represented by the
    multiplicities.
    """

    mk = dict(((value, 0) for value in symbol_counts.values()))
    number_of_observed_symbols = np.count_nonzero([value for value in symbol_counts.values()])
        
    for symbol in symbol_counts.keys():
        mk[symbol_counts[symbol]] += 1

    # the number of symbols that have not been observed in the data
    mk[0] = alphabet_size - number_of_observed_symbols 
    
    return mk


def bayesian_bias_criterion(H_nsb, H_plugin, H_uncond, bbc_tolerance):
    """
    Get whether the Bayesian bias criterion (bbc) is passed.
    
    :param H_NSB: NSB entropy
    :param H_plugin: Plugin entropy
    :param H_uncond: (Unconditional) entropy of the spike train, aka H_spiking
    :param bbc_tolerance: tolerance for the Bayesian bias criterion
    """

    if get_bbc_term(H_nsb, H_plugin, H_uncond) < bbc_tolerance:
        return 1
    else:
        return 0


def get_bbc_term(H_nsb, H_plugin, H_uncond):
    """
    Get the bbc-tolerance-independent term of the Bayesian bias
    criterion (bbc).
    
    :param H_NSB: NSB entropy
    :param H_plugin: Plugin entropy
    :param H_uncond: (Unconditional) entropy of the spike train, aka H_spiking
    """
    
    if H_uncond > 0:
        return np.abs(H_nsb - H_plugin) / H_uncond
    else:
        return np.inf
    
def bbc_estimator(symbol_counts,
                  past_symbol_counts,
                  alphabet_size,
                  alphabet_size_past,
                  H_uncond,
                  bbc_tolerance=None,
                  return_ais=False):
    """
    Estimate the entropy of a system using the BBC estimator.
    """

    mk = get_multiplicities(symbol_counts,
                            alphabet_size)
    mk_past = get_multiplicities(past_symbol_counts,
                                 alphabet_size_past)

    N = sum((mk[n] * n for n in mk.keys()))
                                        
    H_nsb_joint = nsb_entropy(mk, alphabet_size, N)
    H_nsb_past = nsb_entropy(mk_past, alphabet_size_past, N)

    H_nsb_cond = H_nsb_joint - H_nsb_past
    I_nsb = H_uncond - H_nsb_cond
    history_dependence = I_nsb / H_uncond
    
    H_plugin_joint = plugin_entropy(mk, N)

    if return_ais:
        ret_val = np.float(I_nsb)
    else:
        ret_val = np.float(history_dependence)

    if not bbc_tolerance == None:
        if bayesian_bias_criterion(H_nsb_joint, H_plugin_joint, H_uncond, bbc_tolerance):
            return ret_val
        else:
            return None
    else:
        return ret_val, np.float(get_bbc_term(H_nsb_joint,
                                              H_plugin_joint,
                                              H_uncond))
