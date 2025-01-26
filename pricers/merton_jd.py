# Merton Jump-diffusion model

from math import exp, sqrt, log, factorial

from pricers.blackscholes import BlackScholes

def merton_jd_price(
        s: float,
        k: float,
        r: float,
        q: float,
        sigma: float,
        t: float,
        lambda_jump: float,
        mean_jump: float,
        sigma_jump: float,
        n: int
):
    # expected jump value
    kappa = exp(mean_jump + 0.5*sigma_jump**2) - 1

    p = 0
    j = 0
    while j < n:
        vol = sqrt(sigma**2+j*sigma_jump**2/t)
        drift = r-lambda_jump*kappa+j*log(1+kappa)
        fwd = s*exp((drift-q)*t)
        p_bs = BlackScholes(fwd, k, t, r, q).price(vol, 'call')
        lambda_jump_new = lambda_jump*(1+kappa)
        poisson_probability = exp(-lambda_jump_new*t)*(lambda_jump_new*t)**j/factorial(j)

        p+= poisson_probability*p_bs
        j += 1

    return p
