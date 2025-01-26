
import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

def spline_interpolation(xp, fp, xvals):
    """ Adjust Scipy's Cubic Spline interpolation with flat extrapolation
    """
    cs = CubicSpline(xp, fp, extrapolate=False)
    xinp = cs(xvals)
    # flat extrapolation
    k = 0
    while k < len(xinp):
        if xvals[k] > max(xp):
            xinp[k] = cs(max(xp))
        elif xvals[k] < min(xp):
            xinp[k] = cs(min(xp))
        k += 1
    return xinp

class BlackScholes:
    def __init__(self, underlying='spot', verbose=False):
        self.underlying = underlying
        self.verbose = verbose

    def coefficients(self, S, K, T, r, vol):
        d1 = np.log(S/K) + 0.5*vol**2/2*T
        d1 += r*T if self.underlying == 'spot' else 0
        d1 = 1/(vol*np.sqrt(T))*d1
        d2 = d1 - vol*np.sqrt(T)
        S_new = S*np.exp(r*T) if self.underlying == 'spot' else S
        return d1, d2, S_new

    def price(self, S, K, T, r, vol, optionType='call'):
        d1, d2, S2 = self.coefficients(S, K, T, r, vol)
        sign = 1 if optionType == 'call' else -1
        return np.exp(-r*T)*(sign*S2*norm.cdf(sign*d1)-sign*K*norm.cdf(sign*d2))

    def delta(self, S, K, T, r, vol, optionType='call'):
        d1, d2, S2 = self.coefficients(S, K, T, r, vol)
        sign = 1 if optionType == 'call' else -1
        return sign*norm.cdf(sign*d1)

    def calculate_ITM_probability(self, S, K, T, r, vol):
        S_new = S*np.exp(r*T) if self.underlying == 'spot' else S
        d = (1/(vol*np.sqrt(T)))*(np.log(K/S_new) - (r-vol**2/2)*T)
        return norm.cdf(d)

    def vega(self, S, K, T, r, vol):
        d1, d2, S2 = self.coefficients(S, K, T, r, vol)
        return S2*norm.pdf(d1)*np.sqrt(T)

    def gamma(self, S, K, T, r, vol):
        d1, d2, S2 = self.coefficients(S, K, T, r, vol)
        return 1/(S2*vol*np.sqrt(T))*norm.pdf(d1)

    def theta(self, S, K, T, r, vol, optionType):
        d1, d2, S2 = self.coefficients(S, K, T, r, vol)
        sign = 1 if optionType == 'call' else -1
        return -(S2*norm.pdf(d1)*vol)/(2*np.sqrt(T))-sign*r*K*np.exp(-r*T)*norm.cdf(sign*d2)

    def calculate_implied_vol(self, target, S, K, T, r, optionType='call', initial_guess = 0.2, max_iter=10**3):
        """ Newton-Raphson method to compute implied volatility to match target price:
        """
        precision = 1.0e-5
        vol = initial_guess
        for i in range(0, max_iter):
            price = self.price(S, K, T, r, vol, optionType)
            vega = self.vega(S, K, T, r, vol)
            diff = target - price
            if self.verbose:
                print(f'iter {i} -- vol = {vol}, target = {target}, price = {price}, ab(diff) = {diff}')
            if abs(diff) < precision:
                return vol
            vol += diff/vega
        return vol
