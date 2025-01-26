# Option pricing using FFT

from abc import ABC, abstractmethod
from math import exp, log, pi
import numpy as np
from scipy.fft import fft
from scipy.interpolate import CubicSpline

class CharFunc(ABC):
    @abstractmethod
    def __init__(self, s, e, r, q, k):
        # vanilla option parameters
        self.spot = s
        self.expiry = e
        self.rfr = r
        self.div_yield = q
        self.strikes = k

    def build(self, *args):
        pass


class BlackScholes(CharFunc):
    """ Class to implement methods specific to the Black-Scholes model
    """
    def __init__(self, s, e, r, q, k):
        super().__init__(s, e, r, q, k)

    def build(self, u, sigma):
        return np.exp(u*1j*(np.log(self.spot)
                            +(self.rfr-self.div_yield-0.5*sigma**2)*self.expiry)-0.5*sigma**2*self.expiry*u**2)


class MertonJumpDiffusion(CharFunc):
    """ Class to implement methods specific to the Black-Scholes model
    """
    def __init__(self, s, e, r, q, k):
        super().__init__(s, e, r, q, k)

    def build(self, u, sigma, lamda, mjump, voljump):
        k = exp(mjump+0.5*voljump**2) - 1
        return (np.exp(u*1j*(np.log(self.spot)
                            +(self.rfr-self.div_yield-0.5*sigma**2 - lamda*k)*self.expiry)-0.5*sigma**2*self.expiry*u**2)
                *np.exp(self.expiry*lamda*(np.exp(1j*u*mjump-0.5*voljump**2*u**2))-1))


class VG(CharFunc):
    """ Class to implement methods specific to the VG model
    """
    def __init__(self, s, e, r, q, k):
        super().__init__(s, e, r, q, k)

    def build(self, u, sigma, v, theta):
        w = (1/v)*np.log(1-theta*v-0.5*sigma**2*v)
        return (np.exp(1j*u*(np.log(self.spot) + self.rfr-self.div_yield + w)*self.expiry)
                *(1-1j*u*theta*v+0.5*u**2*sigma**2*v)**(-self.expiry/v))


class Heston(CharFunc):
    """ Class to implement methods specific to the Heston Stochastic Volatility Model
    """
    def __init__(self, s, e, r, q, k):
        super().__init__(s, e, r, q, k)

    def build(self, u, v0, kappa, theta, sigma, rho):
        # risk neutrality of params
        kappa = kappa
        tau = self.expiry

        # helper vars
        m = np.sqrt((rho*sigma*1j*u-kappa)**2+sigma**2*(1j*u+u**2))
        n = (rho*sigma*1j*u-kappa-m)/(rho*sigma*1j*u-kappa+m)

        # coefficients
        A = self.rfr*1j*u*tau+(kappa*theta)/sigma**2*(-(rho*sigma*1j*u-kappa-m)*tau-2*np.log((1-n*np.exp(m*tau))/(1-n)))
        B = 0
        C = ((np.exp(m*tau)-1)*(rho*sigma*1j*u-kappa-m))/(sigma**2*(1-n*np.exp(m*tau)))

        return np.exp(A+B*np.log(self.spot)+C*v0+1j*u*np.log(self.spot))

    def build2(self, u, v0, kappa, theta, sigma, rho):
        # risk neutrality of params
        kappa = kappa
        tau = self.expiry

        zeta = kappa-1j*u*rho*sigma
        d = np.sqrt(zeta**2+u*sigma**2*(1j+u))
        A1 = u*(1j+u)*np.sinh(d*tau/2)
        A2 = (1/v0)*(d*np.cosh(d*tau/2)+zeta*np.sinh(d*tau/2))
        G = np.log(d/v0) + 0.5*(kappa-d)*tau-np.log((d+zeta)/(2*v0) + ((d-zeta)*np.exp(-d*tau))/(2*v0))
        return np.exp(1j*u*tau*(self.rfr-(kappa*theta*rho)/sigma)-(A1/A2) + (2*kappa*theta*G)/sigma**2)

class NumericalIntegrationScheme:
    """ Class to implement numerical integration schemes
    """
    def __init__(self, N):
        self.N = N

    def trapezoid(self):
        w = np.ones(self.N)
        w[0] = w[-1] = 1/2
        return w

    def simpson(self):
        raise NotImplementedError


class CarrMadanFFT(CharFunc):
    """ Class to price vanilla options using the Carr Madan formula to price using characteristic function of the log
        asset price using Fast Fourtier Transformations (FFT).
    """
    # supported char functions
    models = {
            'BlackScholes': BlackScholes,
            'Heston': Heston,
            'MertonJumpDiffusion': MertonJumpDiffusion,
            'VG': VG
    }

    def __init__(self, s, e, r, q, k, N=None) -> None:
        super().__init__(s, e, r, q, k)

        # FFT parameters
        self.alpha = 1.5
        self.delta_u = 0.25

        if not N:
            self.N = 4096

    def build(self):
        pass

    def grid(self):
        # equidistant integration grid u
        u = self.delta_u * np.arange(0, self.N)

        # grid of strikes
        delta_k = (2 * pi) / (self.N * self.delta_u)
        b = (self.N * delta_k) / 2
        k = -b + delta_k * np.arange(0, self.N)

        return [u, k, b]

    def price(self, model, option_type='call', *args):

        # grids for var and strikes (k)
        [u, k, b] = self.grid()

        # integration weights
        w = NumericalIntegrationScheme(self.N).trapezoid()

        if model in CarrMadanFFT.models:

            # char function
            cf = CarrMadanFFT.models[model].build(self, u - (self.alpha + 1) * 1j, *args)

            # FFT
            psi = cf * exp(-self.rfr * self.expiry)/(self.alpha ** 2 + self.alpha - u ** 2 + 1j * (2 * self.alpha + 1) * u)
            z = np.exp(1j * u * b) * self.delta_u * psi * w
            call_prices_grid = np.exp(-self.alpha * k) * (1 / pi) * fft(z)
            cs = CubicSpline(k, call_prices_grid.real)

            if option_type =='call':
                return cs(np.log(self.strikes))
            elif option_type == 'put':
                return cs(np.log(self.strikes))-self.spot+self.strikes*np.exp(-self.rfr*self.expiry)

        else:
            raise NotImplementedError(f"The char function for the {model} model is not implemented!")
