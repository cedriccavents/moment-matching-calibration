# model calibration

from datetime import datetime, date
from math import log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import time

from pricers.blackscholes import *
from pricers.charfunction import CarrMadanFFT
from market_moments import OptionData, MarketImpliedMoments

class ModelCalibration:
    def __init__(self, spot, r, yte, model, quotes, verbose=False):
        self.s = spot
        self.r = r
        self.yte = yte
        self.model = model
        self.verbose = verbose
        self.quotes = quotes

        # calibration updates
        self.calib_guess = []
        self.calib_obs = None
        self.calib_model = []
        self.calib_strikes = None

    @staticmethod
    def goodness_of_fit(x_market: np.array, x_observed: np.array, fun='RMSE') -> float:
        obj_fun = {
            'RMSE': np.sqrt(1/len(x_market)*np.sum((x_market-x_observed)**2)),
            'AAE': np.sum(1/len(x_market)*np.absolute(x_market-x_observed)),
            'ARPE': (1/len(x_market))*np.sum(np.absolute(x_market-x_observed)/x_market)
        }

        try:
            return obj_fun[fun]
        except Exception:
            raise NotImplementedError(
                f'The objective function {fun} is not known! Please choose from {list(obj_fun.keys())}'
            )

    def find_root(self, args, target='CallPrice') -> float:
        """ Distance metric between observed quotes and model quotes
        """
        bs = BlackScholes('forward')
        strikes = np.array(list(self.quotes['prices'].keys()))
        option_types = np.array([list(self.quotes['prices'][x].keys())[0] for x in self.quotes['prices']])
        fft = CarrMadanFFT(self.s, self.yte, self.r, 0, strikes)
        c_model = fft.price(self.model, 'call', *args)
        p_model = fft.price(self.model, 'put', *args)

        if target == 'CallPrice':
            # price target
            alpha = 10e-2
            x_obs = np.array([list(self.quotes['prices'][x].values())[0] for x in self.quotes['prices']])
            strikes = strikes[option_types=='call']
            x_obs = x_obs[option_types=='call']
            x_model = c_model[option_types=='call']
        else:
            # implied vol target
            alpha = 1
            x_obs = np.array(list(self.quotes['vols'].values()))
            i = 0
            x_model = []
            while i < len(strikes):
                p = c_model[i] if option_type[i] == 'call' else p_model[i]
                model_vol = bs.calculate_implied_vol(
                    p, self.s, strikes[i], self.yte, self.r, option_types[i], initial_guess=0.5)
                x_model.append(model_vol)
                i += 1

        # distance metric
        fval = alpha*self.goodness_of_fit(np.array(x_obs), np.array(x_model), 'RMSE')

        if self.verbose:
            print(f'Calibration -- params: {args} -- fval: {fval}')

        self.calib_guess.append(args)
        self.calib_model.append(x_model)
        self.calib_obs = x_obs
        self.calib_strikes = strikes

        return fval

    def constraints(self) -> list:
        """ Define constraints per model
        """
        if self.model == 'Heston':
            # bounds = ((0.0, 1), (0.01, 5), ((0.01, 5), (0.01, 1), (-0.99, 0.99)))
            cons = [
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 0.99-x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                {'type': 'ineq', 'fun': lambda x: x[2] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 0.99-x[2]},
                {'type': 'ineq', 'fun': lambda x: x[3] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 5-x[3]},
                {'type': 'ineq', 'fun': lambda x: -x[4]},
                {'type': 'ineq', 'fun': lambda x: 1+x[4]}
            ]

        if self.model == 'MertonJumpDiffusion':
            cons = [
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 1.99-x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 5-x[1]},
                {'type': 'ineq', 'fun': lambda x: x[2] + 10},
                {'type': 'ineq', 'fun': lambda x: 10-x[2]},
                {'type': 'ineq', 'fun': lambda x: x[3] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 1.99-x[3]},
            ]

        if self.model == 'VG':
            cons = [
                {'type': 'ineq', 'fun': lambda x: x[0] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 1.99 - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[1] - 0.001},
                {'type': 'ineq', 'fun': lambda x: 4.99 - x[1]}
            ]
        return cons

    def calibrate(self, initial_guess: np.array, target='CallPrice') -> dict:
        """ Calibrate pricing model using minimisation procedure
        """

        cons = self.constraints()

        print(f'Calibrating {self.model} ...')
        res = minimize(
            fun = self.find_root,
            x0 = initial_guess,
            args=target,
            constraints=cons,
            method='SLSQP',
            tol=1e-6
        )

        return {
            'param': res.x,
            'message': res.message,
            'success': res.success,
            'eval': res.fun
        }

    def performance(self) -> None:
        """ Calibration plot: observed vs model quantities
        """
        raise NotImplementedError


class MomentMatchingCalibration(MarketImpliedMoments):
    """ Moment matching market implied calibration method
    """
    def _init__(self, spot, calc_date, expiry):
        super().__init__(spot, calc_date, expiry)


    def VG(self):
        """ VG model Moment Matching market implied Calibration
        """
        v_mkt, s_mkt, k_mkt = self.calculate_central_moments()

        # polynomial coefficients
        c1 = (1/(v_mkt*s_mkt**2))*(k_mkt/3-1)*self.yte
        c2 = (k_mkt-3)/(s_mkt**2) - 1
        c3 = 2*(v_mkt**2)/self.yte**2*(1-2*(k_mkt/3-1)/s_mkt**2)

        # analytical solution to cubic equation (Cardano)
        q = -c2**2/(9*c1**2)
        r = -c3/(2*c1) - (c2/(3*c1))**3
        d = q**3 + r**3

        sigma = np.sqrt(-c2/(3*c1)+(r+np.sqrt(d))**(1/3) + (r-np.sqrt(d))**(1/3))
        v = (s_mkt**2*v_mkt**3)/(self.yte**2*(sigma**2+2*(v_mkt/self.yte))**2*(v_mkt/self.yte-sigma**2))
        theta = np.sign(s_mkt)*np.sqrt((1/v_mkt)*(v_mkt/self.yte-sigma**2))

        return sigma, v, theta


if __name__ == '__main__':

    # Moment Matching Calibration
    mm = MomentMatchingCalibration(spot=5868.25, calc_date='2025-01-17', expiry='2025-03-20')
    print(mm.VG())

    # Standard Calibration
    object = OptionData(spot=5868.25, calc_date='2025-01-17', expiry='2025-03-20')
    quotes = object.get_liquid_option_quotes()
    heston = ModelCalibration(5868.25, object.get_rfr(), object.yte, 'Heston', quotes)
    mertonjd = ModelCalibration(5868.25, object.get_rfr(), object.yte, 'MertonJumpDiffusion', quotes, verbose=True)
    vg = ModelCalibration(5868.25, object.get_rfr(), object.yte, 'VG', quotes, verbose=True)

    # calibrate
    heston.calibrate(initial_guess=[0.2, 0.5, 0.04099, 0.1, -0.59], target='CallPrice')
    vg.calibrate(initial_guess=[0.2, 0.8, -1], target='CallPrice')
    # mertonjd.calibrate(initial_guess=[0.2, 1, -0.1, 0.6], target='CallPrice')

    # calibration fit
    n = len(heston.calib_guess)
    m = len(mertonjd.calib_guess)
    p = len(vg.calib_guess)
    plt.plot(heston.calib_strikes, heston.calib_obs, '*', label='market')
    plt.plot(heston.calib_strikes, heston.calib_model[n-1], "--", label=f'Heston')
    # plt.plot(mertonjd.calib_strikes, mertonjd.calib_model[m-1], "--", label=f'Merton JD')
    plt.plot(vg.calib_strikes, vg.calib_model[m-1], "--", label=f'VG')
    plt.legend()
    plt.title(f'Calibration fit ({object.e.strftime('%Y-%m-%d')})')
    plt.show()


