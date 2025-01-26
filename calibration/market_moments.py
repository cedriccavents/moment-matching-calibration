# Calculation of market implied moments from vanilla option quotes
from datetime import datetime, date
from math import log
import pandas as pd
import numpy as np
import statsmodels.api as sm

from pricers.blackscholes import *

class OptionData:
    def __init__(self, spot, calc_date, expiry) -> None:
        self.s = spot
        self.calc_date = datetime.strptime(calc_date, '%Y-%m-%d')
        self.e = datetime.strptime(expiry, '%Y-%m-%d')

        # year-to-maturity
        self.yte = (self.e-self.calc_date).days/365

        # execute
        self.df = self.read_data()
        self.call_grid, self.put_grid = self.clean_data()

    def read_data(self) -> pd.DataFrame:
        # read option data and process
        calc_date = self.calc_date.strftime('%m-%d-%Y')
        try:
            df = pd.read_csv(
                rf"./data/esh25-options-monthly-options-exp-"
                rf"{self.e.strftime('%m_%d_%y')}-show-all-stacked-intraday-{calc_date}.csv"
                )
            #  remove last row (redundant)
            df = df[:-1]
            return df
        except Exception as e:
            raise Exception(e)

    def clean_data(self):
        """ Basic cleanup of option quotes: (1) remove all options with 0 volume and (2) remove all options with 0
            bid prices
        """
        df = self.df[self.df['Volume'] > 0].reset_index(drop=True)
        df = df[df['Bid'] > 0].reset_index(drop=True)
        df['Strike'] = df['Strike'].apply(lambda x: float(x.replace('.00C', '').replace('.00P', '').replace(',','.'))*1000, 1)
        df['Price'] = 0.5*(df['Bid'] + df['Ask'])
        call_grid = df[df['Type'] == 'Call'].reset_index(drop=True)
        put_grid = df[df['Type'] == 'Put'].reset_index(drop=True)

        return call_grid, put_grid

    def get_rfr(self) -> float:
        """ Derive risk-free-rate from Put-Call Parity
        """
        common_strikes = list(set(self.call_grid['Strike']) & set(self.put_grid['Strike']))
        call_grid_common = self.call_grid[self.call_grid['Strike'].isin(common_strikes)].reset_index(drop=True)
        put_grid_common = self.put_grid[self.put_grid['Strike'].isin(common_strikes)].reset_index(drop=True)

        # regress Fwd vs strikes
        y = call_grid_common['Price'] - put_grid_common['Price']
        X = sm.add_constant(call_grid_common['Strike'])
        model = sm.OLS(y, X).fit()

        return (-1/self.yte)*log(-model.params[1])

    def get_liquid_option_quotes(self) -> dict:
        """ Determine list of liquid option quotes
        """
        liquid_strikes = list(set(self.call_grid['Strike']).union(self.put_grid['Strike']))
        liquid_strikes.sort()
        market_prices = {}
        for k in liquid_strikes:
            if (k/self.s < 0.75) & (k/self.s > 0.4) & (k in list(self.put_grid['Strike'])):
                v_put = self.put_grid[self.put_grid['Strike'] == k]['Price'].values[0]
                market_prices[k] = {'put': v_put}
            elif k in list(self.call_grid['Strike']):
                v_call = self.call_grid[self.call_grid['Strike'] == k]['Price'].values[0]
                market_prices[k] = {'call': v_call}
            else:
                print(f'Remove strike: {k}')

        # calibrate BS implied vol
        imp_vols = {}
        for k in market_prices:
            option_type = list(market_prices[k].keys())[0]
            bs = BlackScholes('forward')
            imp_vols[k] = bs.calculate_implied_vol(
                market_prices[k][option_type], self.s, k, self.yte, self.get_rfr(), option_type, initial_guess=0.5
            )

        return {'prices': market_prices, 'vols': imp_vols}


class MarketImpliedMoments(OptionData):
    """ Implementation of Market Implied Moments base
    """
    def __init__(self, spot, calc_date, expiry):
        super().__init__(spot, calc_date, expiry)

    def get_forward(self) -> [float, float]:
        """ calculate forward using Put-Call Parity where difference between Put and Call price is closest to 0.
        """
        common_strikes = list(set(self.call_grid['Strike']) & set(self.put_grid['Strike']))
        call_grid_common = self.call_grid[self.call_grid['Strike'].isin(common_strikes)].reset_index(drop=True)
        put_grid_common = self.put_grid[self.put_grid['Strike'].isin(common_strikes)].reset_index(drop=True)

        # find K where Put price is closest to Call price
        cp_diff = np.abs(np.array(call_grid_common['Price']) - np.array(put_grid_common['Price']))
        idx = np.where(cp_diff== np.min(cp_diff))
        k0 = call_grid_common.loc[idx].Strike.values[0]
        call_k0 = call_grid_common.loc[idx].Price.values[0]
        put_k0 = put_grid_common.loc[idx].Price.values[0]
        F = k0 + np.exp(self.get_rfr()*self.yte)*(call_k0 - put_k0)

        return F, k0

    def calculate_moments(self, N) -> np.array:
        """  Calculate N-th moment (based on option spanning formula of Bakshi and Madan)
            numerical integration using Trapezoidal rule
        """
        m = np.array(np.zeros(N+1))
        r = self.get_rfr()
        T = self.yte

        # strike grid
        F, k0 = self.get_forward()
        call_otm = self.call_grid[self.call_grid.Strike > k0].reset_index(drop=True)
        put_otm = self.put_grid[self.put_grid.Strike < k0].reset_index(drop=True)
        deltap = 0.5*(np.diff(put_otm.Strike))
        deltac = 0.5*(np.diff(call_otm.Strike))

        # integrands
        pf2_left = np.array(put_otm.Price)[:-1]/np.array(put_otm.Strike)[:-1]**2
        pf2_right =  np.array(put_otm.Price)[1:]/np.array(put_otm.Strike)[1:]**2
        cf2_left = np.array(call_otm.Price)[:-1]/np.array(call_otm.Strike)[:-1]**2
        cf2_right =  np.array(call_otm.Price)[1:]/np.array(call_otm.Strike)[1:]**2

        # central moments
        n = 2
        m[0] = np.log(k0/self.s) - 1 + F/k0 - 0.5*np.exp(r*T)*(
                np.sum((pf2_left + pf2_right)*deltap) + np.sum((cf2_left + cf2_right)*deltac))
        while n < N+1:
            p_left = (np.array(put_otm.Price)[:-1]/np.array(put_otm.Strike)[:-1]**2
                      *((n-1)*np.log(np.array(put_otm.Strike)[:-1]/self.s)**(n-2)
                        - np.log(np.array(put_otm.Strike)[:-1]/self.s)**(n-1)))
            p_right = (np.array(put_otm.Price)[1:]/np.array(put_otm.Strike)[1:]**2
                       *((n-1)*np.log(np.array(put_otm.Strike)[1:]/self.s)**(n-2)
                         - np.log(np.array(put_otm.Strike)[1:]/self.s)**(n-1)))
            c_left = (np.array(call_otm.Price)[:-1]/np.array(call_otm.Strike)[:-1]**2
                      *((n-1)*np.log(np.array(call_otm.Strike)[:-1]/self.s)**(n-2)
                        - np.log(np.array(call_otm.Strike)[:-1]/self.s)**(n-1)))
            c_right = (np.array(call_otm.Price)[1:]/np.array(call_otm.Strike)[1:]**2
                       *((n-1)*np.log(np.array(call_otm.Strike)[1:]/self.s)**(n-2)
                         - np.log(np.array(call_otm.Strike)[1:]/self.s)**(n-1)))

            m[n-1] = (np.log(k0/self.s)**n + n*np.log(k0/self.s)**(n-1)*(F/k0-1)
                      + np.exp(r*T)*n*0.5*(np.sum((p_left + p_right)*deltap) + np.sum((c_left + c_right)*deltac)))
            n += 1

        return m

    def calculate_central_moments(self) -> [float, float, float]:
        m = self.calculate_moments(N=3)
        mkt_var = m[1] - m[0]**2
        mkt_skew = (m[2]-3*m[0]*m[1]+2*m[0]**3)/(mkt_var)**(3/2)
        mkt_kurt = (m[3] - 4*m[0]*m[2] + 6*m[0]**2*m[1] - 3*m[0]**4)/(mkt_var)**2
        return mkt_var, mkt_skew, mkt_kurt


if __name__ == '__main__':

    market = MarketImpliedMoments(spot=5868.25, calc_date='2025-01-17', expiry='2025-03-20')
    print(market.calculate_central_moments())
