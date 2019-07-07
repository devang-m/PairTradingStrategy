import pandas
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from scipy.stats import normaltest

def plot_util(dataframe, x, y, title, xlabel, ylabel):
    graph = dataframe.plot(x=x, y=y, title=title)
    graph.set(xlabel=xlabel, ylabel=ylabel)
    plt.show()

class TradingStrategy:

    def __init__(self):
        self.london_data = pandas.read_csv('data/RIO LN Equity.csv', parse_dates=['Dates'])
        self.australian_data = pandas.read_csv('data/RIO AU Equity.csv', parse_dates=['Dates'])
        self.forex_USD_AUD = pandas.read_csv('data/AUDUSD Curncy.csv', parse_dates=['Dates'])
        self.forex_USD_GBP = pandas.read_csv('data/GBPUSD Curncy.csv', parse_dates=['Dates'])

    def cleanData(self):
        self.london_data['PX_LAST_GBP'] = self.london_data['PX_LAST'] / (100)

        # Dates in forex conversion dataset for which the stock prices are there
        self.forex_USD_GBP = self.forex_USD_GBP[self.forex_USD_GBP['Dates'].isin(self.london_data['Dates'])]
        self.forex_USD_AUD = self.forex_USD_AUD[self.forex_USD_AUD['Dates'].isin(self.australian_data['Dates'])]

        # Resetting the index
        self.forex_USD_AUD.reset_index(inplace=True, drop=True)
        self.forex_USD_GBP.reset_index(inplace=True, drop=True)

        # Resetting the dates to only the common dates when both stocks were traded
        self.australian_data = self.australian_data[self.australian_data['Dates'].isin(self.london_data['Dates'])]
        self.australian_data.reset_index(inplace=True, drop=True)
        self.london_data = self.london_data[self.london_data['Dates'].isin(self.australian_data['Dates'])]
        self.london_data.reset_index(inplace=True, drop=True)

        # Finding the quote in USD to make the data comparable
        self.australian_data['PX_LAST_USD'] = self.australian_data['PX_LAST'] / self.forex_USD_AUD['PX_LAST']
        self.london_data['PX_LAST_USD'] = self.london_data['PX_LAST_GBP'] / self.forex_USD_GBP['PX_LAST']

    def splitTestAndTrain(self):
        ratio = (int)(0.6 * len(self.australian_data.index))
        self.australian_train = self.australian_data[:ratio]
        self.london_train = self.london_data[:ratio]
        self.australian_test = self.australian_data[ratio:]
        self.london_test = self.london_data[ratio:]
        self.australian_test.reset_index(inplace=True, drop=True)
        self.london_test.reset_index(inplace=True, drop=True)

    # Just did some visualisation to understand the data. Not used in the final model
    def visualisationAndBasicAnalysis(self, dataframe_london, dataframe_aus):
        # Viewing the time series
        plt.plot(dataframe_aus['Dates'], dataframe_aus['PX_LAST_USD'], 'b-', label='Australia')
        plt.plot(dataframe_aus['Dates'], dataframe_london['PX_LAST_USD'], 'g-', label='London')
        plt.legend(loc='upper left')
        plt.xlabel('Dates')
        plt.ylabel('Price in USD')

        # Get returns percentage change
        dataframe_aus['Returns'] = dataframe_aus['PX_LAST_USD'].pct_change()
        dataframe_london['Returns'] = dataframe_london['PX_LAST_USD'].pct_change()

        # Getting cumulative return
        dataframe_aus['Cumulative'] = (1 + dataframe_aus['Returns']).cumprod()
        plot_util(dataframe_aus, 'Dates', 'Cumulative', 'Cumulative Return for Australian stock', "Dates", "Return")

        dataframe_london['Cumulative'] = (1 + dataframe_london['Returns']).cumprod()
        plot_util(dataframe_london, 'Dates', 'Cumulative', 'Cumulative Return for London stock', "Dates", "Return")

        # Getting rolling mean
        dataframe_aus['RollingMean30'] = dataframe_aus['PX_LAST_USD'].rolling(window=30).mean()
        plot_util(dataframe_aus, 'Dates', 'RollingMean30', 'Rolling Mean for Australian stock', "Dates", "Rolling Mean for 30 days in USD")

        dataframe_london['RollingMean30'] = dataframe_london['PX_LAST_USD'].rolling(window=30).mean()
        plot_util(dataframe_london, 'Dates', 'RollingMean30', 'Rolling Mean for London stock', "Dates", "Rolling Mean for 30 days in USD")

    def correlation(self, dataframe_london, dataframe_aus):
        # Correlation coeff
        corr = np.corrcoef(list(dataframe_london['PX_LAST_USD']), list(dataframe_aus['PX_LAST_USD']))[0, 1]
        print("Correlation Coeff: " + str(corr))
        # Checking if the 2 stocks have sufficient correlation
        if corr < 0.9:
            print("Not enough correlation")
            exit(0)

    def regression(self, dataframe_london, dataframe_aus):
        # OLS Regression
        price_ols = sm.OLS(list(dataframe_london['PX_LAST_USD']), list(dataframe_aus['PX_LAST_USD'])).fit()
        price_ols_robust = price_ols.get_robustcov_results()
        print("OLS t-statistic: " + str(price_ols_robust.tvalues[0]) + "\n")
        return price_ols_robust

    def cointegrationCheck(self, dataframe_london, dataframe_aus):
        # Check for correlation
        self.correlation(dataframe_london, dataframe_aus)

        # Check for co-integration
        score, pv, cv = coint(dataframe_aus['PX_LAST_USD'], dataframe_london['PX_LAST_USD'])
        print("Co integration check: " + str(pv < 0.1) + "\nP value: " + str(pv) + "\n")
        if pv > 0.1:
            print("No co-integration")
            exit(0)

        # Check for regression
        model = self.regression(dataframe_london, dataframe_aus)
        self.beta = model.params[0]

        self.common_df = pd.DataFrame()
        self.common_df['Dates'] = dataframe_aus['Dates']
        # From the OLS formula: london price = beta * aus price + residual
        self.common_df['residual'] = dataframe_london['PX_LAST_USD'] - (self.beta * dataframe_aus['PX_LAST_USD'])
        self.common_df['ratio'] = dataframe_london['PX_LAST_USD']/dataframe_aus['PX_LAST_USD']
        residual_mean = self.common_df['residual'].mean()
        residual_sd = self.common_df['residual'].std()
        # Checking if the residuals are normally distributed to apply Z test
        stat, p = normaltest(self.common_df['residual'])
        print("P-value for normality check: " + str(p) + "\nResiduals are normally distributed: " + str(p > 0.1) + "\n")
        if p < 0.1:
            print("Residuals don't look gaussian")
            exit(0)

        self.common_df['normalised'] = (self.common_df['residual'] - residual_mean)/residual_sd
        plt.plot(self.common_df['Dates'], self.common_df['normalised'], 'green', label="Co-integration movement of normalised returns")
        plt.xlabel("Dates")
        plt.ylabel("Normalised Residual from OLS Regression (SD from Mean)")
        plt.axhline(0, color='black')
        plt.axhline(1, color='red', linestyle='--')
        plt.axhline(-1, color='red', linestyle='--')
        plt.show()

    def marketSignals(self):
        # If the normalised residual is lower by one sd, then long london, short AUS
        self.common_df['longLondon'] = self.common_df['normalised'] < -1
        # If the normalised residual is higher by one sd, then short london, long AUS
        self.common_df['shortLondon'] = self.common_df['normalised'] > 1
        # If the normalised residual is within 0.25 sd, then exit position
        self.common_df['exit'] = (-0.25 < self.common_df['normalised']) & (self.common_df['normalised'] < 0.25)
        # If the normalised residual is over abs 3 sd, then trigger stop loss
        self.common_df['stopLoss'] = (3 < self.common_df['normalised']) | (self.common_df['normalised'] < -3)

    def tradingStrat(self, dataframe_london, dataframe_aus):
        total_realised_pnl = 0
        # investment_current - Stores the total capital required to start the trading.
        # Since ours is a hedged position, the total capital is the money used to but the long position
        # and 50% of that for the margin account
        investment_current = 0
        investment_max = 0
        shares_london = 0
        shares_aus = 0
        self.common_df['total'] = 0
        for i, row in enumerate(self.common_df.iterrows()):
            # this column has the total unrealised PnL for the trades
            self.common_df['total'][i] = total_realised_pnl + shares_aus * dataframe_aus['PX_LAST_USD'][i] + shares_london * dataframe_london['PX_LAST_USD'][i]
            if row[1]['longLondon']:
                # Ensuring that we do not take too much of a position
                if shares_london > 1000:
                    continue
                investment_current += 1.5 * 100 * dataframe_london['PX_LAST_USD'][i]
                shares_london += 100
                shares_aus -= 100 * self.common_df['ratio'][i]
            if row[1]['shortLondon']:
                # Ensuring that we do not take too much of a position
                if shares_london < -1000:
                    continue
                investment_current -= 1.5 * 100 * dataframe_london['PX_LAST_USD'][i]
                shares_london -= 100
                shares_aus += 100 * self.common_df['ratio'][i]
            # Remove position if we meet stoploss or exit criteria or we reach end of trading
            if row[1]['exit'] or row[1]['stopLoss'] or i == len(self.common_df)-1:
                total_realised_pnl += shares_london*dataframe_london['PX_LAST_USD'][i] + shares_aus*dataframe_aus['PX_LAST_USD'][i]
                shares_london = 0
                shares_aus = 0
                investment_max = max(investment_max, abs(investment_current))
                investment_current = 0
        plot_util(self.common_df, 'Dates', 'total', 'Cummulative Profit Graph', "Dates", "Profit in USD")
        print("Net Profit: $" + str(total_realised_pnl) +"\n")
        print("Net Return is: " + str((100*total_realised_pnl)/(investment_max)) + "%\n")

obj = TradingStrategy()
obj.cleanData()
obj.splitTestAndTrain()
#obj.visualisationAndBasicAnalysis(obj.london_train, obj.australian_train)

# Training dataset
obj.cointegrationCheck(obj.london_train, obj.australian_train)
obj.marketSignals()
obj.tradingStrat(obj.london_train, obj.australian_train)

# Test dataset
obj.cointegrationCheck(obj.london_test, obj.australian_test)
obj.marketSignals()
obj.tradingStrat(obj.london_test, obj.australian_test)
