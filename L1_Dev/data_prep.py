from L0_Library.config import *


class DataPrep:
    """
    DataPrep class for preparing data using the databento API.

    Attributes:
        API_key (str): The API key for accessing the databento API.
        dataset (str): The name of the dataset.
        start_date (datetime): The start date for data retrieval.
        start_hour (int): The start hour for data retrieval.
        start_minute (int): The start minute for data retrieval.
        end_date (datetime): The end date for data retrieval.
        end_hour (int): The end hour for data retrieval.
        end_minute (int): The end minute for data retrieval.
        time_zone (str): The time zone to use for data retrieval and processing.
    """

    def __init__(self, API_key: str, dataset: str, time_zone: str, start_date: datetime, start_hour: int,
                 start_minute: int, end_date: datetime, end_hour: int, end_minute: int):

        self.API_key = API_key
        self.dataset = dataset
        self.start_date = start_date
        self.start_hour = start_hour
        self.start_minute = start_minute
        self.end_date = end_date
        self.end_hour = end_hour
        self.end_minute = end_minute
        self.time_zone = time_zone

    def get_data(self, symbol):
        """
        Retrieve data from DataBento's schema mbp-10 using class attributes.

        Parameters:
            symbol (str): The symbol for which to retrieve data.

        Returns:
            df_result (pd.DataFrame): A DataFrame containing the retrieved data.
            trading_days (int): The number of trading days processed.
            trading_dates (list): A list of trading dates.
        """
        df_result = pd.DataFrame()
        trading_days = 0
        trading_dates = []

        start_date = datetime(self.start_date.year, self.start_date.month, self.start_date.day, self.start_hour,
                              self.start_minute)
        end_date = datetime(self.end_date.year, self.end_date.month, self.end_date.day, self.end_hour, self.end_minute)

        # Number of days between start_date until end_date inclusive on both dates.
        num_days = (end_date - start_date).days + 1

        for i in tqdm(range(num_days), desc='Processing', ncols=100):
            start = start_date + timedelta(days=i)
            end = start_date + timedelta(days=i)
            end = end.replace(hour=self.end_hour, minute=self.end_minute, second=0, microsecond=0)

            start_iso = pytz.timezone(self.time_zone).localize(start).isoformat()
            end_iso = pytz.timezone(self.time_zone).localize(end).isoformat()

            # ================================= databento API =================================================
            client = db.Historical(self.API_key)
            data = client.timeseries.get_range(dataset=self.dataset,
                                               start=start_iso,
                                               end=end_iso,
                                               symbols=symbol,
                                               schema='mbp-10')
            data = data.to_df()
            data = data.set_index('ts_event')
            # ===================================================================================================

            if len(data.index) != 0:
                trading_days += 1
                trading_dates.append(pd.unique(data.index.date)[0])
                data = data[['symbol', 'action', "side", "depth", "price", "size",
                             'bid_px_00', 'ask_px_00', "bid_sz_00", "ask_sz_00",
                             'bid_px_01', 'ask_px_01', "bid_sz_01", "ask_sz_01"]]
                df_result = pd.concat(objs=[df_result, data], axis=0)

        # Convert the timezone of the index to time_zone
        df_result.index = df_result.index.tz_convert(self.time_zone)
        # Convert the index to datetime format and remove timezone information
        df_result.index = pd.to_datetime(df_result.index).tz_localize(None)

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result, trading_days, trading_dates

    def mid_price(self, df_input: pd.DataFrame, resample_freq: str, type_mid: str, drop_na: bool):
        """
        Calculate the mid-price of a given DataFrame with Limit Order Book data with options to specify the type of
        mid-price calculation, resampling frequency, and handling of missing values.

        Parameters:
            df_input (pd.DataFrame): The input DataFrame containing bid and ask price and size data.
            resample_freq (str): The resampling frequency for aggregating data (e.g., '1T' for 1-minute bars).
            type_mid (str): The type of mid-price calculation:
                            - 'last': Use the last observation for mid-price.
                            - 'mean': Use the average of bid and ask prices for mid-price.
                            - 'vwmp': Use the Volume Weighted Mid-Price (VWMP) calculation.
            drop_na (bool): If True, drop rows with missing values after resampling. If False, keep rows with
                            NaN values.

        Returns:
            df_result (pd.DataFrame): A DataFrame containing the calculated mid-price series with date and time
                                      as columns.
        """
        df_input = df_input.copy()[['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00']]
        df_result = pd.DataFrame()

        # Calculate the mid-price using the last observation
        if type_mid == 'last':
            df_result[f'mid-price_{type_mid}'] = (df_input['bid_px_00'] + df_input['ask_px_00']) * 0.5
            if drop_na:
                df_result = df_result.resample(resample_freq).last().dropna()
            else:
                df_result = df_result.resample(resample_freq).last()

        # Calculate the mid-price using the average mid-price
        if type_mid == "mean":
            df_result[f"mid-price_{type_mid}"] = (df_input["bid_px_00"] + df_input["ask_px_00"]) * 0.5
            if drop_na:
                df_result = df_result.resample(resample_freq).mean().dropna()
            else:
                df_result = df_result.resample(resample_freq).mean()

        # Calculate the mid-price using Volume Weighted Mid-Price (VWMP)
        if type_mid == "vwmp":
            df_input["vwbp_denominator"] = (df_input["bid_px_00"] * df_input["bid_sz_00"])
            df_input["vwap_denominator"] = (df_input["ask_px_00"] * df_input["ask_sz_00"])

            df_resampled = df_input.resample(resample_freq).agg(['sum', 'count'])
            df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]

            mask_index = df_resampled.index
            mask = np.all([(df_resampled[f"{col}_sum"] == 0) & (df_resampled[f"{col}_count"] == 0)
                           for col in df_input.columns], axis=0)
            mask = pd.Series(mask, index=mask_index)

            if drop_na:
                df_resampled = df_resampled.loc[~mask]
            else:
                df_resampled = df_resampled.where(~mask, np.nan)

            vwbp = df_resampled["vwbp_denominator_sum"] / df_resampled["bid_sz_00_sum"]
            vwap = df_resampled["vwap_denominator_sum"] / df_resampled["ask_sz_00_sum"]
            df_result[f"mid-price_{type_mid}"] = (vwbp + vwap) * 0.5

        start_date = datetime(self.start_date.year, self.start_date.month, self.start_date.day, self.start_hour,
                              self.start_minute)
        end_date = datetime(self.end_date.year, self.end_date.month, self.end_date.day, self.end_hour, self.end_minute)

        # .resample() creates dates in between dates (i.e: adds np.NaN every minute between 4PM and 9:30AM of next day)
        df_result = df_result[(df_result.index.time >= start_date.time()) & (df_result.index.time < end_date.time())]

        df_result['Date'] = pd.to_datetime(df_result.index.date, format='%m/%d/%Y')
        df_result['Hour'] = pd.to_datetime(df_result.index.strftime('%H:%M:%S'), format='%H:%M:%S').time

        df_result = df_result.pivot(index='Hour', columns='Date', values=f"mid-price_{type_mid}")

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def col_return(col: pd.Series):

        log_returns = []
        counter = 1

        log_returns.append(np.nan)

        for i in range(1, len(col)):

            if np.isnan(col[i]):
                log_returns.append(np.nan)
                counter += 1
            else:
                log_return = 1 / np.sqrt(counter) * np.log(col[i] / col[i - counter])
                log_returns.append(log_return)
                counter = 1

        s_result = pd.Series(log_returns, index=col.index)

        return s_result

    def returns(self, df_input: pd.DataFrame):

        df_result = pd.DataFrame()
        for column in df_input.columns:
            df_result[column] = self.col_return(df_input[column])
            df_result.columns.name = 'Date'

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def bipower_variation(arr: pd.Series, K: int):
        """
        Calculate the Bipower Variation of a pandas' series.

        Parameters:
            arr (pandas.Series): Input time series of returns.
            K (int): Parameter used in the calculation.

        Returns:
            bipower_var (float): Bipower Variation value.

        Reference:
        - [46] K. Boudt, C. Croux, and S. Laurent, Journal of Empirical Finance 18, 353 (2011)
          | Barndorff-Nielsen and Shephard (2004) Power and bipower variation with stochastic volatility and jumps.
          J. Financ. Econometrics 2, 1–37.

        The realized bipower variation (RBV) is a measure used in financial econometrics to estimate local volatility
        in the presence of jumps. The bipower variation is robust to jumps and can be used to separate
        continuous and jump components of volatility.

        In the paper, they use a rolling time window of length K = 390 (i.e. one day worth of data, but dropping any
        overnight contribution).
        """
        bipower_var = (np.pi / (2 * (K - 1))) * np.sum(np.abs(arr) * np.abs(arr.shift(-1)))

        return bipower_var

    def rolling_bipower_variation(self, df_input: pd.DataFrame, K: int):
        """
        Calculate the rolling Bipower Variation for a DataFrame of returns.

        Parameters:
            df_input (pd.DataFrame): Input DataFrame containing time series of returns.
            K (int): Parameter used in the calculation.

        Returns:
            df_result (pd.DataFrame): DataFrame containing rolling Bipower Variation values.
        """
        df_temp = df_input.unstack()
        df_temp.dropna(inplace=True)
        # Calculate the rolling bipower variation using a lambda function
        df_temp = df_temp.rolling(window=K).apply(lambda x: self.bipower_variation(x, K=K))
        df_temp = df_temp.reset_index()
        df_temp.rename(columns={0: 'bipower_variation'}, inplace=True)
        df_result = df_input.unstack().reset_index().merge(df_temp, on=['Date', 'Hour'], how='left').drop(0, axis=1)
        df_result = df_result.pivot(index='Hour', columns='Date', values=f'bipower_variation')

        pd.options.display.float_format = '{:,.10f}'.format

        return df_result

    @staticmethod
    def periodicity(row: pd.Series, threshold: float):
        """
        Calculate the non-parametric periodicity estimator based on Taylor and Xu (1997).

        Parameters:
            row (pandas.Series): A pandas Series containing standardized returns.
            threshold (float): A threshold value to determine the periodicity.

        Returns:
            s_result (pandas.Series): The calculated periodicity measure.

        Reference:
            The non-parametric periodicity estimator proposed by Taylor and Xu (1997) is based on the standard deviation
            (SD) of all standardized returns belonging to the same local window. It calculates the periodicity measure for
            a single row of standardized returns.
        """
        squared_function = lambda x: x ** 2 if not np.isnan(x) else np.nan
        squared_returns = row.apply(squared_function)

        step_function = lambda x: 1 if x <= threshold else 0
        weights = squared_returns.apply(step_function)

        s_result = np.sqrt(1.081 * ((weights * squared_returns).sum() / weights.sum()))

        return s_result

    def df_periodicity(self, df_input: pd.DataFrame, threshold: float):
        """
        Calculate the periodicity estimator for a DataFrame of standardized returns.

        Parameters:
            df_input (pandas.DataFrame): A DataFrame containing standardized returns in columns.
            threshold (float): value to remove outliers.
        Returns:
            df_result (pandas.DataFrame): A DataFrame with the periodicity measure calculated for each column.
        """
        cols = df_input.columns
        s_result = df_input.apply(lambda x: self.periodicity(x, threshold=threshold), axis=1)
        s_result = s_result / np.sqrt((s_result ** 2).sum() / 389)
        df_result = pd.concat([s_result] * len(cols), axis=1)
        df_result.columns = cols

        df_result = df_result.where(df_input.notna(), np.nan)

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def jump_score(df_returns: pd.DataFrame, df_bipower_variation: pd.DataFrame, df_f: pd.DataFrame):

        df_result = df_returns / (np.sqrt(df_bipower_variation) * df_f)

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def get_jumps(df_input: pd.DataFrame):

        df_result = df_input.applymap(lambda x: np.nan if np.isnan(x) else 1 if np.abs(x) > 4.36 else 0)
        df_result.iloc[:15, :] = 0
        df_result.iloc[375:, :] = 0
        df_result = df_result.where(df_input.notna(), np.nan)

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result
