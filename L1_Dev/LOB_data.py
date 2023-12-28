from L0_Library.config import *


class DataLOB:
    """
    ts_recv: The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
    ts_event: The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
    rtype: The record type. Each schema corresponds with a single rtype value.
    publisher_id: The publisher ID assigned by Databento, which denotes dataset and venue.
    instrument_id: The numeric instrument ID.
    action: The event action. Can be [A]dd, [C]ancel, [M]odify, clea[R] book, or [T]rade.
    side: The order side. Can be [A]sk, [B]id or [N]one.
    depth: The book level where the update event occurred.
    price: The order price expressed as a signed integer where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.
    size: The order quantity.
    flags: A combination of packet end with matching engine status.
    ts_in_delta: The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv.
    sequence: The message sequence number assigned at the venue.
    action: The event action. Will always be [T]rade in the TBBO schema.
    side: The aggressive order's side in the trade. Can be [A]sk, [B]id or [N]one.
    depth: The book level where the update event occurred.
    price: The order price expressed as a signed integer where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.
    size: The order quantity.
    bid_px_N: The bid price at level N (top level if N=00).
    ask_px_N: The ask price at level N (top level if N=00).
    bid_sz_N: The bid size at level N (top level if N=00).
    ask_sz_N: The ask size at level N (top level if N=00).
    bid_ct_N: The number of bid orders at level N (top level if N=00).
    ask_ct_N: The number of ask orders at level N (top level if N=00).
    """

    def __init__(self, dataset: str, API_key: str, resample_freq: str, start_date: datetime, end_date: datetime,
                 start_hour: int, end_hour: int, start_minute: int, end_minute: int, time_zone: str, schema: str):

        self.dataset = dataset
        self.API_key = API_key
        self.resample_freq = resample_freq
        self.start_date = start_date
        self.end_date = end_date
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.start_minute = start_minute
        self.end_minute = end_minute
        self.time_zone = time_zone
        self.schema = schema

    ################################################################################
    ########################### Pull Data from Databento ###########################
    ################################################################################
    
    def build_data_set(self, symbol):
        """
        This function gets data from databento

        symbol:str
        start_date:date object (YYYY, MM, DD)
        start_hour:int
        start_minute:int
        end_date:date object (YYYY, MM, DD)
        end_hour:int
        end_minute:int
        """

        start_date = datetime(self.start_date.year, self.start_date.month, self.start_date.day, self.start_hour, self.start_minute)
        end_date = datetime(self.end_date.year, self.end_date.month, self.end_date.day, self.end_hour, self.end_minute)

        df = pd.DataFrame()
        count_days = 0
        index_with_data = []

        # Number of days between start_date until end_date inclusive on both dates.
        t_days = (end_date - start_date).days + 1

        for i in tqdm(range(t_days), desc="Processing", ncols=100):
            start = start_date + timedelta(days=i)
            end = start_date + timedelta(days=i)
            end = end.replace(hour=self.end_hour, minute=self.end_minute, second=0, microsecond=0)

            start_iso = pytz.timezone(self.time_zone).localize(start).isoformat()
            end_iso = pytz.timezone(self.time_zone).localize(end).isoformat()

            if self.schema == 'mbp-10':
                # ================================= databento API =================================================
                """
                MBP (Market-by-Price) provides changes to and snapshots of aggregated book depth, keyed by price.
                This is limited to a fixed number of levels from the top. We denote number of levels displayed
                in the schema with a suffix, such as mbp-1 and mbp-10.
                """

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
                    count_days += 1
                    index_with_data.append(pd.unique(data.index.date)[0])
                    data = data[['symbol', 'action', "side", "depth", "price", "size",
                                 'bid_px_00', 'ask_px_00', "bid_sz_00", "ask_sz_00",
                                 'bid_px_01', 'ask_px_01', "bid_sz_01", "ask_sz_01"]]
                    df = pd.concat([df, data], axis=0)

            else:
                print(f'Schema {self.schema} is not supported at the moment.')
                break

        # Convert the timezone of the index to time_zone
        df.index = df.index.tz_convert('US/Eastern')
        # Convert the index to datetime format and remove timezone information
        df.index = pd.to_datetime(df.index).tz_localize(None)

        return df, count_days, index_with_data

    ###############################################################################
    ################################## Mid-Price ##################################
    ###############################################################################

    def get_mid_price(self, df_orders: pd.DataFrame, type_mid: str, drop_na: bool):
        df = df_orders.copy()[["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]]
        df_prices = pd.DataFrame()

        # Calculate the mid-price using the last observation
        if type_mid == "last":
            df_prices[f"mid-price_{type_mid}"] = (df["bid_px_00"] + df["ask_px_00"]) * 0.5
            if drop_na:
                df_prices = df_prices.resample(self.resample_freq).last().dropna()
            else:
                df_prices = df_prices.resample(self.resample_freq).last()

        # Calculate the mid-price using the average mid-price
        if type_mid == "mean":
            df_prices[f"mid-price_{type_mid}"] = (df["bid_px_00"] + df["ask_px_00"]) * 0.5
            if drop_na:
                df_prices = df_prices.resample(self.resample_freq).mean().dropna()
            else:
                df_prices = df_prices.resample(self.resample_freq).mean()

        # Calculate the mid-price using Volume Weighted Mid-Price (VWMP)
        if type_mid == "vwmp":
            df["vwbp_denominator"] = (df["bid_px_00"] * df["bid_sz_00"])
            df["vwap_denominator"] = (df["ask_px_00"] * df["ask_sz_00"])

            df_resampled = df.resample(self.resample_freq).agg(['sum', 'count'])
            df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
            
            mask_index = df_resampled.index
            mask = np.all([(df_resampled[f"{col}_sum"] == 0) & (df_resampled[f"{col}_count"] == 0)
                           for col in df.columns], axis=0)
            mask = pd.Series(mask, index=mask_index)

            if drop_na:
                df_resampled = df_resampled.loc[~mask]
            else:
                df_resampled = df_resampled.where(~mask, np.nan)

            vwbp = df_resampled["vwbp_denominator_sum"] / df_resampled["bid_sz_00_sum"]
            vwap = df_resampled["vwap_denominator_sum"] / df_resampled["ask_sz_00_sum"]
            df_prices[f"mid-price_{type_mid}"] = (vwbp + vwap) * 0.5

        start_date = datetime(self.start_date.year, self.start_date.month, self.start_date.day, self.start_hour, self.start_minute)
        end_date = datetime(self.end_date.year, self.end_date.month, self.end_date.day, self.end_hour, self.end_minute)

        df_prices = self.filter_and_mask_df(df_prices, start_date, end_date)

        df_prices['Date'] = df_prices.index.date
        df_prices['Hour'] = df_prices.index.strftime('%H:%M:%S')
        
        df_prices = df_prices.pivot(index='Hour', columns='Date', values=f"mid-price_{type_mid}")
        
        return df_prices

    @staticmethod
    def filter_and_mask_df(df, start_date, end_date):
        # Filter return_df on mask to remove not desired dates for the DataFrame
        # RESAMPLE creates dates in between! --> np.NaN for every minute between 4PM and 9:30AM of next day
        mask = (df.index.time >= start_date.time()) & \
               (df.index.time < end_date.time())
        return df[mask]

    ###############################################################################
    ################################# Log-Returns #################################
    ###############################################################################

    def get_return(self, df_prices: pd.DataFrame):
        """
        [62] L. M. Calcagnile, G. Bormetti, M. Treccani, S. Marmi,
        and F. Lillo, Quantitative Finance 18, 237 (2018).
        """

        df = df_prices.copy()
        df_returns = pd.DataFrame()
        
        for column in df.columns:
            df_returns[column] = self.minute_log_return(df[column])

        return df_returns

    @staticmethod
    def minute_log_return(series):
        array = np.array(series)
        log_returns = []
        
        if np.isnan(array).all():
            log_returns_series = pd.Series([np.nan] * len(series), index=series.index)
            return log_returns_series        
    
        counter = 1
        for i in range(1, len(array)):
            if not np.isnan(array[i]):
                log_return = 1 / np.sqrt(counter) * np.log(array[i] / array[i - counter])
                log_returns.append(log_return)
                counter = 1
            else:
                log_returns.append(np.nan)
                counter += 1

        log_returns_series = pd.Series(log_returns, index=series.index[1:])
        return log_returns_series

    ###############################################################################
    ############################## Bipower Variation ##############################
    ###############################################################################

    def get_bipower_variation_estimator(self, df_returns: pd.DataFrame):
        """
        [46] K. Boudt, C. Croux, and S. Laurent, Journal of Empirical Finance 18, 353 (2011) | Barndorff-Nielsen and
        Shephard (2004) Power and bipower variation with stochastic volatility and jumps. J. Financ. Econometrics 2, 1–37.

        The realized bipower variation (RBV) is a measure used in financial econometrics to estimate local volatility
        in the presence of jumps. The bipower variation is robust to jumps and can be used to separate
        continuous and jump components of volatility.

        In the paper, they use a rolling time window of length K = 390 (i.e. one day worth of data, but dropping any
        overnight contribution)
        """

        df = df_returns.copy()
        df_bpv = pd.DataFrame()
        
        for column in df.columns:
            df_bpv[column] = self.day_bipower_variation(df[column])
    
        return df_bpv

    @staticmethod
    def day_bipower_variation(series):
        k = len(series) - 1
        bipower_variation = np.sqrt((np.pi / (2 * k)) * np.sum(np.abs(series[:-1]) * np.abs(series.shift(-1)[:-1])))

        bipower_variation_series = pd.Series([bipower_variation] * len(series), index=series.index)
        return bipower_variation_series

    ###############################################################################
    ################################# Periodicity #################################
    ###############################################################################

    def periodicity_estimator(self, df_returns: pd.DataFrame, df_bpv: pd.DataFrame, xs: list):
        """
        The non-parametric periodicity estimator proposed by Taylor and Xu (1997) is based on the standard deviation
        (SD) of all standardized returns belonging to the same local window
        """
        
        df_f0, df_f1 = self.fs(df_returns, df_bpv, xs)
        df_f = df_f0 * df_f1

        return df_f

    def fs(self, df_returns: pd.DataFrame, df_bpv: pd.DataFrame, xs: list):
        T = df_returns.shape[0]

        # f_0
        df_norm_returns_0 = df_returns / df_bpv
        df_W0 = self.W(df_norm_returns_0, df_bpv, x=xs[0])
        den0 = np.sqrt(T**(-1) * np.sum(df_W0.iloc[:, 0]**2))
        df_f0 = df_W0 / den0

        # f_1
        df_norm_returns_1 = df_norm_returns_0 / df_f0
        df_W1 = self.W(df_norm_returns_1, df_bpv, x=xs[1])
        den1 = np.sqrt(T**(-1) * np.sum(df_W1.iloc[:, 0]**2))
        df_f1 = df_W1 / den1
        
        return df_f0, df_f1

    def W(self, df_norm_returns: pd.DataFrame, df_bpv: pd.DataFrame, x: float):
        df_W = pd.DataFrame(index=df_norm_returns.index, columns=df_norm_returns.columns)
        
        for index, row in df_norm_returns.iterrows():
            df_W_num = self.frac_W(row, x, term="num")
            df_W_den = self.frac_W(row, x, term="den")
            df_W.loc[index, :] = np.sqrt(1.081 * df_W_num/df_W_den)
        
        return df_W

    @staticmethod
    def frac_W(norm_ret_row, x, term):
        value = 0
        threshold = -norm_ret_row**2 + x
        
        if term == "num":
            value = norm_ret_row**2
        elif term == "den":
            value = 1
            
        return np.sum(np.where(threshold > 0, value, 0))

    ##############################################################################
    ################################# Jump Score #################################
    ##############################################################################

    def jump_score(self, df_returns: pd.DataFrame, df_bpv: pd.DataFrame, df_f: pd.DataFrame):
        df_jumps = df_returns / (df_bpv * df_f)
        return df_jumps

    ##############################################################################
    ############################ DataFrame Compilation ###########################
    ##############################################################################

    def compilation(self, generate_data: bool, df_orders_pickle: pd.DataFrame, symbol: str, type_mid: str):
        if generate_data:
            df_orders = self.build_data_set(symbol)
        else:
            df_orders = df_orders_pickle
            
        df_prices = self.get_mid_price(df_orders, type_mid, drop_na=False)
        df_returns = self.get_return(df_prices)
        df_bpv = self.get_bipower_variation_estimator(df_returns)
        df_periodicity = self.periodicity_estimator(df_returns, df_bpv, [16, 6.635])
        df_jumps = self.jump_score(df_returns, df_bpv, df_periodicity)

        df_dict = {
            "prices": df_prices,
            "returns": df_returns,
            "bipower_variation": df_bpv,
            "periodicity": df_periodicity,
            "jumps": df_jumps
        }
    
        return df_dict
