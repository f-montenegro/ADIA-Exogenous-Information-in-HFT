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

    def __init__(self, dataset: str, resample_freq: str,
                 date: datetime, start: str, end: str, end_hour: int, end_minute: int, t_days: int):

        self.dataset = dataset
        self.resample_freq = resample_freq
        self.start = start
        self.end = end
        self.date = date
        self.end_hour = end_hour
        self.end_minute = end_minute
        self.t_days = t_days

    def get_mbp_data(self, symbol):
        """
        MBP (Market-by-Price) provides changes to and snapshots of aggregated book depth, keyed by price.
        This is limited to a fixed number of levels from the top. We denote number of levels displayed
        in the schema with a suffix, such as mbp-1 and mbp-10.
        """

        client = db.Historical("db-SEbmRhQ3ekjnrdRQLfK4iDYJhVkrL")
        data_mbp = client.timeseries.get_range(dataset=self.dataset,
                                               start=self.start,
                                               end=self.end,
                                               symbols=symbol,
                                               schema="mbp-10")

        data_mbp = data_mbp.to_df()
        data_mbp = data_mbp.set_index('ts_event')

        return data_mbp

    def build_data_set(self, symbol):
        """
        date: datetime object datetime(YYYY, MM, DD, HH, MM) for (MM/DD/YYYY), at HH:MM AM
        """

        data_df = pd.DataFrame()
        count_days = 0

        for i in range(self.t_days):
            start = self.date + timedelta(days=i)
            end = self.date + timedelta(days=i)
            end = end.replace(hour=self.end_hour, minute=self.end_minute, second=0, microsecond=0)

            start_iso = start.isoformat()[:16]
            end_iso = end.isoformat()[:16]

            data = self.get_mbp_data(dataset="XNAS.ITCH", start=start_iso, end=end_iso, symbol=symbol)

            if len(data.index) != 0:
                count_days += 1
                data = data[["symbol", "action", "side", "depth", "price", "size",
                             "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00",
                             "bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01", "bid_ct_01", "ask_ct_01"]]

                mask = (data["action"] == "T") & (data["depth"] == 0)
                data = data[mask]
                data_df = pd.concat([data_df, data], axis=0)

        return data_df, count_days

    def get_mid_price(self, df_input: pd.DataFrame, type_mid: str):
        """
        Calculate mid price per minute
        """

        df = df_input.copy()
        return_df = pd.DataFrame()

        if type_mid == "last":
            df_resampled = df.resample(self.resample_freq).last().dropna()
            return_df[f"mid-price_{type_mid}"] = (df_resampled["bid_px_00"] + df_resampled["ask_px_00"]) * 0.5
            return return_df

        if type_mid == "mean":
            return_df[f"mid-price_{type_mid}"] = (df["bid_px_00"] + df["ask_px_00"]) * 0.5
            return_df = return_df.resample(self.resample_freq).mean().dropna()
            return return_df

        if type_mid == "vwmp":
            df["vwbp_denominator"] = (df["bid_px_00"] * df["bid_sz_00"])
            df["vwap_denominator"] = (df["ask_px_00"] * df["ask_sz_00"])

            df_resampled = df.resample(self.resample_freq).agg(['sum', 'count'])
            df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]

            mask = np.all([(df_resampled[f"{col}_sum"] == 0) & (df_resampled[f"{col}_count"] == 0)
                           for col in df.columns], axis=0)

            df_resampled = df_resampled.loc[~mask]

            vwbp = df_resampled["vwbp_denominator_sum"] / df_resampled["bid_sz_00_sum"]
            vwap = df_resampled["vwap_denominator_sum"] / df_resampled["ask_sz_00_sum"]
            return_df[f"mid-price_{type_mid}"] = (vwbp + vwap) * 0.5
            return return_df

    def periodicity_estimator_taylor_and_xu(self, df_input, window_size, count_days):
        """
        The non-parametric periodicity estimator proposed by Taylor and Xu (1997) is based on the standard deviation (SD) of all
        standardized returns belonging to the same local window
        """

        df = df_input.copy()
        df["across_days_std"] = [df[df.index.time == i]["log_return"].std() for i in
                                 pd.unique(df.index.time)] * count_days
        df["across_days_std_2"] = df["across_days_std"] ** 2

        kwargs = {'column_name': "across_days_std_2",
                  "window_size": window_size}

        df = df.groupby(df.index.date, group_keys=False).apply(self.rolling_std, **kwargs)
        df["periodicity"] = df["across_days_std"] / np.sqrt(df["rolling_std"] / window_size)
        df.drop(columns=["across_days_std", "across_days_std_2", "rolling_std"], inplace=True)

        return df

    @staticmethod
    def sqrt_root_average_realised_bipower_variation(df_input: pd.DataFrame, column_name: str, k: int):
        """
        [46] K. Boudt, C. Croux, and S. Laurent, Journal of Empirical Finance 18, 353 (2011) | Barndorff-Nielsen and
        Shephard (2004) Power and bipower variation with stochastic volatility and jumps. J. Financ. Econometrics 2, 1–37.

        The realized bipower variation (RBV) is a measure used in financial econometrics to estimate integrated
        volatility in the presence of jumps. The bipower variation is robust to jumps and can be used to separate
        continuous and jump components of volatility.
        """

        df = df_input.copy()
        nan_mask = df[column_name].isna()
        df = df.fillna(method='ffill')
        df["sigma_squared"] = np.nan

        if len(df) < k + 1:
            return "Time series is too short for the given window size."

        for t in range(k, len(df)):
            sigma_squared = 0
            for i in range(1, k + 1):
                sigma_squared += abs(df.at[df.index[t - i], column_name]) * abs(df.at[df.index[t - i + 1], column_name])

            sigma_squared *= math.pi / (2 * k)
            df.at[df.index[t], "sigma_squared"] = sigma_squared

        df = df.where(~nan_mask, np.nan)

        return df

    @staticmethod
    def calculate_return(df_input: pd.DataFrame, column_name: str):
        """
        [62] L. M. Calcagnile, G. Bormetti, M. Treccani, S. Marmi,
        and F. Lillo, Quantitative Finance 18, 237 (2018).
        """

        df = df_input.copy()
        array = np.array(df[column_name])

        counter = 1
        for i in range(1, len(array)):
            if not np.isnan(array[i]):
                df.at[df.index[i], column_name] = 1 / np.sqrt(counter) * np.log(array[i] / array[i - counter])
                counter = 1
            else:
                counter += 1

        df = df.iloc[1:, :]

        return df

    @staticmethod
    def rolling_std(df_input, column_name, window_size):
        """
        Create a rolling object with a given window size and minimum period 1
        min_periods: specifies the minimum number of non-NA/null observations in the window required to have a valid result.
        """

        df = df_input.copy()
        if window_size < 3:
            print("The window size should be greater than 3")

        df['rolling_std'] = df[column_name].rolling(window=window_size, min_periods=1, center=True).std()

        return df