from L0_Library.config import *


class Clusters:

    def __init__(self):
        pass

    @staticmethod
    def data_timestamp(df_input: pd.DataFrame) -> list:
        list_result = []
        for i in df_input.columns:
            if (df_input[i] == 1).any():
                list_result.append(i.strftime('%m/%d/%Y'))
        return list_result

    @staticmethod
    def calculate_data_inter_times(df_input: pd.DataFrame, list_input: list):
        df_result = df_input.copy()
        for i in list_input:
            if df_input[i].sum() > 1:
                mask = np.array(df_input[i] == 1)
                series = pd.Series(range(len(df_input[i])))[mask]
                shifted_shifted = series - series.shift(1) - 1
                shifted_shifted = shifted_shifted.replace(np.nan, np.inf)
                df_result[i].iloc[shifted_shifted.index] = shifted_shifted
            else:
                df_result[i] = df_input[i].replace(1, np.inf)

        pd.options.display.float_format = '{:,.3f}'.format

        return df_result

    @staticmethod
    def min_bernoulli_trials_p(row, epsilon):
        if row['data'] > epsilon:
            return epsilon
        else:
            return row['data']

    def bernoulli_trials_p(self, df_input: pd.DataFrame, rolling_window: int, epsilon: float):
        df_input_unstack = df_input.unstack().reset_index().rename(columns={0: 'data'})
        df_input_unstack['data'] = np.nan

        s_bernoulli_trials_p = df_input.unstack().dropna().rolling(rolling_window).sum() / rolling_window
        s_bernoulli_trials_p = s_bernoulli_trials_p.reset_index().rename(columns={0: 'data'})

        df_result = pd.merge(df_input_unstack, s_bernoulli_trials_p,
                             on=['Date', 'Hour'], how='left')[['Date', 'Hour', 'data_y']]
        df_result = df_result.rename(columns={'data_y': 'data'})

        df_result['data'] = df_result.apply(lambda row: self.min_bernoulli_trials_p(row, epsilon), axis=1)

        df_result = df_result.pivot(index='Hour', columns='Date', values=f'data')

        pd.options.display.float_format = '{:,.5f}'.format

        return df_result

    @staticmethod
    def bernoulli_hypothesis_threshold(df_input: pd.DataFrame, epsilon: float):
        df_threshold = (np.log(1 - epsilon) / np.log(1 - df_input)) - 1

        pd.options.display.float_format = '{:,.5f}'.format

        return df_threshold

    def cluster_data(self, df_input: pd.DataFrame, df_inter_times: pd.DataFrame, df_threshold: pd.DataFrame):
        df_result = df_inter_times.where(df_threshold.notna(), np.nan)

        df_input = df_input.where(df_threshold.notna(), np.nan)
        l_data_timestamp = self.data_timestamp(df_input)

        for i in l_data_timestamp:
            s_inter_times = df_inter_times[i]
            s_threshold = df_threshold[i]
            s_clustered_data = df_result[i]
            for j in range(len(s_inter_times)):
                if s_inter_times[j] == np.inf:
                    s_clustered_data[j] = 1
                elif s_inter_times[j] < s_threshold[j]:
                    s_clustered_data[j] = 0
                elif s_inter_times[j] >= s_threshold[j]:
                    s_clustered_data[j] = 1
            df_result[i] = s_clustered_data

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    @staticmethod
    def clustered_data_list(df_input: pd.DataFrame, cluster: str):
        df_result = df_input
        df_result = pd.DataFrame(df_result.unstack()[df_result.unstack() == 1]).rename(columns={0: f'{cluster}'})

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    ##############################################################################
    ############################ DataFrame Compilation ###########################
    ##############################################################################

    def compilation(self, df_input: pd.DataFrame, cluster: str):

        l_data_timestamp = self.data_timestamp(df_input=df_input)

        df_inter_times = self.calculate_data_inter_times(df_input, l_data_timestamp)
        df_bernoulli_prob = self.bernoulli_trials_p(df_input, rolling_window=7780, epsilon=0.05)
        df_bernoulli_threshold = self.bernoulli_hypothesis_threshold(df_bernoulli_prob, epsilon=0.05)
        df_clustered_data = self.cluster_data(df_input, df_inter_times, df_bernoulli_threshold)
        df_clustered_data_list = self.clustered_data_list(df_clustered_data, cluster)

        df_dict = {
            f"inter_times_{cluster}": df_inter_times,             # Inter-Times
            "bernoulli_prob": df_bernoulli_prob,                  # Bernoulli Hypothesis-Test Implied Probability
            "bernoulli_threshold": df_bernoulli_threshold,        # Bernoulli Hypothesis-Test Implied Threshold
            f"clustered_{cluster}": df_clustered_data,            # Clustered Data
            f"clustered_{cluster}_list": df_clustered_data_list,  # Clustered Data List
        }

        return df_dict
