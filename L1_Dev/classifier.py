from L0_Library.config import *


class Classifier:

    def __init__(self):
        pass

    @staticmethod
    def jumps_classification(df_jumps: pd.DataFrame, df_news: pd.DataFrame, inf_bound='1', sup_bound='20'):
        # JUMPS
        df_jumps = df_jumps.reset_index()
        df_jumps.columns = ["Date", "Hour", "News"]
        df_jumps = df_jumps[["Date", "Hour"]].rename(columns={"Hour": "Hour_Jump"})

        # NEWS
        df_news = df_news.reset_index()
        df_news.columns = ["Date", "Hour", "News"]
        df_news = df_news[["Date", "Hour"]].rename(columns={"Hour": "Hour_News"})

        df_news['inf_hour'] = pd.to_datetime(df_news['Hour_News'].astype(str)) - pd.to_timedelta(f'{inf_bound} minute')
        df_news['inf_hour'] = df_news['inf_hour'].dt.strftime('%H:%M:%S')

        df_news['sup_hour'] = pd.to_datetime(df_news['Hour_News'].astype(str)) + pd.to_timedelta(f'{sup_bound} minutes')
        df_news['sup_hour'] = df_news['sup_hour'].dt.strftime('%H:%M:%S')

        # JUMPS and NEWS relationship
        df_jumps['Date'] = pd.to_datetime(df_jumps['Date'].astype(str))
        df_news['Date'] = pd.to_datetime(df_news['Date'].astype(str))

        df_news_related_jumps = pd.merge(df_jumps, df_news, on='Date', how='left')

        df_news_related_jumps["Hour_Jump"] = pd.to_datetime(df_news_related_jumps['Hour_Jump'].astype(str))
        df_news_related_jumps["inf_hour"] = pd.to_datetime(df_news_related_jumps['inf_hour'])
        df_news_related_jumps["sup_hour"] = pd.to_datetime(df_news_related_jumps['sup_hour'])

        df_news_related_jumps["News_Related"] = np.where(
            (df_news_related_jumps["inf_hour"] <= df_news_related_jumps["Hour_Jump"]) &
            (df_news_related_jumps["Hour_Jump"] <= df_news_related_jumps["sup_hour"]), 1, 0
        )

        df_news_related_jumps['inf_hour'] = df_news_related_jumps['inf_hour'].dt.strftime('%H:%M:%S')
        df_news_related_jumps['sup_hour'] = df_news_related_jumps['sup_hour'].dt.strftime('%H:%M:%S')
        df_news_related_jumps['Hour_Jump'] = df_news_related_jumps['Hour_Jump'].dt.strftime('%H:%M:%S')

        df_result = df_news_related_jumps.groupby(["Date", "Hour_Jump"])["News_Related"].max().reset_index()
        df_result = df_result.merge(df_news_related_jumps[["Date", "Hour_Jump", "Hour_News", "News_Related"]],
                                    on=["Date", "Hour_Jump", "News_Related"],
                                    how="left")
        df_result = df_result[["Date", "Hour_Jump", "Hour_News", "News_Related"]]
        df_result["Hour_News"] = np.where(df_result["News_Related"] == 1, df_result["Hour_News"], np.nan)
        df_result["Date"] = df_result["Date"].dt.strftime('%Y-%m-%d')

        return df_result

    ##############################################################################
    ############################ DataFrame Compilation ###########################
    ##############################################################################

    def compilation(self, df_jumps_clustered_list: pd.DataFrame, df_news_clustered_list: pd.DataFrame):

        df_jumps_news_related = self.jumps_classification(df_jumps_clustered_list, df_news_clustered_list,
                                                          inf_bound='1', sup_bound='20')

        df_dict = {
            "jumps_news_related": df_jumps_news_related,    # List of Jumps with Classification (News Related or not)
        }

        return df_dict
