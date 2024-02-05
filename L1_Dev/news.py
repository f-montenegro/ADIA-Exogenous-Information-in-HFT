from L0_Library.config import *
from L1_Dev.utils import df_index_columns_correction


class News:
    def __init__(self, base_url: str, API_key: str, start_date: datetime, end_date: datetime, time_zone: str):

        self.base_url = base_url
        self.API_key = API_key
        self.start_date = start_date - timedelta(days=40)
        self.end_date = end_date
        self.time_zone = time_zone

    def get_raw_news(self, symbol: list, news_type: str, all_pages: bool, items: int = None, pages: int = None):
        """
        Get news from StockNewsAPI for a list of stocks and over a period of time.
        User can choose between pulling all news or a limited number of news.

        Parameters:
            symbol (list): The symbol list for which to retrieve data.
            news_type (str): Parameter required by the API pulling function. Usual value is "article".
            all_pages (bool): Get news from all pages
            items (int): Quantity of news from each page. If all_pages = True, we assume items=100 (maximum per page)
            pages (int): Quantity of pages for each symbol. If all_pages = True, we assume pages=ALL (all pages)

        Returns:
            df_result (pd.DataFrame): A DataFrame containing all the retrieved news for the symbol list.
                                      The DataFrame contains the following columns:
                                            -   date (index).
                                            -   title: news title.
                                            -   source_name: who posted the news.
                                            -   sentiment: Positive / Negative.
                                            -   type: same as news_type parameter.
                                            -   tickers: all the tickers affected by the news.
        """

        start_date = self.start_date.strftime('%m%d%Y')
        end_date = self.end_date.strftime('%m%d%Y')
        date = f'{start_date}-{end_date}'

        if all_pages:
            pages = np.inf
            items = 100

        page = 1
        df_result = pd.DataFrame()

        while page <= pages:
            try:
                params = {'tickers': [symbol],
                          'items': items,
                          'page': page,
                          'date': date,
                          'type': news_type,
                          'token': self.API_key}

                response = requests.get(self.base_url, params=params)

                response = response.json()['data']
                response = pd.DataFrame(response)
                response = response[['date', 'title', 'text', 'source_name', 'sentiment', 'type', 'tickers']]
                df_result = pd.concat([df_result, response] if page > 1 else [response], ignore_index=True)
                page += 1

            except requests.exceptions.RequestException:
                break

            except KeyError:
                break

        df_result = df_result.set_index('date')

        return df_result

    def get_news(self, df_input: pd.DataFrame):
        df_input.index = pd.to_datetime(df_input.index, utc=True).tz_convert(self.time_zone).tz_localize(None)
        date_series = pd.to_datetime(df_input.index).sort_values(ascending=True)

        start_hour = pd.to_datetime("09:30:00", format="%H:%M:%S")
        end_hour = pd.to_datetime("15:59:00", format="%H:%M:%S")

        hour_range = pd.date_range(start=start_hour, end=end_hour, freq="1min")
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")

        df_rows = pd.to_datetime(hour_range).strftime('%H:%M:%S')
        df_cols = pd.to_datetime(date_range).strftime('%Y-%m-%d')

        df_result = pd.DataFrame(index=df_rows, columns=df_cols)
        df_result = df_result.fillna(0)

        for date in date_series.unique():
            row = pd.to_datetime(date).replace(second=0).strftime('%H:%M:%S')
            col = pd.to_datetime(date).strftime('%Y-%m-%d')

            try:
                df_result.at[row, col] = 1

            except requests.exceptions.RequestException:
                continue

            except KeyError:
                continue

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    @staticmethod
    def news_list(df_input: pd.DataFrame):
        df_result = df_input
        df_result = pd.DataFrame(df_result.unstack()[df_result.unstack() == 1]).rename(columns={0: 'News'})

        pd.options.display.float_format = '{:,.0f}'.format

        return df_result

    ##############################################################################
    ############################ DataFrame Compilation ###########################
    ##############################################################################

    def compilation(self, generate_data: bool, df_news_storage: pd.DataFrame, symbol: str):
        if generate_data:
            df_raw_news = self.get_raw_news(symbol=symbol, news_type='article', all_pages=True)
        else:
            df_raw_news = df_news_storage

        df_news = self.get_news(df_raw_news)
        df_news = df_index_columns_correction(df_news)

        df_news_list = self.news_list(df_news)

        df_dict = {
            "raw_news": df_raw_news,      # StockNewsAPI news description
            "news": df_news,              # News Table
            "news_list": df_news_list     # List of News
        }

        return df_dict
