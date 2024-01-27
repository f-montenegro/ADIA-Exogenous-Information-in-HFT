from L0_Library.config import *


class News:
    def __init__(self, base_url: str, API_key: str, start_date: datetime, end_date: datetime):

        self.base_url = base_url
        self.API_key = API_key
        self.start_date = start_date
        self.end_date = end_date

    def get_news(self, symbol: list, news_type: str, all_pages: bool, items: int = None, pages: int = None):
        start_date = self.start_date.strftime('%m%d%Y')
        end_date = self.end_date.strftime('%m%d%Y')
        date = f'{start_date}-{end_date}'

        if all_pages:
            pages = np.inf
            items = 100

        page = 1
        df = pd.DataFrame()

        while page <= pages:
            try:
                params = {'tickers': symbol,
                          'items': items,
                          'page': page,
                          'date': date,
                          'type': news_type,
                          'token': self.API_key}

                response = requests.get(self.base_url, params=params)

                response = response.json()['data']
                response = pd.DataFrame(response)
                response = response[['date', 'title', 'text', 'source_name', 'sentiment', 'type', 'tickers']]
                df = pd.concat([df, response] if page > 1 else [response], ignore_index=True)
                page += 1

            except requests.exceptions.RequestException:
                break

            except KeyError:
                break

        return df.set_index('date')

    def get_dummy_news(self, response):
        date_series = pd.to_datetime(response.reset_index()["date"].unique(), utc=True).sort_values(ascending=True)

        start_hour = pd.to_datetime("09:30:00", format="%H:%M:%S")
        end_hour = pd.to_datetime("15:59:00", format="%H:%M:%S")

        hour_range = pd.date_range(start=start_hour, end=end_hour, freq="1min")
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")

        df_rows = pd.to_datetime(hour_range).strftime('%H:%M:%S')
        df_cols = pd.to_datetime(date_range).strftime('%Y-%m-%d')

        df = pd.DataFrame(index=df_rows, columns=df_cols)
        df = df.fillna(0)

        for date in date_series.unique():
            row = pd.to_datetime(date).replace(second=0).strftime('%H:%M:%S')
            col = pd.to_datetime(date).strftime('%Y-%m-%d')

            try:
                df.at[row, col] += 1

            except:
                continue
