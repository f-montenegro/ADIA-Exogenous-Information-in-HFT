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
