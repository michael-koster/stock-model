from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from datetime import date, timedelta


def stock_news(stock):
    news = {}
    # copy the finviz url 
    # (it may change over time so make sure url ending is correct)
    url = f'https://finviz.com/quote.ashx?t={stock}&p=d'
    print(f'Fetching news for {stock} from {url}')
    request = Request(url=url, headers={'user-agent': 'news_scraper'})
    response = urlopen(request)

    # parse the data
    html = BeautifulSoup(response, features='html.parser')
    finviz_news_table = html.find(id='news-table')
    news[stock] = finviz_news_table

    # filter and store neede in news_parsed
    news_parsed = []
    for stock, news_item in news.items():
        for row in news_item.findAll('tr'):
            try:
                headline = row.a.getText()
                source = row.span.getText()
                news_parsed.append([stock, headline])
            except:
                pass

    # convert to a dataframe for data analysis
    df = pd.DataFrame(news_parsed, columns=['Stock', 'Headline'])

    print(df)

    df.to_csv(f'/stock_data/news-{stock}.csv', index=False, header=True)
                                        


def main():
    stock_news('AAPL')


if __name__ == '__main__':
    main()
    
