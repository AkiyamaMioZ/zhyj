import yfinance as yf
import pandas as pd
import datetime as dt


def download_stock_data(ticker, start_date, end_date):
    # 下载指定股票的历史数据
    stock_data = yf.download(tickers=ticker, start=start_date, end=end_date)

    # 选择我们感兴趣的列
    data = stock_data[['Open', 'High', 'Low', 'Close']]

    return data


def save_to_csv(data, filename):
    # 将数据保存到CSV文件
    data.to_csv(filename)


# 定义开始日期和结束日期
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=5 * 365)  # 回溯五年

# 腾讯的股票代码
ticker = "0700.HK"

# 下载数据
stock_data = download_stock_data(ticker, start_date, end_date)

# 保存数据到CSV文件
save_to_csv(stock_data, "tencent_stock_data.csv")

print("Data has been saved to byd_stock_data.csv")