import datetime
import os
import pandas as pd
import yfinance as yf

DATA_FOLDER = 'StockData'


def generate_file_path(ticker, start_date, end_date):
	return f'{DATA_FOLDER}/{ticker}_{start_date}_{end_date}.csv'


class StockData:
	def __init__(self, ticker, start_date, end_date):
		self.ticker = ticker
		self.start_date = start_date
		self.end_date = end_date
		self.path = generate_file_path(ticker, start_date, end_date)
		self.df = None
		self.get_data()

	def __getattr__(self, attr):
		return getattr(self.df, attr)
	
	def get_data(self):
		if os.path.exists(self.path):
			self.df = pd.read_csv(self.path, index_col=0)
			self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
		else:
			self.df = yf.download(self.ticker, self.start_date, self.end_date)
			self.df.index = pd.to_datetime(self.df.index, format='%Y-%m-%d')
			self.df.to_csv(self.path)

	def point(self, index, column='Adj Close'):
		return self.df.iloc[index][column]


class Simulation:
	def __init__(self, data, start_balance=100_000, monthly_income=10_000, save_percent=0.1,
				raise_percent=0.04, inflation=0.025):
		self.data = data
		self.start_balance = start_balance
		self.monthly_income = monthly_income
		self.save_percent = save_percent
		self.raise_percent = raise_percent
		self.inflation = inflation

		self.current_balance = start_balance
		self.current_shares = 0
		self.current_date = data.index[0]
		self.current_index = 0

		self.total_investment = start_balance

	def networth(self):
		return self.current_balance + self.current_shares * self.price()

	def price(self):
		return self.data.point(self.current_index, column='Adj Close')

	def update(self):
		roi = (self.networth() - self.total_investment) / self.total_investment * 100
		print(f'{self.current_date.strftime(format="%Y-%m-%d")} | ${self.price():,.2f} | ${self.networth():,.2f} | {roi:,.2f}%')

	def trade(self, shares=None):
		if shares is None:
			shares = self.max_shares()
		if not shares:
			return
			
		price = self.price()
		amount = price * shares
		
		if shares > 0 and amount > self.current_balance:
			raise ValueError('Attmepting to purchase too many shares.')
		if shares < 0 and shares > self.current_shares:
			raise ValueError('Attemtping to sell too many shares.')

		self.current_balance -= amount
		self.current_shares += shares

	def max_shares(self, amount=None):
		if amount is None:
			amount = self.current_balance
		price = self.price()
		return amount // price

	def run(self):
		self.trade()
		while self.current_date < self.data.index[-1]:
			self.tick()

	def tick(self):
		self.current_date += datetime.timedelta(days=1)
		if self.current_date.day == 1:
			if self.current_date.month == 1:
				self.yearly()
			self.monthly()
		if self.current_index % 1 == 0:
			self.daily()

	def daily(self):
		if self.current_date in self.data.index:
			self.current_index += 1
			self.trade()

	def monthly(self):
		self.update()
		self.current_balance += self.monthly_income * self.save_percent
		self.total_investment += self.monthly_income * self.save_percent

	def yearly(self):
		self.monthly_income *= (1 + self.raise_percent)
	

def main():
	ticker = 'SPY'
	start_date = '2002-01-01'
	end_date = '2022-01-01'
	data = StockData(ticker, start_date, end_date)
	simulation = Simulation(data)
	simulation.run()
	simulation.update()


if __name__ == '__main__':
	main()
	