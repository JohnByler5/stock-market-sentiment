import asyncio
import datetime
import os
import re
import time
from io import StringIO

import asyncpraw
import jsonlines
import nltk
import openai
import pandas as pd
import pytz
import torch
from aiohttp import client_exceptions
from aiostream import stream
from asyncprawcore import exceptions
from markdown import Markdown
from nltk.corpus import cmudict
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# OpenAI
openai.api_key = os.environ['openai_api_key']

# NLTK
nltk.download("cmudict")
cmudict = set(cmudict.dict())
with open('extra_words.txt') as f:
	extra_words = set(f.read().splitlines())
with open('allowed_tickers.txt') as f:
	allowed_tickers = set(x.lower() for x in f.read().splitlines())
WORDS = (cmudict | extra_words) - allowed_tickers
nltk_tokenizer = RegexpTokenizer(r'\w+')

# Symbols
stocks = pd.read_csv('us_symbols.csv')
stocks.set_index('ticker', drop=True, inplace=True)
etfs = pd.read_csv('ETFs.csv')
etfs.set_index('fund_symbol', drop=True, inplace=True)
mutual_funds = pd.read_csv('MutualFunds.csv')
mutual_funds.set_index('fund_symbol', drop=True, inplace=True)
TICKER_SET = set(stocks.index) | set(etfs.index) | set(mutual_funds.index)

# Sentiment
sentiment_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Reddit
CLIENT_ID = os.environ['reddit_client_id']
CLIENT_SECRET = os.environ['reddit_client_secret']
USER_AGENT = os.environ['reddit_user_agent']

with open('subreddits.txt', 'r') as f:
	SUBREDDITS = f.read().splitlines()

# Regex
WHITESPACE = re.compile(r"\s+")
PUNCTUATION = re.compile(r"\.+")
WEB_ADDRESS = re.compile(r"(?i)http(s):\/\/[a-z0-9.~=?&_\-\/]+")
USER = re.compile(r"(?i)/?u/[a-z0-9_-]+")
SUB_REDDIT = re.compile(r"(?i)/?r/[a-z0-9_-]+")


def log(*args, sep=' ', end='\n'):
	print(*args, sep=sep, end=end)
	with open('log.txt', 'a') as f:
		dt = datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")
		f.write(f'[{dt}] {sep.join(args)}{end}')


def unmark_element(element, stream=None):
	if stream is None:
		stream = StringIO()
	if element.text:
		stream.write(element.text)
	for sub in element:
		unmark_element(sub, stream)
	if element.tail:
		stream.write(element.tail)
	return stream.getvalue()


Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
	return __md.convert(text)


def get_sentiment(text):
	tokens = sentiment_tokenizer(text,
	                             padding=True,
	                             truncation=True,
	                             return_tensors='pt')
	output = model(**tokens)
	scores = torch.nn.functional.softmax(output.logits, dim=-1)
	scores = [score.tolist() for score in scores[0]]
	return (scores[0], scores[1], scores[2])


def clean_text(text):
	text = unmark(text)
	text = WEB_ADDRESS.sub('', text)
	text = USER.sub('', text)
	text = SUB_REDDIT.sub('', text)
	text = text.replace('%20', ' ')
	text = WHITESPACE.sub(' ', text)
	text = PUNCTUATION.sub(' ', text)
	return text.strip()


def is_ticker_format(token, text):
	index = text.index(token)
	end_index = index + len(token)
	if index >= 1 and text[index - 1] not in [' ', '$', '(', '-']:
		return False
	if end_index + 1 < len(text) and text[end_index] not in [
	  ' ', ')', '-'
	] and text[end_index + 1] not in [' ', '-']:
		return False
	return True


def is_guaranteed_ticker(token, text):
	if f'${token}' in text:
		return True
	return False


def is_word(token):
	token = token.lower()
	if token in WORDS:
		return True
	if token.endswith('s') and token[:-1].isupper():
		return True
	return False


def get_mentions(text):
	tokens = set(nltk_tokenizer.tokenize(text))
	tickers = set()
	for token in tokens:
		if not is_ticker_format(token, text):
			continue
		if is_word(token) and not is_guaranteed_ticker(token, text):
			continue
		token = token.upper()
		if token in TICKER_SET:
			tickers.add(token)
	return tickers


async def submission_stream(subreddit):
	async for item in subreddit.stream.submissions(skip_existing=True):
		result = {
		 'datetime': item.created_utc,
		 'post_id': item.id,
		 'type': 'submission',
		 'author': item.author.name,
		 'subreddit': item.subreddit.display_name,
		 'score': item.score,
		 'text': f'{item.title} - {item.selftext}'
		}
		yield result


async def comment_stream(subreddit):
	async for item in subreddit.stream.comments(skip_existing=True):
		result = {
		 'datetime': item.created_utc,
		 'post_id': item.id,
		 'type': 'comment',
		 'author': item.author.name,
		 'subreddit': item.subreddit.display_name,
		 'score': item.score,
		 'text': item.body
		}
		yield result


async def reddit_stream(subreddits=SUBREDDITS):
	async with asyncpraw.Reddit(client_id=CLIENT_ID,
	                            client_secret=CLIENT_SECRET,
	                            user_agent=USER_AGENT) as reddit:
		subreddit = await reddit.subreddit('+'.join(subreddits))
		combine = stream.merge(submission_stream(subreddit),
		                       comment_stream(subreddit))
		async with combine.stream() as streamer:
			async for result in streamer:
				yield result


async def monitor_stream(async_stream):
	attempts, monitored, detected, start = 0, 0, 0, time.time()
	log('Monitoring reddit posts...')
	while True:

		try:
			async for result in async_stream():
				if attempts:
					attempts = 0
				monitored += 1

				result['datetime'] = datetime.datetime.fromtimestamp(
				 result['datetime'], tz=pytz.utc).astimezone(
				  pytz.timezone('US/Eastern')).strftime('%m-%d-%Y %H:%M:%S')
				result['text'] = clean_text(result['text'])
				result['tickers'] = list(get_mentions(result['text']))
				if not result['tickers']:
					continue
				result['sentiment'] = get_sentiment(result['text'])
				result['text'] = result.pop('text')

				detected += 1
				s = time.time() - start
				m, s = divmod(s, 60)
				h, m = divmod(m, 60)
				duration = f'{h:02.0f}:{m:02.0f}:{s:02.0f}'
				log(f'  ({duration}) - {detected}/{monitored} - {result["tickers"]}')
				yield result

		except (exceptions.RequestException, exceptions.ServerError,
		        client_exceptions.ClientPayloadError) as e:
			attempts += 1
			if attempts == 3:
				raise e
			log(f'  Encountered Reddit error. Retrying... (attempts={attempts}/3)')


async def main():
	log('Starting up program...')
	with jsonlines.open('results.jsonl', 'a') as writer:
		while True:
			async for result in monitor_stream(reddit_stream):
				writer.write(result)


if __name__ == '__main__':
	asyncio.run(main())
