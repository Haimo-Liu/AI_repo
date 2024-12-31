import os
import time
import json
import signal
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
from urllib.parse import quote_plus
from random import randint, uniform
import tweepy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Stock symbols to track
STOCKS = ['RKLB', 'PLTR', 'APP', 'HIMS', 'SNOW', 'KVYO', 'SHOP', 'TTAN']

class Post:
    """Represents a social media post with sentiment analysis capabilities."""
    def __init__(self, text, platform, url="", score=0, timestamp=None):
        self.text = text
        self.platform = platform
        self.url = url
        self.score = score
        self.timestamp = timestamp or datetime.now()
        self._sentiment = None
        
    @property
    def sentiment(self):
        """Analyze and cache the sentiment of the post."""
        if self._sentiment is None:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(self.text)
            self._sentiment = 'bullish' if scores['compound'] > 0.05 else 'bearish' if scores['compound'] < -0.05 else 'neutral'
        return self._sentiment

class StockAnalyzer:
    """Analyzes stock mentions and sentiment across social media platforms."""
    
    def __init__(self, data_dir="data"):
        """Initialize the analyzer with empty data structures."""
        self.posts = []  # List of Post objects
        self.results = {}  # Results per stock
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Rate limiting settings
        self.rate_limits = {
            'reddit': {'requests_per_minute': 30},
            'twitter': {'requests_per_minute': 2},  # Extremely conservative Twitter API rate
            'nitter': {'requests_per_minute': 5}  # Very conservative limit for Nitter instances
        }
        self.last_request_time = {}
        self.consecutive_failures = {}
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
    def _enforce_rate_limit(self, platform):
        """Enforce rate limiting for a specific platform with exponential backoff."""
        if platform not in self.last_request_time:
            self.last_request_time[platform] = time.time()
            self.consecutive_failures[platform] = 0
            return
        
        # Calculate time since last request
        now = time.time()
        elapsed = now - self.last_request_time[platform]
        
        # Calculate minimum delay based on rate limit and failures
        requests_per_minute = self.rate_limits[platform]['requests_per_minute']
        min_delay = 60.0 / requests_per_minute
        
        # Add exponential backoff if there have been failures
        if platform in self.consecutive_failures:
            failures = self.consecutive_failures[platform]
            if failures > 0:
                # Exponential backoff: 2^failures seconds, plus base delay
                min_delay += (2 ** failures)
                logging.info(f"Adding exponential backoff for {platform}: {min_delay} seconds")
        
        # If we need to wait, do so
        if elapsed < min_delay:
            wait_time = min_delay - elapsed
            logging.info(f"Rate limiting {platform}: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        self.last_request_time[platform] = time.time()

    def _safe_request(self, url, platform='nitter', retries=3, base_delay=3):
        """Make a request with retry logic and rate limiting."""
        for attempt in range(retries):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit(platform)
                
                # Add random query parameter to avoid caching
                cache_buster = f"{'&' if '?' in url else '?'}_={int(time.time())}"
                full_url = f"{url}{cache_buster}"
                
                response = requests.get(full_url, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    # Reset failure count on success
                    if platform in self.consecutive_failures:
                        self.consecutive_failures[platform] = 0
                    return response
                elif response.status_code == 429:  # Rate limited
                    self.consecutive_failures[platform] = self.consecutive_failures.get(platform, 0) + 1
                    wait_time = min(300, 30 * (2 ** self.consecutive_failures[platform]))  # Exponential waiting with 5-minute cap
                    logging.warning(f"Rate limited on {platform}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.warning(f"HTTP {response.status_code} for {url}")
                    if response.status_code == 403:  # Forbidden
                        self.consecutive_failures[platform] = self.consecutive_failures.get(platform, 0) + 1
                        wait_time = min(300, 10 * (2 ** self.consecutive_failures[platform]))
                        logging.warning(f"Access forbidden on {platform}. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        
            except Exception as e:
                logging.error(f"Error fetching {url}: {str(e)}")
                
            # Exponential backoff with jitter
            backoff = base_delay * (2 ** attempt) + uniform(0, 3)
            time.sleep(backoff)
            
        return None
        
    def collect_stocktwits_posts(self):
        """Collect posts from StockTwits."""
        for stock in STOCKS:
            logging.info(f"Collecting StockTwits posts for {stock}...")
            
            url = f'https://stocktwits.com/symbol/{stock}'
            response = self._safe_request(url)
            
            if response:
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    posts = soup.find_all('div', {'class': 'st_3FuWe'})  # StockTwits message container class
                    
                    for post in posts:
                        # Get message text
                        message_div = post.find('div', {'class': 'st_3FNfw'})  # StockTwits message text class
                        text = message_div.get_text(strip=True) if message_div else ""
                        
                        # Get likes count as score
                        likes_div = post.find('div', {'class': 'st_1NPqO'})  # StockTwits likes class
                        score = int(likes_div.get_text(strip=True)) if likes_div else 0
                        
                        # Extract timestamp from post
                        timestamp_div = post.find('time', {'class': 'st_3mpF9'})  # StockTwits timestamp class
                        post_time = datetime.now()  # Default to now if can't parse time
                        if timestamp_div and timestamp_div.get('datetime'):
                            try:
                                post_time = datetime.fromisoformat(timestamp_div.get('datetime').replace('Z', '+00:00'))
                            except ValueError:
                                pass
                        
                        # Only include posts from last 24 hours
                        if datetime.utcnow() - post_time <= timedelta(hours=24):
                            if text:
                                self.add_post(
                                    text=text,
                                    platform='stocktwits',
                                    url=url,
                                    score=score,
                                    timestamp=post_time
                                )
                    
                    if posts: 
                        logging.info(f"Collected {len(posts)} posts about {stock} from StockTwits")
                        
                except Exception as e: 
                    logging.error(f"Error parsing StockTwits data: {str(e)}")
                    continue
                
                # Be nice to StockTwits
                time.sleep(uniform(3, 5))
                
    def collect_yahoo_finance_posts(self):
        """Collect posts from Yahoo Finance message boards."""
        for stock in STOCKS:
            logging.info(f"Collecting Yahoo Finance posts for {stock}...")
            
            url = f'https://finance.yahoo.com/quote/{stock}/community'
            response = self._safe_request(url)
            
            if response:
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    posts = soup.find_all('div', {'class': 'C($c-fuji-grey-l) Mb(2px) Fz(14px) Lh(20px)'})
                    
                    for post in posts:
                        text = post.get_text(strip=True)
                        
                        # Get reactions count as score
                        reactions = post.find_parent('div', {'class': 'Py(14px) Pstart(10px)'})
                        score = 0
                        if reactions:
                            score_spans = reactions.find_all('span', {'class': 'Mstart(4px)'})
                            score = sum(int(span.get_text(strip=True)) for span in score_spans if span.get_text(strip=True).isdigit())
                        
                        # Extract timestamp from post
                        timestamp_div = post.find('span', {'class': 'C($c-fuji-grey-j) Fz(12px) Fw(n) D(ib) Mstart(8px)'})
                        post_time = datetime.now()  # Default to now if can't parse time
                        if timestamp_div:
                            try:
                                time_text = timestamp_div.get_text(strip=True)
                                # Yahoo Finance shows relative time, convert to absolute
                                if 'minutes ago' in time_text:
                                    minutes = int(time_text.split()[0])
                                    post_time = datetime.now() - timedelta(minutes=minutes)
                                elif 'hours ago' in time_text:
                                    hours = int(time_text.split()[0])
                                    post_time = datetime.now() - timedelta(hours=hours)
                            except ValueError:
                                pass
                        
                        # Only include posts from last 24 hours
                        if datetime.utcnow() - post_time <= timedelta(hours=24):
                            if text:
                                self.add_post(
                                    text=text,
                                    platform='yahoo_finance',
                                    url=url,
                                    score=score,
                                    timestamp=post_time
                                )
                    
                    if posts:
                        logging.info(f"Collected {len(posts)} posts about {stock} from Yahoo Finance")
                        
                except Exception as e:
                    logging.error(f"Error parsing Yahoo Finance data: {str(e)}")
                    continue
                
                # Be nice to Yahoo Finance
                time.sleep(uniform(3, 5))
                        
    def init_twitter_client(self):
        """Initialize Twitter API client if credentials are available."""
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        logging.info("Initializing Twitter client...")
        
        if not bearer_token:
            logging.error("Twitter bearer token not found in environment variables")
            return None
            
        try:
            logging.info("Creating Twitter client with bearer token...")
            client = tweepy.Client(
                bearer_token=bearer_token,
                wait_on_rate_limit=True
            )
            logging.info("Twitter client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Twitter client: {str(e)}")
            logging.error(f"Bearer token format: {bearer_token[:10]}...{bearer_token[-10:]}")
        return None

    def collect_x_posts(self, stocks):
        """
        Collect posts from X/Twitter using multiple methods.
        
        Args:
            stocks (list): List of stock symbols to collect posts for
        """
        logging.info("Starting data collection from Twitter API...")
        posts_collected = 0
        save_threshold = 25  # Save more frequently
        stocks_collected = set()  # Track which stocks we've collected enough data for
        
        # Try Twitter API first
        client = self.init_twitter_client()
        if client:
            for stock in stocks:
                if stock in stocks_collected:
                    continue
                    
                logging.info(f"Collecting X/Twitter posts for {stock} using API...")
                try:
                    queries = [
                        f"{stock} stock lang:en -is:retweet",
                        f"{stock} trading lang:en -is:retweet",
                        f"{stock} price lang:en -is:retweet"
                    ]
                    
                    total_tweets = 0
                    for query in queries:
                        try:
                            # Enforce rate limit with much longer delay
                            self._enforce_rate_limit('twitter')
                            delay = 180  # 3 minutes base delay
                            jitter = uniform(30, 60)  # 30-60 seconds random jitter
                            total_delay = delay + jitter
                            logging.info(f"Waiting {total_delay:.2f} seconds before Twitter API request...")
                            time.sleep(total_delay)
                            
                            max_retries = 3
                            retry_delay = 60  # Start with 1 minute delay
                            
                            for retry in range(max_retries):
                                try:
                                    logging.info(f"Searching Twitter for query: {query}")
                                    # Set up signal handler for timeout
                                    def timeout_handler(signum, frame):
                                        raise TimeoutError("Request timed out")
                                    
                                    signal.signal(signal.SIGALRM, timeout_handler)
                                    signal.alarm(60)  # 60 second timeout
                                    
                                    try:
                                        # Calculate start time for 24-hour window
                                start_time = datetime.utcnow() - timedelta(hours=24)
                                tweets = client.search_recent_tweets(
                                            query=query,
                                            max_results=5,  # Reduced batch size for better reliability
                                            tweet_fields=['public_metrics', 'created_at', 'text'],
                                            start_time=start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                                        )
                                        signal.alarm(0)  # Disable alarm
                                    except TimeoutError:
                                        signal.alarm(0)  # Disable alarm
                                        logging.warning("Twitter API request timed out")
                                        raise
                                    logging.info("Twitter API request successful")
                                    
                                    if tweets and hasattr(tweets, 'data'):
                                        for tweet in tweets.data:
                                            metrics = tweet.public_metrics
                                            score = sum(metrics.values()) if metrics else 0
                                            
                                            self.add_post(
                                                text=tweet.text,
                                                platform='x',
                                                url=f"https://twitter.com/search?q={quote_plus(query)}",
                                                score=score,
                                                timestamp=tweet.created_at
                                            )
                                            total_tweets += 1
                                            posts_collected += 1
                                            
                                            if posts_collected >= save_threshold:
                                                self.save_posts()
                                                posts_collected = 0
                                                logging.info(f"Saved current progress - {len(self.posts)} total posts collected")
                                                # Add extra delay after saving to be more conservative
                                                time.sleep(uniform(15, 30))
                                        
                                        # Successfully got data, break retry loop
                                        break
                                        
                                except Exception as e:
                                    if "429" in str(e):
                                        if retry < max_retries - 1:  # Don't sleep on last retry
                                            wait_time = min(3600, retry_delay * (2 ** retry))  # Exponential backoff capped at 1 hour
                                            logging.warning(f"Rate limited, waiting {wait_time} seconds...")
                                            time.sleep(wait_time)
                                            # Save progress before waiting
                                            self.save_posts()
                                            continue
                                    raise  # Re-raise if not a rate limit error or on last retry
                            
                        except Exception as e:
                            logging.warning(f"Error with query '{query}' for {stock}: {str(e)}")
                            time.sleep(60)  # Wait on error
                            continue
                    
                    if total_tweets > 0:
                        logging.info(f"Collected {total_tweets} tweets about {stock} from Twitter API")
                        stocks_collected.add(stock)
                        
                except Exception as e:
                    logging.error(f"Error collecting tweets via API for {stock}: {str(e)}")
                    time.sleep(60)  # Long delay on error
        
        # Only use Nitter as fallback for remaining stocks
        remaining_stocks = set(stocks) - stocks_collected
        if remaining_stocks:
            logging.info(f"Using Nitter as fallback for stocks: {remaining_stocks}")
            
            nitter_instances = [
                'https://nitter.privacydev.net',
                'https://nitter.fdn.fr',
                'https://nitter.1d4.us'
            ]
        
        # Initialize rate limiting with ultra-conservative settings
        self.rate_limits['nitter'] = {
            'requests_per_minute': 1,
            'min_delay': 120,  # 2 minutes between requests
            'max_retries': 3,
            'max_backoff': 900  # 15 minutes max backoff
        }
        
        def try_nitter_collection(stock, instance, query):
            """Helper function to handle Nitter collection attempt"""
            url = f'{instance}/search?f=tweets&q={quote_plus(query)}'
            
            try:
                time.sleep(self.rate_limits['nitter']['min_delay'])
                response = self._safe_request(url, platform='nitter', retries=1)
                
                if response and response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    tweets = soup.find_all('div', class_='tweet-content')
                    
                    new_posts = 0
                    for tweet in tweets[:10]:  # Limit to 10 tweets per query
                        text = tweet.get_text().strip()
                        if text:
                            stats = tweet.find_parent('div', class_='tweet').find('div', class_='tweet-stats')
                            score = 0
                            if stats:
                                numbers = [int(''.join(filter(str.isdigit, s))) for s in stats.strings if any(c.isdigit() for c in s)]
                                score = sum(numbers) if numbers else 0
                            
                            self.add_post(
                                text=text,
                                platform='x',
                                url=url,
                                score=score
                            )
                            new_posts += 1
                            nonlocal posts_collected
                            posts_collected += 1
                    
                    return new_posts
            except Exception as e:
                logging.error(f"Error with {instance} for {stock} (query: {query}): {str(e)}")
                raise
            
            return 0
        
        # Process remaining stocks one at a time with Nitter
        for stock in remaining_stocks:
                logging.info(f"Collecting posts about {stock} using Nitter...")
                success = False
                posts_before = len(self.posts)
                
                for instance in nitter_instances:
                    if success:
                        break
                    
                    search_queries = [
                        f'{stock}+stock+lang:en',
                        f'{stock}+trading+lang:en',
                        f'{stock}+investor+lang:en'
                    ]
                    
                    for query in search_queries:
                        if success:
                            break
                        
                        for attempt in range(self.rate_limits['nitter']['max_retries']):
                            try:
                                new_posts = try_nitter_collection(stock, instance, query)
                                
                                if new_posts > 0:
                                    logging.info(f"Collected {new_posts} tweets about {stock} from {instance}")
                                    
                                    if posts_collected >= save_threshold:
                                        self.save_posts()
                                        posts_collected = 0
                                        logging.info("Saved current progress")
                                    
                                    posts_after = len(self.posts)
                                    if posts_after - posts_before >= 10:
                                        logging.info(f"Collected sufficient posts for {stock}")
                                        success = True
                                        stocks_collected.add(stock)
                                        break
                                
                            except Exception as e:
                                if "429" in str(e):
                                    wait_time = min(self.rate_limits['nitter']['max_backoff'], 60 * (2 ** attempt))
                                    logging.warning(f"Rate limited, waiting {wait_time} seconds...")
                                    time.sleep(wait_time)
                                elif "timeout" in str(e).lower():
                                    wait_time = min(300, 45 * (attempt + 1))
                                    logging.warning(f"Network timeout, waiting {wait_time} seconds...")
                                    time.sleep(wait_time)
                                else:
                                    wait_time = min(120, 30 * (attempt + 1))
                                    logging.warning(f"Error occurred, waiting {wait_time} seconds...")
                                    time.sleep(wait_time)
                                continue
                        
                        if success:
                            break
                    
                    if not success:
                        time.sleep(uniform(30, 45))  # More conservative delays between instances
                if success:
                    break
                    
                # End of Nitter collection for this stock
                if not success:
                    logging.warning(f"Failed to collect sufficient posts for {stock} from any Nitter instance")
                    continue  # Move to next stock
                
                # Longer delay between stocks for Nitter fallback
                time.sleep(uniform(60, 90))  # Very conservative delays
                
                # Nitter collection complete for this stock
                if success:
                    stocks_collected.add(stock)
                    logging.info(f"Successfully collected posts for {stock} from Nitter")
        
        # Save any remaining posts
        if posts_collected > 0:
            self.save_posts()
        
        return stocks_collected
        
    def save_posts(self):
        """Save collected posts to a JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.data_dir, f'posts_{timestamp}.json')
        
        posts_data = [{
            'text': post.text,
            'platform': post.platform,
            'url': post.url,
            'score': post.score,
            'sentiment': post.sentiment  # This will calculate and cache the sentiment
        } for post in self.posts]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved {len(posts_data)} posts to {filename}")
        return filename
    
    def load_posts(self, filename):
        """Load posts from a JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                posts_data = json.load(f)
            
            for post_data in posts_data:
                post = Post(
                    text=post_data['text'],
                    platform=post_data['platform'],
                    url=post_data['url'],
                    score=post_data['score']
                )
                post._sentiment = post_data.get('sentiment')  # Restore cached sentiment
                self.posts.append(post)
                
            logging.info(f"Loaded {len(posts_data)} posts from {filename}")
            
        except Exception as e:
            logging.error(f"Error loading posts from {filename}: {str(e)}")
    
    def add_post(self, text, platform, url="", score=0):
        """Add a new post for analysis."""
        post = Post(text, platform, url, score)
        self.posts.append(post)
        
        # Save posts periodically (every 100 posts)
        if len(self.posts) % 100 == 0:
            self.save_posts()
        
    def analyze_stock(self, stock):
        """Analyze mentions and sentiment for a specific stock."""
        # Filter posts mentioning this stock
        stock_posts = [
            post for post in self.posts 
            if stock.upper() in post.text.upper()
        ]
        
        # Sort by engagement score for top 10
        top_posts = sorted(stock_posts, key=lambda x: x.score, reverse=True)[:10]
        
        return {
            'total_bullish': sum(1 for post in stock_posts if post.sentiment == 'bullish'),
            'total_bearish': sum(1 for post in stock_posts if post.sentiment == 'bearish'),
            'top10_bullish': sum(1 for post in top_posts if post.sentiment == 'bullish'),
            'top10_bearish': sum(1 for post in top_posts if post.sentiment == 'bearish'),
            'total_mentions': len(stock_posts)
        }

    def collect_reddit_posts(self, target_stock=None):
        """
        Collect posts from Reddit using PRAW.
        
        Args:
            target_stock (str, optional): If provided, only collect posts for this stock
        """
        import praw
        
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = 'StockSentimentBot/1.0'
        
        if not (client_id and client_secret):
            logging.warning("Reddit API credentials not found. Skipping Reddit data collection.")
            return
        
        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']
            
            stocks_to_search = [target_stock] if target_stock else STOCKS
            
            for stock in stocks_to_search:
                logging.info(f"Collecting Reddit posts about {stock}...")
                
                for subreddit_name in subreddits:
                    try:
                        subreddit = reddit.subreddit(subreddit_name)
                        
                        # Search for posts containing the stock symbol
                        search_query = f"${stock} OR {stock}"
                        posts = subreddit.search(search_query, limit=100, sort='hot')
                        
                        for post in posts:
                            # Calculate engagement score
                            score = post.score + post.num_comments
                            
                            # Only include posts from last 24 hours
                            post_time = datetime.fromtimestamp(post.created_utc)
                            if datetime.utcnow() - post_time <= timedelta(hours=24):
                                self.add_post(
                                    text=f"{post.title}\n{post.selftext}",
                                    platform='reddit',
                                    url=f"https://reddit.com{post.permalink}",
                                    score=score,
                                    timestamp=post_time
                                )
                            
                            # Get top comments
                            post.comments.replace_more(limit=0)  # Remove MoreComments objects
                            for comment in post.comments.list()[:10]:  # Top 10 comments
                                if len(comment.body.strip()) > 0:
                                    comment_time = datetime.fromtimestamp(comment.created_utc)
                                    if datetime.utcnow() - comment_time <= timedelta(hours=24):
                                        self.add_post(
                                            text=comment.body,
                                            platform='reddit',
                                            url=f"https://reddit.com{comment.permalink}",
                                            score=comment.score,
                                            timestamp=comment_time
                                        )
                        
                        # Respect rate limits
                        self._enforce_rate_limit('reddit')
                        
                    except Exception as e:
                        logging.error(f"Error collecting from r/{subreddit_name}: {str(e)}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error initializing Reddit client: {str(e)}")
    
    def analyze_all_stocks(self, stocks):
        """
        Analyze the specified list of stocks and store results.
        
        Args:
            stocks (list): List of stock symbols to analyze
        """
        # Initialize empty stats
        platform_stats = {}
        stock_stats = {stock: {'total': 0, 'reddit': 0, 'x': 0, 'stocktwits': 0, 'yahoo_finance': 0} for stock in stocks}
        
        # Track collection progress
        stocks_completed = set()
        save_interval = 50  # Save more frequently
        last_save = time.time()
        
        while stocks_completed != set(stocks):
            remaining_stocks = [s for s in stocks if s not in stocks_completed]
            
            # Try Reddit collection first (higher rate limits)
            for stock in remaining_stocks:
                try:
                    posts_before = len(self.posts)
                    self.collect_reddit_posts([stock])
                    posts_after = len(self.posts)
                    
                    
                    posts_added = posts_after - posts_before
                    if posts_added > 0:
                        logging.info(f"Added {posts_added} Reddit posts for {stock}")
                    
                    # Save periodically
                    if time.time() - last_save >= 300:  # Save every 5 minutes
                        self.save_posts()
                        last_save = time.time()
                        logging.info("Saved current progress")
                        
                except Exception as e:
                    logging.error(f"Error collecting Reddit posts for {stock}: {str(e)}")
                    time.sleep(30)  # Wait before retrying
            
            # Always try Twitter collection for all stocks
            try:
                twitter_collected = self.collect_x_posts(stocks)
                if twitter_collected:
                    logging.info(f"Collected Twitter data for: {twitter_collected}")
                    # Mark stocks as completed only if we have both Reddit and Twitter data
                    for stock in twitter_collected:
                        if any(p.platform == 'reddit' and stock.upper() in p.text.upper() for p in self.posts):
                            stocks_completed.add(stock)
                            logging.info(f"Completed both Reddit and Twitter collection for {stock}")
            except Exception as e:
                logging.error(f"Error in Twitter collection: {str(e)}")
                time.sleep(60)  # Longer wait for Twitter errors
            
            # Save progress
            if time.time() - last_save >= 300:
                self.save_posts()
                last_save = time.time()
                logging.info("Saved current progress")
            
            # If we haven't completed any new stocks, wait before retrying
            if not (stocks_completed - set(stocks)):
                logging.warning("No new stocks completed in this iteration, waiting before retry...")
                time.sleep(300)  # 5 minute wait
        
        # Final save of all collected posts
        self.save_posts()
        
        # Update statistics
        for post in self.posts:
            platform_stats[post.platform] = platform_stats.get(post.platform, 0) + 1
            for stock in stocks:
                if stock.upper() in post.text.upper():
                    stock_stats[stock]['total'] += 1
                    stock_stats[stock][post.platform] += 1
        
        # Log collection statistics
        logging.info("\nData Collection Summary:")
        logging.info("\nPosts by Platform:")
        for platform, count in platform_stats.items():
            logging.info(f"{platform}: {count} posts")
        
        logging.info("\nPosts by Stock:")
        for stock, stats in stock_stats.items():
            logging.info(f"\n{stock}:")
            logging.info(f"  Total mentions: {stats['total']}")
            for platform, count in stats.items():
                if platform != 'total':
                    logging.info(f"  {platform}: {count} posts")
        
        # Analyze sentiment for each stock
        results = {}
        for stock in stocks:
            logging.info(f"\nAnalyzing mentions for {stock}...")
            results[stock] = self.analyze_stock(stock)
        
        return results

def format_results_table(results):
    """Format results into a pretty table."""
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Stock'
    return df.to_string()

if __name__ == "__main__":
    # Create analyzer
    analyzer = StockAnalyzer()
    
    try:
        # Collect and analyze data
        logging.info("Starting data collection and analysis...")
        results = analyzer.analyze_all_stocks()
        
        # Print results
        print("\nStock Mention Analysis Results:")
        print("=" * 80)
        print(format_results_table(results))
        print("=" * 80)
        
        logging.info("Analysis complete!")
        
    except KeyboardInterrupt:
        logging.info("\nStopping analysis...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
