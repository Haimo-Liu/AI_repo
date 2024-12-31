#!/usr/bin/env python3
import os
import logging
from datetime import datetime
from stock_sentiment import StockAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Set up environment variables for API access
    os.environ['TWITTER_BEARER_TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAAefiAAAAAAAUQ6D8XH9G8QmI3ur3BWCwdvfDmA%3DicLnZh0R2C0HJxPOFrJRH3MyteE2reeCCXqTx36CIAtWEedHR4'
    os.environ['REDDIT_CLIENT_ID'] = 'SVT4b14ZZpg4NzogTfLQUg'
    os.environ['REDDIT_CLIENT_SECRET'] = 'VRFJmecbAwyc-FzFiBKJK9kz7aqrQA'

    # Initialize analyzer with more conservative rate limits
    analyzer = StockAnalyzer()
    
    # Adjust rate limits to be more conservative
    analyzer.rate_limits = {
        'twitter': {
            'requests_per_minute': 5,  # Much more conservative Twitter API limit
            'min_delay': 30,  # 30 seconds between requests
            'max_retries': 5
        },
        'reddit': {
            'requests_per_minute': 30,  # Reddit API limit
            'min_delay': 2,  # 2 seconds between requests
            'max_retries': 3
        },
        'nitter': {
            'requests_per_minute': 1,
            'min_delay': 120,  # 2 minutes between requests
            'max_retries': 3,
            'max_backoff': 900  # 15 minutes max backoff
        }
    }
    
    # List of stocks to analyze
    stocks = ['RKLB', 'PLTR', 'APP', 'HIMS', 'SNOW', 'KVYO', 'SHOP', 'TTAN']
    
    try:
        # Start data collection
        logging.info("Starting data collection and analysis...")
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Clean start for data collection
        analyzer.posts = []
        logging.info("Starting fresh data collection...")
        
        # Collect and analyze data
        results = analyzer.analyze_all_stocks(stocks)
        
        # Save results to a file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            f.write("Stock Sentiment Analysis Results\n")
            f.write("=" * 80 + "\n\n")
            f.write("Collection Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Posts Collected: {len(analyzer.posts)}\n")
            f.write("\nDetailed Results:\n")
            f.write("=" * 80 + "\n")
            
            for stock, stats in results.items():
                f.write(f"\n{stock}:\n")
                f.write(f"Total Bullish Mentions: {stats.get('total_bullish', 0)}\n")
                f.write(f"Total Bearish Mentions: {stats.get('total_bearish', 0)}\n")
                f.write(f"Top 10 Discussions - Bullish: {stats.get('top10_bullish', 0)}\n")
                f.write(f"Top 10 Discussions - Bearish: {stats.get('top10_bearish', 0)}\n")
                f.write(f"Total Mentions: {stats.get('total_mentions', 0)}\n")
                f.write("-" * 40 + "\n")
        
        logging.info(f"Results saved to {results_file}")
        
    except KeyboardInterrupt:
        logging.info("\nStopping data collection...")
        # Save partial results on interrupt
        analyzer.save_posts()
    except Exception as e:
        logging.error(f"Error during data collection: {str(e)}")
        # Save partial results on error
        analyzer.save_posts()
        raise

def send_email(results, recipient):
    """Send analysis results via email."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from datetime import datetime
    
    sender = "stock.sentiment.bot@gmail.com"
    password = os.getenv('SMTP_PASSWORD')
    
    if not password:
        logging.error("SMTP password not found in environment variables")
        return
    
    msg = MIMEMultipart()
    msg['Subject'] = f'Stock Sentiment Analysis Report - {datetime.now().strftime("%Y-%m-%d")}'
    msg['From'] = sender
    msg['To'] = recipient
    
    # Format results as a table
    table = "| Stock | Total Bullish | Total Bearish | Top 10 Bullish | Top 10 Bearish |\n"
    table += "|-------|--------------|---------------|----------------|----------------|\n"
    
    for stock, data in results.items():
        table += f"| {stock} | {data['total_bullish']} | {data['total_bearish']} | "
        table += f"{data['top10_bullish']} | {data['top10_bearish']} |\n"
    
    body = f"""
Stock Sentiment Analysis Report for the last 24 hours:

{table}

Note: This is an automated report generated daily.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        logging.info(f"Email sent successfully to {recipient}")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, help='Email address to send results to')
    args = parser.parse_args()
    
    results = main()
    
    if args.email:
        send_email(results, args.email)
