import pandas as pd
import feedparser
from datetime import datetime

companies = ["Microsoft", "Apple", "Google", "Amazon"]
all_headlines = []

for company in companies:
    rss_url = f'https://news.google.com/rss/search?q={company}'
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            date_str = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
        else:
            date_str = ''
        all_headlines.append({
            'date': date_str,
            'headline': entry.title,
            'company': company
        })

# Save to CSV
if all_headlines:
    df = pd.DataFrame(all_headlines)
    print(df.head())
    df.to_csv("tech_news.csv", index=False)
else:
    print("No headlines found. Try a different company or source.")

