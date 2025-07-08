import pandas as pd

def get_samsung_news():
    """Get Samsung news from sample data only"""
    news_data = []
    sample_dates = [
        "2025-06-13", "2025-06-12", "2025-06-11", "2025-06-10", 
        "2025-06-09", "2025-06-06", "2025-06-05", "2025-06-04"
    ]
    sample_headlines = [
        "Samsung Galaxy S25 Ultra receives rave reviews from tech critics",
        "Samsung announces new AI features for upcoming foldable phones",
        "Samsung's chip division reports strong Q2 earnings",
        "Samsung partners with Google for enhanced Android integration",
        "Samsung's display technology leads industry innovation",
        "Samsung expands 5G network infrastructure globally",
        "Samsung's memory chip sales exceed market expectations",
        "Samsung unveils new smart home ecosystem",
        "Samsung's mobile market share grows in key regions",
        "Samsung invests heavily in next-generation display technology",
        "Samsung's semiconductor business faces supply chain challenges",
        "Samsung launches new sustainability initiatives",
        "Samsung's foldable phone sales hit record numbers",
        "Samsung collaborates with major gaming companies",
        "Samsung's AI research division makes breakthrough",
        "Samsung's TV division introduces revolutionary features",
        "Samsung expands manufacturing facilities in Asia",
        "Samsung's smartphone camera technology praised by experts",
        "Samsung announces new partnership with automotive industry",
        "Samsung's wearable technology gains market traction"
    ]
    for i, date in enumerate(sample_dates):
        for j in range(2):
            headline_idx = (i * 2 + j) % len(sample_headlines)
            news_data.append({
                'date': date,
                'headline': sample_headlines[headline_idx]
            })
    df = pd.DataFrame(news_data)
    df.to_csv('samsung_news.csv', index=False)
    print(f"✅ Saved {len(df)} sample news headlines to samsung_news.csv")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df

if __name__ == "__main__":
    get_samsung_news()

