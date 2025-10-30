import praw
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from psaw import PushshiftAPI
import datetime as dt

api = PushshiftAPI()

start_epoch = int((datetime.now() - timedelta(days=5*365)).timestamp())

submissions = api.search_submissions(
    after=start_epoch,
    subreddit='Bitcoin',
    filter=['title', 'created_utc', 'score', 'num_comments'],
    limit=10000  # Increase as needed
)

posts = []
for submission in submissions:
    posts.append({
        "title": submission.title,
        "created_utc": datetime.utcfromtimestamp(submission.created_utc),
        "score": submission.score,
        "num_comments": submission.num_comments
    })

# Convert to DataFrame
df = pd.DataFrame(posts)
print("#"*30)
print(len(df))
print(df.head())
print(df.tail())
print("#"*30)
df['date'] = pd.to_datetime(df['created_utc']).dt.date
daily_mentions = df.groupby('date').size().reset_index(name='mention_count')

# Save and show
# daily_mentions.to_csv('reddit_mentions_bitcoin.csv', index=False)
print(daily_mentions.head())
print(daily_mentions.tail())
print(len(daily_mentions))


# Plot daily mentions
plt.figure(figsize=(14, 6))
plt.plot(daily_mentions['date'], daily_mentions['mention_count'], color='darkorange', linewidth=1.5)
plt.title('Daily Reddit Mentions of Bitcoin (r/Bitcoin)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Number of Mentions')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig("reddit_mentions_plot.png")  # Save to file (optional)
plt.show()