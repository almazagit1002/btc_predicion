from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt

# Set up pytrends
pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ['bitcoin']

# Fetch 5-year historical data
pytrends.build_payload(kw_list, timeframe='today 5-y')
df = pytrends.interest_over_time()
print("#"*50)
print(len(df)/365)
print("#"*50)
print("#"*50)
print(len(df))
print("#"*50)
# Clean up
df = df.drop(columns=['isPartial'], errors='ignore')
print("#"*50)
print(len(df)/365)
print("#"*50)
print("#"*50)
print(len(df))
print("#"*50)
# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['bitcoin'], color='orange', linewidth=2)
plt.title('Google Search Interest for "bitcoin" (Past 5 Years)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Search Interest')
plt.grid(True)
plt.tight_layout()
plt.show()
