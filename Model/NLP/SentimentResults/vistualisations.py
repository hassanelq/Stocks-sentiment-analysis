# date,text,sentiment_label,sentiment_score
# 2024-11-04 11:26:00,trump media stock prediction markets move eve election day,Neutral,0.9999934434890747
# 2024-11-04 10:22:00,tesla uber boost ev incentives offering 000 eco friendly driving,Positive,0.9999908208847046
# 2024-11-04 10:15:00,elon musk predicts fun second trump presidency mark cuban responds cratering economy blast,Neutral,0.9999591112136841
# 2024-11-04 09:54:00,tesla fsd fails detect deer nighttime collision raising safety concerns,Negative,0.9999966621398926
# 2024-11-04 09:46:00,tesla stock triggers sell signal booming byd eyes bev crown,Positive,0.9549660682678223
# 2024-11-04 09:44:00,barry diller elon musk deserved megalomaniac,Neutral,0.9988621473312378
# 2024-11-04 09:26:00,tesla ev sales china fall 5 3 october,Neutral,0.9921596646308899
# 2024-11-04 09:22:00,elon musk heads doge may unsettle tesla stock analyst,Neutral,0.9986594915390015
# 2024-11-04 09:19:00,elon musk big gamble tesla stock plunge kamala harris wins election,Neutral,0.9879649877548218
# 2024-11-04 08:20:00,chinese electric vehicle stocks rise deliveries jump october,Positive,0.9994152784347534
# 2024-11-04 08:10:00,teslas stock faces new risk trump defeat,Negative,0.9998800754547119
# 2024-11-04 07:27:00,tesla stock sixth straight day,Neutral,0.9999048709869385

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


results = pd.read_csv("twitter_sentiment.csv")

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=results, x="sentiment_label")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Plot sentiment score distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=results, x="sentiment_score", bins=20)
plt.title("Sentiment Score Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()

# Plot sentiment score distribution by sentiment label
plt.figure(figsize=(8, 6))
sns.histplot(data=results, x="sentiment_score", bins=20, hue="sentiment_label")
plt.title("Sentiment Score Distribution by Sentiment Label")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()
