import praw
import config
import pandas as pd

credentials = config.global_config

reddit = praw.Reddit(client_id=credentials['client_id'], client_secret=credentials['client_secret'], password=credentials["password"], user_agent=credentials["user_agent"], username=credentials["username"])

all_flair = ["AskIndia", "Non-Political", "[R]eddiquette", "Scheduled", "Photography", "Science/Technology", "Politics", "Business/Finance", "Policy/Economy", "Sports", "Food"]
subreddit = reddit.subreddit('india')
all_data = {"flair": [] , "comments": [], "timestamp":[], "isSelf": [], "commentNumber": [], "over_18": [], "upvotes": [], "body": [], "title":[]}

for flair in all_flair:
   flair_posts_list = subreddit.search(flair, limit = 100)
   for post in flair_posts_list:
       all_data["flair"].append(flair)
       all_data["title"].append(post.title)
       all_data["timestamp"].append(post.created_utc)
       all_data["isSelf"].append(int(post.is_self))
       all_data["commentNumber"].append(post.num_comments)
       all_data["over_18"].append(int(post.over_18))
       all_data["upvotes"].append(post.score)
       all_data["body"].append(post.selftext)
       mainStr = ''
       post.comments.replace_more(limit=None)
       for comment in post.comments:
           mainStr += comment.body + ' '
       all_data["comments"].append(mainStr)

dataStore = pd.DataFrame(all_data)
dataStore.to_csv('../csv_data/data.csv', index=False)
