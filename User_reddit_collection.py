import praw
from praw.models import MoreComments
import pandas as pd

class UserCollection():

    def __init__(self, subreddit, num_of_submissions=1):
        self.user_agent = "Sentiment data collector 1.0"
        self.reddit = praw.Reddit(
    client_id="kT9UUS3q7lspfNMK4epo6g",
    client_secret="lzd-5QrjADwJe2Zme-IghWc2yx492g",
    user_agent = self.user_agent,
    check_for_async = False
        )
        self.subreddit = subreddit
        self.num_of_submissions = num_of_submissions

    def data_collection(self):
        df = pd.DataFrame()
        sub = self.subreddit
        for submission in self.reddit.subreddit(sub).hot(limit=self.num_of_submissions):
            for comment in submission.comments:
                if isinstance(comment, MoreComments): #Avoids known error that can occur when collenting comments
                    continue
                df = df.append([[comment.body, comment.created_utc, comment.score, comment.subreddit]],ignore_index=True)

        df.columns = ['Comment','Date', 'Score', 'Subreddit']
        df.Comment.astype(str)
        return df
