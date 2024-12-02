import tweepy
from config import (
    BEARER_TOKEN,
    CONSUMER_KEY,
    CONSUMER_SECRET,
    ACCESS_TOKEN,
    ACCESS_TOKEN_SECRET
)

class TwitterClient:
    def __init__(self):
        self.api = self.authenticate()

    def authenticate(self):
        auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        try:
            api.verify_credentials()
            print("Authentication with Twitter successful.")
            return api
        except tweepy.TweepyException as e:
            print(f"Error during authentication: {e}")
            raise e

    def search_mentions(self, query, since_id=None):
        return self.api.search_tweets(q=query, since_id=since_id, tweet_mode='extended')

    def post_tweet(self, status, in_reply_to_status_id=None):
        return self.api.update_status(status=status, in_reply_to_status_id=in_reply_to_status_id)