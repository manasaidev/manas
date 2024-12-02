import time
from datetime import datetime, timezone
from twitter import TwitterClient
from generator import CustomCryptoResponseGenerator

MONITORED_ACCOUNT = 'manas'
SLEEP_INTERVAL = 60

def process_mentions(twitter_client, generator, mentions):
    for mention in mentions:
        username = mention.user.screen_name
        prompt = mention.full_text.replace(f'@{MONITORED_ACCOUNT}', '').strip()

        if prompt:
            response = generator.generate_response(prompt)
            status = f"@{username} {response}"
            twitter_client.post_tweet(status, in_reply_to_status_id=mention.id_str)
        else:
            print(f"Skipping empty prompt from @{username}")

def main():
    start_time = datetime.now(timezone.utc)
    print(f"Bot started at {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    twitter_client = TwitterClient()
    generator = CustomCryptoResponseGenerator()

    generator.load_external_data()
    generator.prepare_data()
    generator.train_transformer_model()

    since_id = None

    while True:
        try:
            mentions = twitter_client.search_mentions(f'@{MONITORED_ACCOUNT}', since_id=since_id)

            if mentions:
                since_id = mentions[0].id_str
                process_mentions(twitter_client, generator, mentions)

            print(f"Sleeping for {SLEEP_INTERVAL} seconds...")
            time.sleep(SLEEP_INTERVAL)

        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(f"Sleeping for {SLEEP_INTERVAL} seconds before retrying...")
            time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    main()