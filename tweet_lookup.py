# fetch tweets for specific user account
import json
import requests
import user_lookup
from secret import authorize

bearer_token = authorize['bearer_token']


# Method required by bearer token authentication
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r


# Endpoint URL for TWEET LOOKUP
def create_url(id):
    return "https://api.twitter.com/2/users/{}/tweets".format(id)


def connection_to_endpoint(url, params=None):
    response = requests.request(
        method="GET", url=url, params=params, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception("Request returned an error: {} {}".format(
            response.status_code, response.text))
    return response.json()


def get_tweets_timeline(username: str, max_results: int):
    id = user_lookup.get_user_by_username(username)['data']['id']
    url = create_url(id)
    params = {"exclude": "retweets,replies",
              "max_results": max_results, "tweet.fields": "public_metrics"}
    response = connection_to_endpoint(url, params)
    return response['data']


if __name__ == "__main__":
    response = get_tweets_timeline(input('Username: '), input('Tweets: '))
    print(json.dumps(response, indent=4, sort_keys=True))
