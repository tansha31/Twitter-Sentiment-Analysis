# fetch tweets by keyword posted in last 7 days
import json
import requests
import streamlit as st

bearer_token = st.secrets['bearer_token']


# Method required by bearer token authentication
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r


# Endpoint URL for RECENT SEARCH
def create_url(query, tweet_fields, max_results):
    return "https://api.twitter.com/2/tweets/search/recent?query={}&tweet.fields={}&max_results={}".format(
        query, tweet_fields, max_results)


def connection_to_endpoint(url, params=None):
    response = requests.request(
        method="GET", url=url, params=params, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception("Request returned an error: {} {}".format(
            response.status_code, response.text))
    return response.json()


def get_recent_tweets(keyword: str, max_results: int):
    tweet_fields = 'text,author_id,created_at,public_metrics'
    url = create_url(keyword, tweet_fields, max_results)
    response = connection_to_endpoint(url)
    return response['data']


if __name__ == '__main__':
    response = get_recent_tweets(input('Keyword: '), input('Tweets: '))
    print(json.dumps(response, indent=4, sort_keys=True))
