# used by tweet_lookup.py for extracting user id
import requests
from secret import authorize

bearer_token = authorize['bearer_token']


# Method required by bearer token authentication
def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r


# Endpoint URL for USER LOOKUP
def create_url(username):
    return "https://api.twitter.com/2/users/by/username/{}".format(username)


def connection_to_endpoint(url):
    response = requests.request(method="GET", url=url, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception("Request returned an error: {} {}".format(
            response.status_code, response.text))
    return response.json()


def get_user_by_username(username):
    url = create_url(username)
    response = connection_to_endpoint(url)
    return response


if __name__ == "__main__":
    user = get_user_by_username(input('Username: '))
    print(user['data'])
