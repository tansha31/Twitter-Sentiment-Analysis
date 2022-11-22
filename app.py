import re
import json
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords
from search_tweets import get_recent_tweets
from tweet_lookup import get_tweets_timeline
from nltk.stem.porter import PorterStemmer
from spacy.lang.en import stop_words as stop_words1
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# setup
nltk.download('vader_lexicon')
model = SentimentIntensityAnalyzer()
twitter_img = Image.open('./resources/twitter.png')
pending = Image.open('./resources/pending.png')
positive = Image.open('./resources/positive.png')
negative = Image.open('./resources/negative.png')
neutral = Image.open('./resources/neutral.png')


# main
st.title('Twitter Tweet Sentiment Analysis')
# st.markdown("<h1 style='text-align: center; font-size : 2.7rem'>Twitter Tweet Sentiment Analysis</h2>", unsafe_allow_html=True)
st.markdown("""This app fetches tweets from twitter based on the option selected.
            It then processes the tweets through machine learing pipeline function for sentiment analysis.
            The resulting sentiments and corresponding tweets are then put in a dataframe for display and visualised through graphs.""")

st.markdown('')
st.subheader('How do you wanna scrape tweets? 🤔')
search_by = st.selectbox(label='Choose from the drop down', options=[
                         'Tweets Lookup', 'Search Tweets'], help='twitter-v2 api')
st.markdown("""**Tweet Lookup:** returns a variety of information about a single tweet specified by the requested ID.   
            **Search Tweets:** returns a list of tweets from the last seven days that match a specified search keyword.""")
st.markdown('')

# sidebar configuration
st.sidebar.image(twitter_img, width=75)
st.sidebar.title('Sentiment Analysis . . . 🚀')
st.sidebar.markdown('- Text Input')
st.sidebar.markdown('- Tokenization')
st.sidebar.markdown('- Stop Words Filtering')
st.sidebar.markdown('- Stemming/Lemmitization')
st.sidebar.markdown('- Classification')
st.sidebar.markdown('- Sentiment Class')
st.sidebar.markdown('')

option = st.sidebar.selectbox(label='Choose from the drop down', options=[
                              'Display tweets', 'Run sentiment analyzer'])

if option == 'Display tweets':
    raw = st.sidebar.checkbox(label='View Raw')

else:
    wordcloud = st.sidebar.checkbox(label='Generate Word Cloud')

st.sidebar.markdown('')
if st.sidebar.button('Contact Developer'):
    st.sidebar.markdown('**Tanmay Sharma** (Data Scientist)')
    st.sidebar.markdown(
        """[![image](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/tansha31)
        [![image](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tanmay-sharma-75718b195/)""")


@st.cache
def sentiment_analyzer(text):
    polarity = model.polarity_scores(text)['compound']

    if polarity < 0:
        return 'Negative'

    if polarity == 0:
        return 'Neutral'

    return 'Positive'


@st.cache
def textPreprocessing(input_txt):
    if pd.isnull(input_txt) == True:
        return input_txt

    # convert all text to lowercase
    input_txt = input_txt.lower()

    # remove usernames
    r = re.findall("@[\w]|@[][\w]", input_txt)
    # print(r)
    for word in r:
        input_txt = re.sub(word, "", input_txt)

    # remove punctuation, special character and numbers
    input_txt = input_txt.replace("[^a-zA-Z#]", " ")

    # tokenization of tweets
    input_txt = input_txt.split()

    # stopwords removal
    stop_words2 = stop_words1.STOP_WORDS
    tokenized_tweet = []
    for word in input_txt:
        if word not in stop_words2 and len(word) > 2:
            tokenized_tweet.append(word)

    # stemming
    stemmer = PorterStemmer()
    tokens = []
    for word in tokenized_tweet:
        tokens.append(stemmer.stem(word))
    tokenizedTweet = tokens

    # combine into sentence
    tokenizedTweet = " ".join(tokenizedTweet)

    return tokenizedTweet


@st.cache
def overall_sentiment(df):
    return df['sentiment'].value_counts().index[0]


@st.cache
def convert_to_df(response: list):
    """
    Converts json to pandas dataframe
    """
    tweets = []
    like_count = []
    reply_count = []
    quote_count = []
    retweet_count = []

    for tweet in response:
        tweets.append(tweet['text'])
        retweet_count.append(tweet['public_metrics']['retweet_count'])
        reply_count.append(tweet['public_metrics']['reply_count'])
        like_count.append(tweet['public_metrics']['like_count'])
        quote_count.append(tweet['public_metrics']['quote_count'])

    data = {
        'tweet': tweets,
        'retweet_count': retweet_count,
        'reply_count': reply_count,
        'like_count': like_count,
        'quote_count': quote_count
    }

    df = pd.DataFrame(data=data)
    return df


def generateWC(df):
    text = df['tweet'].apply(textPreprocessing)
    all_words = " ".join([sentence for sentence in text])
    wordcloud = WordCloud(width=800, height=500, random_state=42,
                          max_font_size=100).generate(all_words)

    # plot the graph
    fig = plt.figure(figsize=(9, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)


def analyze(df):
    st.success('Average Like Count: {}'.format(
        round(sum(df['like_count']) / len(df), 2)))
    st.info('Average Retweet Count: {}'.format(
        round(sum(df['retweet_count']) / len(df), 2)))
    st.warning('Average Reply Count: {}'.format(
        round(sum(df['reply_count']) / len(df), 2)))
    st.error('Average Quote Count: {}'.format(
        round(sum(df['quote_count']) / len(df), 2)))


if search_by == 'Tweets Lookup':
    st.subheader('Tweets Lookup')
    username = st.text_input(
        label='Enter the username for which you want to fetch the tweets.', placeholder='Google')
    max_results = st.slider(
        label='Choose the number of tweets for which you want to analyze sentiment (Max 100).', min_value=5, max_value=100, value=25)

    if st.button(label='Submit', help='Click to fetch data'):
        response = get_tweets_timeline(username, max_results)
        df = convert_to_df(response)

        if option == 'Display tweets':
            st.dataframe(df)
            analyze(df)

            st.download_button(label='Download data as CSV', data=df.to_csv().encode(
                'utf-8'), file_name='{}.csv'.format(username))

            if raw:
                st.code(body=json.dumps(obj=response, indent=4, sort_keys=True))
                st.download_button(label='Download raw', data=json.dumps(
                    response, indent=4, sort_keys=True), file_name='{}_raw.txt'.format(username))

        elif option == 'Run sentiment analyzer':
            st.subheader('Sentiment Analysis')
            # st.image(pending, caption='kal ana kal', width=700)
            df['sentiment'] = np.vectorize(sentiment_analyzer)(df['tweet'])
            st.write(df)

            st.download_button(label='Download data as CSV', data=df.to_csv().encode(
                'utf-8'), file_name='{}.csv'.format(username))

            if wordcloud:
                st.markdown('')
                st.markdown('')
                st.subheader('Word Cloud')
                generateWC(df)

            st.markdown('')
            st.markdown('')

            st.subheader('Visualizing through graph')

            fig = plt.figure(figsize=(9, 7))
            sns.countplot(x='sentiment', data=df, palette='Set2')
            plt.title('Sentiment')
            st.pyplot(fig)

            st.markdown('')

            sentiment = overall_sentiment(df)

            if sentiment == 'Positive':
                st.success('Overall sentiment: Positive')
                # st.image(positive, width=600)

            elif sentiment == 'Neutral':
                st.warning('Overall sentiment: Neutral')
                # st.image(neutral)

            else:
                st.error('Overall sentiment: Negative')
                # st.image(negative)


if search_by == 'Search Tweets':
    st.subheader('Search Tweets')
    keyword = st.text_input(
        label='Enter the keyword for which you want to fetch the tweets.', placeholder='World War 3')
    max_results = st.slider(
        label='Choose the number of tweets for which you want to analyze sentiment (Max 100).', min_value=10, max_value=100, value=30)

    if st.button(label='Submit', help='Click to fetch data'):
        response = get_recent_tweets(keyword, max_results)
        df = convert_to_df(response)

        if option == 'Display tweets':
            st.dataframe(df)
            analyze(df)

            st.download_button(label='Download data as CSV', data=df.to_csv().encode(
                'utf-8'), file_name='{}.csv'.format(keyword))

            if raw:
                st.code(body=json.dumps(obj=response, indent=4, sort_keys=True))
                st.download_button(label='Download raw', data=json.dumps(
                    response, indent=4, sort_keys=True), file_name='{}_raw.txt'.format(keyword))

        elif option == 'Run sentiment analyzer':
            st.subheader('Sentiment Analysis')
            # st.image(pending, caption='kal ana kal', width=700)
            df['sentiment'] = np.vectorize(sentiment_analyzer)(df['tweet'])
            st.write(df)

            st.download_button(label='Download data as CSV', data=df.to_csv().encode(
                'utf-8'), file_name='{}.csv'.format(keyword))

            if wordcloud:
                st.markdown('')
                st.markdown('')
                st.subheader('Word Cloud')
                generateWC(df)

            st.markdown('')
            st.markdown('')

            st.subheader('Visualizing through graph')

            fig = plt.figure(figsize=(9, 7))
            sns.countplot(x='sentiment', data=df, palette='Set2')
            plt.title('Sentiment')
            st.pyplot(fig)

            st.markdown('')

            sentiment = overall_sentiment(df)

            if sentiment == 'Positive':
                st.success('Overall sentiment: Positive')
                # st.image(positive, width=600)

            elif sentiment == 'Neutral':
                st.warning('Overall sentiment: Neutral')
                # st.image(neutral)

            else:
                st.error('Overall sentiment: Negative')
                # st.image(negative)


st.markdown('')
st.markdown('')
st.markdown('---')
st.markdown('')
st.markdown('')
st.markdown('')
st.markdown("<h6 style='text-align: center; color: black;'>© Twitter Tweet Sentiment Analyzer. All Rights Reserved.</h6>", unsafe_allow_html=True)
