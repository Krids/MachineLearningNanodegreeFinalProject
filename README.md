# Content: Udacity Nanodegree Engenheiro de Machine Learning

## Project: Usando notícias para prever o movimento de acões

### Install

This project requires **Python 3.X** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [seaborn](https://seaborn.pydata.org/)
- [scipy](https://www.scipy.org/)


You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 3.x installer and not the Python 2.7 installer. 

### Code

Template code is provided in the `kernel.ipynb` notebook file. To run this kernel you need to use Kaggle platform, because the data is imported using Kaggle`s functions and also the data is only available in there.

### Run

In a terminal or command window, navigate to the top-level project directory `Udacity-FinalProject/` (that contains this README) and run one of the following commands:

```bash
ipython notebook kernel.ipynb
```  
or
```bash
jupyter notebook kernel.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

____

## Data

 #### Market Data

  * `time(datetime64[ns, UTC])` - the current time (in marketdata, all rows are taken at 22:00 UTC)
  * `assetCode(object)` - a unique id of an asset
  * `assetName(category)` - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
  * `universe(float64)` - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
  * `volume(float64)` - trading volume in shares for the day
  * `close(float64)` - the close price for the day (not adjusted for splits or dividends)
  * `open(float64)` - the open price for the day (not adjusted for splits or dividends)
  * `returnsClosePrevRaw1(float64)` - see returns explanation above
  * `returnsOpenPrevRaw1(float64)` - see returns explanation above
  * `returnsClosePrevMktres1(float64)` - see returns explanation above
  * `returnsOpenPrevMktres1(float64)` - see returns explanation above
  * `returnsClosePrevRaw10(float64)` - see returns explanation above
  * `returnsOpenPrevRaw10(float64)` - see returns explanation above
  * `returnsClosePrevMktres10(float64)` - see returns explanation above
  * `returnsOpenPrevMktres10(float64)` - see returns explanation above
  * `returnsOpenNextMktres10(float64)` - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.

 #### News Data

  * `time(datetime64[ns, UTC])` - UTC timestamp showing when the data was available on the feed (second precision)
  * `sourceTimestamp(datetime64[ns, UTC])` - UTC timestamp of this news item when it was created
  * `firstCreated(datetime64[ns, UTC])` - UTC timestamp for the first version of the item
  * `sourceId(object)` - an Id for each news item
  * `headline(object)` - the item's headline
  * `urgency(int8)` - differentiates story types (1: alert, 3: article)
  * `takeSequence(int16)` - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.
  * `provider(category)` - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)
  * `subjects(category)` - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.
  * `audiences(category)` - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)
  * `bodySize(int32)` - the size of the current version of the story body in characters
  * `companyCount(int8)` - the number of companies explicitly listed in the news item in the subjects field
  * `headlineTag(object)` - the Thomson Reuters headline tag for the news item
  * `marketCommentary(bool)` - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries
  * `sentenceCount(int16)` - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
  * `wordCount(int32)` - the total number of lexical tokens (words and punctuation) in the news item
  * `assetCodes(category)` - list of assets mentioned in the item
  * `assetName(category)` - name of the asset
  * `firstMentionSentence(int16)` - the first sentence, starting with the headline, in which the scored asset is mentioned.
	1: headline
	2: first sentence of the story body
	3: second sentence of the body, etc
	0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.
  * `relevance(float32)` - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.
  * `sentimentClass(int8)` - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
  * `sentimentNegative(float32)` - probability that the sentiment of the news item was negative for the asset
  * `sentimentNeutral(float32)` - probability that the sentiment of the news item was neutral for the asset
  * `sentimentPositive(float32)` - probability that the sentiment of the news item was positive for the asset
  * `sentimentWordCount(int32)` - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.
  * `noveltyCount12H(int16)` - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
  * `noveltyCount24H(int16)` - same as above, but for 24 hours
  * `noveltyCount3D(int16)` - same as above, but for 3 days
  * `noveltyCount5D(int16)` - same as above, but for 5 days
  * `noveltyCount7D(int16)` - same as above, but for 7 days
  * `volumeCounts12H(int16)` - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
  * `volumeCounts24H(int16)` - same as above, but for 24 hours
  * `volumeCounts3D(int16)` - same as above, but for 3 days
  * `volumeCounts5D(int16)` - same as above, but for 5 days
  * `volumeCounts7D(int16)` - same as above, but for 7 days