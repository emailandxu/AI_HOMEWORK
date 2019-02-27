import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("./data/labeledTrainData.tsv", header=0,
                    delimiter="\t",  quoting=3)

def review_to_words(text):
    # remove html tag
    text = BeautifulSoup(text).get_text()
    # remove punctuation and number
    letters_only = re.sub("[^a-zA-Z]"," ",text)

    # convert text into lower case
    lower_case = letters_only.lower()
    # split text into words
    words = lower_case.split()
    stops = set(stopwords.words("english"))
    # remove stop words from "words"
    meaningful_words = [w for w in words if w not in
             stops ]

    # return the result that joined the words
    # back into one string separated by space
    return " ".join(meaningful_words)


clean_train_reviews = []
num_reviews = train["review"].size

for i in range(num_reviews):
    # for showing status
    if (i+1)%1000==0:
        print("Reivew: %d/%d"%(i+1,num_reviews))

    clean_train_reviews.append( review_to_words(
        train["review"][i]
    ))

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

print("Training the random forest..")

forest = RandomForestClassifier(n_estimators=100)

# train Random forest
forset = forest.fit(train_data_features, train["sentiment"])


test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t",
                   quoting=3)
print("test set shape:",test.shape)

num_reviews = test["review"].size
clean_test_reviews = []

print("Cleaning and parsing the test set moview reviews...")

for i in range(num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"],"sentiment":result})

output.to_csv("Bag_of_words_model.csv", index=False, quoting=3)