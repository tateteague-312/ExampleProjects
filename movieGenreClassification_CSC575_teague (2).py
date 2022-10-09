import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial 
from collections import Counter
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
stops = stopwords.words('english')
# nltk.download('punkt')
# nltk.download('stopwords')



def textCleaning(df):
    '''Takes in dataframe with description column and performs text cleaning(stop word removal and stemming) and tokenization'''
    porter = nltk.stem.PorterStemmer()
    ### Remove non alphabetic characters
    df['descr'] = df['descr'].str.replace('[^a-zA-Z]',' ', regex=True)
    ### Remove Stop words
    df['noStopWords'] = df['descr'].apply(lambda x: [word.lower() for word in x.split() if word.lower() not in stops])
    ### Stem Words
    df['stemmed'] = df['noStopWords'].apply(lambda y:[porter.stem(word) for word in y])

    ### Get unique words and flatten into single list to look for names, foreign words and typos
    uniqueWords_by_doc = np.asarray(df['stemmed'].apply(lambda z: list(set(z))))
    flatUnique = [word for sent in uniqueWords_by_doc for word in sent]
    docFreq = Counter(flatUnique)
    uncommonWords_byDoc = Counter({key: c for key,c in docFreq.items() if c < 3})
    print(f'\n# of uncommon words across documents to remove: {len(uncommonWords_byDoc):,}')

    df['reduced'] = df['stemmed'].apply(lambda a: ' '.join([word for word in a if word not in uncommonWords_byDoc]))
    ### Understand scope and size of final word list
    wordList = np.asarray(df['reduced'].apply(lambda x: x.split()))
    totalWords = [word for sent in wordList for word in sent]
    uniqWords = Counter(totalWords)

    print(f'\nTotal # of words: {len(totalWords):,}')
    print(f'# of unique words: {len(uniqWords):,}')

    return df


def doc_tf(df):
    '''Creates a document by term frequency dataframe, term frequency matrix'''
    vec = CountVectorizer()  
    vectors = vec.fit_transform(df.reduced)
    ### decrease memory usage and we know because these are counts only upper bound for negative values use int8
    doctf = pd.DataFrame(vectors.todense().astype('uint16')) 
    doctf.columns = vec.get_feature_names_out()

    return doctf

def tf_x_idf(df):
    '''Create a function to find the TF x IDF weights for a given tf matrix'''
    N = df.shape[1]
    #count non zero inputs as the term occurs
    df['df'] = df.apply(np.count_nonzero,axis=1)
    #compute inverse term frequency 
    df['idf'] = np.log2(N/df['df']).astype(np.float16)
    #compute final weight matrix 
    tf_idf = df.iloc[:,:-2].multiply(df['idf'],axis='index')

    return tf_idf.astype(np.float16)

def RocchioTrain(df):
    '''Takes in a TF x IDF matrix with appended labels returns Prototype Vector & unique classes'''
    numWords = df.drop('_label_', axis = 1).shape[1]

    labels = np.asarray(df['_label_'].values)
    unique = np.unique(labels)

    prototypeVector = np.zeros((len(unique),numWords))
    for i in unique:

        prototypeVector[i-1] = np.asarray(df[df['_label_'] == i].sum())[:-1]

    return prototypeVector, unique


def RocchioClassification(prototypes, classes, testInstance):
    '''Given a test vectorized word list and prototype vectors of each class performs classification of words'''
    try:
        pred = {}
        for proto,class_ in zip(prototypes,classes):
            proto = proto.reshape(1, -1)
            prob = 1-spatial.distance.cosine(proto, testInstance)
            pred[class_] = prob

        maxKey = max(pred, key = pred.get)
        actual = testInstance[-1:]
        return maxKey, actual[0], pred[maxKey]
    except:
        arr = testInstance[:-1].reshape(1,-1)
        pred = {}
        for proto,class_ in zip(prototypes,classes):
            proto = proto.reshape(1, -1)
            prob = 1-spatial.distance.cosine(proto, arr)
            pred[class_] = prob

        maxKey = max(pred, key = pred.get)
        actual = testInstance[-1:]
        return maxKey, actual[0], pred[maxKey]


def accuracy(df, ptv, classes):
    '''Takes a test data set, posteriors and priors fr a trained doc x term matrix and returns accuracy of predictions'''
    testSet = df.values
    ### can use numpy sum with true/false & list comprehension to get one liner
    acc = np.asarray([RocchioClassification(ptv,classes,i)[0] == RocchioClassification(ptv,classes,i)[1] for i in testSet])
    
    return acc.sum() / len(acc)  

def recommendations(df,q):
    '''Takes in a vectorized movie description and returns a dataframe of cosine similarities between its description and all other movies in training set'''
    docRank_q = []
    for i in range(df.shape[0]):
        doc = df.iloc[i].values
        query = q.values
        docRank_q.append(1-spatial.distance.cosine(doc,query))
    finDocRank = pd.DataFrame({'DocRank':docRank_q})

    return finDocRank

def main():

    colNames = ['id','title','genre','descr']
    dfMovies = pd.read_csv('C:/Users/tate5/OneDrive/CSC 575/Final Project/train_data.txt',delimiter=':::',names=colNames,engine='python')
    dfMovies['genre'] = dfMovies['genre'].apply(lambda x: x.strip())


    ### Class generalization, broadening classes to reduce overall number and increase with group samples. Based off common movie genre type groupings
    dfMovies['_label_'] = dfMovies.genre.copy()
    dfMovies['_label_'].replace('action',1,inplace=True)
    dfMovies['_label_'].replace('thriller', 1,inplace=True)
    dfMovies['_label_'].replace('adventure', 2,inplace=True)
    dfMovies['_label_'].replace('mystery', 2,inplace=True)
    dfMovies['_label_'].replace('comedy', 3,inplace=True)
    dfMovies['_label_'].replace('crime', 4,inplace=True)
    dfMovies['_label_'].replace('drama', 5,inplace=True)
    dfMovies['_label_'].replace('romance', 5,inplace=True)
    dfMovies['_label_'].replace('adult', 5,inplace=True)
    dfMovies['_label_'].replace('history', 6,inplace=True)
    dfMovies['_label_'].replace('biography', 6,inplace=True)
    dfMovies['_label_'].replace('war', 6,inplace=True)
    dfMovies['_label_'].replace('news', 6,inplace=True)
    dfMovies['_label_'].replace('horror', 7,inplace=True)
    dfMovies['_label_'].replace('music', 8,inplace=True)
    dfMovies['_label_'].replace('musical', 8,inplace=True)
    dfMovies['_label_'].replace('sci-fi', 9,inplace=True)
    dfMovies['_label_'].replace('fantasy', 9,inplace=True)
    dfMovies['_label_'].replace('documentary', 10,inplace=True)
    dfMovies['_label_'].replace('short', 10,inplace=True)
    dfMovies['_label_'].replace('reality-tv', 11,inplace=True)
    dfMovies['_label_'].replace('family', 11,inplace=True)
    dfMovies['_label_'].replace('animation', 11,inplace=True)
    dfMovies['_label_'].replace('sport', 11,inplace=True)
    dfMovies['_label_'].replace('talk-show', 11,inplace=True)
    dfMovies['_label_'].replace('game-show', 11,inplace=True)
    dfMovies['_label_'].replace('western', 12,inplace=True)

    dfMovies = dfMovies.groupby('_label_').head(500).reset_index(drop=True)
    ### Create Label mapping for later use
    labelMap = {1:'Action',2:'Adventure',3:'Comedy',4:'Crime',5:'Drama',6:'History',7:'Horror',8:'Music',9:'Sci-Fi',10:'Documentary',11:'Other',12:'Western'}
    ### Pre process text
    cleanedDf = textCleaning(dfMovies)
    docTf = doc_tf(cleanedDf)
    tf_idf = tf_x_idf(docTf.T)
    ### Split and prepare data for model
    full = pd.concat([tf_idf.T,dfMovies['_label_']],axis=1)
    train, test = train_test_split(full, test_size=0.2,stratify=full['_label_'])
    ### Run model
    ptv, classes = RocchioTrain(train)
    ### Metrics and Evaluation
    print(f'Accuracy over all test set is: {accuracy(test, ptv, classes):.2%}')
    test['pred'] = [RocchioClassification(ptv,classes,x)[0] for x in test.values]
    print(metrics.classification_report(test['_label_'],test['pred'],target_names=['Action','Adventure','Comedy','Crime','Drama','History','Horror','Music','Sci-Fi','Documentary','Other','Western']))

    cf_mat = metrics.confusion_matrix(test['_label_'],test['pred'])
    ls = ['Action','Adventure','Comedy','Crime','Drama','History','Horror','Music','Sci-Fi','Documentary','Other','Western']
    fig, ax = plt.subplots(figsize=(10,10))
    mat = sns.heatmap(cf_mat,annot=True,cmap=plt.cm.Blues,xticklabels=ls, yticklabels=ls)
    plt.show()

#################################################################################################################
###################################### USE CASE EXAMPLE #########################################################
#################################################################################################################
    usr = 1
    while usr != '999':
        usr = input('Enter 1 for Test Example and 0 to enter your own movie. 999 to exit:\n')
        if usr == '1':
            descrip = '''Tate Noble returns to the town of his youth where as a boy his parents were murdered. His childhood friend Samuel, now the sheriff of La Mesa knows who is responsible, and Tate's arrival sparks hostility between Samuel and his father Judge Carter. As the mystery unravels, Tate and Samuel enlist help from an unlikely source, the mob, in order to bring to justice the man ultimately responsible, the evil Harcourt Simms.'''
            title = 'Gunfight at La Mesa (2010)'
            print('Title:\n',title)
            print('\nDescription:\n',descrip)

            testTmp = pd.DataFrame({'id':99999999,'title':title,'genre':'western','descr':descrip}, index=[0])
            dfQuery = pd.concat([dfMovies,testTmp])
            a = textCleaning(dfQuery)
            b = doc_tf(a)
            c = tf_x_idf(b.T)

            testInstance = c.T.iloc[-1]

            pred = RocchioClassification(ptv,classes,testInstance)[0]
            rec = recommendations(c.T,testInstance)
            lookup = rec.sort_values('DocRank',ascending=False).index[1:6].values
            sim = rec.sort_values('DocRank',ascending=False).values[1:6]
            print(f'This movie seems like a {labelMap[pred]}')
            print(dfMovies[['title','genre','descr']].loc[lookup])
            print(f'Similarity scores: {sim}')

        elif usr == '0':
            title = input('Movie Title:')
            descrip = input('Please input a description of a movie:')
            
            testTmp = pd.DataFrame({'id':99999999,'title':title,'genre':'western','descr':descrip}, index=[0])
            dfQuery = pd.concat([dfMovies,testTmp])
            a = textCleaning(dfQuery)
            b = doc_tf(a)
            c = tf_x_idf(b.T)

            testInstance = c.T.iloc[-1]

            pred = RocchioClassification(ptv,classes,testInstance)[0]
            rec = recommendations(c.T,testInstance)
            lookup = rec.sort_values('DocRank',ascending=False).index[1:6].values
            sim = rec.sort_values('DocRank',ascending=False).values[1:6]
            print(f'This movie seems like a {labelMap[pred]}')
            print(dfMovies[['title','genre','descr']].loc[lookup])
            print(f'Similarity scores: {sim}')

if __name__ == "__main__":
    main()




