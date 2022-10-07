import nltk
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sqlalchemy import types 
from collections import Counter
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sqlalchemy.dialects import registry
from sklearn.preprocessing import LabelEncoder
from snowflake.connector.pandas_tools import pd_writer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('wordnet')
nltk.download('omw-1.4')
warnings.filterwarnings('ignore')
registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')




class EmploymentClassifier:
  
    def __init__(self,Q):
        self.engine = create_engine(
            URL(
                account = 'revantage.us-east-1',
                user = dbutils.secrets.get(scope = "insightsTeam_pass",key='SnowUser'),
                password = dbutils.secrets.get(scope = "insightsTeam_pass",key='SnowPass'),
                database = 'LIV_SANDBOX',
                schema = 'LC',
                warehouse = 'LIV_ANALYST_WH',
                role='LIV_ANALYST',
            )
        )
        self.query = Q
        self.getData() 
        self.split_prep_run()
        self.predict()
        self.clean_Map()
        self.snowflake()
        
    
    def getData(self):
        '''Pulls training data from local DBFS & from Snowflake to get the live data for predictions'''
        train = pd.read_csv('/dbfs/FileStore/tables/TrainingData/Employment_train_V2.csv')
        train.iloc[:,1:] = train.iloc[:,1:].astype('category')
        live = pd.read_sql(self.query, con = self.engine)
        train.columns = train.columns.str.strip().str.upper()
        live.columns = live.columns.str.strip().str.upper()
        
        encoder = LabelEncoder()
        train['Target'] = encoder.fit_transform(train['LABEL'])
        self.map_ = dict(zip(encoder.classes_,encoder.transform(encoder.classes_)))
        
        self.dfTrain = self.cleanText(train)
        self.df = self.cleanText(live)
        
    def cleanText(self,df):
        '''Clean data and basic corpus metrics'''
        df['text'] = df['POSITION'].str.replace('[^a-zA-Z0-9]',' ', regex=True)
        df['text'] = df['text'].apply(lambda x: str(x).lower().split()) 

        lemmatizer = nltk.stem.WordNetLemmatizer()
        df['text'] = df['text'].apply(lambda y:[lemmatizer.lemmatize(word) for word in y])

        all = np.asarray(df['text'])
        totalWords = [word for sent in all for word in sent]
        uniqWords = Counter(totalWords)
        print(f'\nTotal # of words: {len(totalWords):,}')
        print(f'# of unique words: {len(uniqWords):,}')

        df['text'] = df['text'].apply(lambda z: ' '.join(z))

        return df

    def split_prep_run(self):

        y = self.dfTrain['Target']
        x = self.dfTrain['text']

        xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.2,random_state=23,stratify=y)
        xtrain = np.asarray(xtrain)
        xtest = np.asarray(xtest)

        text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=None, split='whitespace', output_mode='count')
        text_vectorizer.adapt(xtrain, batch_size=512)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), dtype='string'),
            text_vectorizer,
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.map_), activation='softmax'),
        ])

        print(self.model.summary())
        self.model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(xtrain, ytrain, batch_size=256, epochs=20, validation_data=(xtest, ytest))

        train_preds = self.model.predict(xtrain)
        test_preds = self.model.predict(xtest)

        print('Train Accuracy : {}'.format(accuracy_score(ytrain, np.argmax(train_preds, axis=1))))
        print('Test  Accuracy : {}'.format(accuracy_score(ytest, np.argmax(test_preds, axis=1))))
        print('\nClassification Report : ')
        print(classification_report(ytest, np.argmax(test_preds, axis=1), target_names=self.map_))

    def predict(self):

        predictions = self.model.predict(self.df['text'])
        self.df['Prediction'] = np.argmax(predictions,axis=1)
        inv_map = {val: key for key, val in self.map_.items()}
        self.df['Prediction'] = self.df['Prediction'].map(inv_map)
        
    
    def clean_Map(self):
        
        industryMap = pd.read_csv('/dbfs/FileStore/tables/TrainingData/industryMap_V2.csv')
        industryMap['EMPLOYERINDUSTRY'] = industryMap['EMPLOYERINDUSTRY'].apply(lambda x: x.lower())
        indMap = dict(zip(industryMap.EMPLOYERINDUSTRY,industryMap.LABEL))
        self.df['EMPLOYERINDUSTRY'] = self.df['EMPLOYERINDUSTRY'].apply(lambda x: x.lower() if x else x)
        self.df['MAPPED_INDUSTRY'] = self.df['EMPLOYERINDUSTRY'].map(indMap)
        self.df['INDUSTRY'] = self.df['MAPPED_INDUSTRY'].combine_first(self.df['Prediction'])
        self.df['EARLIESTMOVEIN'] =  pd.to_datetime(self.df['EARLIESTMOVEIN'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        self.df['LEASESTARTDATE']=  pd.to_datetime(self.df['LEASESTARTDATE'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        self.df['LEASEENDDATE']=  pd.to_datetime(self.df['LEASEENDDATE'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        self.df['MOVEOUTDATE']=  pd.to_datetime(self.df['MOVEOUTDATE'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
        
        self.df['EARLIESTMOVEIN']=  self.df['EARLIESTMOVEIN'].dt.date
        self.df['LEASESTARTDATE']=  self.df['LEASESTARTDATE'].dt.date
        self.df['LEASEENDDATE']=  self.df['LEASEENDDATE'].dt.date
        self.df['MOVEOUTDATE']=  self.df['MOVEOUTDATE'].dt.date
        
    def snowflake(self):
        self.df.drop(['text','Prediction','MAPPED_INDUSTRY'],axis=1,inplace=True)
        self.df.to_sql(name='EMPLOYMENT'.lower(),
                       if_exists='replace',
                       con=self.engine,
                       index = False,
                       method=pd_writer,
                       dtype={'EARLIESTMOVEIN':types.Date,
                              'LEASESTARTDATE':types.Date,
                              'LEASEENDDATE':types.Date,
                              'MOVEOUTDATE':types.Date})

ec = EmploymentClassifier('Select * from LIV_SANDBOX.LC.ResidentMigration_VW')
