import csv
import sys
import pandas as pd
import numpy as np

from geopy.distance import distance
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine


class distanceFinder:

    def __init__(self,usr,pass_, infile, outfile):
        '''Pass in credentials to SF and name input file and output file'''
        self.infile = infile
        self.outfile = outfile
        self.engine = create_engine(
            URL(
            account = 'revantage.us-east-1',
            user = usr,
            password = pass_,
            database = 'LIV_SANDBOX',
            schema = 'MAPPING',
            warehouse = 'LIV_ANALYST_WH',
            role='LIV_ANALYST',
            authenticator='externalbrowser'#https://myportal.okta.com can use this if we want to not open browser and in office #externalbrowser
           )
        )
        self.getPoints()

    def assetList(self):
        query = '''SELECT BU, PROPERTY_NAME, STATE, "Longitude", "Latitude"  
        FROM BX_CONTROL W
        WHERE ACTIVE = 1 
        AND ACQUISITION_DATE IS NOT NULL 
        AND PROPERTY_NAME NOT LIKE '%Retail%'
        AND STATE <> 'NY'
        AND COMBO IS NULL 
        '''
        self.dfLC = pd.read_sql_query(query,self.engine)

    def distanceFinder(self,x1,y1,x2,y2):
        
        return distance((x1,y1),(x2,y2)).miles

    def getPoints(self):
        self.assetList()
        self.dfNew = pd.read_excel(self.infile,skiprows=1)

   
        ### External Locations
        nameNew = np.asarray(self.dfNew['Property Name'])
        longNew = np.asarray(self.dfNew['Longitude'])
        latNew = np.asarray(self.dfNew['Latitude'])
        stateNew = np.asarray(self.dfNew['Address'].str[-8:-6])
        unitsNew = np.asarray(self.dfNew['Units'])
        newStart = np.asarray(self.dfNew['Start'].astype(str))
        newFinish = np.asarray(self.dfNew['Finish'].astype(str))
        matrixNew = np.vstack([nameNew,longNew,latNew,stateNew,unitsNew,newStart,newFinish]).T

        ### Livcor Matrix
        buLC = np.asarray(self.dfLC['bu'])
        nameLC = np.asarray(self.dfLC['property_name'])
        longLC = np.asarray(self.dfLC['Longitude'])
        latLC = np.asarray(self.dfLC['Latitude'])
        stateLC = np.asarray(self.dfLC['state'])
        matrixLC = np.vstack([buLC,nameLC, longLC, latLC,stateLC]).T

        
        with open(self.outfile, mode='w', newline='',encoding="utf-8") as outFile:
            writer = csv.writer(outFile, delimiter=',')
            writer.writerow(['BU','Livcor Property', 'External Name','External Units','Distance (miles)','Start Date','finish Date', 'Latitude','Longitude','Type'])
            for i in matrixNew:
                for j in matrixLC:
                    if i[3] == j[4]:
                        try:
                            dis = self.distanceFinder(i[2],i[1],j[3],j[2])
                            writer.writerow([ j[0],j[1],i[0],i[4],dis,i[5],i[6],i[2],i[1],'New'])
                        except:
                            print(f'LC or New lat, long combo not valid \nLC:{j[3],j[2]}\nNew: {i[2],i[1]}')
        
            for line in matrixLC:
                writer.writerow([line[0],line[1],'','','','','',line[3],line[2],'LC'])



distanceFinder(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
