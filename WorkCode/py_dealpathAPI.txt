import re
import requests
import numpy as np
import pandas as pd
from datetime import date
import sqlalchemy as sql
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from snowflake.connector.pandas_tools import pd_writer
from sqlalchemy.dialects import registry
registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')
import warnings
warnings.filterwarnings('ignore')

usr = dbutils.secrets.get('insightsTeam_pass', 'SnowUser')
pass_ = dbutils.secrets.get('insightsTeam_pass', 'SnowPass')
auth = dbutils.secrets.get('insightsTeam_pass', 'stecDealPathKey')

engine = create_engine(
                      URL(
  account = 'revantage.us-east-1',
  user = usr,
  password = pass_,
  database = 'LIV_SANDBOX',
  schema = 'LC',
  warehouse = 'LIV_ANALYST_WH',
  role='LIV_ANALYST',
  )
)

headers = { "Accept" : "application/vnd.dealpath.api.v1+json", "Authorization" : "Bearer %s" % auth}
variableIDs =[753200,695583,753548,753544,728555,335134,374612,695561,233640,345984,695542,603119,233598,695537,233667,901693,901687,901681,901672,901657,659184,659181,659160,659154,659148,659136,233579,233576,233570,233567,340272,822458,297235,353360, 822460,233602,298493,294703,949831,387162,967970]


 class dealpath():


  def __init__(self,url = 'https://api.dealpath.com/',varid = variableIDs,h=headers):
    self.headers = h
    self.varid = varid
    self.url = url
#     self.varlist = self.buildDataFrame('fields',url+'fields')
#     self.deal = self.buildDataFrame('deals',url+'deals')
#     self.properties = self.buildDataFrame('properties',url+'properties')
#     self.cleanDataFrame()
#     self.pushSnowflake()
    
  def getAllFieldList(self):
    ext_ = 'field_definitions'
    url = self.url + ext_
    data_full,nxt= self.getData(ext_,url)
    while nxt:
      data, nxt = self.getData(ext_,url+'?next_token='+nxt)
      data_full.extend(data)
    
    return  pd.json_normalize(data_full)
  
  # Cleanup Column Names
  def renameColumns(self,word):
      cap = word.upper()
      word = re.sub(r'[^A-Z0-9]','_',cap)

      if word[-1] == '_':
        tmp = word[:-1]
        tmp1 = tmp[:4]
        new = tmp[5:]+ '_'+tmp1
        return new
      else:
        return word


  def getData(self,ext_,url):
    resp = requests.get(url,headers = self.headers)
    return resp.json()[ext_]['data'], resp.json()[ext_]['next_token']
  
  def buildDataFrame(self, ext_, url):
    if ext_ != 'fields':
      self.data,nxt= self.getData(ext_,url)
      while nxt:
        data, nxt = self.getData(ext_,url+'?next_token='+nxt)
        self.data.extend(data)
    
    else:
      self.data = []
      for var in self.varid:
        data,nxt = self.getData(ext_,url+'/field_definition/'+str(var)+'?sparse=true')
        self.data.extend(data)
        while nxt:
          data,nxt = self.getData(ext_,url+'/field_definition/'+str(var)+'?sparse=true'+'&next_token='+nxt)
          self.data.extend(data)
          
    return pd.json_normalize(self.data)
    
  def cleanDataFrame(self):
    dfPivot = self.varlist
    dfPivot['ID'] = dfPivot['deal_id'].combine_first(dfPivot['property_id'])
    dfPivot.drop(['deal_id','property_id','field_definition_id','edit_value'],axis=1,inplace=True)
    dfPivot = dfPivot.pivot_table(
      values = 'value',
      index = 'ID',
      columns = 'name',
      aggfunc='first'
    )

    dfPivot.reset_index(inplace=True)
    dfPivot.replace(0,np.nan,inplace=True)
    dfPivot['ID'] =dfPivot['ID'].astype(str)
    dfPivot['ID'] = dfPivot['ID'].apply(lambda x: x[:-2])
    dfPivot['Basis/SF'] =dfPivot['Basis/SF'].astype('float64')
    dfPivot['B/SF RC%'] =dfPivot['B/SF RC%'].astype('float64')
    dfPivot['Basis/Unit'] =dfPivot['Basis/Unit'].astype('float64')
    dfPivot['B/Unit RC%'] =dfPivot['B/Unit RC%'].astype('float64')
    dfPivot['MTM Cap'] =dfPivot['MTM Cap'].astype('float64')
    dfPivot['MTM Rent'] =dfPivot['MTM Rent'].astype('float64')
    dfPivot['ROI Cap'] =dfPivot['ROI Cap'].astype('float64')
    dfPivot['Y1 Cap (Formula)'] =dfPivot['Y1 Cap (Formula)'].astype('float64')
    dfPivot['Avg SF'] = dfPivot['Avg SF'].astype('float64')
    dfPivot['IP Cap'] = dfPivot['IP Cap'].astype('float64')

    dfPivot.rename({'Basis/SF': 'BASIS_SF',
                    'B/SF RC%':'Basis_SF_RC',
                    'B/Unit RC%':'BASIS_UNIT_RC',
                    'Basis/Unit':'BASIS_UNIT',
                    'Constr. Type':'CONSTRUCTION_TYPE', 
                    'Avg SF':'AVG_UNIT_SF',
                    'IP Cap':'IP_CAP_RATE',
                    'Y1 Cap (Formula)':'Y1_CAP'},axis=1,inplace=True)

    self.deal['id'] = self.deal['id'].astype(str)
    self.properties['id'] = self.properties['id'].astype(str)

    dealFull = self.deal.merge(dfPivot, left_on='id',right_on='ID')
    propertiesFull = self.properties.merge(dfPivot, left_on='id',right_on='ID')

    dealFull.drop(['ID','address.address_2','last_updated'],axis=1,inplace=True)
    propertiesFull.drop(['ID','address.address_2'],axis=1,inplace=True)
    dealFull.replace(0,np.nan,inplace=True)
    propertiesFull.replace(0,np.nan,inplace=True)

    dealFull.rename({'address.address_1':'ADDRESS',
                     'address.city':'CITY',
                     'address.state':'STATE',
                     'address.country':'COUNTRY',
                     'address.postal_code':'ZIPCODE',
                     'address.lat':'LATITUDE',
                     'address.lng':'LONGITUDE'},axis=1,inplace=True)

    propertiesFull.rename({'address.address_1':'ADDRESS',
                           'address.city':'CITY',
                           'address.state':'STATE',
                           'address.country':'COUNTRY',
                           'address.postal_code':'ZIPCODE',
                           'address.lat':'LATITUDE',
                           'address.lng':'LONGITUDE'},axis=1,inplace=True)

    dealFull.rename(columns= lambda x: self.renameColumns(x),inplace=True)
    propertiesFull.rename(columns= lambda x: self.renameColumns(x),inplace=True)

    dealFull.rename(columns= lambda x: re.sub(r'__','_',x),inplace=True)
    propertiesFull.rename(columns= lambda x: re.sub(r'__','_',x),inplace=True)


    self.DEALS = dealFull.explode('PROPERTY_IDS')
    self.PROPS = propertiesFull.explode('DEAL_IDS')

  def pushSnowflake(self):
    self.DEALS['ETLDATE'] = date.today()
    self.PROPS['ETLDATE'] = date.today()

    self.DEALS.to_sql(
      'dealpath_deals',
      con=engine,
      index=False,
      method=pd_writer,
      if_exists='append',
      dtype={'MODEL_DATE':sql.types.Date,
             'CFO_DATE':sql.types.Date,
             'CLOSING_DATE':sql.types.Date,
             'ACTUAL_CLOSING_DATE':sql.types.Date,
             'CLOSED_TO_ANOTHER_BUYER_DATE':sql.types.Date,
             'AWARDED_TO_ANOTHER_BUYER_DATE':sql.types.Date,
             'DEAL_AWARDED':sql.types.Date,
             'CREATION_DATE':sql.types.Date})

    self.PROPS.to_sql(
      'dealpath_properties',
      con=engine,
      index=False,
      method=pd_writer,
      if_exists='append',
      dtype={'MODEL_DATE':sql.types.Date,
             'CFO_DATE':sql.types.Date,
             'CLOSING_DATE':sql.types.Date,
             'ACTUAL_CLOSING_DATE':sql.types.Date,
             'CLOSED_TO_ANOTHER_BUYER_DATE':sql.types.Date,
             'AWARDED_TO_ANOTHER_BUYER_DATE':sql.types.Date,
             'DEAL_AWARDED':sql.types.Date,
             'CREATION_DATE':sql.types.Date})
        

tmp = dealpath()

