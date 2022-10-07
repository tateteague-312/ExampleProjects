### Packages & Config settings
import pandas as pd
import pyspark
pd.options.mode.chained_assignment = None
import datetime
from itertools import product
from pyspark.sql import Window
from pyspark.sql import functions as f 

userid = dbutils.secrets.get(scope = "insightsTeam_pass",key='SnowUser')
password = dbutils.secrets.get(scope = "insightsTeam_pass",key='SnowPass')

options = {
  "sfUrl": "revantage.us-east-1.snowflakecomputing.com/",
  "sfUser": userid,
  "sfPassword": password,
  "sfDatabase": "LIV_SANDBOX",
  "sfSchema": "LC",
  "sfWarehouse": "LIV_ANALYST_WH"
}

### QUERY FOR SNOWFLAKE
query = '''
WITH ff as(
	SELECT DISTINCT 
		 concat( pm.LIVCORBUPROPERTYIDENTIFIER, u.unitname) as unitid
		,MARKETRENTRATE 
		,to_date(u.ETLCREATEDDATE) AS date
		,u.unitid AS actual 
		,concat(concat( pm.LIVCORBUPROPERTYIDENTIFIER, u.unitname),to_date(u.ETLCREATEDDATE))AS id
 		FROM DNA_SDW_PD.LIV_ODS.PROPERTYMAPPING  pm
 		LEFT JOIN (SELECT * FROM DNA_SDW_PD.LIV_ODS.UNIT WHERE MARKETRENTRATE > 0) u ON pm.PROPERTYASSETID = u.PROPERTYASSETID 
	WHERE 
		(ISEXCLUSIONFLAG = 0 OR ISEXCLUSIONFLAG IS NULL) 
		AND u.UNITNAME NOT IN (
			'WAIT%'
			,'%OFFICE%'
			,'%model%'
			,'%COMMON%'
			,'%GARAGE%'
		)
		
	ORDER BY concat( pm.LIVCORBUPROPERTYIDENTIFIER, u.unitname),date
)

	SELECT 
		unitid
		,actual
		,marketrentrate
		,date

	FROM (
		SELECT *
			,ROW_NUMBER() OVER (PARTITION BY id ORDER BY marketrentrate DESC) AS dup
		FROM ff
	)
	WHERE dup = 1 
	ORDER BY unitid, date 
'''

df = spark.read.format("snowflake").options(**options).option("query",query).load() \
  .withColumn("UNITID",f.col("UNITID").cast(pyspark.sql.types.StringType())) \
  .withColumn("ACTUAL",f.col("ACTUAL").cast(pyspark.sql.types.StringType())) \
  .withColumn("MARKETRENTRATE",f.col("MARKETRENTRATE").cast("int")) \
  .withColumn("DATE",f.col("DATE").cast(pyspark.sql.types.DateType()))


### Create Calendar For everyday between lowest and max date 

### Grab Min & Max days to create date range 
mindate = df.agg({'DATE':'min'}).collect()[0][0]
maxdate = df.agg({'DATE':'max'}).collect()[0][0]
### Create all dates inbetween
dayRange = [(mindate + datetime.timedelta(days=d)) for d in range((maxdate - mindate).days + 1)]
### Create DF with open colum to cross join units
dates = spark.createDataFrame(((d,) for d in dayRange), ('cal',))
### Get individual untis
units = df.select("UNITID").distinct()

### Cross Join & Forward Fill

### Crossjoin 
datesUnits = dates.crossJoin(units)
datesUnits = datesUnits.withColumnRenamed('UNITID','ID')
### Join the calendar table with the original 
cond = [datesUnits.cal == f.to_date(df.DATE), datesUnits.ID == df.UNITID]
df2 = (datesUnits.join(df, cond,"left"))
### Create window to move over rents
wind = (Window.partitionBy('ID').rangeBetween(Window.unboundedPreceding, -1).orderBy(f.unix_timestamp('cal')))
df3 = df2.withColumn('lastRent', f.last('MARKETRENTRATE', ignorenulls=True).over(wind))\
         .withColumn('lastUNITID', f.last('ACTUAL', ignorenulls=True).over(wind))
### Coalesce to get new values if there is no original value
df4 = df3.select( df3.ID, \
                 f.coalesce(df3.ACTUAL,df3.lastUNITID).alias('UNITID'), \
                 f.coalesce(df3.DATE, df3.cal).alias('DATE'), \
                 f.coalesce(df3.MARKETRENTRATE, df3.lastRent).alias('RENT'))

## Push To Snowflake 

df4 = df4.dropna()
df4 = df4.withColumn('ETLDATE',f.lit(datetime.datetime.now()))
df4 = df4.withColumnRenamed('ACTUAL','UNITID')
df4.write.format("snowflake").options(**options).option("dbtable","ASKINGRENT").mode('overwrite').save()


























