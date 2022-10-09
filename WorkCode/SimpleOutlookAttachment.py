import re
import sys
import requests
import win32com.client
from datetime import date

############### Change Object initiation to use outlook live instead of outlook application ###########################
############# For reference: https://docs.microsoft.com/en-us/exchange/exchange-online #####################################

outlook = win32com.client.Dispatch('Outlook.Application').GetNamespace('MAPI')
#################################################################################################################


#open outlook inbox object at index 6 
inbox = outlook.GetDefaultFolder(6).folders('OccupancyReports') # <------------------ This OccupancyReports is just the name of the subfolder I created in my inbox 
#Create inbox object                                                                  with basic email rules you can do the same and name as you like
messages = inbox.Items
#grab most recent email
mostRecentEmail = messages.GetFirst()

#strange email format couldn't figure out how to decode so just wrote it to a file, probably better method
with open('hyperlinkaddress.txt', 'w', encoding='utf-8') as f:
    f.write(mostRecentEmail.Body)
f.close()

with open('hyperlinkaddress.txt','r') as infile:
    lines = infile.readlines()
    link = lines[18]
infile.close()

#clean URL
link = str(link).strip()
cleanedLink = re.sub('[<>]','',link)

#Add today's date to file name
today = date.today()
d = today.strftime("%m_%d_%y")
#send request to auto download link
report = requests.get(cleanedLink,allow_redirects=True)
open(f'OccupancyReport_{d}.csv','wb').write(report.content)

