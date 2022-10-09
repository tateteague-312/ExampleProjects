from urllib.parse import parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
# from notify_run import Notify
from fake_useragent import UserAgent

import pandas as pd
import numpy as np
import time
import sys   
import os
import csv
import warnings

#NEW Line addition

#Measure Success
success = 0
count = 0
### Get push notifications sent to you 
# notify = Notify()
## Pass Captcha
fakeAgent = UserAgent()
### Supress Warnings
warnings.filterwarnings("ignore")
### Selenium Parameters
options = Options()
options.add_argument('--lang=en')
options.add_argument('--disable-geolocation')
options.add_argument('--disable-search-geolocation-disclosure')
options.add_experimental_option('excludeSwitches', ['enable-logging']) #hides dev debug text
options.add_argument(f'user-agent={fakeAgent.random}')
driver = webdriver.Chrome(executable_path="S:/16. STRATEGY & PLANNING/7. Misc/Tate Remote Desktop/SnowFlake/chromedriver.exe", options=options)



df = pd.read_csv(sys.argv[1])
prop = np.asarray(df['PROPERTY_NAME'])
add_ = np.asarray(df['PROPERTY_ADDRESS'])
zip_ = np.asarray(df['PROPERTY_ZIPCODE'])
state = np.asarray(df['PROPERTY_STATE'])
unit = np.asarray(df['PROPERTY_UNITS'])
propertyList = np.vstack((prop, add_,zip_,state,unit)).T

fileName = str(sys.argv[2])
with open(fileName, mode='w', newline='',encoding="utf-8") as outFile:
    writer = csv.writer(outFile, delimiter=',')
    writer.writerow(['Property Name','Unit Count','Address','Zip','State','Average Review','Number of Reviews','Stars','Date','Review'])
    for line in propertyList:


        base_url = 'https://www.google.com/maps/place/'
        url = base_url + str(line[0]) +' '+ str(line[1])
        driver.get(url) 
        count+=1
        time.sleep(1)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent":str(fakeAgent.random)})

        if 'sorry' not in str(driver.current_url): 
            try:
                ### Leverage Google Search to find result based off Property name and Address forced through URL
                WebDriverWait(driver,2).until(EC.presence_of_element_located((By.XPATH,"//button[@aria-label='Search']")))
                driver.find_element_by_xpath("//button[@aria-label='Search']").click()
                #time.sleep(2)
        
                ### Note the total number of reviews 
                number_of_reviews = WebDriverWait(driver,2).until(EC.presence_of_element_located((By.XPATH, '//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span/span[2]/span[1]/button'))).text
                number_of_reviews = int(number_of_reviews.split(' ')[0])
                ### Average Review of Property
                avgReview = WebDriverWait(driver,2).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/div[2]/span/span/span'))).text 
                max_count = 1 + (number_of_reviews//7) 
                try:
                    WebDriverWait(driver,2).until(EC.presence_of_element_located((By.XPATH, '//*[@id="pane"]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span[1]/span/span/span[2]/span[1]/button'))).click()
                    success+=1
                except:
                    print("Failed at Number of Reviews Click")
                x = 0
                
                ### Scroll to bottom of reviews                                                                    
                try:
                    while(x<max_count): 
                        scroll_div = WebDriverWait(driver,2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class='m6QErb DxyBCb cYB2Ge-oHo7ed cYB2Ge-ti6hGc']"))) # It gets the section of the scroll bar.
                        try:
                            driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scroll_div) # Scroll it to the bottom.
                        except:
                            print("Scroll fail")
                            pass
                        x+=1
                        time.sleep(2)
                    ### Expand reviews to show the full contents of comment 
                    
                        click_ = WebDriverWait(driver,2).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "[class='w8nwRe gXqMYb-hSRGPd']")))
                        for i in click_:
                            i.click()
                except:
                    pass
            
              
                ### Loop through Reviews
                try:    #//*[@id="pane"]/div/div[1]/div/div/div[2]/div[9]

                    for item in WebDriverWait(driver,3).until(EC.visibility_of_all_elements_located((By.CSS_SELECTOR, "[class='jftiEf NIyLF-haAclf fontBodyMedium']"))):
                        ### What elements of Reviews do we need to grab (Currently just stars, date, review)
                        stars = WebDriverWait(item,1).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class='kvMYJc']"))).get_attribute("aria-label")#[1]

                        date = WebDriverWait(item,1).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class='rsqaWe']"))).text        

                        try:
                            review = WebDriverWait(item,2).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[class='wiI7pd']"))).text

                        except:
                            print("No review comment")
                            review = 'None'
                        # print(x,stars, date , review)
                        writer.writerow([line[0],line[1],line[2],line[3],line[4], avgReview, number_of_reviews, stars, date, review]) 
                except:
                    print("Couldn't get individual Review details")
              
                ### WRite out rows in Real time that way if any problems can restart from last scraped Property
                
                success+=1
            except:
                print('Could not find Apartment')
                avgReview = "None"
                number_of_reviews = "None"
                stars = "None"
                date = "None"
                review = "None"
                writer.writerow([line[0],line[1],line[2],line[3],line[4], avgReview, number_of_reviews, stars, date, review])
        else:
            print('YOU\'VE BEEN POWNED BY GOOGLE')
            break  

size = os.path.getsize(fileName)
### Descriptive Stats Output
rate = success/count
print('################# STATS ################')
print('Number of Succesful Properties Scraped: {}'.format(success))
print('Total Number of Properites Searched: {}'.format(count))
print('Success Rate: {:.2%}'.format((rate)))
print('File Size: {} (Bytes)'.format(size))
print('########################################')
#driver.close()
# notify.send('Successes: {} Total: {} Success Rate: {:.2%} File Size: {} (Bytes)'.format(success,count,rate,size))
