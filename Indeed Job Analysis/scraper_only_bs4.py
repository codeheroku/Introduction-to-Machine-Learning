from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup

## Download the chromedriver from link in description
## And give the location of executable here
driver = webdriver.Chrome("./chromedriver")

dataframe = pd.DataFrame(columns=["Title","Location","Company","Salary","Sponsored","Description"])


for i in range(0,500,10):

	##Step1: Get the page
	driver.get("https://www.indeed.co.in/jobs?q=data+science&l=India&start="+str(i))
	driver.implicitly_wait(4)

	all_jobs = driver.find_elements_by_class_name('result')

	for job in all_jobs:

		result_html = job.get_attribute('innerHTML')
		soup = BeautifulSoup(result_html,'html.parser')

		try:
			title = soup.find("a",class_="jobtitle").text.replace('\n','')
		except:
			title = 'None'

		try:
			location = soup.find(class_="location").text
		except:
			location = 'None'

		try:
			company = soup.find(class_="company").text.replace("\n","").strip()
		except:
			company = 'None'

		try:
			salary = soup.find(class_="salary").text.replace("\n","").strip()
		except:
			salary = 'None'

		try:
			sponsored = soup.find(class_="sponsoredGray").text
			sponsored = "Sponsored"
		except:
			sponsored = "Organic"

		
		sum_div = job.find_elements_by_class_name("summary")[0]
		try:
			sum_div.click()
		except:
			close_button = driver.find_elements_by_class_name("popover-x-button-close")[0]
			close_button.click()
			sum_div.click()
			

		job_desc = driver.find_element_by_id('vjs-desc').text


		dataframe = dataframe.append({'Title':title,'Location':location,"Company":company,
						"Sponsored":sponsored,"Description":job_desc},
						ignore_index=True)


dataframe.to_csv("datascience.csv",index=False)




















