import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as rqs

all_notices = []

for id in range(0,16):
	url = 'http://www.vssut.ac.in/notice-list.php?page='+str(id)
	vssut_url = rqs.urlopen(url).read()
	vssut_soup = BeautifulSoup(vssut_url,'lxml')
	vssut_soup.prettify()
	vssut_element = vssut_soup.find('table',class_='table')
	notices = []
	for link in vssut_element.find_all('td'):
		notices.extend(link.findAll(text=True))
	cleaned_notices = [notice.rstrip() for notice in notices if len(notice)>10]
	all_notices += cleaned_notices
notices_df = pd.DataFrame(all_notices,columns=['Notice_Name'])
notices_df.to_csv('all_notices.csv',index=False)