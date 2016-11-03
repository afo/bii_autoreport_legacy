import urllib
import json
import gspread
import os
from oauth2client.client import SignedJwtAssertionCredentials
import hashlib
import pandas as pd
import numpy as np
from flask import Flask
app = Flask(__name__)
from werkzeug.contrib.fixers import ProxyFix
from bii_data import *

def get_data_ind(code):
	json_key = json.load(open(''))
	scope = ['https://spreadsheets.google.com/feeds']

	credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)
	gc = gspread.authorize(credentials)
	sh = gc.open_by_key("")


	worksheet = sh.get_worksheet(0)
	worksheet2 = sh.get_worksheet(1)
	header = worksheet2.row_values(1)




	# get column index for email
	email_ind = header.index("MAIL") + 1
	# get a list of all emails
	li = worksheet2.col_values(email_ind)

	## get all data for current mail
	ind = (len(li) - 1) - li[::-1].index(code) + 1
	
	values_all = worksheet2.row_values(ind) #extract values from Worksheet2

	# get indices in spreadsheet for each of the categories
	names = ["TRUST", "RESILIENCE", "DIVERSITY", "BELIEF", "PERFECTION", "COLLABORATION", "INNOVATION ZONE", "COMFORT ZONE", "SCORE"]
	values_ind = [header.index(i) for i in names]

	# get only data for the categories
	values=[values_all[i] for i in values_ind]
	values = map(float, values)

	return values



def get_data_group(code, group_type):
	json_key = json.load(open(''))
	scope = ['https://spreadsheets.google.com/feeds']

	credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)
	gc = gspread.authorize(credentials)
	sh = gc.open_by_key("")
	worksheet = sh.get_worksheet(1)
	#header_json = worksheet.row_values(1)

	all_data = worksheet.get_all_records()
	nbr_rows_new = len(all_data)

	update_json = False
	with open("nbr_rows.txt", "r") as f_rows:
		nbr_rows_old = int(f_rows.read())
		if nbr_rows_old != nbr_rows_new:
			update_json = True

	if update_json:
		os.remove("nbr_rows.txt")
		f_new = open("nbr_rows.txt", "w")
		f_new.write(str(nbr_rows_new))

	if update_json:
		with open('biidata.json', 'w') as outfile:
			json.dump(all_data, outfile, ensure_ascii=False,sort_keys=True, indent=4, separators=(',', ': '))

	df0 = bii_data(group_type, code)

	names = ["TRUST", "RESILIENCE", "DIVERSITY", "BELIEF", "PERFECTION", "COLLABORATION", "INNOVATION ZONE", "COMFORT ZONE", "SCORE"]
	values = [df0[i] for i in names]

	return values





