import urllib
import json
import gspread
from oauth2client.client import SignedJwtAssertionCredentials
import hashlib
import pandas as pd
from flask import Flask
app = Flask(__name__)

@app.route('/fetchdata')
def data_fetch():

	# Credential logins for the Google Drive
	json_key = json.load(open('Key-file.json'))
	scope = ['https://spreadsheets.google.com/feeds']

	credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)

	gc = gspread.authorize(credentials)

	sh = gc.open_by_key("1ePsWRoYTn5zoONbjvpadDCb_CP21umdEXgaJHpU-a7k")

	worksheet = sh.get_worksheet(0)



	'''Uncomment the sections below to copy all the data in the Spreadsheet'''

	#a = worksheet.get_all_records()
	#with open('biidata.json', 'a') as outfile:
	#	json.dump(a, outfile, ensure_ascii=False,sort_keys=True, indent=4, separators=(',', ': '))




	# Append latest entry, run everytime
	header_json = worksheet.row_values(1)

	# Convert to integers
	k=worksheet.row_count
	data_json = worksheet.row_values(k) # Be sure to sort the data correctly

	k=0
	for item in data_json: 
	    try:
	        data_json[k] = int(item)
	    except Exception:
	        pass
	    k+=1

	data = dict(zip(header_json, data_json))

	with open('biidata.json', mode='r') as feedsjson:
	    feeds = json.load(feedsjson)




	with open('biidata.json', mode='w') as outfile:
	    feeds.append(data)
	    json.dump(feeds,outfile, ensure_ascii=False,sort_keys=True, indent=4, separators=(',', ': '))
        
        return 'Done!'


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)