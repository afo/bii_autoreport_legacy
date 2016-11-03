from flask import Flask, request, render_template, send_from_directory
from werkzeug.contrib.fixers import ProxyFix
import numpy as np

# Local import
from invertid import invertid
from nocache import nocache
from aux_functions import *




app = Flask(__name__)


# Flask auxiliary functions

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def page_not_found(e):
    return render_template('error.html'), 500

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response



# Flask main functions

@app.route('/')
@nocache
def startpage():
	"""
	Startpage for report generation
	"""
	return render_template('start.html')

@app.route('/report', methods=['POST'])
@nocache
def startpage_post():
	"""
	Takes input from url.
	"""

	email = str(request.form['email'])
	part = str(request.form['part'])
	proj = str(request.form['proj'])
	comp = str(request.form['comp'])

	return report(email, part, proj, comp)

@app.route('/createReport')
@nocache
def report(email, part, proj, comp):

	"""
	This function collects the correct data given the input from url and creates a report.
  	'ident' can be:
  	'ind' - individual report without comparison
  	'gruop' - group report without comparison
  	'comp' - individual report with comparison
  	'compGroup' - group report with comparison

  	"""

	# Remove old image files
	remove_old()

	# Set variables to None if no value
	part, proj, comp = set_none(part, proj, comp)

	# Identify type of report requested and get data
	ident, part, proj, code, data = get_data(email, part, proj)
	
	# Error if empty data set
	if np.size(data[8]) == 0:
		page_not_found()

	[trust,res,div,bel,resall,collab,iz, comfort ,score] = data
	
	# Get comparison data
	ident, data_comp, comp = get_data_comp(ident, comp)	

	# Create graphs
	ident = create_graphs(ident, code, data, data_comp, comp, part, proj)

  	# Render templates
  	if len(code) == 2 and not isinstance(code, basestring):
  		code = "Partner Code: " + code[0] + "     Project Code: " + code[1]

  	if ident == 'ind':
  		return render_template('reportind.html', name = code, index_score=round(np.mean(score),2), tru = round(np.mean(trust),2), col = round(np.mean(collab),2), res_all = round(np.mean(resall),2), div = round(np.mean(div),2), men_st = round(np.mean(bel),2), cz = round(np.mean(comfort),2), in_zone = round(np.mean(iz),2), res = round(np.mean(res),2))
  	elif ident == 'group':
  		return render_template('reportgroup.html', name = code, index_score=round(np.mean(score),2), tru = round(np.mean(trust),2), col = round(np.mean(collab),2), res_all = round(np.mean(resall),2), div = round(np.mean(div),2), men_st = round(np.mean(bel),2), cz = round(np.mean(comfort),2), in_zone = round(np.mean(iz),2), res = round(np.mean(res),2))
  	elif ident == 'comp':
  		return render_template('reportcompind.html', comp = comp, name = code, index_score=round(np.mean(score),2), tru = round(np.mean(trust),2), col = round(np.mean(collab),2), res_all = round(np.mean(resall),2), div = round(np.mean(div),2), men_st = round(np.mean(bel),2), cz = round(np.mean(comfort),2), in_zone = round(np.mean(iz),2), res = round(np.mean(res),2))
  	elif ident == 'compGroup':
  		return render_template('reportcomp.html', comp = comp, name = code, index_score=round(np.mean(score),2), tru = round(np.mean(trust),2), col = round(np.mean(collab),2), res_all = round(np.mean(resall),2), div = round(np.mean(div),2), men_st = round(np.mean(bel),2), cz = round(np.mean(comfort),2), in_zone = round(np.mean(iz),2), res = round(np.mean(res),2))





@app.route('/<user_id>')
@nocache
def main(user_id):
	"""
	Given an url with the ending 'user_id' this function generates an individual report.
	"""

	remove_old()



	# Find the user email from the user_id
	code = invertid(user_id)

	# Get data for individual

	data = get_data_ind(code)

	if np.size(data[8]) == 0:
		page_not_found()

	[trust,res,div,bel,resall,collab,comfort,iz,score] = data

	# Create graphs
	bii_radar(code, data, None, None, None, None)
  	bii_hbar(False, code, data)

  	# Render template
  	return render_template('reportind.html', name = code, index_score=round(np.mean(score),2), tru = round(np.mean(trust),2), col = round(np.mean(collab),2), res_all = round(np.mean(resall),2), div = round(np.mean(div),2), men_st = round(np.mean(bel),2), in_zone = round(np.mean(iz),2), cz = round(np.mean(comfort),2), res = round(np.mean(res),2))


app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
	app.run(debug=True)
	#app.run(host = '0.0.0.0', port =80,debug=True)
