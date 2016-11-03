
import os
import shutil

# Local import
from get_data import *
from bii_plot.bii_radar import bii_radar
from bii_plot.bii_hist import bii_hist
from bii_plot.bii_hbar import bii_hbar
from bii_plot.bii_hbar2 import bii_hbar2

def remove_old():
	"""
	Remove old graphs from server.
	"""

	if os.path.isfile('static/hbar.png'):
	    os.remove('static/hbar.png')
	if os.path.isfile('static/radar.png'):
	    os.remove('static/radar.png')
	if os.path.isfile('static/hist_cz.png'):
	    os.remove('static/hist_cz.png')
	if os.path.isfile('static/hist_score.png'):
	    os.remove('static/hist_score.png')
	if os.path.isfile('static/hbar2.png'):
	    os.remove('static/hbar2.png')

	return

def set_none(part, proj, comp):
	"""
	Set value to None for parameters if no input
	"""

	if not part:
		part = None
	if not proj:
		proj = None
	if not comp:
		comp = None
	elif comp == ' ':
		comp = None

	return part, proj, comp

def get_data(email, part, proj):
	"""
	Collect correct data and determines type of report to produce from input data given from website. 
	Returns:
	ident - indicator for which type of data (individual, workgroup, subset of workgroup)
	data - the data corresponding to input
	code - name of report to be generated
	"""

	if not part and not proj:
		ident = 'ind'
		code = email
		data = get_data_ind(code)
	elif part and not proj:
		ident = 'group'
		code = part
		data = get_data_group(code, 'part')
	elif not part and proj:
		ident = 'group'
		code = proj
		data = get_data_group(code, 'proj')
	else:
		ident = 'group'
		code = [part,proj]
		data = get_data_group(code, 'projAndPart')

	return ident, part, proj, code, data

def get_data_comp(ident, comp):
	"""
	Get the correct comparison data.
	Returns new identity if comparison variable exist.
	If no comparison variable returns empty data set.
	"""

	if comp is not None:
		ident = 'comp'
		data_comp = get_data_group(comp,ident)
		comp = reassign_comp(comp)
	else:
		data_comp = None

	return ident, data_comp, comp

def reassign_comp(comp):
	"""
	Rename the comparison variable to be presented in report.
	"""	
	if comp == 'all':
  		comp = 'All'
	elif comp == 'male':
  		comp = 'Male'
	elif comp == 'female':
  		comp = 'Female'
	elif comp == 'study1':
  		comp = "Engineering"
	elif comp == 'study2':
  		comp = "Manager"
	elif comp == 'study3':
  		comp = "Arts/Humanities"
  	return comp

def create_graphs(ident, code, data, data_comp, comp, part, proj):
	"""
	Creates and saves graphs. Returns updated identity.
	"""
	score = data[-1]
	comfort = data[-3]
	if ident == 'ind':
 		bii_radar(code, data, data_comp, comp, part, proj)
 		bii_hbar(False, code, data)
 	elif ident == 'group':
  		bii_radar(code, data, data_comp, comp, part, proj)
  		bii_hbar(True, code, data)
  		bii_hist(code,score,'Innovation Index Score')
  		bii_hist(code,comfort,'Comfort Zone Score')
  	elif ident == 'comp':
  	    bii_radar(code, data, data_comp, comp, part, proj)
  	    bii_hbar2(data, data_comp, code, comp)
  	    if part is not None or proj is not None:
  			bii_hist(code,score,'Innovation Index Score')
  			bii_hist(code,comfort,'Comfort Zone Score')
  			ident = 'compGroup'
  	return ident