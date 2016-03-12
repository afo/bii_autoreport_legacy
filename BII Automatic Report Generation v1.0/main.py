# -*- coding: utf-8 -*-
"""
Created March 2016

@author: Alexander Fred Ojala & Johan Eng Larsson
UC Berkeley, SCET
"""


from flask import Flask, request, render_template

import os
import shutil
import matplotlib
import time
matplotlib.use('Agg')

import pandas as pd
from sys import argv
from pylab import *
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.mlab as mlab

# Local import
from invertid import invertid
from bii_data import bii_data
from bii_plot.bii_radar import bii_radar
from bii_plot.bii_hist import bii_hist
from bii_plot.bii_hbar import bii_hbar
from bii_plot.bii_hbar2 import bii_hbar2
from nocache import nocache

def reassign_comp(comp):	
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

app = Flask(__name__)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route('/')
@nocache
def my_form():
	return render_template('test.html')

@app.route('/report', methods=['POST'])
@nocache
def my_form_post():
	email = str(request.form['email'])
	wg = str(request.form['wg'])
	proj = str(request.form['proj'])
	comp = str(request.form['comp'])

	return test(email, wg, proj, comp)
	#return email + wg + comp

@app.route('/createReport')
@nocache
def test(email, wg, proj, comp):


	if not wg and not proj:
		ident = 1
		wg = None
		proj = None
		code = email
		data = bii_data(ident,email)
	elif wg and not proj:
		ident = 2
		proj = None
		code = wg
		data = bii_data(ident, wg)
	elif not wg and proj:
		ident = 5
		wg = None
		code = proj
		data = bii_data(ident, proj)
	else:
		ident = 6
		code = [wg,proj]
		data = bii_data(ident,code)

	if np.size(data) == 0:
		page_not_found()
	

	if os.path.isfile('/home/ubuntu/BII/static/hbar.png'):
	    os.remove('/home/ubuntu/BII/static/hbar.png')
	if os.path.isfile('/home/ubunut/BII/static/radar.png'):
	    os.remove('/home/ubuntu/BII/static/radar.png')
	if os.path.isfile('/home/ubuntu/BII/static/hist_cz.png'):
	    os.remove('/home/ubuntu/BII/static/hist_cz.png')
	if os.path.isfile('/home/ubuntu/BII/static/hist_score.png'):
	    os.remove('/home/ubuntu/BII/static/hist_score.png')
	if os.path.isfile('/home/ubuntu/BII/static/hbar2.png'):
	    os.remove('/home/ubuntu/BII/static/hbar2.png')
		

	if comp == " ":
		comp = None
	elif not comp:
		comp = None


	if comp is not None:
		ident = 3
		data_comp = bii_data(ident,comp)
		[trust_c,res_c,div_c,bel_c,collab_c,resall_c,czx_c,comfort_c,iz_c,score_c] = data_comp
	else: 
		data_comp = None

	[trust,res,div,bel,collab,resall,czx,comfort,iz,score] = data

	comp = reassign_comp(comp)



	if ident == 1:
		bii_radar(ident, code, data, data_comp, comp, wg, proj)
		bii_hbar(ident, code, data)
  	elif ident == 2 or ident == 5 or ident == 6:
  		ident = 2
  		bii_radar(ident, code, data, data_comp, comp, wg, proj)
  		bii_hbar(ident, code, data)
  		bii_hist(ident,code,score,'Innovation Index Score', 'green',1)
  		bii_hist(ident,code,comfort,'Comfort Zone Score', 'yellow',2)
  	elif ident == 3:
  	    bii_radar(ident, code, data, data_comp, comp, wg, proj)
  	    bii_hbar2(ident, data, data_comp, code, comp)
  	    if wg is not None or proj is not None:
  			ident = 2
  			bii_hist(ident,code,score,'Innovation Index Score', 'green',1)
  			bii_hist(ident,code,comfort,'Comfort Zone Score', 'yellow',2)
  			ident = 4

  

  	# Render templates

  	if len(code) == 2 and not isinstance(code, basestring):
  		code = "Partner: " + code[0] + "    Project: " + code[1]

  	if ident == 1:
  		return render_template('reportind.html', wg = code, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))
  	elif ident == 2:
  		return render_template('reportwg.html', wg = code, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))
  	elif ident == 3:
  		return render_template('reportcompind.html', comp = comp, wg = code, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))
  	elif ident == 4:
  		return render_template('reportcomp.html', comp = comp, wg = code, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))

@app.route('/<user_id>')
@nocache
def main(user_id):

	
        if os.path.isfile('home/ubuntu/BII/static/hbar.png'):
                os.remove('home/ubuntu/BII/static/hbar.png')
        if os.path.isfile('home/ubunut/BII/static/radar.png'):
                os.remove('home/ubuntu/BII/static/radar.png')
        if os.path.isfile('home/ubuntu/BII/static/hist_cz.png'):
                os.remove('home/ubuntu/BII/static/hist_cz.png')
        if os.path.isfile('home/ubuntu/BII/static/hist_score.png'):
                os.remove('home/ubuntu/BII/static/hist_score.png')
        if os.path.isfile('home/ubuntu/BII/static/hbar2.png'):
                os.remove('home/ubuntu/BII/static/hbar2.png')
	
	ident = 1
	comp = None
	wg = None
	proj = None

	data_comp = None
	#get mail from user_id

	code = invertid(user_id)


	# this will be done by finding the correct email that matches the user_id
	# when we have id everything is pretty straightforward
	# the json-file will simply return a dictionary, i.e. we can use the dictionary functions to search for the value

	data = bii_data(ident, code)

	if np.size(data) == 0:
		page_not_found()

	[trust,res,div,bel,collab,resall,czx,comfort,iz,score] = data

	bii_radar(ident, code, data, data_comp, comp, wg, proj)
  	bii_hbar(ident, code, data)

  	return render_template('reportind.html', wg = code, index_score=round(mean(score),2), tru = round(mean(trust),2), col = round(mean(collab),2), res_all = round(mean(resall),2), div = round(mean(div),2), men_st = round(mean(bel),2), in_zone = round(mean(iz),2), res = round(mean(res),2))





@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def page_not_found(e):
    return render_template('error.html'), 500




if __name__ == '__main__':
	app.run(host = '0.0.0.0', port = 80)