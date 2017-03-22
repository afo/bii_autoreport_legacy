import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def bii_hbar2(in_data, in_data_comp, code, comp):

	if len(code) == 2 and not isinstance(code, basestring):
		code = code[0] + " " + code[1]

	[trust,res,div,bel,collab,resall,comfort,iz,score] = in_data
	[t,r,d,b,col,ra,cz,izc,score_c] = in_data_comp
	data1 = [np.mean(trust), np.mean(res), np.mean(div), np.mean(bel), np.mean(collab), np.mean(resall), np.mean(comfort), np.mean(iz)][-1::-1]
	data2 = [np.mean(t), np.mean(r), np.mean(d), np.mean(b), np.mean(col), np.mean(ra), np.mean(cz), np.mean(izc)][-1::-1]
	
	if isinstance(trust,float):
		err1 = [0,0,0,0,0,0,0,0]
	else:
		err1 = [np.std(trust), np.std(res), np.std(div), np.std(bel), np.std(collab), np.std(resall), np.std(comfort), np.std(iz)][-1::-1]
	


	err2 = [np.std(t), np.std(r), np.std(d), np.std(b), np.std(col), np.std(ra), np.std(cz), np.std(izc)][-1::-1]


	df = pd.DataFrame(dict(graph=['Tru', 'Res', 'Ment Str', 'Conf', 'Collab', 'Res All', 'Com Zone', 'In Zone'][-1::-1], 
									d1=data1, d2 = data2))
	ind = np.arange(len(df))
	width = 0.4

	fig, ax = plt.subplots()
	ax.barh(ind + width, df.d1, width, color='blue', label=code)
	ax.errorbar(data1,ind + 0.6, xerr=err1, color='g', fmt='o')
	ax.barh(ind, df.d2, width, color='red', label=comp)
	ax.errorbar(data2,ind + 0.2, xerr=err2, label="Std", color='g', fmt='o')

	ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
	lgd = ax.legend(loc='upper center', shadow=True, fontsize='x-large',bbox_to_anchor=(1.1, 1.1),borderaxespad=0.)
	plt.xlabel('Score')
	plt.title('Results for ' + code, fontweight='bold', y=1.01)
	plt.xlabel(r'$\mathrm{Total \ Innovation \ Index \ Score:}\ %.3f$' %(np.mean(score)),fontsize='18')
	ax.set_xlim([0,10])
	file_name = 'hbar2'
	path_name = "static/%s" %file_name

	#path_name = "[path]/bii/mod/static/%s" %file_name
	plt.savefig(path_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
