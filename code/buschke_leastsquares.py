#######################################################
# Python program name	: 
#Description	: buschke_leastsquares.py
#Args           : Estimate the Buschke model parameters that best fit with MRI data  
#				 Plots  plt_boxplot_brain_volumes                                                                                     
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#######################################################

import os, sys, pdb, operator
import datetime
import time
import numpy as np
import pandas as pd
import importlib
import sys
import warnings
from subprocess import check_output
#sys.path.append('/Users/jaime/github/papers/EDA_pv/code')
import area_under_curve 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
#sys.path.append('/Users/jaime/github/code/tensorflow/production')
#import descriptive_stats as pv

def compute_buschke_integral_df(dataframe, features_dict=None):
	""" compute_buchske_integral_df compute new Buschke 
	Args: dataframe with the columns fcsrtrl1_visita[1-7]
	Output:return the dataframe including the columns bus_visita[1-7]"""

	import scipy.stats.mstats as mstats
	print('Compute the Buschke aggregate \n')
	S = [0] * dataframe.shape[0]
	# arithmetic, gemoetric mean and sum of Bischke scores
	nb_years = 7
	mean_a, mean_g, suma = S[:], S[:], S[:]
	#longit_pattern= re.compile("^fcsrtrl[1-3]_+visita+[1-7]")
	#longit_status_columns = [x for x in dataframe.columns if (longit_pattern.match(x))]
	for i in range(1, nb_years+1):
		coda='visita'+ format(i)
		#bus_scores = ['fcsrtrl1_visita1', 'fcsrtrl2_visita1', 'fcsrtrl3_visita1']
		bus_scores = ['fcsrtrl1_'+coda, 'fcsrtrl2_'+coda,'fcsrtrl3_'+coda]
		df_year = dataframe[bus_scores]
		df_year = df_year.values
		#bus_scores = ['fcsrtrl1_visita2', 'fcsrtrl2_visita2', 'fcsrtrl3_visita2']
		for ix, y in enumerate(df_year):	
			#print(row[bus_scores[0]], row[bus_scores[1]],row[bus_scores[2]])
			#pdb.set_trace()
			bes = area_under_curve.buschke_aggregate(y)
			S[ix]=bes[0]
			mean_a[ix] = np.mean(y)
			mean_g[ix] = mstats.gmean(y)
			suma[ix] = np.sum(y) 
			print('Total Aggregate S=', bes[0])
			print('arithmetic mean:', mean_a[ix], ' Geometric mean:', mean_g[ix], ' Sum:',suma[ix])
			print('Poly1d exponents decreasing' ,bes[-1])
			print('Poly2 exponents decreasing',bes[-2])
			print('\n')
		coda_col= 'bus_int_'+ coda
		# add bus_visita[1-7] in features_dict
		#features_dict['CognitivePerformance'].append(coda_col);
		#features_dict['CognitivePerformance'].append('bus_sum_'+coda);
		#features_dict['CognitivePerformance'].append('bus_meana_'+coda);
		dataframe[coda_col] = S
		dataframe['bus_sum_'+coda] = suma
		dataframe['bus_meana_'+coda] = mean_a
	
	return dataframe


def identify_outliers(df, dictio):
	"""identify_outliers: 
	Args: dictionary of keys - tissues (3) or subcortical (9)
	Output: df with rows of outliers eliminated dictionary {'L_Accu_visita1':[[list too small ixs],[too large ixs]]
	"""

	n = 3 #number of std deviations
	df_noout = df.copy()

	quantiles = [0.01, 0.99] #1%, 99% and Maximum
	for col in dictio.keys():
		statcol = df[col].describe()
		#statcol['mean']/100
		#statcol['mean'] + n*statcol['std']
		lower_limit = df[col].quantile([quantiles[0], quantiles[1]])[quantiles[0]]
		upper_limit = df[col].quantile([quantiles[0], quantiles[1]])[quantiles[1]]
		print('Outliers for ', col, 'lower:', lower_limit,'upper_limit:',upper_limit, 'Max:', statcol['max'])
		listlow= df.index[df[col] < lower_limit].tolist()
		listhigh = df.index[df[col] > upper_limit].tolist()
		print('Index of outliers lower bound:', listlow,'upper bound:', listhigh)
		dictio[col] =  [listlow, listhigh]
		#pdb.set_trace()
		df_noout[col] = df_noout[col].loc[~df_noout[col].index.isin(listlow)]
		df_noout[col] = df_noout[col].loc[~df_noout[col].index.isin(listhigh)]
	print('df original shape:', df.shape, 'df outliers 1-99$\\%$ extreme values removed shape:', df_noout.shape)	
	return df_noout, dictio

def plt_boxplot_brain_volumes(df, cols, title=None ):
	"""plt_boxplot_brain_volumes: boxplot of brain volumes
	Args: df, cols (tissue or subcortical features)
	Output: None, save a figure 
	"""
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	f, ax = plt.subplots(1, 1, figsize=(16, 14))
	ax.set_xticklabels(ax.get_xticklabels(),fontsize=8)
	ax.set_ylabel('$mm^3$')
	df.describe()
	df = df[cols]	
	print df.describe()
	df.dropna(inplace=True)

	boxplot = df.boxplot(column=cols)
	#props = {"rotation" : 45}
	#plt.setp(ax.get_xticklabels(), **props)
	plt.tight_layout()
	if len(cols) <= 2 :
		coda = 'brain'
	elif len(cols) <= 3 :
		coda = 'tissue'
	else:
		coda = 'subcortical'
		
	figname = 'boxplot_' + str(title) + coda + '.png'
	plt.savefig(os.path.join(figures_dir,figname), dpi=120)
	#plt.show()
	#boxplot = df.boxplot(column=['Col1', 'Col2'], by=['X', 'Y'])

def plot_correlation_tissue(df, target, mri_tissue_cols):
	"""
	"""
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	fig, axis = plt.subplots(1, 1, figsize=(8, 6))
	df_t = df[mri_tissue_cols + target]
	# df_t.dropna(inplace=True)
	# for col in df_t.columns:
	# 	df_t[col] = df_t[col].astype(float)
	corr = 	df_t.corr()
	print corr
	ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1, vmax=1,annot=True,cmap="seismic")
	ax.set_xticklabels(ax.get_xticklabels(),fontsize=7)
	ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
	props = {"rotation" : 0}
	plt.setp(ax.get_yticklabels(), **props)
	title = 'Correlation Tissue volume ' + target[0]
	ax.set_title(title)
	plt.tight_layout()
	#remove '_visita1'
	figname = 'tissue_' + target[0][:7] + 'heatmap.png'
	plt.savefig(os.path.join(figures_dir,figname), dpi=120)
	return 0


def plot_correlation_subcortical(df, target, mri_subcortical_cols):
	"""plot_correlation_brain_volumes: plot correlationn matrix of tissue, brain 
	and cortical structures
	Args: df from csv created by fsl_psotproccessing, target: feature dep variable to comapre with
	Output: df

	"""
	# plotting brain vol and Buschke 
	# plotting tissue and Buschke 
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	fig, axis = plt.subplots(1, 1, figsize=(8, 6))
	df_sub = df[mri_subcortical_cols + target]
	
	#df_sub.dropna(inplace=True)
	# convert to float to compute correlation
	# for col in df_sub.columns:
	# 	df_sub[col] = df_sub[col].astype(float)	
	corr = 	df_sub.corr()
	print corr

	# plotting subcortical and Buschke 
	fig, axis = plt.subplots(1, 1, figsize=(18, 14))
	corr = df_sub.corr()
	print corr
	ax = sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns,vmin=-1, vmax=1,annot=True,cmap="seismic")
	ax.set_xticklabels(ax.get_xticklabels(),fontsize=7)
	ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
	props = {"rotation" : 0}
	plt.setp(ax.get_yticklabels(), **props)
	title = 'Correlation subcortical volume ' + target[0]
	ax.set_title(title)
	plt.tight_layout()
	figname = 'subcortical_' + target[0][:7] + 'heatmap.png'
	plt.savefig(os.path.join(figures_dir,figname), dpi=120)
	return 0

def plot_scatter_brain(dataframe, x,y,colors=None, title=None, figname=None, cmap=None):
	"""
	"""
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	#figname = 'bus-scatterMNIxy.png'
	#YS convert brain volumes to float or integer when the csv is created, so
	# you dont need to do this here
	#dataframe[y] = dataframe[y].values.astype(int)

	if cmap is None:
		dataframe.plot.scatter(x, y,c=colors, title =title,alpha=0.2)
	else:
		dataframe.plot.scatter(x, y,c=colors, title =title,alpha=0.2, cmap=cmap)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240)

def plot_histogram_tissues(df_nooutliers_t, figures_path):
	#tissues
	fig, axarr = plt.subplots(figsize=(12,8),nrows=1, ncols=3)
	csf_cm, wm_cm, gm_cm = df_nooutliers_t['csf_volume_visita1']/1000,df_nooutliers_t['wm_volume_visita1']/1000,df_nooutliers_t['gm_volume_visita1']/1000
	csf_cm.plot(kind='hist', ax=axarr[0], title='csf $cm^3$')
	csf_cm.plot(kind='kde', ax=axarr[0], secondary_y=True)
	gm_cm.plot(kind='hist',ax=axarr[1], title='gm $cm^3$')
	gm_cm.plot(kind='kde', ax=axarr[1], secondary_y=True)
	wm_cm.plot(kind='hist',ax=axarr[2], title='wm $cm^3$')
	wm_cm.plot(kind='kde', ax=axarr[2], secondary_y=True)
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path,'kde_3_tissues.png' ), dpi=240)

def plot_histogram_subcort(df_nooutliers, figures_path):
	#subcoritcal structures
	fig, axarr = plt.subplots(figsize=(22,22),nrows=7, ncols=2)
	hipp_L, hipp_R = df_nooutliers['L_Hipp_visita1']/1000, df_nooutliers['R_Hipp_visita1']/1000 
	amyg_L, amyg_R = df_nooutliers['L_Amyg_visita1']/1000, df_nooutliers['R_Amyg_visita1']/1000 
	caud_L, caud_R = df_nooutliers['L_Caud_visita1']/1000, df_nooutliers['R_Caud_visita1']/1000 
	accu_L, accu_R = df_nooutliers['L_Accu_visita1']/1000, df_nooutliers['R_Accu_visita1']/1000 
	pall_L, pall_R = df_nooutliers['L_Pall_visita1']/1000, df_nooutliers['R_Pall_visita1']/1000 
	thal_L, thal_R = df_nooutliers['L_Thal_visita1']/1000, df_nooutliers['R_Thal_visita1']/1000 
	puta_L, puta_R = df_nooutliers['L_Puta_visita1']/1000, df_nooutliers['R_Puta_visita1']/1000 
	hipp_L.plot(kind='hist', ax=axarr[0,0], title='hippL $cm^3$')
	hipp_L.plot(kind='kde', ax=axarr[0,0], secondary_y=True)
	hipp_R.plot(kind='hist', ax=axarr[0,1], title='hippR $cm^3$')
	hipp_R.plot(kind='kde', ax=axarr[0,1], secondary_y=True)
	amyg_L.plot(kind='hist',ax=axarr[1,0], title='amygL $cm^3$')
	amyg_L.plot(kind='kde', ax=axarr[1,0], secondary_y=True)
	amyg_R.plot(kind='hist',ax=axarr[1,1], title='amygR $cm^3$')
	amyg_R.plot(kind='kde', ax=axarr[1,1], secondary_y=True)
	caud_L.plot(kind='hist', ax=axarr[2,0], title='caudL $cm^3$')
	caud_L.plot(kind='kde', ax=axarr[2,0], secondary_y=True)
	caud_R.plot(kind='hist',ax=axarr[2,1], title='caudR $cm^3$')
	caud_R.plot(kind='kde', ax=axarr[2,1], secondary_y=True)
	accu_L.plot(kind='hist',ax=axarr[3,0], title='accuL $cm^3$')
	accu_L.plot(kind='kde', ax=axarr[3,0], secondary_y=True)
	accu_R.plot(kind='hist',ax=axarr[3,1], title='accR $cm^3$')
	accu_R.plot(kind='kde', ax=axarr[3,1], secondary_y=True)
	pall_L.plot(kind='hist', ax=axarr[4,0], title='pallL $cm^3$')
	pall_L.plot(kind='kde', ax=axarr[4,0], secondary_y=True)
	pall_R.plot(kind='hist',ax=axarr[4,1], title='pallR $cm^3$')
	pall_R.plot(kind='kde', ax=axarr[4,1], secondary_y=True)
	thal_L.plot(kind='hist',ax=axarr[5,0], title='thalL $cm^3$')
	thal_L.plot(kind='kde', ax=axarr[5,0], secondary_y=True)
	thal_R.plot(kind='hist',ax=axarr[5,1], title='thalR $cm^3$')
	thal_R.plot(kind='kde', ax=axarr[5,1], secondary_y=True)
	puta_L.plot(kind='hist',ax=axarr[6,0], title='putaL $cm^3$')
	puta_L.plot(kind='kde', ax=axarr[6,0], secondary_y=True)
	puta_R.plot(kind='hist',ax=axarr[6,1], title='putaR $cm^3$')
	puta_R.plot(kind='kde', ax=axarr[6,1], secondary_y=True)
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path,'kde_subcortical.png' ), dpi=240)

def normal_gaussian_test(rv_values, rv_name =None, method=None, plotcurve=False):

	import pylab
	from scipy import stats
	report_test, sha_report, kol_report = [], [], []
	header = '***** normal_gaussian_test for variable:%s ***** \n'%(rv_name)
	report_test.append(header) 
	p_threshold = 0.05
	rv_values = rv_values.dropna()
	[t_shap, p_shap] = stats.shapiro(rv_values)
	
	if p_shap < p_threshold:
		test_r = '\tShapiro-Wilk test: Reject null hypothesis that sample comes from Gaussian distribution \n'
		print(test_r, rv_name)
	else:
		test_r = '\tShapiro-Wilk test: DO NOT Reject null hypothesis that sample comes from Gaussian distribution \n'+\
		'\tLikely sample comes from Normal distribution. But the failure to reject could be because of the sample size:%s\n'%(str(rv_values.shape[0]))
		print(test_r, rv_name)	
	sha_report = '\tShapiro-Wilk test: t statistic:%s and p-value:%s \n'%(str(t_shap), str(p_shap))
	print(sha_report)
	report_test.append(test_r)
	report_test.append(sha_report)

	[t_kol, p_kol] = stats.kstest(rv_values, 'norm', args=(rv_values.mean(), rv_values.std()))
	if p_kol < p_threshold:
		test_r = 'KolmogorovSmirnov: Reject null hypothesis that sample comes from Gaussian distribution \n' 
		print(test_r, rv_name)
	else:
		test_r = '\tKolmogorov-Smirnov: DO NOT Reject null hypothesis that sample comes from Gaussian distribution \n'+ \
		'\tLikely sample comes from Normal distribution. But the failure to reject could be because of the sample size:%s\n'%(str(rv_values.shape[0]))
		print(test_r, rv_name)
	kol_report = '\tKolmogorov test: t statistic:%s and p-value:%s \n'%(str(t_kol), str(p_kol))
	report_test.append(test_r)
	report_test.append(kol_report)
	print(kol_report)
	#Comparing CDF for KS test
	if plotcurve is True:
		#quantile-quantile (QQ) plot
		#If the two distributions being compared are from a common distribution, the points in from QQ plot the points in 
		#the plot will approximately lie on a line, but not necessarily on the line y = x		
		sm.qqplot(rv_values, loc = rv_values.mean(), scale = rv_values.std(), line='s')
		plt.title('Shapiro-Wilk: '+ rv_name+ ' . Shapiro p-value='+ str(p_shap))
		pylab.show()
		length = len(rv_values)
		plt.figure(figsize=(12, 7))
		plt.plot(np.sort(rv_values), np.linspace(0, 1, len(rv_values), endpoint=False))
		plt.plot(np.sort(stats.norm.rvs(loc=rv_values.mean(), scale=rv_values.std(), size=len(rv_values))), np.linspace(0, 1, len(rv_values), endpoint=False))
		plt.legend('top right')
		plt.legend(['Data', 'Theoretical Gaussian Values'])
		plt.title('Comparing CDFs for KS-Test: '+ rv_name + ' . KS p-value='+str(p_kol))
	return report_test

def plot_histograma_cardiovascular_categorical(df, target_variable=None):
	"""plot_histograma_cardiovascular_categorical: 
	"""
	this_function_name = sys._getframe(  ).f_code.co_name
	print('Calling to {}',format(this_function_name))
	list_cardio=['hta', 'hta_ini', 'glu', 'lipid', 'tabac', 'tabac_cant', 'tabac_fin', \
	'tabac_ini', 'sp', 'cor', 'cor_ini', 'arri', 'arri_ini', 'card', 'card_ini', 'tir', \
	'ictus', 'ictus_num', 'ictus_ini', 'ictus_secu']
	df['hta'] = df['hta'].astype('category').cat.rename_categories(['NoHypArt', 'HypArt'])
	df['glu'] = df['glu'].astype('category').cat.rename_categories(['NoGlu', 'DiabMel','Intoler.HydroC'])
	df['tabac'] = df['tabac'].astype('category').cat.rename_categories(['NoSmoker', 'Smoker', 'ExSomoker'])
	df['sp'] = df['sp'].astype('category').cat.rename_categories(['NoOW', 'OverWeight', 'NP'])
	df['cor'] = df['cor'].astype('category').cat.rename_categories(['NoHeartPb', 'Angina', 'Infartion', 'NP'])
	df['arri'] = df['arri'].astype('category').cat.rename_categories(['NoArri', 'FibrAur', 'Arrhythmia', 'NP'])
	df['card'] = df['card'].astype('category').cat.rename_categories(['NoCardDis', 'CardDis', 'NP'])
	df['tir'] = df['tir'].astype('category').cat.rename_categories(['NoTyr', 'HiperTyr','HipoTir', 'NP'])
	df['ictus'] = df['ictus'].astype('category').cat.rename_categories(['NoIct', 'IschIct','HemoIct', 'NP'])
	fig, ax = plt.subplots(2,5)
	fig.set_size_inches(15,10)
	ax[-1, -1].axis('off')
	fig.suptitle('Conversion absolute numbers for cardiovascular')
	d = df.groupby([target_variable, 'hta']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'glu']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'tabac']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,2])
	d = df.groupby([target_variable, 'sp']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,3])
	d = df.groupby([target_variable, 'cor']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,4])
	d = df.groupby([target_variable, 'arri']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'card']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'tir']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,2])
	d = df.groupby([target_variable, 'ictus']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,3])

	# in relative numbers
	fig, ax = plt.subplots(2,5)
	ax[-1, -1].axis('off')
	fig.set_size_inches(15,10)
	fig.suptitle('Conversion relative numbers for cardiovascular')
	d = df.groupby([target_variable, 'hta']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'glu']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'tabac']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,2])
	d = df.groupby([target_variable, 'sp']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,3])
	d = df.groupby([target_variable, 'cor']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,4])
	d = df.groupby([target_variable, 'arri']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'card']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'tir']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,2])
	d = df.groupby([target_variable, 'ictus']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,3])
	plt.show()


def plot_histograma_psychiatrichistory_categorical(df, target_variable=None):
	list_psychiatric_h=['depre', 'depre_ini', 'depre_num', 'depre_trat', 'ansi', 'ansi_ini', 'ansi_num', 'ansi_trat']
	df['depre_num_cat'] = pd.cut(df['depre_num']*df['depre'],4)
	df['ansi_num_cat'] = pd.cut(df['ansi_num'],4)
	fig, ax = plt.subplots(1,2)
	fig.suptitle('Conversion absolute numbers for psychiatric history')
	d = df.groupby([target_variable, 'depre_num_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'ansi_num_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1])

	fig, ax = plt.subplots(1,2)
	fig.suptitle('Conversion relative numbers for psychiatric history')
	d = df.groupby([target_variable, 'depre_num_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'ansi_num_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])

def plot_histograma_sleep_categorical(df, target_variable=None):

	this_function_name = sys._getframe(  ).f_code.co_name
	print('Calling to {}',format(this_function_name))

	list_sleep= ['sue_con', 'sue_dia', 'sue_hor', 'sue_man', 'sue_mov', 'sue_noc', 'sue_pro', \
	'sue_rec', 'sue_ron', 'sue_rui', 'sue_suf']
	df['sue_noc_cat'] = pd.cut(df['sue_noc'], 4) # hours of sleep night
	df['sue_dia_cat'] = pd.cut(df['sue_dia'],4) # hours of sleep day
	fig, ax = plt.subplots(2,4)
	#ax[-1, -1].axis('off')
	fig.set_size_inches(10,10)
	fig.suptitle('Conversion absolute numbers for Sleep')
	datag = df.groupby([target_variable, 'sue_noc_cat']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[0,0])
	datag = df.groupby([target_variable, 'sue_dia_cat']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[0,1])
	datag = df.groupby([target_variable, 'sue_con']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[0,2])
	datag = df.groupby([target_variable, 'sue_suf']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[0,3])
	datag = df.groupby([target_variable, 'sue_pro']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[1,0])
	datag = df.groupby([target_variable, 'sue_ron']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[1,1])
	datag = df.groupby([target_variable, 'sue_rec']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[1,2])
	datag = df.groupby([target_variable, 'sue_hor']).size()
	p = datag.unstack(level=1).plot(kind='bar', ax=ax[1,3])

	fig, ax = plt.subplots(2,4)
	#ax[-1, -1].axis('off')
	fig.set_size_inches(10,10)
	fig.suptitle('Conversion relative numbers for Sleep')
	datag = df.groupby([target_variable, 'sue_noc_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,0])
	datag = df.groupby([target_variable, 'sue_dia_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,1])
	datag = df.groupby([target_variable, 'sue_con']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,2])
	datag= df.groupby([target_variable, 'sue_suf']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,3])
	datag = df.groupby([target_variable, 'sue_pro']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,0])
	datag = df.groupby([target_variable, 'sue_ron']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,1])
	datag = df.groupby([target_variable, 'sue_rec']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,2])
	datag = df.groupby([target_variable, 'sue_hor']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,3])


def plot_histograma_anthropometric_categorical(df, target_variable=None):

	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by anthropometric\n')

	this_function_name = sys._getframe(  ).f_code.co_name
	print('Calling to {}',format(this_function_name))
	#list_anthropometric = ['lat_manual', 'pabd', 'peso', 'talla', 'audi', 'visu', 'imc']
	list_anthropometric = ['imc', 'pabd', 'peso', 'talla']

	df['pabd_cat'] = pd.qcut(df['pabd'], 4, precision=1) # abdo perimeter
	df['peso_cat'] = pd.qcut(df['peso'], 4, precision=1) # weight
	df['talla_cat'] = pd.qcut(df['talla'], 4, precision=1) # height
	df['imc_cat'] = pd.qcut(df['imc'], 4, precision=1) # height
	# Relative terms only	
	fig, ax = plt.subplots(2,2, figsize=(12,10))
	#ax[-1, -1].axis('off')
	#fig.set_size_inches(10,10)
	#fig.suptitle('Conversion relative for Anthropometric')
	d = df.groupby([target_variable, 'pabd_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'peso_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'talla_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'imc_cat']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,0])
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_anthropometric_mci.png'), dpi=240)


def plot_histograma_engagement_categorical(df, target_variable=None):	
	lista_engag_ext_w = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', \
	'a10', 'a11', 'a12', 'a13', 'a14']
	#physical activities 1, creative 2, go out with friends 3,travel 4, ngo 5,church 6, social club 7,
	# cine theater 8,sport 9, listen music 10, tv-radio (11), books (12), internet (13), manualidades(14)
	df['physicaltrain_cat'] = pd.cut(df['a01'] + df['a09'], 3) # phys exer sport
	df['creative_cat'] = pd.cut(df['a02'] + df['a14'], 3) # creative manualidades
	df['sociallife_cat'] = pd.cut(df['a03'] + df['a05']+ df['a07'] + df['a08'], 3)
	#church, books, music , techno
	fig, ax = plt.subplots(2,4)
	ax[-1, -1].axis('off')
	fig.set_size_inches(15,10)
	fig.suptitle('Conversion absolute numbers for Engagement external world')
	d = df.groupby([target_variable, 'physicaltrain_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,0],label='physical exercise')

	d = df.groupby([target_variable, 'creative_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'sociallife_cat']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,2])
	d = df.groupby([target_variable, 'a06']).size() # church goers
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0,3])
	d = df.groupby([target_variable, 'a12']).size() #read books
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'a10']).size() #listen to music
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'a13']).size() #internet
 	p = d.unstack(level=1).plot(kind='bar', ax=ax[1,2],label='Internet')
	ax[1,2].legend() 	
 	# 
	fig, ax = plt.subplots(2,4)
	ax[-1, -1].axis('off')
	fig.set_size_inches(15,10)
	fig.suptitle('Conversion relative numbers for Engagement external world')
	datag = df.groupby([target_variable, 'physicaltrain_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,0])
	datag = df.groupby([target_variable, 'creative_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,1])
	datag = df.groupby([target_variable, 'sociallife_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,2])
	datag = df.groupby([target_variable, 'a06']).size().unstack(level=1) # church goers
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,3])
	datag = df.groupby([target_variable, 'a12']).size().unstack(level=1) #read books
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,0])
	datag = df.groupby([target_variable, 'a10']).size().unstack(level=1) #listen to music
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,1])
	datag = df.groupby([target_variable, 'a13']).size().unstack(level=1) #internet
 	datag = datag / datag.sum()
 	p = datag.plot(kind='bar', ax=ax[1,2])


def plot_histograma_demographics_categorical(df, target_variable=None):
	d, p= pd.Series([]), pd.Series([])

	df['nivel_educativo'] = df['nivel_educativo'].astype('category').cat.rename_categories(['~Pr', 'Pr', 'Se', 'Su'])
	df['familial_ad'] = df['familial_ad'].astype('category').cat.rename_categories(['NoFam', 'Fam'])
	df['nivelrenta'] = df['nivelrenta'].astype('category').cat.rename_categories(['Baja', 'Media', 'Alta'])
	df['edad_visita1'] = pd.cut(df['edad_visita1'], range(0, 100, 10), right=False)
	#in absolute numbers
	fig, ax = plt.subplots(1,5)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by absolute numbers, for various demographics')

	d = df.groupby([target_variable, 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'familial_ad']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'nivelrenta']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[3])
	d = df.groupby([target_variable, 'edad_visita1']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[4])

	#in relative numbers
	fig, ax = plt.subplots(1,5)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers, for various demographics')
	d = df.groupby([target_variable, 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'nivel_educativo']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'familial_ad']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])
	d = df.groupby([target_variable, 'nivelrenta']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[3])
	d = df.groupby([target_variable, 'edad_visita1']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[4])
	plt.show()

def plot_histograma_genetics(df, target_variable=None):
	#in absolute numbers
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Genetics\n')
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	#fig.set_size_inches(20,5)
	#fig.suptitle('Conversion by absolute numbers, Genetics')
	#abs numbers
	d = df.groupby([target_variable, 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0], title='APOE4-conversion Absolute')
	#in relative numbers
	d = df.groupby([target_variable, 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1],title='APOE4-conversion Relative')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_genetics_mci.png'), dpi=240)
	#plt.show()

def plot_histograma_diet_categorical(df, target_variable=None):
	""" plot_histograma_diet_categorical
	Args: datafame, target_variable
	Output:None
	"""
	# df['alfrut'] = df['alfrut'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['alcar'] = df['alcar'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['aldulc'] = df['aldulc'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	# df['alverd'] = df['alverd'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	
	# 4 groups:: 'dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable'
	nb_of_categories = 4
	#df['dietaproteica_cut'] = pd.cut(df['dietaproteica'],nb_of_categories)
	#df['dietagrasa_cut'] = pd.cut(df['dietagrasa'],nb_of_categories)
	# df['dietaketo_cut'] = pd.cut(df['dietaketo'],nb_of_categories)
	
	df['dietaglucemica_cut'] = pd.cut(df['dietaglucemica'],nb_of_categories)
	df['dietasaludable_cut']= pd.cut(df['dietasaludable'],nb_of_categories)
	df['dietaproteica_cut']= pd.cut(df['dietaproteica'],nb_of_categories)
	df['dietagrasa_cut']= pd.cut(df['dietagrasa'],nb_of_categories)
	#diet in relative numbers
	fig, ax = plt.subplots(1,4)
	fig.set_size_inches(20,5)
	fig.suptitle('Conversion by relative numbers for Alimentation')
	d = df.groupby([target_variable, 'dietaglucemica_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0])
	d = df.groupby([target_variable, 'dietasaludable_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1])
	d = df.groupby([target_variable, 'dietaproteica_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2])
	p = d.plot(kind='bar', ax=ax[3])
	d = df.groupby([target_variable, 'dietagrasa_cut']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[3])

def plot_histograma_bygroup_categorical(df, type_of_group, target_variable):
	""" plot_histograma_bygroup_categorical
	"""
	if type_of_group is 'Genetics_s': plot_histograma_genetics(df, target_variable)
	if type_of_group is 'Demographics_s': plot_histograma_demographics_categorical(df, target_variable)
	if type_of_group is 'Diet_s': plot_histograma_diet_categorical(df, target_variable)
	if type_of_group is 'EngagementExternalWorld_s': plot_histograma_engagement_categorical(df, target_variable)
	if type_of_group is 'Anthropometric_s': plot_histograma_anthropometric_categorical(df, target_variable)
	if type_of_group is 'Sleep_s': plot_histograma_sleep_categorical(df, target_variable)
	if type_of_group is 'PsychiatricHistory_s': plot_histograma_psychiatrichistory_categorical(df, target_variable)
	if type_of_group is 'Cardiovascular_s': plot_histograma_cardiovascular_categorical(df, target_variable)
	if type_of_group is 'TraumaticBrainInjury_s': plot_histograma_traumaticbraininjury_categorical(df, target_variable)

def main():
	# Open csv with MRI data
	plt.close('all')
	csv_path = '/Users/jaime/Downloads/test_code/PV7years_T1vols.csv'
	csv_path = '~/vallecas/data/BBDD_vallecas/PVDB_pve_sub.csv'
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures/'
	dataset = pd.read_csv(csv_path)
	dataframe = compute_buschke_integral_df(dataset)
	#dataframe.replace('=', np.nan, axis=1, inplace=True)

	# select buschke and MRI cols
	mri_brain_cols = ['scaling2MNI_visita1', 'volume_bnative_visita1','volume_bMNI_visita1']
	
	mri_subcortical_cols = ['L_Thal_visita1','R_Thal_visita1','L_Puta_visita1','R_Puta_visita1','L_Amyg_visita1',\
	'R_Amyg_visita1', 'L_Pall_visita1','R_Pall_visita1', 'L_Caud_visita1','R_Caud_visita1','L_Hipp_visita1',\
	'R_Hipp_visita1','L_Accu_visita1','R_Accu_visita1','BrStem_visita1']
	# remove BrStem because there are many 0 (error when segmentation) 
	mri_subcortical_cols = mri_subcortical_cols[:-1]
	
	mri_tissue_cols = ['csf_volume_visita1', 'gm_volume_visita1', 'wm_volume_visita1']
	buschke_features = ['fcsrtlibdem_visita1', 'fcsrtlibdem_visita2', 'fcsrtlibdem_visita3', 'fcsrtlibdem_visita4', \
	'fcsrtlibdem_visita5', 'fcsrtlibdem_visita6', 'fcsrtlibdem_visita7', 'fcsrtrl1_visita1', 'fcsrtrl1_visita2', \
	'fcsrtrl1_visita3', 'fcsrtrl1_visita4', 'fcsrtrl1_visita5', 'fcsrtrl1_visita6', 'fcsrtrl1_visita7', 'fcsrtrl2_visita1', \
	'fcsrtrl2_visita2', 'fcsrtrl2_visita3', 'fcsrtrl2_visita4', 'fcsrtrl2_visita5', 'fcsrtrl2_visita6', 'fcsrtrl2_visita7', \
	'fcsrtrl3_visita1', 'fcsrtrl3_visita2', 'fcsrtrl3_visita3', 'fcsrtrl3_visita4', 'fcsrtrl3_visita5', 'fcsrtrl3_visita6', \
	'fcsrtrl3_visita7','bus_int_visita1', 'bus_sum_visita1', 'bus_int_visita2', 'bus_sum_visita2', 'bus_int_visita3', \
	'bus_sum_visita3', 'bus_int_visita4', 'bus_sum_visita4', 'bus_int_visita5', 'bus_sum_visita5', 'bus_int_visita6', \
	'bus_sum_visita6', 'bus_int_visita7', 'bus_sum_visita7'] 
	
	buschke_features = ['bus_int_visita1']
	conversion_features =['conversionmci','dx_corto_visita1', 'dx_corto_visita2','dx_corto_visita3',\
	'dx_corto_visita4','dx_corto_visita5','dx_corto_visita6','dx_corto_visita7']
	conversion_features =['conversionmci', 'scdgroups_visita1']

	### identify and remove 1-99 pc extreme values
	# outliers for brain tissue 
	lst = mri_tissue_cols + buschke_features + conversion_features
	dictio_t = {'csf_volume_visita1':[], 'gm_volume_visita1':[], 'wm_volume_visita1':[]}
	df_nooutliers_t, outliers_dictio = identify_outliers(dataframe[lst], dictio_t)	
	# outliers for subcortical structures
	lst = mri_subcortical_cols + buschke_features + conversion_features
	dictio_s = {'L_Accu_visita1':[], 'L_Amyg_visita1':[], 'L_Caud_visita1':[],'L_Hipp_visita1':[],\
	'L_Pall_visita1':[], 'L_Puta_visita1':[],'L_Thal_visita1':[],'R_Accu_visita1':[], \
	'R_Amyg_visita1':[], 'R_Caud_visita1':[],'R_Hipp_visita1':[], 'R_Pall_visita1':[], \
	'R_Puta_visita1':[],'R_Thal_visita1':[]}
	df_nooutliers, outliers_dictio = identify_outliers(dataframe[lst], dictio_s)

	### Correlation GroupBy analysis
	types_of_groups = ['Anthropometric_s', 'Genetics_s', 'Demographics_s', 'Diet_s','EngagementExternalWorld_s','Sleep_s',\
	'PsychiatricHistory_s','Cardiovascular_s'] #,'TraumaticBrainInjury_s'] 
	target_variable = 'conversionmci'

	for type_of_group in types_of_groups:
		plot_histograma_bygroup_categorical(dataframe, type_of_group, target_variable)
		pdb.set_trace()
	return	
	### boxplot of brain size
	plt_boxplot_brain_volumes(dataframe, mri_brain_cols[1:3] )
	# boxplot of brain strucures
	plt_boxplot_brain_volumes(df_nooutliers_t, mri_tissue_cols,'no_outliers')
	plt_boxplot_brain_volumes(dataframe,  mri_subcortical_cols)
	plt_boxplot_brain_volumes(df_nooutliers,  mri_subcortical_cols, 'no_outliers')
	
	## plot histogram KDE
	plot_histogram_tissues(df_nooutliers_t, figures_path)
	plot_histogram_subcort(df_nooutliers, figures_path)

	normal_gaussian_test(dataframe['volume_bnative_visita1'], rv_name ='volume_bnative_visita1')
	normal_gaussian_test(df_nooutliers_t['csf_volume_visita1'], rv_name ='csf volume')
	normal_gaussian_test(df_nooutliers_t['gm_volume_visita1'], rv_name ='gm volume')
	normal_gaussian_test(df_nooutliers_t['wm_volume_visita1'], rv_name ='wm volume')

		
	### Scatter plots	
	print('\n\n Plotting Scatter Plots between pairs of variables ....\n\n')
	lst = buschke_features + mri_brain_cols + conversion_features
	lst = buschke_features + mri_brain_cols + mri_subcortical_cols + mri_tissue_cols + conversion_features
	dataframe = dataframe[lst]
	#dataframe.dropna(inplace=True)
	colors = np.where(dataframe['conversionmci']==1,'r','b')

	#dataframe.plot.scatter('bus_int_visita1', 'volume_bMNI_visita1',c=colors, title ='Busche Integral ~ Brain-std mm3',alpha=0.1)
	xvar, yvar = 'bus_int_visita1', 'volume_bMNI_visita1'
	title = 'Buschke Integral ~ Brain MNI (mm$^3$)'	
	plot_scatter_brain(dataframe, xvar, yvar, colors, title, 'bus-scatterMNI.png')
	xvar, yvar = 'bus_int_visita1', 'volume_bnative_visita1'
	title = 'Buschke Integral ~ Brain native (mm$^3$)'	
	plot_scatter_brain(dataframe, xvar, yvar, colors, title, 'bus-scatternative.png')

	xvar, yvar = 'bus_int_visita1', 'wm_volume_visita1'
	title = 'Buschke Integral ~ WM vol mm3'
	plot_scatter_brain(df_nooutliers_t, xvar, yvar, colors, title, 'bus-scatterWN.png')
	xvar, yvar = 'bus_int_visita1', 'csf_volume_visita1'
	title = 'Buschke Integral ~ CSF vol mm3'
	plot_scatter_brain(df_nooutliers_t, xvar, yvar, colors, title, 'bus-scatterCSF.png')
	xvar, yvar = 'bus_int_visita1', 'gm_volume_visita1'
	title = 'Buschke Integral ~ GM vol mm3'
	plot_scatter_brain(df_nooutliers_t, xvar, yvar, colors, title, 'bus-scatterGM.png')

	color3 = pd.cut(dataframe.scdgroups_visita1, [-0.1, 0, 1,2], labels=['b','g','r'])
	plot_scatter_brain(df_nooutliers_t, xvar, yvar, color3, title, 'bus-scatterGM-scd.png')

	xvar, yvar = 'R_Hipp_visita1', 'L_Hipp_visita1'
	title = 'R/L Hippocampus mm3'
	qua = df_nooutliers['bus_int_visita1'].quantile([0.25,0.5, 0.75])
	plot_scatter_brain(df_nooutliers, xvar, yvar, 'bus_int_visita1', title, 'bus-scatterHippo.png', 'copper')

	### pair plot
	# heat map of correlation betwween brain volumes and target features
	target = 'bus_int_visita1'
	plot_correlation_tissue(df_nooutliers_t, [target], mri_tissue_cols)
	plot_correlation_subcortical(dataframe, [target], mri_subcortical_cols)
	target = ['conversionmci', 'scdgroups_visita1']
	hipposystem = ['L_Hipp_visita1','R_Hipp_visita1', 'L_Amyg_visita1', 'R_Amyg_visita1','L_Accu_visita1','R_Accu_visita1'] + [target[1]]
	basal_system = ['R_Thal_visita1','L_Thal_visita1','L_Puta_visita1','R_Puta_visita1','L_Pall_visita1',\
	'R_Pall_visita1', 'L_Caud_visita1','R_Caud_visita1'] + [target[1]]

	#gh = sns.pairplot(df_nooutliers[hipposystem].dropna(), hue=target[1])
	#print df_nooutliers[hipposystem].dropna().corr()
	#plt.savefig('/Users/jaime/github/papers/EDA_pv/figures/basal-sys-pairplot-mci.png', dpi=240)
	gh = sns.pairplot(df_nooutliers[hipposystem].dropna(), hue=target[1])
	print df_nooutliers[hipposystem].dropna().corr()
	plt.savefig('/Users/jaime/github/papers/EDA_pv/figures/hipp-sys-pairplot-scd.png', dpi=240)
	gb = sns.pairplot(df_nooutliers[basal_system].dropna(), hue=target[1])
	plt.savefig('/Users/jaime/github/papers/EDA_pv/figures/basal-sys-pairplot-scd.png', dpi=240)

	# estiamte parameters y = betai xi xi =  sum + 12 + 23
	print('\n\n\n END!!!!')	 
if __name__ == "__name__":
	
	main()