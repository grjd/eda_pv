#######################################################
# Python program name	: 
#Description	: buschke_leastsquares.py
#Args           : Estimate the Buschke model parameters that best fit with MRI data  
#				 Plots  plt_boxplot_brain_volumes                                                                                     
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#######################################################
# -*- coding: utf-8 -*-
import os, sys, pdb, operator
import datetime
import time
import numpy as np
import pandas as pd
import importlib
import sys

sys.path.append('/Users/jaime/github/code/tensorflow/production')
import descriptive_stats as pv
#sys.path.append('/Users/jaime/github/papers/EDA_pv/code')
import warnings
from subprocess import check_output
import area_under_curve 
import matplotlib.pyplot as plt
import seaborn as sns


def compare_OLS_models(regs_list, model_names):
	"""
	"""
	from statsmodels.iolib.summary2 import summary_col
	info_dict={'R-squared' : lambda x: x.rsquared,'No. observations' : lambda x: int(x.nobs)}
	results_table = summary_col(results=regs_list,  float_format='%0.2f',stars = True,model_names=model_names,\
		info_dict=info_dict)

	results_table.add_title('Table 2 - OLS Regressions')
	print(results_table)



def fit_buschke_model(df, x, y):
	"""fit_buschke_model: builds regression model and solves it with OLS method
	Args:x list of independent variables , y the dependent variable (column names of df)
	Output: regression model
	"""
	
	import statsmodels.api as sm
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	# build the model
	# to estimate beta_0 we need a column of all 1s
	df['const'] = 1
	exoglist = ['const'] + x
	reg = sm.OLS(endog=df[y], exog=df[exoglist], missing='drop')
	#type(reg) == statsmodels.regression.linear_model.OLS	
	# fit the model to obtain the parameters
	res_ols = reg.fit()
	#type(results)==statsmodels.regression.linear_model.RegressionResultsWrapper
	print(res_ols.summary())
	betas = res_ols.params
	print('the model parameters are:', betas)
	# use the model to predict
	df_plot = df.dropna(subset=[y+x])
	#plot predcited values
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_subplot(111, xlabel='Buschke $\\int$', ylabel='Hippocampal size', title='OLS predicted values')
	if len(x) > 1:
		# for several dimensions plot just one
		x = x[1]
	ax.scatter(df_plot[x], res_ols.predict(), alpha=0.5, label='predicted')
	ax.scatter(df_plot[x], df_plot[y], alpha=0.5, label='observed')
	plt.savefig(os.path.join(figures_dir,'buschke-hippo-fit.png'), dpi=240)
	return res_ols


def compute_buschke_integral_df(dataframe, features_dict=None):
	""" compute_buchske_integral_df compute new Buschke 
	Args: dataframe with the columns fcsrtrl1_visita[1-7]
	Output:return the dataframe including the columns bus_visita[1-7]"""

	import scipy.stats.mstats as mstats
	print('Compute the Buschke aggregate \n')
	# Busche integral (aggreate is clculated with 3 values, totalk integral and partial integral 21 and 32)
	S = [0] * dataframe.shape[0]
	S21 = [0] * dataframe.shape[0]
	S32 = [0] * dataframe.shape[0]
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
			S21[ix]=bes[2]
			S32[ix]=bes[3]
			mean_a[ix] = np.mean(y)
			mean_g[ix] = mstats.gmean(y)
			suma[ix] = np.sum(y) 
			print('Total Aggregate S=', bes[0])
			print('Total sum Sum =', bes[1], ' partial Sum 10=',bes[2], ' Partial Sum 21=',bes[3])
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
		dataframe['bus_parint1_' + coda] = S21
		dataframe['bus_parint2_' + coda] = S32
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

def plot_histograma_tbi_categorical(df, target_variable=None):
	""" physical exercise
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by TBI \n')
	df['tce_total'] = df['tce'][df['tce']<9]*df['tce_num']
	bins = [-np.inf, 0, 1, 2, 3, np.inf]
	names = ['0','1','2', '3', '3+']
	df['tce_cut']= pd.cut(df['tce_total'], bins, labels=names)
	fig, ax = plt.subplots(1, figsize=(4,4))
	d = df.groupby([target_variable, 'tce_cut']).size().unstack(level=1)
	print d; d = d / d.sum(); print d
	p = d.plot(kind='bar', ax=ax, legend='nb of TBI')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_tbi_mci.png'), dpi=240)

def plot_histograma_cardiovascular_categorical(df, target_variable=None):
	"""plot_histograma_cardiovascular_categorical: 
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	this_function_name = sys._getframe(  ).f_code.co_name
	print('Calling to {}',format(this_function_name))
	list_cardio=['hta', 'hta_ini', 'glu', 'lipid', 'tabac', 'tabac_cant', 'tabac_fin', \
	'tabac_ini', 'sp', 'cor', 'cor_ini', 'arri', 'arri_ini', 'card', 'card_ini', 'tir', \
	'ictus', 'ictus_num', 'ictus_ini', 'ictus_secu']

	df['hta'] = df['hta'].astype('category').cat.rename_categories(['NoHypArt', 'HypArt'])
	df['glu'] = df['glu'].astype('category').cat.rename_categories(['NoGlu', 'DiabMel','Intoler.HydroC'])
	df['tabac'] = df['tabac'].astype('category').cat.rename_categories(['NoSmoker', 'Smoker', 'ExSomoker'])
	df['sp'] = df['sp'][df['sp']<9].astype('category').cat.rename_categories(['NoOW', 'OverWeight'])
	df['cor'] = df['cor'][df['cor']<9].astype('category').cat.rename_categories(['NoHeartPb', 'Angina', 'Infarction'])
	df['arri'] = df['arri'][df['arri']<9].astype('category').cat.rename_categories(['NoArri', 'FibrAur', 'Arrhythmia'])
	df['card'] = df['card'][df['card']<9].astype('category').cat.rename_categories(['NoCardDis', 'CardDis'])
	df['tir'] = df['ictus'][df['ictus']<9].astype('category').cat.rename_categories(['NoTyr', 'HiperTyr','HipoTir'])
	df['ictus'] = df['ictus'][df['ictus']<9].astype('category').cat.rename_categories(['NoIct', 'IschIct','HemoIct'])

	# in relative numbers
	fig, ax = plt.subplots(3,3)
	#ax[-1, -1].axis('off')
	fig.set_size_inches(12,12)
	#fig.suptitle('Conversion relative numbers for cardiovascular')
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
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'cor']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'arri']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,2])
	d = df.groupby([target_variable, 'card']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2,0])
	d = df.groupby([target_variable, 'tir']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2,1])
	d = df.groupby([target_variable, 'ictus']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[2,2])
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_cardio_mci.png'), dpi=240)
	# calling to physical ex 
	plot_histograma_physicalex_categorical(df, target_variable)
	# calling to TBI 
	plot_histograma_tbi_categorical(df, target_variable)

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
	"""
	"""
	print('Groupby by sleep\n')
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'

	this_function_name = sys._getframe(  ).f_code.co_name
	print('Calling to {}',format(this_function_name))

	list_sleep= ['sue_con', 'sue_dia', 'sue_hor', 'sue_man', 'sue_mov', 'sue_noc', 'sue_pro', \
	'sue_rec', 'sue_ron', 'sue_rui', 'sue_suf']
	df['sue_noc_cat'] = pd.cut(df['sue_noc'], 4) # hours of sleep night
	df['sue_dia_cat'] = pd.cut(df['sue_dia'],4) # hours of sleep day

	# relative
	fig, ax = plt.subplots(3,3)
	#ax[-1, -1].axis('off')
	fig.set_size_inches(12,12)
	#fig.suptitle('Conversion relative numbers for Sleep')
	
	bins = [-np.inf, 0, 2, 4, np.inf]
	names = ['0', '<2', '2-4','4+']
	df['sue_dia_r']= pd.cut(df['sue_dia'], bins, labels=names)
	bins = [-np.inf, 0, 2, 4, 8, 10, np.inf]
	names = ['0', '<2', '2-4','4-8', '8-10', '10+']
	df['sue_noc_r']= pd.cut(df['sue_noc'], bins, labels=names)
	df['sue_con_r'] = df['sue_con'].astype('category').cat.rename_categories(['Light', 'Moderate', 'Deep'])

	df['sue_suf_r'] = df['sue_suf'][df['sue_suf']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	df['sue_rec_r'] = df['sue_rec'][df['sue_rec']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	df['sue_mov_r'] = df['sue_mov'][df['sue_mov']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	df['sue_ron_r'] = df['sue_ron'][df['sue_ron']<9].astype('category').cat.rename_categories(['No', 'Yes', 'Snore&Breath'])
	df['sue_rui_r'] = df['sue_rui'][df['sue_rui']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	df['sue_hor_r'] = df['sue_hor'][df['sue_hor']<9].astype('category').cat.rename_categories(['No', 'Yes'])

	datag = df.groupby([target_variable, 'sue_dia_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,0])
	datag = df.groupby([target_variable, 'sue_noc_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,1])
	datag = df.groupby([target_variable, 'sue_con_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,2])
	datag= df.groupby([target_variable, 'sue_suf_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,0])
	datag = df.groupby([target_variable, 'sue_rec_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,1])
	datag = df.groupby([target_variable, 'sue_mov_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,2])
	datag = df.groupby([target_variable, 'sue_ron_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[2,0])
	datag = df.groupby([target_variable, 'sue_rui_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[2,1])
	datag = df.groupby([target_variable, 'sue_hor_r']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[2,2])
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_sleep_mci.png'), dpi=240)


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

def plot_histograma_physicalex_categorical(df, target_variable=None):
	""" physical exercise
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Physical exercise \n')
	df['phys_total'] = df['ejfre']*df['ejminut']
	bins = [-np.inf, 60, 120, 180, 240, 360, 420, np.inf]
	names = ['<60', '60-120','120-180', '180-240', '240-360', '360-420', '420+']
	df['phys_total_cut']= pd.cut(df['phys_total'], bins, labels=names)

	fig, ax = plt.subplots(1, figsize=(4,4))
	d = df.groupby([target_variable, 'phys_total_cut']).size().unstack(level=1)
	print d; d = d / d.sum(); print d
	p = d.plot(kind='bar', ax=ax)
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_physex_mci.png'), dpi=240)

def plot_histograma_engagement_categorical(df, target_variable=None):
	"""
	"""	
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	lista_engag_ext_w = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', \
	'a10', 'a11', 'a12', 'a13', 'a14']
	#physical activities 1, creative 2, go out with friends 3,travel 4, ngo 5,church 6, social club 7,
	# cine theater 8,sport 9, listen music 10, tv-radio (11), books (12), internet (13), manualidades(14)

	df['creative_cat'] = pd.cut(df['a02'] + df['a14'], 3) # creative manualidades
	df['creative_cat'] = df['creative_cat'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['friends_cat'] = df['a03'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['travel_cat'] = df['a04'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['ngo_cat'] = df['a05'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['church_cat'] = df['a06'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['club_cat'] = df['a07'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['movies_cat'] = df['a08'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['sports_cat'] = df['a09'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['music_cat'] = df['a10'][df['a10']>0].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['tv_cat'] = df['a11'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['books_cat'] = df['a12'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	df['internet_cat'] = df['a13'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])

 	# 
	fig, ax = plt.subplots(4,3)
	#ax[-1, -1].axis('off')
	fig.set_size_inches(15,10)
	#fig.suptitle('Conversion relative numbers for Engagement external world')
	datag = df.groupby([target_variable, 'creative_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,0])
	datag = df.groupby([target_variable, 'friends_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,1])
	datag = df.groupby([target_variable, 'travel_cat']).size().unstack(level=1)
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[0,2])
	datag = df.groupby([target_variable, 'ngo_cat']).size().unstack(level=1) # church goers
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,0])
	datag = df.groupby([target_variable, 'church_cat']).size().unstack(level=1) #read books
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,1])
	datag = df.groupby([target_variable, 'club_cat']).size().unstack(level=1) #listen to music
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[1,2])
 	datag = df.groupby([target_variable, 'movies_cat']).size().unstack(level=1) #read books
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[2,0])
	datag = df.groupby([target_variable, 'sports_cat']).size().unstack(level=1) #listen to music
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[2,1])
	datag = df.groupby([target_variable, 'music_cat']).size().unstack(level=1) #internet
 	datag = datag / datag.sum()
 	p = datag.plot(kind='bar', ax=ax[2,2])
 	datag = df.groupby([target_variable, 'tv_cat']).size().unstack(level=1) #read books
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[3,0])
	datag = df.groupby([target_variable, 'books_cat']).size().unstack(level=1) #listen to music
	datag = datag / datag.sum()
	p = datag.plot(kind='bar', ax=ax[3,1])
	datag = df.groupby([target_variable, 'internet_cat']).size().unstack(level=1) #internet
 	datag = datag / datag.sum()
 	p = datag.plot(kind='bar', ax=ax[3,2])
 	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_social_mci.png'), dpi=240)


def plot_histograma_demographics_categorical(df, target_variable=None):
	"""
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	d, p= pd.Series([]), pd.Series([])

	df['nivel_educativo'] = df['nivel_educativo'].astype('category').cat.rename_categories(['~Pr', 'Pr', 'Se', 'Su'])
	
	bins = [-np.inf, 0, 1, 2, 3, 4, np.inf]
	names = ['0', '1', '2','3', '4', '5+']
	df['numhij_r']= pd.cut(df['numhij'], bins, labels=names)

	bins = [-np.inf, 0, 10, 20, 30, np.inf]
	names = ['0', '<10', '10-20', '20-30', '30+']
	df['sdatrb_r']= pd.cut(df['sdatrb'], bins, labels=names)

	bins = [ 0, 3, 8, 10]
	names = ['Low', 'Med','Up']
	df['sdeconom_r']= pd.cut(df['sdeconom'], bins, labels=names)
	df['sdestciv'] = df['sdestciv'].astype('category').cat.rename_categories(['Single', 'Married', 'Widow', 'Divorced'])

	bins = [ 0, 2, 4, 6, np.inf]
	names = ['1', '2-4','4-6', '6+']
	dfno9 = df['sdvive'][df['sdvive'] <9]
	df['sdvive_r']= pd.cut(dfno9, bins, labels=names).dropna()
	
	df['nivelrenta'] = df['nivelrenta'].astype('category').cat.rename_categories(['Low', 'Med', 'High'])
	
	#in absolute numbers
	#in relative numbers
	fig, ax = plt.subplots(3,2)
	fig.set_size_inches(14,12)
	#fig.suptitle('Conversion conditioned by various demographics')

	d = df.groupby([target_variable, 'nivel_educativo']).size().unstack(level=1)
	d = d / d.sum();print(d)
	p = d.plot(kind='bar', ax=ax[0,0])
	
	d = df.groupby([target_variable, 'numhij_r']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'sdatrb_r']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'sdeconom_r']).size().unstack(level=1)
	d = d / d.sum();print(d)
	p = d.plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'sdestciv']).size().unstack(level=1)
	print(d)
	d = d / d.sum();print(d)
	p = d.plot(kind='bar', ax=ax[2,0])
	d = df.groupby([target_variable, 'sdvive_r']).size().unstack(level=1)
	print(d)
	d = d / d.sum();print(d)
	p = d.plot(kind='bar', ax=ax[2,1])
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_Demosmci.png'), dpi=240)

def plot_histograma_genetics(df, target_variable=None):
	#in absolute numbers
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Genetics\n')
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	#abs numbers
	d = df.groupby([target_variable, 'apoe']).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0], title='APOE4-conversion Absolute')
	#in relative numbers
	d = df.groupby([target_variable, 'apoe']).size().unstack(level=1)
	d = d / d.sum()
	p = d.plot(kind='bar', ax=ax[1],title='APOE4-conversion Relative')
	print('**** a.rel *****'); print(d)
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_mci_genetics.png'), dpi=240)
	
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	#abs numbers
	d = df.groupby(['apoe', target_variable]).size()
	p = d.unstack(level=1).plot(kind='bar', ax=ax[0], title='*APOE4-conversion Absolute')
	#in relative numbers
	d = df.groupby(['apoe', target_variable]).size().unstack(level=1)
	d = d / d.sum()
	print('**** b.rel *****\n'); print(d)
	p = d.plot(kind='bar', ax=ax[1],title='*APOE4-conversion Relative')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_genetics_mci.png'), dpi=240)
	
	# X-conversion Y familal 
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	df_af = df.groupby([target_variable, 'familial_ad']).size()
	p_af = df_af.unstack(level=1).plot(kind='bar', ax=ax[0], title='Familial AD ~ conversion Absolute')
	df_af = df.groupby([target_variable,'familial_ad']).size().unstack(level=1)
	df_af = df_af / df_af.sum()
	print('**** c.relY *****\n'); print(df_af)
	p_af = df_af.plot(kind='bar', ax=ax[1], title='Familial AD ~ conversion Relative')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_Familial_mciY.png'), dpi=240)

	# X -familal y-conversion
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	df_af = df.groupby(['familial_ad', target_variable]).size()
	p_af = df_af.unstack(level=1).plot(kind='bar', ax=ax[0], title='Familial AD ~ conversion Absolute')
	df_af = df.groupby(['familial_ad', target_variable]).size().unstack(level=1)
	df_af = df_af / df_af.sum()
	print('**** c.rel *****\n'); print(df_af)
	p_af = df_af.plot(kind='bar', ax=ax[1], title='Familial AD ~ conversion Relative')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_Familial_mci.png'), dpi=240)
	
	# group with familial and APOE
	fig, ax = plt.subplots(1, 2, figsize=(8, 6))
	df_af =  df.groupby(['familial_ad', 'conversionmci', 'apoe'])['id'].count()
	df_af.unstack(level=1).plot(kind='bar', ax=ax[0], title='APOE4 & Familial AD ~ conversion Absolute')
	print(df_af)
	df_af =  df.groupby(['familial_ad', 'conversionmci', 'apoe'])['id'].count().unstack(level=1)
	print('**** d.abs *****\n');print(df_af)
	df_af = df_af / df_af.sum()
	df_af.plot(kind='bar', ax=ax[1], title='APOE4 & Familial AD ~ conversion Relative')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_path, 'groupby_geneticsFamilial_mci.png'), dpi=240)
	print('**** d.rel *****\n');print(df_af)
	pdb.set_trace()
	#plt.show()

def plot_histograma_food_categorical(df, target_variable=None):
	"""
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Diet \n')
	df['alfrut_cut'] = df['alfrut'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	df['alcar_cut'] = df['alcar'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	df['aldulc_cut'] = df['aldulc'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
	df['alverd_cut'] = df['alverd'].astype('category').cat.rename_categories(['0', '1-2', '3-5','6-7'])
		#diet in relative numbers
	fig, ax = plt.subplots(2,2)
	fig.set_size_inches(8,8)
	#fig.suptitle('Conversion by relative numbers for Alimentation')
	d = df.groupby([target_variable, 'alfrut_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'alverd_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'alcar_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'aldulc_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,1])
	plt.tight_layout()
	#plt.savefig(os.path.join(figures_path, 'groupby_diet_mci.png'), dpi=240)
	plt.savefig(os.path.join(figures_path, 'groupby_food_mci.png'), dpi=240)


def plot_histograma_diet_categorical(df, target_variable=None):
	""" plot_histograma_diet_categorical
	Args: datafame, target_variable
	Output:None
	"""

	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Diet \n')
	# 4 groups:: 'dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable'
	nb_of_categories = 4
	# cut in 4 groups based on range
	df['dietaglucemica_cut'] = pd.cut(df['dietaglucemica'],nb_of_categories)
	df['dietasaludable_cut']= pd.cut(df['dietasaludable'],nb_of_categories)
	df['dietaproteica_cut']= pd.cut(df['dietaproteica'],nb_of_categories)
	df['dietagrasa_cut']= pd.cut(df['dietagrasa'],nb_of_categories)
	#cut in 4 quartiles
	df['dietaglucemica_cut'] = pd.qcut(df['dietaglucemica'], nb_of_categories, precision=1) 
	df['dietasaludable_cut'] = pd.qcut(df['dietasaludable'], nb_of_categories, precision=1) 
	df['dietaproteica_cut'] = pd.qcut(df['dietaproteica'], nb_of_categories, precision=1) 
	df['dietaproteica_cut'] = pd.qcut(df['dietaproteica'], nb_of_categories, precision=1) 
	#diet in relative numbers
	fig, ax = plt.subplots(2,2)
	fig.set_size_inches(8,8)
	#fig.suptitle('Conversion by relative numbers for Alimentation')
	d = df.groupby([target_variable, 'dietaglucemica_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'dietasaludable_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'dietaproteica_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'dietagrasa_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,1])
	plt.tight_layout()
	#plt.savefig(os.path.join(figures_path, 'groupby_diet_mci.png'), dpi=240)
	plt.savefig(os.path.join(figures_path, 'groupby_diet_mciQuart.png'), dpi=240)
	print('Calling to food...\n)')
	plot_histograma_food_categorical(df, target_variable)

def plot_histograma_T1_categorical(df, target_variable):
	""" group by conversion wioth subcortical structure size
	"""
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures'
	print('Groupby by Subcortical structure size \n')
	nb_of_categories = 4
	#cut in 4 quartiles

	df['L_Thal_visita1_cut'] = pd.qcut(df['L_Thal_visita1'].dropna(), nb_of_categories, precision=1)
	df['R_Thal_visita1_cut'] = pd.qcut(df['R_Thal_visita1'].dropna(), nb_of_categories, precision=1) 
	df['L_Puta_visita1_cut'] = pd.qcut(df['L_Puta_visita1'].dropna(), nb_of_categories, precision=1)  
	df['R_Puta_visita1_cut'] = pd.qcut(df['R_Puta_visita1'].dropna(), nb_of_categories, precision=1)  
	df['L_Caud_visita1_cut'] = pd.qcut(df['L_Caud_visita1'].dropna(), nb_of_categories, precision=1)
	df['R_Caud_visita1_cut'] = pd.qcut(df['R_Caud_visita1'].dropna(), nb_of_categories, precision=1) 

	df['L_Pall_visita1_cut'] = pd.qcut(df['L_Pall_visita1'].dropna(), nb_of_categories, precision=1)  
	df['R_Pall_visita1_cut'] = pd.qcut(df['R_Pall_visita1'].dropna(), nb_of_categories, precision=1) 

	df['L_Hipp_visita1_cut'] = pd.qcut(df['L_Hipp_visita1'].dropna(), nb_of_categories, precision=1)  
	df['R_Hipp_visita1_cut'] = pd.qcut(df['R_Hipp_visita1'].dropna(), nb_of_categories, precision=1)  
	df['L_Amyg_visita1_cut'] = pd.qcut(df['L_Amyg_visita1'].dropna(), nb_of_categories, precision=1)  
	df['R_Amyg_visita1_cut'] = pd.qcut(df['R_Amyg_visita1'].dropna(), nb_of_categories, precision=1)  
	df['L_Accu_visita1_cut'] = pd.qcut(df['L_Accu_visita1'].dropna(), nb_of_categories, precision=1)  
	df['R_Accu_visita1_cut'] = pd.qcut(df['R_Accu_visita1'].dropna(), nb_of_categories, precision=1)  

	fig, ax = plt.subplots(3,2)
	fig.set_size_inches(10,12)
	#fig.suptitle('Conversion by relative numbers for Alimentation')
	d = df.groupby([target_variable, 'L_Hipp_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'R_Hipp_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'L_Amyg_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'R_Amyg_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,1])
	d = df.groupby([target_variable, 'L_Accu_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[2,0])
	d = df.groupby([target_variable, 'R_Accu_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[2,1])
	plt.tight_layout()
	#plt.savefig(os.path.join(figures_path, 'groupby_diet_mci.png'), dpi=240)
	plt.savefig(os.path.join(figures_path, 'groupby_HipAmyAcc_mciQuart.png'), dpi=240)


	fig, ax = plt.subplots(4,2)
	fig.set_size_inches(10,12)
	d = df.groupby([target_variable, 'L_Puta_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,0])
	d = df.groupby([target_variable, 'R_Puta_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[0,1])
	d = df.groupby([target_variable, 'L_Thal_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,0])
	d = df.groupby([target_variable, 'R_Thal_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[1,1])

	d = df.groupby([target_variable, 'L_Caud_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[2,0])
	d = df.groupby([target_variable, 'R_Caud_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[2,1])
	d = df.groupby([target_variable, 'L_Pall_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[3,0])
	d = df.groupby([target_variable, 'R_Pall_visita1_cut']).size().unstack(level=1)
	print(d); d = d / d.sum(); print(d)
	p = d.plot(kind='bar', ax=ax[3,1])
	plt.tight_layout()
	#plt.savefig(os.path.join(figures_path, 'groupby_diet_mci.png'), dpi=240)
	plt.savefig(os.path.join(figures_path, 'groupby_ThalCau_mciQuart.png'), dpi=240)
	#pdb.set_trace()


def plot_histograma_bygroup_categorical(df, type_of_group, target_variable):
	""" plot_histograma_bygroup_categorical
	"""
	if type_of_group is 'Genetics_s': plot_histograma_genetics(df, target_variable)
	if type_of_group is 'Demographics_s': plot_histograma_demographics_categorical(df, target_variable)
	if type_of_group is 'Diet_s': plot_histograma_diet_categorical(df, target_variable) #call to food from within
	if type_of_group is 'EngagementExternalWorld_s': plot_histograma_engagement_categorical(df, target_variable)
	if type_of_group is 'Anthropometric_s': plot_histograma_anthropometric_categorical(df, target_variable)
	if type_of_group is 'Sleep_s': plot_histograma_sleep_categorical(df, target_variable)
	if type_of_group is 'PsychiatricHistory_s': plot_histograma_psychiatrichistory_categorical(df, target_variable)
	if type_of_group is 'Cardiovascular_s': plot_histograma_cardiovascular_categorical(df, target_variable) #call to physical within 


def QQplot(df, feature):
	"""Q-Q plot of the quantiles of x versus the quantiles/ppf of a distribution
	cdf: actual cdf
	fit: model CDF
	"""
	import statsmodels.api as sm
	import scipy.stats as stats

	#mod_fit = sm.OLS(df[feature].dropna(), df[feature].dropna()).fit()
	#res = mod_fit.resid # residuals
	res = df[feature].dropna()
	#fig = sm.qqplot(res)
	#determine parameters for t distribution including the loc and scale:
	#fig = sm.qqplot(res, stats.norm, fit=True, line='45')
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	#fit=True then the parameters for dist are fit automatically using dist.fit
	#fir=False loc, scale, and distargs are passed to the distribution
	graph = sm.graphics.qqplot(res, dist=stats.norm, line='45', fit=True, ax=ax)
	#ax.set_xlim(-6, 6)
	top = ax.get_ylim()[1] * 0.75
	left = -1.8
	txt = ax.text(left, top, "stats.norm, \nline='45', \nfit=True",
              verticalalignment='top')
	txt.set_bbox(dict(facecolor='k', alpha=0.1))
	fig.tight_layout()
	plt.gcf()
	#fig = sm.qqplot(df[feature].dropna(), stats.t, fit=True, line='45')


def Anova1way(df, feature):
	"""
	"""
	import scipy.stats as stats
	import time
	y1, yN = 1,5
	# get indices from visita1+i and get values from 1
	vislabel_v1 = feature
	dictio = {feature+'_visita1': [], feature+'_visita2': [], feature+'_visita3': [], \
	feature+'_visita4': [], feature+'_visita5': []}
	df_oneway = pd.DataFrame(data=dictio)
	df_oneway[feature+'_visita1'] = df[feature].dropna()
	#dx_corto_visitaJ notnan

	for ix in np.arange(2,yN+1):
		x = df['dx_corto_visita' + str(ix)]
		#get vales in visita1 of those indices
		indices = np.argwhere(~np.isnan(x))
		indices = indices.ravel()
		vislabel = feature + '_visita' + str(ix)
		df_oneway[vislabel] = df[vislabel_v1][indices].dropna()
	#ANOVA one way for 	visitasL

	start = time.time()
	F, p = stats.f_oneway(df_oneway[dictio.keys()[0]].dropna(),df_oneway[dictio.keys()[1]].dropna(), \
		df_oneway[dictio.keys()[2]].dropna(),df_oneway[dictio.keys()[3]].dropna(),\
		df_oneway[dictio.keys()[4]].dropna())
	end = time.time()
	print('ANOVA for:', feature, ' run in :', end - start, ' secs')
	print('\n\n F=',F, ' p=', p)


def anova_test_2groups(dataframe, ed, ing):
	"""
	anova_tests_paper: anova_tests_paper with statsmodel , this is better than with sciopy.stats.f_oneway
	#the egression (GLM) mode used in statsmodel does very easily for you the post-hoc, that is, of there is a difference
	where is that difference eg we foan effect in smoking and conversion, but were is it, in the snoker or the nonsmoker or in the exsmoker?
	Args:ed, ing laberls, ed the factor that affect to ing, eg ed=smoke and ing conversionmci
	Output
	"""
	import scipy.stats as stats
	import statsmodels.api as sm
	from statsmodels.formula.api import ols
	from statsmodels.stats.multicomp import pairwise_tukeyhsd
	from statsmodels.stats.multicomp import MultiComparison
	#dataframe[ing].replace({0: 'no', 1: 'hete', 2: 'homo'}, inplace= True)
	#olsmodelstr = ing + ' ~ ' + 'C('+ed+')'
	olsmodelstr = ing + ' ~ ' + ed
	print olsmodelstr
	mod = ols(olsmodelstr, data=dataframe).fit()
	print mod.summary()
	aov_table = sm.stats.anova_lm(mod, typ=2)
	# aov_table isa datafarme aov_table.columns=Index([u'sum_sq', u'df', u'F', u'PR(>F)'], dtype='object')
	print aov_table
	#no effect sizes is calculated when we use Statsmodels.  
	#To calculate eta squared we can use the sum of squares from the table
	aov_table['mean_sq'] = aov_table[:]['sum_sq']/aov_table[:]['df']
	esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
	aov_table['eta_sq'] = aov_table[:-1]['sum_sq']/sum(aov_table['sum_sq'])
	aov_table['omega_sq'] = (aov_table[:-1]['sum_sq']-(aov_table[:-1]['df']*aov_table['mean_sq'][-1]))/(sum(aov_table['sum_sq'])+aov_table['mean_sq'][-1])
	print aov_table

	#Tukeys HSD Post-hoc comparison:ukey HSD post-hoc comparison test controls for type I error 
	#and maintains the familywise error rate at 0.05.
	#the reject column states whether or not the null hypothesis should be rejected
	mc = MultiComparison(dataframe[ing].dropna(), dataframe[ed])
	mc_results = mc.tukeyhsd()
	print(mc_results)
	# qq plot of the linear model fit with ols
	res = mod.resid 
	fig = sm.qqplot(res, line='s')
	return  aov_table
	
def anova_tests_paper(dataframe_conv, features_dict):
	"""
	"""
	features_dict 
	list_clusters, list_features =  features_dict.keys(), features_dict.values()
	static_topics = filter(lambda k: '_s' in k, list_clusters)
	longitudinal_topics = [a for a in list_clusters if (a not in static_topics)]
	# genetic factor
	genlist = [features_dict['Genetics_s'][0]]+ ['familial_ad']
	
	aov_table_apo = anova_test_2groups(dataframe_conv, genlist[0], 'conversionmci')
	aov_table_fam = anova_test_2groups(dataframe_conv, genlist[-1], 'conversionmci')
	df_anova_gen = pd.concat([aov_table_apo, aov_table_fam])
	# anthropometric ['lat_manual', 'pabd', 'peso', 'talla', 'audi', 'visu', 'imc'] bmi, abdo, weight, height
	antlist = features_dict['Anthropometric_s']

	aov_table_lat = anova_test_2groups(dataframe_conv, 'lat_manual', 'conversionmci')
	# Continuous use ANCOVA
	#aov_table_bmi = anova_test_2groups(dataframe_conv, 'imc', 'conversionmci')
	#aov_table_wei = anova_test_2groups(dataframe_conv, 'peso', 'conversionmci')
	#aov_table_hei = anova_test_2groups(dataframe_conv, 'talla', 'conversionmci')
	#aov_table_pab = anova_test_2groups(dataframe_conv, 'pabd', 'conversionmci')
	#df_anova_ant = pd.concat([aov_table_lat,aov_table_bmi,aov_table_wei,aov_table_hei,aov_table_pab])
	
	#Demographics_s
	#['renta', 'nivelrenta', 'educrenta', 'municipio', 'barrio', 'distrito', 'sexo', 'nivel_educativo', 'anos_escolaridad', 'familial_ad', 'sdestciv', 'sdhijos', 'numhij', 'sdvive', 'sdocupac', 'sdresid', 'sdtrabaja', 'sdeconom', 'sdatrb']
	aov_table_ren = anova_test_2groups(dataframe_conv, 'nivelrenta', 'conversionmci')	
	aov_table_sex = anova_test_2groups(dataframe_conv, 'sexo', 'conversionmci')	
	aov_table_edu = anova_test_2groups(dataframe_conv, 'nivel_educativo', 'conversionmci')
	aov_table_civ = anova_test_2groups(dataframe_conv, 'sdestciv', 'conversionmci')
	df_anova_demo = pd.concat([aov_table_ren, aov_table_sex, aov_table_edu, aov_table_civ])
	
	# Cardiovascular_s
	#['hta', 'hta_ini', 'glu', 'lipid', 'tabac', 'tabac_cant', 'tabac_fin', 'tabac_ini', 'sp', 'cor', 'cor_ini', 'arri', 'arri_ini', 'card', 'card_ini', 'tir', 'ictus', 'ictus_num', 'ictus_ini', 'ictus_secu']
	aov_table_hta = anova_test_2groups(dataframe_conv, 'hta', 'conversionmci')
	aov_table_glu =  anova_test_2groups(dataframe_conv, 'glu', 'conversionmci')
	aov_table_tabac = anova_test_2groups(dataframe_conv, 'tabac', 'conversionmci')
	aov_table_sp = anova_test_2groups(dataframe_conv, 'sp', 'conversionmci')
	aov_table_cor = anova_test_2groups(dataframe_conv, 'cor', 'conversionmci')
	aov_table_arri = anova_test_2groups(dataframe_conv, 'arri', 'conversionmci')
	aov_table_card = anova_test_2groups(dataframe_conv, 'card', 'conversionmci')
	aov_table_tir = anova_test_2groups(dataframe_conv, 'tir', 'conversionmci')
	aov_table_ictus = anova_test_2groups(dataframe_conv, 'ictus', 'conversionmci')
	df_anova_demo = pd.concat([aov_table_hta, aov_table_glu, aov_table_tabac, aov_table_sp,\
		aov_table_cor,aov_table_arri])
	# PhysicalExercise_s
	dataframe_conv['phys_total'] = dataframe_conv['ejfre']*dataframe_conv['ejminut']
	bins = [-np.inf, 60, 120, 180, 240, 360, 420, np.inf]
	names = ['<60', '60-120','120-180', '180-240', '240-360', '360-420', '420+']
	dataframe_conv['phys_total_cut']= pd.cut(dataframe_conv['phys_total'], bins, labels=names)
	aov_table_phys = anova_test_2groups(dataframe_conv, 'phys_total_cut', 'conversionmci')
	df_anova_phys = pd.concat([aov_table_phys])
	
	#PsychiatricHistory_s
	dataframe_conv['depre_num_cat'] = pd.cut(dataframe_conv['depre_num']*dataframe_conv['depre'],4)
	dataframe_conv['ansi_num_cat'] = pd.cut(dataframe_conv['ansi_num'],4)
	aov_table_dep = anova_test_2groups(dataframe_conv, 'depre_num_cat', 'conversionmci')	
	aov_table_ansi = anova_test_2groups(dataframe_conv, 'ansi_num_cat', 'conversionmci')
	df_anova_psy = pd.concat([aov_table_dep, aov_table_glu, aov_table_ansi])	
	#Diet_s
	#cut in 4 quartiles
	nb_of_categories = 4
	dataframe_conv['dietaglucemica_cut'] = pd.qcut(dataframe_conv['dietaglucemica'], nb_of_categories, labels=["Q1", "Q2", "Q3", "Q4"], precision=1) 
	dataframe_conv['dietasaludable_cut'] = pd.qcut(dataframe_conv['dietasaludable'], nb_of_categories, labels=["Q1", "Q2", "Q3", "Q4"], precision=1) 
	dataframe_conv['dietaproteica_cut'] = pd.qcut(dataframe_conv['dietaproteica'], nb_of_categories, labels=["Q1", "Q2", "Q3", "Q4"], precision=1) 
	aov_table_dglu = anova_test_2groups(dataframe_conv, 'dietaglucemica_cut', 'conversionmci')	
	aov_table_dsal = anova_test_2groups(dataframe_conv, 'dietasaludable_cut', 'conversionmci')	
	aov_table_dpro = anova_test_2groups(dataframe_conv, 'dietaproteica_cut', 'conversionmci')
	df_anova_diet = pd.concat([aov_table_dglu, aov_table_dsal, aov_table_dpro])
	#SocialEngagement_s
	dataframe_conv['creative_cat'] = pd.cut(dataframe_conv['a02'] + dataframe_conv['a14'], 3) # creative manualidades
	dataframe_conv['creative_cat'] = dataframe_conv['creative_cat'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['friends_cat'] = dataframe_conv['a03'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['travel_cat'] = dataframe_conv['a04'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['ngo_cat'] = dataframe_conv['a05'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['church_cat'] = dataframe_conv['a06'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['club_cat'] = dataframe_conv['a07'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['movies_cat'] = dataframe_conv['a08'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['sports_cat'] = dataframe_conv['a09'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['music_cat'] = dataframe_conv['a10'][dataframe_conv['a10']>0].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['tv_cat'] = dataframe_conv['a11'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['books_cat'] = dataframe_conv['a12'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	dataframe_conv['internet_cat'] = dataframe_conv['a13'].astype('category').cat.rename_categories(['Never', 'Few', 'Often'])
	
	aov_table_crea = anova_test_2groups(dataframe_conv, 'creative_cat', 'conversionmci')	
	aov_table_fri = anova_test_2groups(dataframe_conv, 'friends_cat', 'conversionmci')	
	aov_table_trav = anova_test_2groups(dataframe_conv, 'travel_cat', 'conversionmci')
	aov_table_ngo = anova_test_2groups(dataframe_conv, 'ngo_cat', 'conversionmci')	
	aov_table_chu = anova_test_2groups(dataframe_conv, 'church_cat', 'conversionmci')	
	aov_table_clu = anova_test_2groups(dataframe_conv, 'club_cat', 'conversionmci')
	aov_table_mov = anova_test_2groups(dataframe_conv, 'movies_cat', 'conversionmci')	
	aov_table_spo = anova_test_2groups(dataframe_conv, 'sports_cat', 'conversionmci')	
	aov_table_mus = anova_test_2groups(dataframe_conv, 'music_cat', 'conversionmci')
	aov_table_tv = anova_test_2groups(dataframe_conv, 'tv_cat', 'conversionmci')	
	aov_table_book = anova_test_2groups(dataframe_conv, 'books_cat', 'conversionmci')	
	aov_table_int = anova_test_2groups(dataframe_conv, 'internet_cat', 'conversionmci')
	df_anova_engage = pd.concat([aov_table_crea, aov_table_fri, aov_table_trav, aov_table_ngo,\
		aov_table_chu, aov_table_clu, aov_table_mov,aov_table_spo,aov_table_mus,aov_table_tv,\
		aov_table_book, aov_table_int])
	print df_anova_engage

	#TraumaticBrainInjury_s
	dataframe_conv['tce_total'] = dataframe_conv['tce'][dataframe_conv['tce']<9]*dataframe_conv['tce_num']
	bins = [-np.inf, 0, 1, 2, 3, np.inf]
	names = ['0','1','2', '3', '3+']
	dataframe_conv['tce_cut']= pd.cut(dataframe_conv['tce_total'], bins, labels=names)
	aov_table_tbi = anova_test_2groups(dataframe_conv, 'tce_total', 'conversionmci')
	df_anova_engage = pd.concat([aov_table_tbi])
	#EngagementExternalWorld_s

	#Sleep
	bins = [-np.inf, 0, 2, 4, np.inf]
	names = ['0', '<2', '2-4','4+']
	dataframe_conv['sue_dia_r']= pd.cut(dataframe_conv['sue_dia'], bins, labels=names)
	bins = [-np.inf, 0, 2, 4, 8, 10, np.inf]
	names = ['0', '<2', '2-4','4-8', '8-10', '10+']
	dataframe_conv['sue_noc_r']= pd.cut(dataframe_conv['sue_noc'], bins, labels=names)
	dataframe_conv['sue_con_r'] = dataframe_conv['sue_con'].astype('category').cat.rename_categories(['Light', 'Moderate', 'Deep'])
	dataframe_conv['sue_suf_r'] = dataframe_conv['sue_suf'][dataframe_conv['sue_suf']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	dataframe_conv['sue_rec_r'] = dataframe_conv['sue_rec'][dataframe_conv['sue_rec']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	dataframe_conv['sue_mov_r'] = dataframe_conv['sue_mov'][dataframe_conv['sue_mov']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	dataframe_conv['sue_ron_r'] = dataframe_conv['sue_ron'][dataframe_conv['sue_ron']<9].astype('category').cat.rename_categories(['No', 'Yes', 'Snore&Breath'])
	dataframe_conv['sue_rui_r'] = dataframe_conv['sue_rui'][dataframe_conv['sue_rui']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	dataframe_conv['sue_hor_r'] = dataframe_conv['sue_hor'][dataframe_conv['sue_hor']<9].astype('category').cat.rename_categories(['No', 'Yes'])
	aov_table_suenoc = anova_test_2groups(dataframe_conv, 'sue_noc_r', 'conversionmci')	
	aov_table_suecon = anova_test_2groups(dataframe_conv, 'sue_con_r', 'conversionmci')	
	aov_table_suesuf = anova_test_2groups(dataframe_conv, 'sue_suf_r', 'conversionmci')
	aov_table_suerec = anova_test_2groups(dataframe_conv, 'sue_rec_r', 'conversionmci')	
	aov_table_suemov = anova_test_2groups(dataframe_conv, 'sue_mov_r', 'conversionmci')	
	aov_table_sueron = anova_test_2groups(dataframe_conv, 'sue_ron_r', 'conversionmci')	
	aov_table_suerui = anova_test_2groups(dataframe_conv, 'sue_rui_r', 'conversionmci')
	aov_table_suehor = anova_test_2groups(dataframe_conv, 'sue_hor_r', 'conversionmci')
	df_anova_sleep = pd.concat([aov_table_suenoc, aov_table_suecon, aov_table_suesuf, aov_table_suerec,\
		aov_table_suemov,aov_table_sueron,aov_table_suerui,aov_table_suehor ])
	print df_anova_sleep
	print aov_table_suenoc
	return

def main():
	# Open csv with MRI data
	plt.close('all')
	csv_path = '/Users/jaime/Downloads/test_code/PV7years_T1vols.csv'
	csv_path = '~/vallecas/data/BBDD_vallecas/PVDB_pve_sub.csv'
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures/'
	dataframe = pd.read_csv(csv_path)
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	print('Build dictionary with features ontology and check the features are in the dataframe\n') 
	features_dict = pv.vallecas_features_dictionary(dataframe)



	# only study rows with conversionmci to some value
	dataframe_conv = dataframe[dataframe['conversionmci'].notnull()]
	
	### Normnality test
	QQplot(dataframe_conv, 'a13')

	Anova1way(dataframe_conv, 'a13')
	pdb.set_trace()	
	### ANOVA ttest
	anova_tests_paper(dataframe_conv, features_dict)
	pdb.set_trace()

	dataframe = compute_buschke_integral_df(dataframe)
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
	'bus_sum_visita6', 'bus_int_visita7', 'bus_sum_visita7', 'bus_parint1_visita1','bus_parint1_visita2','bus_parint1_visita3',\
	'bus_parint1_visita4', 'bus_parint1_visita5','bus_parint1_visita6','bus_parint1_visita7', 'bus_parint2_visita1', 'bus_parint2_visita2',\
	'bus_parint2_visita3','bus_parint2_visita4', 'bus_parint2_visita5','bus_parint2_visita6','bus_parint2_visita7'] 
	
	buschke_features = ['bus_int_visita1', 'bus_sum_visita1','bus_parint1_visita1', 'bus_parint2_visita1']
	#buschke_features = ['bus_int_visita1']
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
	



	###########
	lst2plotgroup = mri_subcortical_cols + ['conversionmci']
	plot_histograma_T1_categorical(df_nooutliers[lst2plotgroup], 'conversionmci')
	pdb.set_trace()


	# Build the model to fit the hippocampal volume
	y = ['L_Hipp_visita1']
	x = ['bus_int_visita1']
	res_ols1 = fit_buschke_model(df_nooutliers, x, y)
	x = buschke_features[1:]
	res_ols2 = fit_buschke_model(df_nooutliers, x, y)
	compare_OLS_models([res_ols1,res_ols2], ['int', 'int and partial'])

	### Correlation GroupBy analysis
	# group by static variables
	types_of_groups = ['Cardiovascular_s', 'EngagementExternalWorld_s','Diet_s', 'Sleep_s','Demographics_s', 'Anthropometric_s', 'Genetics_s', \
	'PsychiatricHistory_s'] #,'TraumaticBrainInjury_s'] 
	target_variable = 'conversionmci'
	for type_of_group in types_of_groups:
		plot_histograma_bygroup_categorical(dataframe, type_of_group, target_variable)
	# group by brain structure size	



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