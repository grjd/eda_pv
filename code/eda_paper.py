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

#sys.path.append('/Users/jaime/github/code/tensorflow/production')
#import descriptive_stats as pv
#sys.path.append('/Users/jaime/github/papers/EDA_pv/code')
import warnings
from subprocess import check_output
#import area_under_curve 
import matplotlib.pyplot as plt
import seaborn as sns

def vallecas_features_dictionary(dataframe):
	"""vallecas_features_dictionary: builds a dictionary with the feature clusters of PV
	NOTE: hardcoded y1 yo year 6, same function in descriptive_stats.py to year7. 
	YS: do a btter version of this function not hardcoded, with number of years as option
	Args: None
	Output: cluster_dict tpe is dict
	""" 
	cluster_dict = {'Demographics':['edad_visita1','edad_visita2', 'edad_visita3', 'edad_visita4', 'edad_visita5', \
	'edad_visita6', 'edad_visita7', 'edadinicio_visita1', 'edadinicio_visita2', 'edadinicio_visita3',\
	'edadinicio_visita4', 'edadinicio_visita5', 'edadinicio_visita6'],'Demographics_s':\
	['renta','nivelrenta','educrenta', 'municipio', 'barrio','distrito','sexo','nivel_educativo',\
	'anos_escolaridad','familial_ad','sdestciv','sdhijos', 'numhij','sdvive','sdocupac', 'sdresid', \
	'sdtrabaja','sdeconom','sdatrb'],'SCD':['scd_visita1', \
	'scd_visita2', 'scd_visita3', 'scd_visita4', 'scd_visita5', 'scd_visita6', \
	'scdgroups_visita1', 'scdgroups_visita2', 'scdgroups_visita3', 'scdgroups_visita4', \
	'scdgroups_visita5', 'scdgroups_visita6', 'peorotros_visita1', \
	'peorotros_visita2', 'peorotros_visita3', 'peorotros_visita4', 'peorotros_visita5', \
	'peorotros_visita6', 'preocupacion_visita1', 'preocupacion_visita2',\
	'preocupacion_visita3', 'preocupacion_visita4', 'preocupacion_visita5', 'preocupacion_visita6',\
	'eqm06_visita1', 'eqm06_visita2', 'eqm06_visita3', 'eqm06_visita4', \
	'eqm06_visita5', 'eqm06_visita6', 'eqm07_visita1', 'eqm07_visita2', \
	'eqm07_visita3', 'eqm07_visita4', 'eqm07_visita5','eqm81_visita1', 'eqm81_visita2', \
	'eqm81_visita3', 'eqm81_visita4', 'eqm81_visita5', 'eqm82_visita1', 'eqm82_visita2', \
	'eqm82_visita3', 'eqm82_visita4', 'eqm82_visita5', 'eqm83_visita1', 'eqm83_visita2', \
	'eqm83_visita3', 'eqm83_visita4', 'eqm83_visita5', 'eqm84_visita1', 'eqm84_visita2', \
	'eqm84_visita3', 'eqm84_visita4', 'eqm84_visita5', 'eqm85_visita1', 'eqm85_visita2', \
	'eqm85_visita3', 'eqm85_visita4', 'eqm85_visita5', 'eqm86_visita1', 'eqm86_visita2', \
	'eqm86_visita3', 'eqm86_visita4', 'eqm86_visita5','eqm09_visita1', 'eqm09_visita2', \
	'eqm09_visita3', 'eqm09_visita4', 'eqm09_visita5', 'eqm10_visita1', 'eqm10_visita2',\
	'eqm10_visita3', 'eqm10_visita4', 'eqm10_visita5', 'eqm10_visita6', \
	'act_aten_visita1', 'act_aten_visita2', 'act_aten_visita3', 'act_aten_visita4', \
	'act_aten_visita5', 'act_aten_visita6','act_orie_visita1',\
	'act_orie_visita2', 'act_orie_visita3', 'act_orie_visita4', 'act_orie_visita5',\
	'act_orie_visita6','act_mrec_visita1', 'act_mrec_visita2', \
	'act_mrec_visita3', 'act_mrec_visita4', 'act_mrec_visita5', 'act_mrec_visita6',\
	'act_expr_visita1', 'act_expr_visita2', 'act_expr_visita3', \
	'act_expr_visita4', 'act_expr_visita5', 'act_expr_visita6', \
	'act_memt_visita1', 'act_memt_visita2', 'act_memt_visita3', 'act_memt_visita4', \
	'act_memt_visita5', 'act_memt_visita6', 'act_prax_visita1', \
	'act_prax_visita2', 'act_prax_visita3', 'act_prax_visita4', 'act_prax_visita5', \
	'act_prax_visita6','act_ejec_visita1', 'act_ejec_visita2', \
	'act_ejec_visita3', 'act_ejec_visita4', 'act_ejec_visita5', 'act_ejec_visita6',\
	'act_comp_visita1', 'act_comp_visita2', 'act_comp_visita3', \
	'act_comp_visita4', 'act_comp_visita5', 'act_comp_visita6', \
	'act_visu_visita1', 'act_visu_visita2', 'act_visu_visita3', 'act_visu_visita4', \
	'act_visu_visita5', 'act_visu_visita6'],'Neuropsychiatric':\
	['act_ansi_visita1', 'act_ansi_visita2', 'act_ansi_visita3', 'act_ansi_visita4',\
	'act_ansi_visita5', 'act_ansi_visita6','act_apat_visita1',\
	'act_apat_visita2', 'act_apat_visita3', 'act_apat_visita4', 'act_apat_visita5', \
	'act_apat_visita6','act_depre_visita1', 'act_depre_visita2',\
	'act_depre_visita3', 'act_depre_visita4', 'act_depre_visita5', 'act_depre_visita6',\
	'gds_visita1', 'gds_visita2', 'gds_visita3', 'gds_visita4', \
	'gds_visita5', 'gds_visita6', 'stai_visita1', 'stai_visita2', \
	'stai_visita3', 'stai_visita4', 'stai_visita5', 'stai_visita6'],\
	'CognitivePerformance':['animales_visita1', 'animales_visita2', 'animales_visita3', \
	'animales_visita4','animales_visita5','animales_visita6',\
	'p_visita1', 'p_visita2', 'p_visita3', 'p_visita4','p_visita5','p_visita6',\
	'mmse_visita1', 'mmse_visita2', 'mmse_visita3', 'mmse_visita4','mmse_visita5', 'mmse_visita6', \
	'reloj_visita1', 'reloj_visita2','reloj_visita3', 'reloj_visita4', 'reloj_visita5', 'reloj_visita6', \
	#'faq_visita1', 'faq_visita2', 'faq_visita3', 'faq_visita4', 'faq_visita5', 'faq_visita6','faq_visita7',\
	'fcsrtlibdem_visita1', 'fcsrtlibdem_visita2', 'fcsrtlibdem_visita3', \
	'fcsrtlibdem_visita4', 'fcsrtlibdem_visita5', 'fcsrtlibdem_visita6', \
	'fcsrtrl1_visita1', 'fcsrtrl1_visita2', 'fcsrtrl1_visita3', 'fcsrtrl1_visita4', 'fcsrtrl1_visita5',\
	'fcsrtrl1_visita6', 'fcsrtrl2_visita1', 'fcsrtrl2_visita2', 'fcsrtrl2_visita3',\
	'fcsrtrl2_visita4', 'fcsrtrl2_visita5', 'fcsrtrl2_visita6', 'fcsrtrl3_visita1', \
	'fcsrtrl3_visita2', 'fcsrtrl3_visita3', 'fcsrtrl3_visita4', 'fcsrtrl3_visita5', 'fcsrtrl3_visita6', \
	'cn_visita1', 'cn_visita2', 'cn_visita3', 'cn_visita4','cn_visita5', 'cn_visita6',\
	#'cdrsum_visita1', 'cdrsum_visita2', 'cdrsum_visita3', 'cdrsum_visita4', 'cdrsum_visita5','cdrsum_visita6', 'cdrsum_visita7'
	],'QualityOfLife':['eq5dmov_visita1', 'eq5dmov_visita2', 'eq5dmov_visita3',\
	'eq5dmov_visita4', 'eq5dmov_visita5', 'eq5dmov_visita6','eq5dcp_visita1', 'eq5dcp_visita2',\
	'eq5dcp_visita3', 'eq5dcp_visita4', 'eq5dcp_visita5', 'eq5dcp_visita6','eq5dact_visita1',\
	'eq5dact_visita2', 'eq5dact_visita3', 'eq5dact_visita4', 'eq5dact_visita5', 'eq5dact_visita6',\
	'eq5ddol_visita1', 'eq5ddol_visita2', 'eq5ddol_visita3', 'eq5ddol_visita4', 'eq5ddol_visita5', 'eq5ddol_visita6', \
	'eq5dans_visita1', 'eq5dans_visita2', 'eq5dans_visita3', 'eq5dans_visita4', 'eq5dans_visita5',\
	'eq5dans_visita6',  'eq5dsalud_visita1', 'eq5dsalud_visita2', 'eq5dsalud_visita3', \
	'eq5dsalud_visita4', 'eq5dsalud_visita5', 'eq5dsalud_visita6', 'eq5deva_visita1', \
	'eq5deva_visita2', 'eq5deva_visita3', 'eq5deva_visita4', 'eq5deva_visita5', 'eq5deva_visita6', \
	'valcvida2_visita1', 'valcvida2_visita2', 'valcvida2_visita3', 'valcvida2_visita4',\
	'valcvida2_visita6', 'valsatvid2_visita1', 'valsatvid2_visita2', 'valsatvid2_visita3',\
	'valsatvid2_visita4', 'valsatvid2_visita5', 'valsatvid2_visita6', 'valfelc2_visita1',\
	'valfelc2_visita2', 'valfelc2_visita3', 'valfelc2_visita4', 'valfelc2_visita5', 'valfelc2_visita6' \
	],'SocialEngagement_s':['relafami', 'relaamigo','relaocio_visita1','rsoled_visita1'],'PhysicalExercise_s':['ejfre', 'ejminut'], 'Diet_s':['alaceit', 'alaves', 'alcar', \
	'aldulc', 'alemb', 'alfrut', 'alhuev', 'allact', 'alleg', 'alpan', 'alpast', 'alpesblan', 'alpeszul', \
	'alverd','dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable'],'EngagementExternalWorld_s':\
	['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14'],\
	'Cardiovascular_s':['hta', 'hta_ini','glu','lipid','tabac', 'tabac_cant', 'tabac_fin', 'tabac_ini',\
	'sp', 'cor','cor_ini','arri','arri_ini','card','card_ini','tir','ictus','ictus_num','ictus_ini','ictus_secu'],\
	'PsychiatricHistory_s':['depre', 'depre_ini', 'depre_num', 'depre_trat','ansi', 'ansi_ini', 'ansi_num', 'ansi_trat'],\
	'TraumaticBrainInjury_s':['tce', 'tce_con', 'tce_ini', 'tce_num', 'tce_secu'],'Sleep_s':['sue_con', 'sue_dia', 'sue_hor',\
	'sue_man', 'sue_mov', 'sue_noc', 'sue_pro', 'sue_rec', 'sue_ron', 'sue_rui', 'sue_suf'],'Anthropometric_s':['lat_manual',\
	'pabd','peso','talla','audi','visu', 'imc'],'Genetics_s':['apoe', 'apoe2niv'],'Diagnoses':['conversionmci','dx_corto_visita1', \
	'dx_corto_visita2', 'dx_corto_visita3', 'dx_corto_visita4', 'dx_corto_visita5', 'dx_corto_visita6', 'dx_corto_visita7',\
	'dx_largo_visita1', 'dx_largo_visita2', 'dx_largo_visita3', 'dx_largo_visita4', 'dx_largo_visita5', 'dx_largo_visita6',\
	'dx_largo_visita7', 'dx_visita1', 'dx_visita2', 'dx_visita3', 'dx_visita4', 'dx_visita5', 'dx_visita6', 'dx_visita7',\
	'ultimodx','edad_conversionmci', 'edad_ultimodx','tpo1.2', 'tpo1.3', 'tpo1.4', 'tpo1.5', 'tpo1.6', 'tpo1.7', \
	'tpoevol_visita1', 'tpoevol_visita2', 'tpoevol_visita3', 'tpoevol_visita4', 'tpoevol_visita5', 'tpoevol_visita6',\
	'tpoevol_visita7','tiempodementia', 'tiempomci']}
	#check thatthe dict  exist in the dataset
	for key,val in cluster_dict.items():
		print('Checking if {} exists in the dataframe',key)
		if set(val).issubset(dataframe.columns) is False:
			print('ERROR!! some of the dictionary:{} are not column names!! \n', key)
			print(dataframe[val])
			#return None
		else:
			print('		Found the cluster key {} and its values {} in the dataframe columns\n',key, val)
	# remove features do not need
	#dataset['PhysicalExercise'] = dataset['ejfre']*dataset['ejminut']
	#list_feature_to_remove = []
	#dataframe.drop([list_feature_to_remove], axis=1,  inplace=True)
	return cluster_dict

def plot_figures_static_of_paper(dataframe):
	"""plot figures of EDA paper
	"""
	# dataframe2plot remove 9s no sabe no contesta
	dataframe2plot = dataframe

	figures_dir ='/Users/jaime/github/papers/EDA_pv/figures'

	fig_filename = 'Fig_visits.png'

	print('Plotting Figure 1 (Fig_visits.png): number of visits in 7 years')
	nb_visits= [np.sum(dataframe['mmse_visita1']>0), np.sum(dataframe['mmse_visita2']>0), np.sum(dataframe['mmse_visita3']>0),\
	np.sum(dataframe['mmse_visita4']>0), np.sum(dataframe['mmse_visita5']>0), np.sum(dataframe['mmse_visita6']>0),np.sum(dataframe['mmse_visita7']>0)]
	print(nb_visits)
	fig, ax = plt.subplots()
	x = np.arange(7)
	plt.bar(x, nb_visits)
	plt.xticks(x, ('y1', 'y2', 'y3', 'y4', 'y5','y6','y7'))
	plt.title('Vallecas Project number of visits')
	plt.xlabel('Years')
	plt.grid(axis='y', alpha=0.75)
	plt.ylabel('# Visits')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	
	print('Plotting Figure 2 (Fig_anthro.png): Anthropometric (peso,talla, imc,pabd)')
	fig_filename = 'Fig_sexlat.png'
	phys_lat = ['sexo','lat_manual']
	fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
	newdf = dataframe2plot.groupby('lat_manual')['lat_manual'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.plot(ax=axes[0],kind='bar', color='yellowgreen')
	axes[0].set_title(r'Hand laterality', color='C0')
	axes[0].set_xticklabels([r'Right',r'Left',r'Ambi',r'LeftC'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0].set_xlabel(' ')
	#axes[0].grid(axis='y', alpha=0.75)
	axes[0].get_legend().remove()
	#sex
	newdf = dataframe2plot.groupby('sexo')['sexo'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.plot(ax=axes[1],kind='bar', color='yellowgreen')
	axes[1].set_title(r'Sex', color='C0')
	axes[1].set_xticklabels([r'Male',r'Female'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[1].set_xlabel(' ')
	#axes[1].grid(axis='y', alpha=0.75)
	axes[1].get_legend().remove()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	
	print('Plotting Figure 2 (Fig_anthro.png): Anthropometric (peso,talla,imc,pabd)')
	fig_filename ='Fig_anthro.png'
	list_anthro = ['peso','talla', 'imc','pabd']
	fig = dataframe[list_anthro].hist(figsize=(10, 10), grid=False, bins=None, xlabelsize=8, ylabelsize=8, rwidth=0.9,color = "skyblue")
	#[x.title.set_size(32) for x in fig.ravel()]
	titles=['BMI', 'Abdo perim', 'Weight', 'Height']
	i=0
	list_anthro.sort()
	for x in fig.ravel():
		title=titles[i]
		#x.grid(axis='y', alpha=0.75)
		x.axvline(x=np.mean(dataframe[list_anthro[i]]), color="red", linestyle='--')
		x.set_title(title)
		i=i+1
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	
	#plot_distribution_kde(dataframe, features = ['imc','pabd','peso','talla'])
	#https://matplotlib.org/examples/color/named_colors.html
	print('Plotting Figure 3 (Fig_ages.png):')
	fig_filename = 'Fig_ages.png'
	list_ages = ['edad_visita1','edad_visita6']
	fig = dataframe[list_ages].hist(figsize=(8, 6), grid=False, bins=None, xlabelsize=8, ylabelsize=8, rwidth=0.9, color = "gray")
	titles=['age y1', 'age y6']
	i=0
	list_ages.sort()
	for x in fig.ravel():
		title=titles[i]
		x.set_title(title)
		x.grid(axis='y', alpha=0.75)
		x.axvline(x=np.mean(dataframe[list_ages[i]]), color="red", linestyle='--')
		i=i+1
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	#### Educ Level
	print('Plotting Figure 4 (Fig_demo_s.png): Demographic (sdestciv,numhij, sdatrb,sdeconom,nivel_educativo,sdvive)')
	fig_filename = 'Fig_demo.png'
	list_demo = ['sdestciv','numhij', 'sdatrb','sdeconom','nivel_educativo','sdvive']
	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,12), sharey=False)
	#educ level
	newdf = dataframe2plot.groupby('nivel_educativo')['nivel_educativo'].agg(['count'])
	#dont take 9 no sabe no contesta
	#if newdf.ix[9]['count'] >0: newdf = newdf[0:-1]
	newdf.plot(ax=axes[0,0],kind='bar', color='bisque')
	axes[0,0].set_title(r'educ level', color='C0')
	axes[0,0].set_xticklabels([r'No',r'Prim', r'Sec',r'Univ'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0,0].set_xlabel(' ')
	axes[0,0].get_legend().remove()
	# sons
	bins =[-np.inf,0, 1, 2, 3, 4, 5, np.inf]
	bins = pd.cut(dataframe2plot['numhij'],bins, include_lowest =True)
	newdf = bins.groupby(bins).agg(['count'])
	newdf.plot(ax=axes[0,1],kind='bar', color='bisque')
	axes[0,1].set_title(r'#sons', color='C0')
	axes[0,1].set_xticklabels([r'0',r'(0,1]',r'(1,2]',r'(2,3]',r'(3,4]',r'(4,5]',r'(5,6+]'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0,1].set_xlabel(' ')
	#axes[0,1].grid(axis='y', alpha=0.75)
	axes[0,1].get_legend().remove()
	# years employee
	bins =[-np.inf,0, 10, 20, 30, 40, 50, np.inf]
	bins = pd.cut(dataframe2plot['sdatrb'],bins, include_lowest =True)
	newdf = bins.groupby(bins).agg(['count'])
	newdf.plot(ax=axes[1,0],kind='bar', color='bisque')
	axes[1,0].set_title(r'#years employee', color='C0')
	axes[1,0].set_xticklabels([r'0',r'(0,10]',r'(10,20]',r'(20,30]',r'(30,40]',r'(40,50]',r'(50+)'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[1,0].set_xlabel(' ')
	#axes[1,0].grid(axis='y', alpha=0.75)
	axes[1,0].axvline(x=np.mean(dataframe2plot['sdatrb']), color="red", linestyle='--')
	axes[1,0].get_legend().remove()
	bins = pd.cut(dataframe2plot['sdeconom'],np.arange(0,10,1))
	#newdf = dataframe2plot.groupby(bins)['imc'].agg(['count'])
	dataframe2plot['sdeconom'].plot(ax=axes[1,1], kind='hist',color='bisque',rwidth=0.9);
	axes[1,1].set_xlabel('')
	#axes[1].set_ylabel('# subjects')
	axes[1,1].set_title(r'perceived socioecon. status', color='C0')
	axes[1,1].axvline(x=np.mean(dataframe2plot['sdeconom']), color="bisque", linestyle='--')
	newdf = dataframe2plot.groupby('sdestciv')['sdestciv'].agg(['count'])
	#dont take 9 no sabe no contesta
	#if newdf.ix[9]['count'] >0: newdf = newdf[0:-1]
	newdf.plot(ax=axes[2,0],kind='bar', color='bisque')
	axes[2,0].set_title(r'marital status', color='C0')
	axes[2,0].set_xticklabels([r'Single', r'Married',r'Widowed', r'Divorced'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[2,0].set_xlabel(' ')
	axes[2,0].get_legend().remove()
	bins =[-np.inf, 1, 2, 3, 4, 5, np.inf]
	bins = pd.cut(dataframe2plot['sdvive'],bins)
	newdf = bins.groupby(bins).agg(['count'])
	newdf.plot(ax=axes[2,1], kind='bar',color='bisque');
	#axes[1].set_ylabel('# subjects')
	axes[2,1].set_title(r'#residents home', color='C0')
	axes[2,1].set_xticklabels([r'1',r'2', r'3',r'4', r'5',r'6+'],rotation=0)
	#axes[2,1].axvline(x=np.mean(dataframe2plot['sdvive']), color="red", linestyle='--')
	axes[2,1].set_xlabel('')
	plt.tight_layout()
	axes[2,1].get_legend().remove()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	# EngagementExternalWorld
	fig_filename = 'Fig_engage.png'
	# BUG replace 0 by 1 row 10
	dataframe['a10'].replace(0,1, inplace=True)
	#['a03' amigos, 'a04' travel, 'a05' ong, 'a06' church, 'a08' cine, 'a09' sport, 'a10' music, 'a11' tv, 'a12' read, 'a13' internet]
	engage_list = ['a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13']
	fig= dataframe2plot[engage_list].hist(figsize=(12, 12), bins=None, xlabelsize=8, ylabelsize=8, rwidth=0.9,grid=False, color = "dodgerblue")
	titles = ['creative', 'friends','travel','NGO','church','soc club','movies','sports','music', 'tv/radio', 'books', 'internet']
	i = 0
	points = np.arange(1,4)
	for x in fig.ravel():
		title=titles[i]
		x.set_title(title)
		x.set_xticks(points)
		x.set_xticklabels(('Never', 'Few', 'Often'))
		i+=1
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
 
	fig_filename = 'Fig_tce.png'
	fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
	tce_list = ['tce']
	newdf = dataframe2plot.groupby('tce')['tce'].agg(['count'])
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1] #newdf[0:-1]	
	newdf.plot(ax=axes,kind='bar', color='tan')
	axes.set_title(r'suffered TBI', color='C0')
	axes.set_xticklabels([r'No',r'Yes'])
	axes.set_ylabel('# subjects')
	axes.set_xlabel(' ')
	axes.get_legend().remove()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	fig_filename = 'Fig_phys.png'
	phys_list = ['ejfre', 'ejminut']
	fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
	dataframe2plot[['ejfre']].plot(ax=axes[0],kind='hist',color='firebrick')
	axes[0].set_xlabel('days/week')
	axes[0].set_ylabel('# subjects')
	axes[0].set_title('physical exercise d/w', color='C0')
	axes[0].get_legend().remove()
	points = [0, 30, 60, 120,180]
	bins = pd.cut(dataframe2plot['ejminut'], points)
	newdf = dataframe2plot.groupby(bins)['a01'].agg(['count'])
	newdf.plot(ax=axes[1], kind='bar',color='firebrick');
	axes[1].set_xlabel('minutes/session')
	#axes[1].set_ylabel('# subjects')
	axes[1].set_title(r'avg session minutes', color='C0')
	axes[1].set_xticklabels([r'0-1/2h',r'1/2-1h',r'1-2h',r'2-3h'])
	axes[1].get_legend().remove()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	fig_filename = 'Fig_cardio.png'
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,16), sharey=False)
	#arryt
	newdf = dataframe2plot.groupby('arri')['arri'].agg(['count'])
	#dont take 9 no sabe no contesta
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1] #newdf[0:-1]
	newdf.plot(ax=axes[0,0],kind='bar', color='indianred')
	axes[0,0].set_title(r'arrhythmias', color='C0')
	axes[0,0].set_xticklabels([r'No',r'Auric', r'Arr'])
	#axes[0,0].set_ylabel('# subjects')
	axes[0,0].set_xlabel(' ')
	axes[0,0].get_legend().remove()
	#bmi
	np.arange(15,50,5)
	bins = pd.cut(dataframe2plot['imc'], points)
	#newdf = dataframe2plot.groupby(bins)['imc'].agg(['count'])
	dataframe2plot['imc'].plot(ax=axes[0,1], kind='hist',color='indianred');
	axes[0,1].set_xlabel('')
	#axes[1].set_ylabel('# subjects')
	axes[0,1].set_title(r'BMI', color='C0')
	#axes[0,1].axvline(x=25, color="red", linestyle='--')
	#axes[0,1].axvline(x=30, color="red", linestyle='--')
	#axes[0,1].set_xticklabels([r'0-1/2h',r'1/2-1h',r'1-2h',r'2-3h'])
	# cor angina
	newdf = dataframe2plot.groupby('cor')['cor'].agg(['count'])
	#dont take 9 no sabe no contesta
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1]
	newdf.plot(ax=axes[0,2],kind='bar', color='indianred')
	axes[0,2].set_title(r'stroke', color='C0')
	axes[0,2].set_xticklabels([r'No',r'Angina', r'Stroke'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0,2].set_xlabel(' ')
	axes[0,2].get_legend().remove()
	# diabetes glu
	newdf = dataframe2plot.groupby('glu')['glu'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.plot(ax=axes[1,0],kind='bar', color='indianred')
	axes[1,0].set_title(r'diabetes', color='C0')
	axes[1,0].set_xticklabels([r'No',r'Diabetes mell.', r'Carbs intol.'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[1,0].set_xlabel(' ')
	axes[1,0].get_legend().remove()
	# hta hipertension
	newdf = dataframe2plot.groupby('hta')['hta'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.plot(ax=axes[1,1],kind='bar', color='indianred')
	axes[1,1].set_title(r'blood preassure', color='C0')
	axes[1,1].set_xticklabels([r'No',r'HBP'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[1,1].set_xlabel(' ')
	axes[1,1].get_legend().remove()
	# ictus 
	newdf = dataframe2plot.groupby('ictus')['ictus'].agg(['count'])
	#dont take 9 no sabe no contesta
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1]
	newdf.plot(ax=axes[1,2],kind='bar', color='indianred')
	axes[1,2].set_title(r'ictus', color='C0')
	axes[1,2].set_xticklabels([r'No',r'Ischemic', r'Hemorr'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[1,2].set_xlabel(' ')
	axes[1,2].get_legend().remove()
	# lipid colesterol  
	newdf = dataframe2plot.groupby('lipid')['lipid'].agg(['count'])
	#dont take 9 no sabe no contesta
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1]
	newdf.plot(ax=axes[2,0],kind='bar', color='indianred')
	axes[2,0].set_title(r'cholesterol', color='C0')
	axes[2,0].set_xticklabels(['No', 'Hyper chol', 'Hyper trig', 'Hyper chol&trig'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[2,0].set_xlabel(' ')
	axes[2,0].get_legend().remove()
	#smoke
	newdf = dataframe2plot.groupby('tabac')['tabac'].agg(['count'])
	newdf.plot(ax=axes[2,1],kind='bar', color='indianred')
	axes[2,1].set_title(r'smoke', color='C0')
	axes[2,1].set_xticklabels(['No', 'Smoker', 'Ex smoker'],rotation=0)
	axes[2,1].set_xlabel(' ')
	axes[2,1].get_legend().remove()
	#tiroides
	newdf = dataframe2plot.groupby('tir')['tir'].agg(['count'])
	if newdf.ix[9]['count'] >0: newdf = newdf.ix[0:9-1]
	newdf.plot(ax=axes[2,2],kind='bar', color='indianred')
	axes[2,2].set_title(r'thyroiditis', color='C0')
	axes[2,2].set_xticklabels(['No', 'Hyper thyroiditis', 'Hipo thyroidism'],rotation=0)
	axes[2,2].set_xlabel(' ')
	axes[2,2].get_legend().remove()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	fig_filename = 'Fig_apoe.png'
	fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
	newdf = dataframe2plot.groupby('apoe')['apoe'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.plot(ax=axes,kind='bar', color='darkolivegreen')
	axes.set_title(r'APOE', color='C0')
	axes.set_xticklabels([r'Negative', r'APOE4 Hetero', r'APOE4 Homo'], rotation=0)
	axes.set_ylabel('# subjects')
	axes.set_xlabel(' ')
	axes.get_legend().remove()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	fig_filename = 'Fig_food.png'
	foods_df = dataframe[['alaceit', 'alaves', 'alcar', 'aldulc', 'alemb', 'alfrut', 'alhuev', 'allact', 'alleg', 'alpan', 'alpast', 'alpesblan', 'alpeszul', 'alverd']]
	#sns.pairplot(foods_df, hue='species', size=2.5);
	fig = dataframe[['alaceit', 'alaves', 'alcar', 'aldulc', 'alemb', 'alfrut', 'alhuev', 'allact', 'alleg', 'alpan', 'alpast', 'alpesblan', 'alpeszul', 'alverd']].hist(figsize=(18, 16), bins=None, rwidth=0.9, xlabelsize=8, ylabelsize=8,grid=False, color = "chocolate")
	plt.tight_layout()
	plt.grid(axis='y', alpha=0.75)	
	titles=['olive oil', 'white meat', 'red meat', 'sweets', 'charcuterie', 'fruit', 'eggs', 'lact', 'legumes', 'bread', 'pasta', 'white fish', 'blue fish', 'vegetables']
	i=0
	for x in fig.ravel()[0:-2]:
		title=titles[i]
		# horizontal grid
		#x.grid(axis='y', alpha=0.75)
		x.set_title(title)
		points = np.arange(0,4)
	 	x.set_xticks(points)
	 	x.set_xticklabels(('0d/w', '1-2d/w', '3-5d/w', '6-7d/w'))
		i=i+1	
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	fig_filename = 'Fig_diet.png'
	list_diet = ['dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable']
	fig = dataframe[list_diet].hist(figsize=(8, 8), bins=None, rwidth=0.9, grid=False, xlabelsize=8, ylabelsize=8,color = "khaki")
	titles=['glucemic', 'fat', 'proteic', 'medit']
	i=0
	for x in fig.ravel():
		title=list_diet[i]
		#x.grid(axis='y', alpha=0.75)
		x.set_title(title)
		x.axvline(x=np.mean(dataframe[title]), color="red", linestyle='--')
		i=i+1	
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')

	# Sleep	
	print('Plotting Figure (Fig_sleep.png):Sleep sue_con, sue_dia, sue_hor, sue_man, sue_mov, sue_noc, sue_pro, sue_rec, sue_ron, sue_rui, sue_suf]')
	fig_filename = 'Fig_sleep.png'
	# sue dia
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18,16), sharex=False, sharey=False)
	bins =[-np.inf,0, 1, 2, 3, np.inf]

	bins = pd.cut(dataframe2plot['sue_dia'],bins, include_lowest =True)
	newdf = bins.groupby(bins).agg(['count'])
	newdf.plot(ax=axes[0,0],kind='bar', color='slateblue')
	axes[0,0].set_title(r'#hrs sleep day', color='C0')
	axes[0,0].set_xticklabels([r'0',r'(0,1]',r'(1,2]',r'(2,3]',r'(3,4+]'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0,0].set_xlabel(' ')
	#axes[0,0].grid(axis='y', alpha=0.75)
	axes[0,0].get_legend().remove()
	# sue noc
	bins =[-np.inf,0, 2, 4, 6, 8, 10,np.inf]
	bins = pd.cut(dataframe2plot['sue_noc'],bins, include_lowest =True)
	newdf = bins.groupby(bins).agg(['count'])
	newdf.plot(ax=axes[0,1],kind='bar', color='slateblue')
	axes[0,1].set_title(r'#hrs sleep night', color='C0')
	axes[0,1].set_xticklabels([r'0',r'(0,2]',r'(2,4]',r'(4,6]', r'(6,8]', r'(8,10]', r'(10,12+]'],rotation=0)
	#axes[0,0].set_ylabel('# subjects')
	axes[0,1].set_xlabel(' ')
	#axes[0,1].grid(axis='y', alpha=0.75)
	axes[0,1].get_legend().remove()
	# sue prof 1,2,3
	newdf = dataframe2plot.groupby('sue_pro')['sue_pro'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[0,2],kind='bar', color='slateblue')
	axes[0,2].set_xticklabels([r'Light',r'Moderate', r'Important'],rotation=0)
	axes[0,2].set_title(r'deep sleep', color='C0')
	axes[0,2].get_legend().remove()
	axes[0,2].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_suf')['sue_suf'].agg(['count'])
	#dont take 9 no sabe no contesta
	newdf.ix[0:9-1].plot(ax=axes[1,0],kind='bar', color='slateblue')
	axes[1,0].set_xticklabels([r'No',r'Yes'],rotation=0)
	axes[1,0].set_title(r'sufficient sleep', color='C0')
	axes[1,0].get_legend().remove()
	axes[1,0].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_rec')['sue_rec'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[1,1],kind='bar', color='slateblue')
	axes[1,1].set_xticklabels([r'No',r'Yes'],rotation=0)
	axes[1,1].set_title(r'remember dreams', color='C0')
	axes[1,1].get_legend().remove()
	axes[1,1].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_mov')['sue_mov'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[1,2],kind='bar', color='slateblue')
	axes[1,2].set_xticklabels([r'No',r'Yes'],rotation=0)
	axes[1,2].set_title(r'moves while sleeps', color='C0')
	axes[1,2].get_legend().remove()
	axes[1,2].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_ron')['sue_ron'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[2,0],kind='bar', color='slateblue')
	axes[2,0].set_xticklabels([r'No',r'Yes', r'Snore&Breath int.'],rotation=0)
	axes[2,0].set_title(r'snores while sleeps', color='C0')
	axes[2,0].get_legend().remove()
	axes[2,0].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_rui')['sue_rui'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[2,1],kind='bar', color='slateblue')
	axes[2,1].set_xticklabels([r'No',r'Yes'],rotation=0)
	axes[2,1].set_title(r'make noises while sleeps', color='C0')
	axes[2,1].get_legend().remove()
	axes[2,1].set_xlabel(' ')
	newdf = dataframe2plot.groupby('sue_hor')['sue_hor'].agg(['count'])
	newdf.ix[0:9-1].plot(ax=axes[2,2],kind='bar', color='slateblue')
	axes[2,2].set_xticklabels([r'No',r'Yes'],rotation=0)
	axes[2,2].set_title(r'tingling while sleeps', color='C0')
	axes[2,2].get_legend().remove()
	axes[2,2].set_xlabel(' ')
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	return

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
	F, p = stats.f_oneway(df_oneway[dictio.keys()[1]].dropna(), \
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

def compute_contingency_table(df, labelx, labely):
	"""compute_contingency_table : computes contingency table and Chi2 or fisher test (2x2 table)
	"""
	#https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.fisher_exact.html
	import scipy.stats as stats

	#df['mci'] = df[labelx].astype('category')
	df[labelx] = df[labelx].astype('int')
	ctable = pd.crosstab(index=df[labelx], columns=df[labely])

	ctable.index = df[labelx].unique()# ["died","survived"]
	ctable.columns= df[labely].unique()
	

	if ctable.shape[0] != 2 or ctable.shape[1] != 2: 
		print('contingency table is not 2x2. Chi2 test...\n')
		c, p, dof, expected = stats.chi2_contingency(ctable.values)
		return [c, p, dof, expected]
		
	#A 2x2 contingency table. Elements should be non-negative integers.
	print('Computing Fisher test in 2x2 contingency table \n')
	#P-value, the probability of obtaining a distribution at least as extreme as the one that was actually observed, 
	#assuming that the null hypothesis is true.
	#prior odds ratio and not a posterior estimate
	#https://www.statsmodels.org/stable/contingency_tables.html
	oddsratio, pvalue = stats.fisher_exact(ctable.values)
	return [oddsratio, pvalue ]

def plot_ts_figures_of_paper(dataframe, features_dict, figname=None):
	"""plot_ts_figures_of_paper calls to plot_figures_longitudinal_timeseries_of_paper
	"""
	print('plotting time series QoL longitudinal features with mean and sigma \n\n')
	matching_features = [s for s in features_dict['QualityOfLife'] if "eq5dmov_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)	
	matching_features = [s for s in features_dict['QualityOfLife'] if "eq5ddol_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)	
	matching_features = [s for s in features_dict['QualityOfLife'] if "eq5dsalud_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)	
	matching_features = [s for s in features_dict['QualityOfLife'] if "valfelc2_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)	
	# do not plot time series of neuropsychiatric bnecause they are z-transform
	#matching_features = [s for s in features_dict['Neuropsychiatric'] if "gds_" in s]
	#plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features])	
	#matching_features = [s for s in features_dict['Neuropsychiatric'] if "stai_" in s]
	#plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features])
	print('plotting time series CogPerf longitudinal features with mean and sigma \n\n')
	matching_features = [s for s in features_dict['CognitivePerformance'] if "animales_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)	
	matching_features = [s for s in features_dict['CognitivePerformance'] if "p_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)
	matching_features = [s for s in features_dict['CognitivePerformance'] if "cn_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)
	matching_features = [s for s in features_dict['CognitivePerformance'] if "mmse_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)
	# matching_features = [s for s in features_dict['CognitivePerformance'] if "bus_int" in s]
	# plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features])
	# matching_features = [s for s in features_dict['CognitivePerformance'] if "bus_sum" in s]
	# plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features])
	# matching_features = [s for s in features_dict['CognitivePerformance'] if "bus_meana" in s]
	# plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features])
	matching_features = [s for s in features_dict['CognitivePerformance'] if "fcsrtlibdem_" in s]
	plot_figures_longitudinal_timeseries_of_paper(dataframe[matching_features], figname)
	return 

def plot_figures_longitudinal_timeseries_of_paper(dataframe, figname=None):
	""" plot_figures_longitudinal_timeseries_of_paper
	Args:
	Output:
	"""
	figures_dir ='/Users/jaime/github/papers/EDA_pv/figures'
	fig  = plt.figure(figsize=(8,6))
	cols = dataframe.columns
	fig_filename = cols[0] +'_years'
	if figname is not None:
		fig_filename= fig_filename + figname
	#if sum(dataframe[['tpo1.2','tpo1.3']].notnull().all(axis=1)==False) == 0 is True:
	#	fig_filename = fig_filename +'_loyals'
	nb_years = len(cols)
	x = np.linspace(1, nb_years,nb_years)
	mu_years = [0] * nb_years
	std_years = [0] * nb_years
	title = cols[0][:-1]+'1-'+str(nb_years)
	ylabel = '$\\mu$ +- $\\sigma$'
	for ix, name in enumerate(cols):
		mu_years[ix] = dataframe[name].mean()
		std_years[ix] = dataframe[name].std()
		if cols[0].find('dx_corto_') > -1:
			mu_years[ix] = float(np.sum(dataframe[name]>0))/dataframe[name].count()
			std_years[ix] = mu_years[ix]
			ylabel = 'Ratio subjects with MCI/AD diagnose in each year'
		elif cols[0].find('dx_largo_') > -1:
			#plot scd, scd plus and mci
			print('plot healthy, scd, scd plus, mci and ad') 
			mu_years[ix] = float(np.sum(dataframe[name]==2))/dataframe[name].count()
			std_years[ix] = mu_years[ix]
			ylabel = 'Ratio subjects with SCD + diagnose in each year'
			title = 'SCD Plus visits 1,7'

		#textlegend[ix] = (mu_years[ix],std_years[ix])
	mu_years = np.asarray(mu_years)
	std_years = np.asarray(std_years)
	fill_max, fill_min = mu_years-std_years, mu_years+std_years
	plt.plot(x, mu_years, 'k-')
	# if cols[0].find('stai_') ==0:
	# 	stai_yrs = []
	# 	for i in range(1,nb_years+1):stai_yrs.append('stai_visita'+str(i))
	# 	# stai is a z transform -2, 4
	# 	fill_max, fill_min = dataframe[stai_yrs].max().max(), dataframe[stai_yrs].min().min()
	# 	plt.ylim(top=fill_max+0.2, bottom=fill_min-0.2)
	# else:
	plt.ylim(top=np.max(mu_years)+ np.max(std_years), bottom=0)

	if cols[0].find('dx_') <= -1:plt.fill_between(x, fill_max, fill_min, facecolor='papayawhip', interpolate=True)
	plt.ylabel(ylabel)
	plt.xlabel('years')
	plt.title(title)
	#plt.text(x, mu_years, textlegend, fontdict=font)
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	plt.show()

def plot_histograma_one_longitudinal(df, longit_pattern=None, figname=None):
	""" plot_histogram_pair_variables: plot histograma for each year of a longitudinal variable
	Args: Pandas dataframe , regular expression pattern eg mmse_visita """
	figures_dir = '/Users/jaime/github/papers/EDA_pv/figures'
	if type(longit_pattern) is list:
		longit_status_columns = longit_pattern
	else:
		longit_status_columns = [x for x in df.columns if (longit_pattern.match(x))]
	fig_filename = longit_status_columns[0][:-1]
	df[longit_status_columns].head(10)
	# plot histogram for longitudinal pattern

	nb_rows, nb_cols = 2, 3 #for 7 years :: 2,4  and row,col = int(i/(2**nb_rows)), int(i%(nb_cols))
	fig, ax = plt.subplots(nb_rows, nb_cols, sharey=False, sharex=False)
	fig.set_size_inches(15,10)
	rand_color = np.random.rand(3,)
	for i in range(len(longit_status_columns)):
		# for 7 years
		#row,col = int(i/(2**nb_rows)), int(i%(nb_cols))
		row,col = int(i/nb_cols), int(i%(nb_cols))
		histo  = df[longit_status_columns[i]].value_counts()
		min_r, max_r =df[longit_status_columns[i]].min(), df[longit_status_columns[i]].max()
		#sns.distplot(df[longit_status_columns[i]], color='g', bins=None, hist_kws={'alpha': 0.4})
		ax[row,col].bar(histo.index, histo, align='center', color=rand_color)
		ax[row,col].set_xticks(np.arange(int(min_r),int(max_r+1)))
		ax[row,col].set_xticklabels(np.arange(int(min_r),int(max_r+1)),fontsize=8, rotation='vertical')
		ax[row,col].set_title(longit_status_columns[i])
	plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=1.0)
	#remove axis for 8th year plot
	#ax[-1, -1].axis('off')
	plt.tight_layout()
	if figname is not None:
		fig_filename = fig_filename + figname
	#if sum(df[['tpo1.2','tpo1.3']].notnull().all(axis=1)==False) == 0 is True:
	#	fig_filename = fig_filename+'_loyals'
	plt.savefig(os.path.join(figures_dir, fig_filename), bbox_inches='tight')
	plt.show()

def plot_figures_longitudinal_of_paper(dataframe, features_dict, figname=None):
	"""plot_figures_longitudinal_of_paper plot longitudinal figures of EDA paper
	Args: list of clusters, the actual features to plot are hardcoded 
	Output : 0
	"""
	# dataframe2plot remove 9s no sabe no contesta
	print('Plotting longitudional features....\n')
	list_clusters = features_dict.keys()
	for ix in list_clusters:
		print('Longitudinal histogram of group:{}',format(ix))
		list_longi = features_dict[ix]
		type_of_tests = []
		if ix is 'CognitivePerformance':
			type_of_tests = ['bus_int_', 'bus_sum_','bus_meana_','fcsrtlibdem_','p_', 'cn_', 'animales_', 'mmse_']
			type_of_tests=type_of_tests[3:]
		elif ix is 'Neuropsychiatric':
			#type_of_tests = ['stai_','gds_', 'act_ansi_', 'act_depre_']
			type_of_tests = ['gds_','stai_']
		elif ix is 'QualityOfLife':
			type_of_tests = ['eq5dmov_','eq5ddol_','eq5dsalud_','valfelc2_']
		#elif ix is 'Diagnoses':
		#	type_of_tests = ['dx_corto_', 'dx_largo_']
		#elif ix is 'SCD':
		#dificultad orientarse  86, 84 toma decisiones, 10 perdida memo afecta su vida
		#	type_of_tests = ['scd_','peorotros_', 'preocupacion_', 'eqm86_','eqm84_','eqm10_']
		if len(type_of_tests) > 0:
			for jj in type_of_tests:
				longi_items_per_group = filter(lambda k: jj in k, list_longi)
				df_longi = dataframe[longi_items_per_group]
				plot_histograma_one_longitudinal(df_longi, longi_items_per_group, figname)
	print('DONE...plot_figures_longitudinal_of_paper Exiting\n')
	return 0	

def select_rows_all_visits(dataframe, visits):
	"""select_rows_all_visits
	Args:
	Output: df with the rows of subjects with all visits
	"""
	
	#df2,df3,df4,df5,df6 = dataframe[['tpo1.2']].notnull(),dataframe[['tpo1.3']].notnull(),dataframe[['tpo1.4']].notnull(),dataframe[['tpo1.5']].notnull(),dataframe[['tpo1.6']].notnull()
	#print('Visits per year 2..6:',np.sum(df2)[0], np.sum(df3)[0],np.sum(df4)[0],np.sum(df5)[0],np.sum(df6)[0])
	df_loyals = dataframe[visits].notnull()
	#y4 749 visits[:-2] y5 668 visits[:-1]

	rows_loyals = df_loyals.all(axis=1)
	return dataframe[rows_loyals]

def main():
	# Open csv with MRI data
	plt.close('all')
	csv_path = '/Users/jaime/Downloads/test_code/PV7years_T1vols.csv'
	csv_path = '~/vallecas/data/BBDD_vallecas/PVDB_pve_sub.csv'
	csv_path = '~/vallecas/data/BBDD_vallecas/Vallecas_Index-10March2019.csv'
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures/'
	dataframe = pd.read_csv(csv_path)
	#testing here cut paste
	# select rows with 5 visits
	visits=['tpo1.2', 'tpo1.3','tpo1.4', 'tpo1.5','tpo1.6']
	df_loyals = select_rows_all_visits(dataframe, visits)
	
	#########################
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	print('Build dictionary with features ontology and check the features are in the dataframe\n') 
	#features_dict is the list of clusters, the actual features to plot are hardcoded
	features_dict = vallecas_features_dictionary(dataframe)
	##########################################

	print('Plotting histograms for static variables \n\n')
	#plot_figures_static_of_paper(dataframe)

	##########################################
	
	##########################################
	print('Plotting histograms for longitudinal variables \n\n')
	#plot_figures_longitudinal_of_paper(df_loyals, features_dict, '_loyals')
	#plot_figures_longitudinal_of_paper(dataframe, features_dict)
	
	print('Plotting time series long variables mean + std \n\n')
	plot_ts_figures_of_paper(df_loyals, features_dict, '_loyals')
	plot_ts_figures_of_paper(dataframe, features_dict)
	pdb.set_trace()
	# select rows with 5 visits
	visits=['tpo1.2', 'tpo1.3','tpo1.4', 'tpo1.5','tpo1.6']
	df_loyals = select_rows_all_visits(dataframe, visits)
	pdb.set_trace()
	##########################################

	# only study rows with conversionmci to some value
	dataframe_conv = dataframe[dataframe['conversionmci'].notnull()]
	
	### Fisher test for binomial features Chi sq or Fisher (2x2)
	labelx, labely = 'conversionmci', 'nivel_educativo'
	Fodds_and_p = compute_contingency_table(dataframe_conv, labelx, labely)
	print('p value for ',labelx, ' x ', labely,' is:', Fodds_and_p[1])
	pdb.set_trace()

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