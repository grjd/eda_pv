#######################################################
# Python program name	: 
#Description	: area_under_curve.py
#Args           : Calculates Buchske aggregate and charts for visual representation of area under the curve                                                                                         
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#######################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.mstats as mstats
from matplotlib.patches import Polygon
import pdb
import time
import pandas as pd
import sys
sys.path.append('/Users/jaime/github/code/tensorflow/production')
import descriptive_stats as pv
import seaborn as sns

# def func(x):
#     return (x - 3) * (x - 5) * (x - 7) + 85


def buschke_aggregate(y):
	"""buschke_aggregate: computes the Buschke aggregate 
	Args: y array or list of len 3
	Output:
	"""
	from scipy.integrate import trapz, simps
	from scipy.interpolate import interp1d
	from matplotlib.patches import Polygon
	# The rank of the coefficient matrix in the least-squares fit is deficient. The warning is only raised if full = False.
	# Turn off warning
	#polyfit issues a RankWarning when the least-squares fit is badly conditioned. This implies that the best fit is not well-defined due to numerical error
	import warnings
	warnings.simplefilter('ignore', np.RankWarning)
	from scipy.interpolate import UnivariateSpline
	
	npx, degree = 3, 2
	x = np.array([0,1,2])
	if y is None: y = np.array([0,1,0])
	pointspersector = 100
	interp_points = (npx-1)*pointspersector
	xs = np.linspace(0, npx-1, interp_points)
	if type(y) is list:
		y = np.asarray(y)
	z_fit = np.polyfit(x[:], y[:], deg=degree)
	p_fit = np.poly1d(z_fit)

	area_tot =  simps(y[0:2], x[0:2], even='avg') + simps(y[1:], x[1:], even='avg')
	area_f = simps(y[0:2] -y[0], x[0:2], even='avg')
	area_s = simps(y[1:] -y[1], x[1:], even='avg')
	b_agg = area_tot + area_f + area_s
	print(y, area_tot, area_f,area_s, 'b_agg=', b_agg, '\n')
	ymin, ymax = 0 - 0.1, max(y)+1
	plot_fitted_pol  = False
	if plot_fitted_pol is True:
		# fit polynomial of degree that pass for (x[1:-1], b_list) points
		# z highest power first
		fig, axes = plt.subplots(1, 2)
		axes[0].plot(x, y, '.', xs, p_fit(xs), '-')
		axes[0].set_xticks([0,1,2])
		axes[0].set_title('polyfit')
		#axes[0].set_xlim([xmin,xmax])
		axes[0].set_ylim([ymin,ymax])
		axes[0].grid(True)
		# Degree of the smoothing spline. Must be <= 5. Default is k=3, a cubic spline
		spl = UnivariateSpline(x, y, k=2)
		axes[1].plot(xs, spl(xs), 'b', lw=3)
		axes[1].set_xticks([0,1,2])
		#axes[1].set_xlim([xmin,xmax])
		axes[1].set_ylim([ymin,ymax])
		axes[1].set_title('Spline')
		axes[1].grid(True)
	
	b_values = [b_agg, area_tot, area_f, area_s, z_fit, p_fit]
	return b_values

def main():
	print("Calling to Buscke aggregate....")
	# load csv values
	csv_file = "/Users/jaime/vallecas/data/BBDD_vallecas/Proyecto_Vallecas_7visitas_19_nov_2018.csv"
	dataframe = pv.load_csv_file(csv_file)
	# Feature Selection : cosmetic name changing and select input and output 
	print('Cosmetic cleanup (lowercase, /, remove blanks) e.g. cleanup_column_names(df,rename_dict={},do_inplace=True)\n\n') 
	pv.cleanup_column_names(dataframe, {}, True)
	df_copy = dataframe.copy()
	mci = dataframe['conversionmci']
	mmse = dataframe['mmse_visita1']
	bus_scores = ['fcsrtrl1_visita1', 'fcsrtrl2_visita1', 'fcsrtrl3_visita1']
	dataframe = dataframe[bus_scores]
	dataframe_bus = dataframe[bus_scores]
	dataframe_bus = dataframe_bus.values
	#Y= [np.array([0,0,0]), np.array([0,0,1]), np.array([0,1,0]),np.array([2,0,0]),np.array([1,1,1]), np.array([2,1,0]), np.array([2,1,2]), np.array([10,12,14]), np.array([16,10,0]),np.array([14,12,10]), np.array([16,16,16]), np.array([16,15,14]),np.array([14,15,16]),np.array([15,16,14]),np.array([12,14,16])]
	#for y in Y:
	#for index, row in dataframe_bus.iterrows():
	S = [0] * dataframe_bus.shape[0]
	# arithmetic, gemoetric mean and sum of Bischke scores
	mean_a, mean_g, suma = S[:], S[:], S[:]

	for ix, y in enumerate(dataframe_bus):	
		#print(row[bus_scores[0]], row[bus_scores[1]],row[bus_scores[2]])
		bes = buschke_aggregate(y)
		S[ix]=bes[0]
		mean_a[ix] = np.mean(y)
		mean_g[ix] = mstats.gmean(y)
		suma[ix] = np.sum(y) 
		print('Total Aggregate S=', bes[0])
		print('arithmetic mean:', mean_a[ix], ' Geometric mean:', mean_g[ix], ' Sum:',suma[ix])
		print('Poly1d exponents drecresaing' ,bes[-1])
		print('Poly2 exponents drecreasing',bes[-2])
		print('\n')
		#pdb.set_trace()
	dataframe['S'] = S 
	df_copy['S'] = S 
	df_copy['mean_a'] = mean_a
	df_copy['mean_g'] = mean_g
	df_copy['suma'] = suma
	selected_columns = ['fcsrtrl1_visita1', 'fcsrtrl2_visita1', 'fcsrtrl3_visita1','S', 'mmse_visita1','conversionmci']
	selected_columns = selected_columns + ['mean_a', 'mean_g', 'suma']
	df_copy = df_copy[selected_columns]
	print('\n\n', 'Max S=', np.max(S), ' Max subject ix=', S.index(max(S)),' Min S=',np.min(S),\
		' min subject ix=', S.index(min(S)), ' avg= ',np.mean(S), ' std=', np.std(S) , '\n\n')	
	print('Maximum scores subject:', dataframe.max())
	print('minimum scores subject:', dataframe.min())
	corr = df_copy.corr(method='pearson', min_periods=1)
	print(corr)
	#plt.matshow(df_copy.corr())
	f, ax = plt.subplots(figsize=(10, 8))
	g = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
	#g = sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values)
	g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 8)
	g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 8)
	g.set_title('Correlation Bushcke, MMSE, conversionMCI')
	print('*** DONE!!!!! ***')
	pdb.set_trace()
	#time.sleep(10)
	plt.close('all')

if __name__== "__main__":
  main()

print('EXIT')





# def build_buschke_aggregate(y):
# 	"""build_buschke_aggregate: 
# 	Args:
# 	Output:
	
# 	"""
# 	from scipy.integrate import trapz, simps
# 	from scipy.interpolate import interp1d
# 	from matplotlib.patches import Polygon
# 	#f = interp1d(x, y)
# 	#f2 = interp1d(x, y, kind='cubic')
# 	npx = 3
# 	x = np.linspace(1,npx,npx,dtype=int)
# 	if type(y) is list:
# 		y = np.asarray(y)
# 	# fit polynomial of degree 2 that pass for (x[1:-1], b_list) points
# 	# z highest power first
# 	z = np.polyfit(x[:], y[:], 3)
# 	# line that goes though first and last point to calculate the difference between demorado and first
# 	z_fordemo = np.polyfit(x[0::3], y[0::3], 1)
# 	print('Interpolation 2nd order polynomial is %sx^2 + %sx + %s'%(str(z[0]), str(z[1]),str(z[2])))
# 	pol = np.poly1d(z)
# 	pol_line14 = np.poly1d(z)
# 	# first derivative of the fitting polynimium
# 	polder = np.polyder(pol)
# 	polder_line14 = np.polyder(pol_line14)
# 	# derivative of line that connects x0y0 with x3y3 on points x0 and x1
	
# 	slopedemo = y[-1] - y[0]

# 	#libre1, libre2, libre3, demorado = pol(x[1]), pol(x[2]), pol(x[2]), pol(x[3])
# 	slope1, slope2, slope3 = polder(x[0]), polder(x[1]), polder(x[2])
# 	# compare how the the fitting is libre1 - y[0]
# 	delta = y[-1] - y[0] #demorado - libre1 
# 	print('Demorado: %s - Libre1: %s == %s'%(str(y[-1]), str(y[0]), str(delta)))
# 	#Calculate the area under the polynomial
# 	# simpson method or trapezopid method trapz(y, x)
# 	area_c = simps(y, x) 
# 	print('The surface of the polynomial is {:6.5f}'.format(area_c))
# 	print('The S to maximize is: The sum of the surface under the fitted polynomial + \
# 		the first derivative at points 1,2,3 and the slope of line between first and last point : S + dy/dx(1,2,3) + dline1-4(4)')
# 	s_maximize = area_c + slope1 + slope2 #+ slope3 #+ slopedemo
# 	print('The S to maximize is =%10.3f =%10.3f + %10.3f + %10.3f \n'%(s_maximize, area_c, slope1, slope2))
# 	plot_integral = True
# 	if plot_integral is True:
# 		x = x[0:]
# 		y = y[0:]
# 		xp = x
# 		xp = np.linspace(1, 5, 10)
# 		_ = plt.plot(x, y, '.', xp, pol(xp), '-')
# 		plt.ylim(0,16)
# 		plt.show()
# 	print('The build_buschke_aggregate function is finished with s_maximize=%.3f\n'%(s_maximize))	
# 	return s_maximize



# a, b = 2, 9  # integral limits
# x = np.linspace(0, 10)
# y = func(x)

# fig, ax = plt.subplots()
# plt.plot(x, y, 'r', linewidth=2)
# plt.ylim(ymin=0)

# # Make the shaded region
# ix = np.linspace(a, b)
# iy = func(ix)
# verts = [(a, 0), * zip(ix, iy), (b, 0)]
# poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
# ax.add_patch(poly)

# plt.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
#          horizontalalignment='center', fontsize=20)

# plt.figtext(0.9, 0.05, '$x$')
# plt.figtext(0.1, 0.9, '$y$')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')

# ax.set_xticks((a, b))
# ax.set_xticklabels(('$a$', '$b$'))
# ax.set_yticks([])

# plt.show()