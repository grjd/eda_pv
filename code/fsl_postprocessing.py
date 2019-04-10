#######################################################
# Python program name	: 
#Description	: fsl_postprocessing.py
#Args           : postprocessing of fsl_anat. 
#				  Computes stats of tissue segmentation and subcortical segmentation    
#				  Creates the csv file with brain segmentation columns 	                                                                                    
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

def fsl_anat_postprocessing(images_path, df=None):
	"""fsl_anat_postprocessing: computes fslmaths and other fsl utils 
	Directory with fsl result must have the format pv_ID_yY.anat. (ID 4 digits number)
	Args: path with the results from fsl_anat, expected id_yi.anat directories, df(None)
	Output: datafame with 9 columns for volumes 3 tissues, saved as csv
	"""
	import fnmatch, re
	import glob
	#images_path= '/Users/jaime/Downloads/test_code'
	#visit = 1
	#fslstats -V output <voxels> <volume> (for nonzero voxels)
	T1_vols = {'scaling2MNI':[],'volume_bnative':[], 'volume_bMNI':[] }
	col_scaling = []
	for b in T1_vols.keys():
		col_scaling.append(b+ '_visita1')
	brain_dict = {'csf_volume':[],'gm_volume':[], 'wm_volume':[], 'csf_mni_volume':[],'gm_mni_volume':[], 'wm_mni_volume':[]}
	col_names = []
	for b in brain_dict.keys():
		col_names.append(b+ '_visita1')
	brain_sub_dict = {'BrStem':[],'L_Accu':[], 'L_Amyg':[], 'L_Caud':[],'L_Hipp':[], 'L_Pall':[], 'L_Puta':[],'L_Thal':[],\
	'R_Accu':[], 'R_Amyg':[], 'R_Caud':[],'R_Hipp':[], 'R_Pall':[], 'R_Puta':[],'R_Thal':[]}
	col_sub_names = []
	for b in brain_sub_dict.keys():
		col_sub_names.append(b+ '_visita1')	
	# open csv with subjects 
	if df is None: df = pd.read_csv(os.path.join(images_path, 'PVDB.csv'))
	print('Dataframe stored Shape==', df.shape)

	# remove / last char in id 
	df['id'] = df['id'].astype(str).str[:-1].astype(np.int64)
	# set index the subject id vallecas
	#df.set_index('id', inplace=True, verify_integrity=True)
	# add empty columns for scaling and brain volumes
	for col in col_scaling:
		df[col] = np.nan
	# add empty columns for the results of tissue segmentation
	for col in col_names:
		df[col] = np.nan
	# add empty columns for the results of subcortical segmentation
	for col in col_sub_names:
		df[col] = np.nan
	print('Columns added for storing tissue segmentation Shape==', df.shape)
	# to access by subject id: df.loc[[id]], to access by row df.iloc[[row]]
	# check last row for last subject it is the same df.loc[[1213]] == df.iloc[[df.shape[0]-1]]
	for root, directories, filenames in os.walk(images_path):
		for directory in directories:
			ff = os.path.join(root, directory) 
			anatdir = os.path.basename(os.path.normpath(ff))
			# Expected dir name is pv_ID_yY.anat
			if ff.endswith('.anat') & anatdir.startswith('pv_'):
				print('anat directory at:',ff)
				#id_subject = os.path.basename(os.path.normpath(ff))[0:4]
				id_subject = anatdir.split('_')[1]
				print('ID SUBJECT==',id_subject)
				# read scaling and brain volume from T1_vols.txt file 
				brain_dict_subject = compute_T1vol(ff, T1_vols)
				for col in T1_vols.keys():
					colname_indf = [s for s in col_scaling if col in s][0]
					df.loc[df['id']==int(id_subject), colname_indf] = brain_dict_subject[col]
					#df.iloc[int(id_subject), df.columns.get_loc(colname_indf)] = brain_dict_subject[col]
					print('T1s vols Updated in df for subject id:', id_subject, ' column:', colname_indf, ' key', col, '==', brain_dict_subject[col] )

				# call to compute_tissue_segmentation for PVE of CSG; GM and WM
				brain_dict_subject = compute_tissue_segmentation(ff, brain_dict)
				for col in brain_dict.keys():
					colname_indf = [s for s in col_names if col in s][0]
					df.loc[df['id']==int(id_subject), colname_indf] = brain_dict_subject[col]
					#df.iloc[int(id_subject), df.columns.get_loc(colname_indf)] = brain_dict_subject[col]
					print('Tissue vols Updated in df for subject id:', id_subject, ' column:', colname_indf, ' key', col, '==', brain_dict_subject[col] )
				# call to compute_subcortical_segmentation for Segmentation of subcortical strutures
				brain_dict_subject = compute_subcortical_segmentation(ff, brain_sub_dict)
				for col in brain_sub_dict.keys():
					colname_indf = [s for s in col_sub_names if col in s][0]
					df.loc[df['id']==int(id_subject), colname_indf] = brain_dict_subject[col]
					#df.iloc[int(id_subject), df.columns.get_loc(colname_indf)] = brain_dict_subject[col]
					print('SubCort vols Updated in df for subject id:', id_subject, ' column:', colname_indf, ' key', col, '==', brain_dict_subject[col] )

	#types =  [type(x) == int for x in df['wm_mni_volume_visita1']]
	#pd.isnull(df['gm_volume_visita1']).all()
	# ave dataframe as csv
	csv_name = 'PVDB_pve_sub.csv'
	csv_name = os.path.join(images_path, csv_name)
	
	df.to_csv(csv_name)
	print('Saved csv at:', csv_name, '\\n')

	return df

def compute_T1vol(ff, T1_vols):
	"""compute_T1vol :  read .anat/T1_vols.txt file and get 
	Scaling factor from T1 to MNI, Brain volume in mm^3 (native/original space) an
	Brain volume in mm^3 (normalised to MNI)
	Args:
	Output:T1_vols if T1_vols does not exist return T1_vbols all 0
	"""
	scaling = 'Scaling factor'
	native = 'native/original space'
	normal = 'normalised to MNI'
	T1vols_file = os.path.join(ff, 'T1_vols.txt')
	T1_vols = dict.fromkeys(T1_vols, np.nan)
	#T1_vols['scaling2MNI'],T1_vols['volume_bnative'],T1_vols['volume_bMNI'] = 0,0,0 
	if os.path.exists(T1vols_file) is False:
		print(T1vols_file)
		warnings.warn("WARNING expected T1_vols.txt file  DOES NOT exist!!!! \n\n")
	else:
		with open(T1vols_file,'r') as f:
			for line in f:
				if line.find(scaling) != -1:
					T1_vols['scaling2MNI']= float(line.split()[-1])
				elif line.find(native) != -1:
					T1_vols['volume_bnative']= float(line.split()[-1])
				elif line.find(normal) != -1:
					T1_vols['volume_bMNI']= float(line.split()[-1])
	return T1_vols

def compute_subcortical_segmentation(ff, brain_dict):
	"""compute_subcortical_segmentation: compute statistics of subcortical regions segmented with fsl_anat.
	Convert the mesh (.vtk file) into a .nii file and calculate the volume with fslstats
	Args: ff path where to dind first_resukts/.vtk files for the subcortical structures
	brain_dict: dictionary with keys each subcortical structure
	Output:brain_dict: dictionary with keys each subcortical structure
	"""

	# first_util labels for each subcortical https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide
	first_u_labels = {'BrStem':16,'L_Accu':26, 'L_Amyg':18, 'L_Caud':11,'L_Hipp':17, 'L_Pall':13, 'L_Puta':12,'L_Thal':10,\
	'R_Accu':58, 'R_Amyg':54, 'R_Caud':50,'R_Hipp':53, 'R_Pall':52, 'R_Puta':51,'R_Thal':49}
	first_path = os.path.join(ff, 'first_results')
	vtkhead, vtktail = 'T1_first-', '_first.vtk'
	t1_file = os.path.join(ff, 'T1.nii.gz')
	brain_dict = dict.fromkeys(brain_dict, np.nan)
	if os.path.exists(t1_file) is False:
		warnings.warn("ERROR T1 DOES NOT exist!! Exiting function. \n")	
	if os.path.exists(first_path) is False:
		warnings.warn("ERROR expected first_results directory DOES NOT exist!!Exiting function. \n")		
	else:
		#Convert mesh (vtk) to nii (Volume) and from that calculate the volume of the structure
		for sub in brain_dict.keys():
			vtk = vtkhead + sub + vtktail #T1_first-R_Accu_first.vtk
			vtk = os.path.join(first_path, vtk)

			if os.path.exists(vtk) is False:
				print('\n WARNING::::::Subcortical structure missing:', vtk)
				brain_dict[sub] = 0
				warnings.warn("WARNING vtk file DOES NOT exist !!! Setting to 0 in csv \n\n")
			else:
				label = str(first_u_labels[sub])
				out_vol = sub + "_meshvolume"
				out_vol = os.path.join(first_path, out_vol)
				#first_utils --meshToVol -m mesh.vtk -i t1_image.nii.gz -l fill_value -o output_name
				print("Calling to:: first_utils","--meshToVol", "-m",vtk, "-i", t1_file, "-l", label, "-o", out_vol)
				util_command = check_output(['first_utils','--meshToVol', '-m', vtk, '-i', t1_file, '-l', label, '-o', out_vol])								
				#stats_command = check_output(["fslstats",out_vol, "-M", "-V"])
				lower_ix = str(int(first_u_labels[sub]) - 1)
				upper_ix = str(int(first_u_labels[sub]) + 1)
				#fslstats R_Hipp_meshvol.nii.gz  -l 52 -u 54 -V
				print('Calling to:: fslstats', out_vol, "-l <lower_ix>=",lower_ix, "-u <upper_ix>=", upper_ix, " -V" )
				stats_command = check_output(["fslstats",out_vol, "-l",lower_ix, "-u", upper_ix, "-V"])
				stats_command = stats_command.split(' ')
				#volmm3 = float(stats_command[0])*float(stats_command[2])
				#volvoxels = float(stats_command[0])*float(stats_command[1])
				volmm3 = float(stats_command[1])
				volvoxels = float(stats_command[0])
				brain_dict[sub] = int(round(volmm3))
				print('DONE with::', vtk, ' Mesh Volume ==', volmm3, '\n\n')
				#fslstats mesh_right_hipp -M -V | awk '{ print $1 * $2 }' #mean voxel \times nb voxels
	print('compute_subcortical_segmentation ENDED \n')		
	return brain_dict


def compute_tissue_segmentation(ff, brain_dict):
	"""compute_tissue_segmentation:
	Args:ff is the .anat directory containing the pve files from fsl_anat
	Output: dict: dictionary with tissue measures volume and voxel intensity per tissue
	dict = {'GM_volume':[],'CSF_volume':[], 'WM_volume':[], 'GM_voxel_int':[], 'CSF_voxel_int':[], 'WM_voxel_int':[]}
	"""
	brain_dict = dict.fromkeys(brain_dict, np.nan)
	if os.path.exists(ff) is False:
		warnings.warn("ERROR expected .anat directory: DOES NOT exist")
	else:
		#-M output mean (for nonzero voxels) -V output <voxels> <volume> (for nonzero voxels)
		# volume in mm3. 
		#fslstats structural_bet_pve_1 -M -V | awk '{ print $1 * $3 }'
		file_root, file_coda = 'T1_fast_pve_', '.nii.gz'
		csf_file = file_root+str(0) + file_coda
		csf_mni_file = file_root+str(0) + '_MNI' + file_coda
		gm_file = file_root+str(1) + file_coda
		gm_mni_file = file_root+str(1) + '_MNI' + file_coda
		wm_file = file_root+str(2) + file_coda
		wm_mni_file = file_root+str(2) + '_MNI' + file_coda
		#'T1_fast_pve_0.nii.gz', 'T1_fast_pve_1.nii.gz', 'T1_fast_pve_2.nii.gz'
		#tissues = [csf_file, gm_file, wm_file] 
		for i in np.arange(0,2+1):
			if i==0:
				tif,tifmni  = csf_file, csf_mni_file
				vol_label, vol_mni_label = 'csf_volume', 'csf_mni_volume'
			elif i==1:
				tif,tifmni  = gm_file, gm_mni_file
				vol_label, vol_mni_label = 'gm_volume', 'gm_mni_volume'
			elif i==2:
				tif,tifmni  = wm_file, wm_mni_file
				vol_label, vol_mni_label = 'wm_volume', 'wm_mni_volume'
		#for tif in tissues:
			tifpath = os.path.join(ff, tif)
			out = check_output(["fslstats",tifpath, "-M", "-V"])
			out = out.split(' ')
			#pdb.set_trace()
			volmm3 = float(out[0])*float(out[2])
			volvoxels = float(out[0])*float(out[1])
			brain_dict[vol_label] = int(round(volmm3))
			print('Label for ', tifpath, ' ', vol_label, '==', volmm3)
			# if pve_i_MNI exists 
			tifpath = os.path.join(ff, tifmni)
			if os.path.exists(tifpath) is True:
				out = check_output(["fslstats",tifpath, "-M", "-V"])
				out = out.split(' ')
				volmm3 = float(out[0])*float(out[2])
				volvoxels = float(out[0])*float(out[1])
				brain_dict[vol_mni_label] = int(round(volmm3))
				print('Label for ', tifpath, ' ', vol_mni_label,  '==', volmm3)
			else:
				print('**** pve_i_MNI not found at', tifpath,' skipping...\n')
				# assign dummy values to pve_i_MNI
				brain_dict[vol_mni_label] = np.nan
	return brain_dict

def main():
	#SomeCommand 2>&1 | tee SomeFile.txt
	#python -c 'import fsl_postprocessing; fsl_postprocessing.main()'|tee texto.txt
	print('Calling to fsl_anat_postprocessing(images_path, df=None) to calculate TISSUE and SUBCORTICAL Segmentation \n\n' )
	print("Current date and time: " , datetime.datetime.now(), '\n\n')
	#print('FSL version is: sudo less $FSLDIR/etc/fslversion'). remote 5.0.10, local 5.0.9
	images_path = '/Volumes/Promise_Pegasus2/jg33/fslanat'
	df_path = '/Volumes/Promise_Pegasus2/jg33/fslanat/Proyecto_Vallecas_7visitas_4_dec_2018.csv'
	#txtoutput = '/Volumes/Promise_Pegasus2/jg33/fslanat/postprocessing_output.txt'
	#txtoutput = 'postprocessing_output.txt'
	#sys.stdout = open(txtoutput, "w")
	images_path = '/Users/jaime/Downloads/test_code'
	df_path = '/Users/jaime/Downloads/test_code/Proyecto_Vallecas_7visitas_4_dec_2018.csv'
	#df = pd.read_csv(os.path.join(images_path, 'PVDB.csv'))
	df = pd.read_csv(df_path)
	print('Calling to fsl_anat_postprocessing(images_path, df)', images_path, df_path)
	dataframe = fsl_anat_postprocessing(images_path, df)
	print('\n\n\n PROGRAM FINISHED check csv file::', os.path.join(images_path, 'PVDB_pve_T1vols.csv'))

if __name__ == "__name__":
	
	main()