import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas
import traceback
import json 
import pickle
from astropy.constants import M_sun, M_jup, M_earth, R_sun, R_jup, R_earth, au 
from scipy import stats 
import pandoramoon as pandora 
from pandoramoon.helpers import ld_convert, ld_invert 
import corner 
import time

rcParams['font.family'] = 'serif'
rcParams.update({'font.size': 13})

#-------------------------------

#### this script will generate plots of the systems -- fit light curves, along with ground truth light curves, and look at demographics

#plt.rcParams.update({'font.size': 18,
 #                    'xtick.labelsize' : 18,
  #                   'ytick.labelsize' : 18})

#plt.rcParams.update({'figsize':(8,8)})

def factor(number):
	factors = []
	for i in np.arange(1,number+1,1):
		if number % i == 0:
			factors.append(i)
	return factors 
#-------------------------------

#lcgen_name = 'march29_2023_'
lcgen_name = input("What is the lgcen_name? [ex: may16_2023]: ")
lcgen_name = lcgen_name+'_'
run_name = input('What is the run_name? [ex: run8_10, or combined_runs]: ')


show_model = input('Do you want to show the ground truth model? y/n: ')

#force_ground_truth_impact = input('Do you want to force the modeled light curves to take ground truth values for IMPACT PARAMETER y/n: ')
force_ground_truth_impact = 'n'
#force_ground_truth_ecc = input('Do you want to force the modeled light curves to take ground truth values for ECCENTRICITY? y/n: ')
force_ground_truth_ecc = 'n'
#force_ground_truth_limb_darkening = input('Do you want to force ground truth limb darkening? y/n: ')
force_ground_truth_limb_darkening = 'n'
#use_hires_times = input("Do you want to use hires times? y/n: ")
use_hires_times = 'y'
if use_hires_times == 'n':
	return_original=True 
elif use_hires_times == 'y':
	return_original=False


single_system = input('Do you want to plot a single system? y/n: ')
if single_system == 'y':
	systems_run = [input('Input the system name: ')]

draw_from_posteriors = input('Do you want to draw from the posteriors? y/n: ')
if draw_from_posteriors == 'y':
	ndraws = int(input('How many draws from the posteriors do you want? '))

show_lc_plots = input('Do you want to show light curve plots? y/n: ')
show_final_plots = input('Do you want to show the final (summary) plots? y/n: ')

skip_already_made = input('Skip making plots that you already made? y/n: ')

#multipanel_lcs = input('Do you want each transit to get its own panel? y/n: ') #### NO MORE THAN 16
multipanel_lcs = 'y'
if multipanel_lcs == 'n':
	truncated_plot = False 

if run_name == 'combined_runs':
	use_earlier_models = 'n'
else:
	use_earlier_models = input('Do you want to use earlier planet models, if the most recent run is unavailable? y/n: ')

if use_earlier_models == 'y':
	check_11_10 = 'n'
	show_run8_10 = 'y'
	overplot = input('Do you want to overplot earlier models? y/n: ')

elif use_earlier_models == 'n':
	overplot = 'y' 
	if run_name == 'combined_runs':
		check_11_10 = 'n'
		show_run8_10 = 'n'
	else:
		check_11_10 = input('Do you want to check run8_10 against run11_10? y/n: ')

	if check_11_10 == 'y':
		show_run8_10 = 'n'



fit_line_to_scatter = input('Do you want to fit a line to the scatter plots? y/n: ')
### each file in modeling_resultsdir is its own system, containing two models ('planet_only' and 'planet_moon')
### the posteriors for each run are found in the chains directory ('equal_weighted_post.txt')
### there are additional relevant files in the info folder ('post_summary.csv' and 'results.json')


#-------------------------------
external_projectdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling'
projectdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'

use_external_files = input('Use files on the external? y/n: ')
if use_external_files == 'y':
	projectdir = external_projectdir 
	plot_externally = 'y'
else:
	plot_externally = input('Do you want to send plots to the external hard drive? y/n: ')

"""
if draw_from_posteriors == 'y':
	external_projectdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling'
	projectdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'
elif draw_from_posteriors == 'n':
	projectdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'
"""

timed_out = np.genfromtxt(projectdir+'/run8_10_timed_out_runs.txt', dtype=str)
timed_out_runs = timed_out.T[0]


#modeling_resultsdir = projectdir+'/modeling_results/'+lcgen_name+'runs/'+run_name
if (draw_from_posteriors == 'y'):
	#modeling_resultsdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling/modeling_results_for_download/'+run_name
	modeling_resultsdir = projectdir+'/modeling_results_for_download/'+run_name
	list_of_runs = os.listdir(projectdir+'/modeling_results_for_download')
	run11_10_resultsdir = projectdir+'/modeling_results_for_download/run11_10'
	#run11_10_resultsdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling/modeling_results_for_download/run11_10'
elif draw_from_posteriors == 'n':
	modeling_resultsdir = projectdir+'/modeling_summaries_for_download/'+run_name
	list_of_runs = os.listdir(projectdir+'/modeling_summaries_for_download')
	run11_10_resultsdir = projectdir+'/modeling_summaries_for_download/run11_10'
systems_run = os.listdir(modeling_resultsdir) ### of the form july13_nores_16, etc
run11_10_systems_run = os.listdir(run11_10_resultsdir)


try:
	with open(projectdir+'/'+run_name+'_poad.pickle', 'rb') as poad_pickle:
		planet_only_attributes_dict = pickle.load(poad_pickle)
	po_sims_run = planet_only_attributes_dict['sim']
	loaded_poad = True
	overwrite_poad = input('planet_only_attributes_dict loaded. Overwrite? y/n: ')
	if overwrite_poad == 'y':
		loaded_poad = False

except:
	print('could not load the planet_only_attributes_dict.')
	loaded_poad = False 

try:
	with open(projectdir+'/'+run_name+'_pmad.pickle', 'rb') as pmad_pickle:
		planet_moon_attributes_dict = pickle.load(pmad_pickle)
	pm_sims_run = planet_moon_attributes_dict['sim']
	loaded_pmad = True 
	overwrite_pmad = input('planet_moon_attributes_dict loaded. Overwrite? y/n: ')
	if overwrite_pmad == 'y':
		loaded_pmad = False 

except:
	print('could not load the planet_only_attributes_dict.')
	loaded_pmad = False 


#skip_previously_run = input('Do you want to skip previously run systems? y/n: ')
skip_previously_run = 'n'



#-------------------------------

plotdir = modeling_resultsdir+'/plots'
external_plotdir = external_projectdir+'/plots'
if os.path.exists(plotdir) == False:
	os.system('mkdir '+plotdir)
if os.path.exists(external_plotdir) == False:
	os.system('mkdir '+external_plotdir)

if plot_externally == 'y':
	plotdir = external_plotdir 


#model_lcdir = projectdir+'/model_LCs_'+lcgen_name[:-1] ### leave off the final underscore -- MAY OR MAY NOT MATCH WITH THE LIGHTCURVE! DUE TO AN OFFSET.
model_lcdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling/model_LCs_'+lcgen_name[:-1]
final_lcdir = projectdir+'/final_lightcurves_'+lcgen_name[:-1] ### leave off the final underscore

catalogue_path = projectdir+'/'+lcgen_name+'catalogue.txt' 
catalogue = np.genfromtxt(catalogue_path, skip_header=1, dtype=str) 
catalogue_columns = ['system_file',	 #0
	'kepler_file', #1
	'num_moons', #2 
	'Planet_period(days)', #3
	'P_Transits_present/absent', #4
	'SNR1', #5
	'SNR2', #6
	'SNR3', #7
	'SNR4', #8
	'SNR5', #9
	'M_Transits_present/absent1', #10
	'M_Transits_present/absent2', #11
	'M_Transits_present/absent3', #12
	'M_Transits_present/absent4', #13
	'M_Transits_present/absent5', #14
	'DW_statistic', #15
	'Best_offset', #16
	'Planet_Transit_duration', #17
	't0_bary' #18
	]

catalogue_systems = catalogue.T[0] 
catalogue_SNR1s = catalogue.T[5]
catalogue_SNR2s = catalogue.T[6]
catalogue_SNR3s = catalogue.T[7]
catalogue_SNR4s = catalogue.T[8]
catalogue_SNR5s = catalogue.T[9]
catalogue_planet_transit_durations = catalogue.T[17]

#-------------------------------

moon_options = ['I', 'II', 'III', 'IV', 'V']
#model_colors = ['#E9A029', '#5EBBA2', '#D63F1B'] ### planet_only, planet_moon, ground_truth 
model_colors = ['#4059AD', '#97D8C4', '#F4B942'] 
sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']

#-------------------------------
#catalogue = np.genfromtxt(catalogue_path, skip_header=2, dtype=str) 
#### catalogue contains: system_file	kepler_file	num_moons	Planet_period(days)	P_Transits_present/absent	
###  SNR1	SNR2	SNR3	SNR4	SNR5	M_Transits_present/absent1	M_Transits_present/absent2	M_Transits_present/absent3	
###  M_Transits_present/absent4	M_Transits_present/absent5	DW_statistic	Best_offset	Planet_Transit_duration	t0_bary


reasonable_moon_fits = []
reasonable_moon_fit_nmoons = []
unreasonable_moon_fits = []
unreasonable_moon_fit_nmoons = []
sims_with_planet_model = []
sims_without_planet_model = []
sims_with_moon_model = []
sims_without_moon_model = []

#-------------------------------
#### posterior column names
planet_only_names = [
	'per_bary', 'a_bary', 'r_planet', 'b_bary', 
	'ecc_bary', 'w_bary', 't0_bary_offset', 'q1', 'q2',
	]

planet_only_names_labels = [
	'barycenter period', 'barycenter semimajor axis', r'$R_P / R_{*}$', 'impact parameter', 'barycenter eccentricity', r'barycenter $\omega$', 
	r'barycenter $t_0$ offset', r'$q_1$', r'$q_2$',
	]

planet_only_corner_names = [
	r'$P$', r'$a$', r'$R_P [\oplus]$', r'$b$', r'$e$', r'$\omega$', r'$\Delta t_0$', r'$q_1$', r'$q_2$',
	]

planet_moon_names = [
	'per_bary', 'a_bary', 'r_planet', 'b_bary', 
	'ecc_bary', 'w_bary', 't0_bary_offset', 'M_planet', 'r_moon',
	'per_moon', 'tau_moon', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2',
	]

planet_moon_names_labels = [
	'barycenter period', 'barycenter semimajor axis', r'$R_P / R_{*}$', 'impact parameter', 'barycenter eccentricity', r'barycenter $\omega$', 
	r'barycenter $t_0$ offset', 'planet mass', 'moon radius', 'moon period', r'moon $\tau$', r'moon $\Omega$', 'moon inclination', 'moon mass', r'$q_1$', r'$q_2$',
	]

planet_moon_corner_names = [
	r'$P$', r'$a$', r'$R_P [\oplus]$', r'$b$', r'$e$', r'$\omega$', r'$\Delta t_0$', r'$M_P [\oplus]$', r'$R_S [\oplus]$', r'$P_S$', r'$\tau$', 
	r'$\Omega_S$', r'$i_S$', r'	', r'$q_1$', r'$q_2$', 
	]

planet_ground_truth_names = [
	'm', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 
	'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'nmoons', 'ntransits'
	]

planet_ground_truth_labels = [
	r'$M_P$', r'$R_P$', r'$R_{*}$', r'$R_P / R_{*}$', r'$\rho_{*}$', r'$b_B$', r'$q_1$', r'$q_2$', r'$a_B$', r'$P_B$', 
	r'$\rho_P$', r'$a_B / R_{P}$', r'$e_B$', r'$\mathrm{inc}_B$', r'$\varpi$', r'$f$', r'$P$', r'$R_{\mathrm{Hill}}$', '# of moons', '# of transits'
	]

moon_ground_truth_names = [
	'm', 'r', 'msmp', 'rsrp', 'a', 'aRp', 'e', 'inc', 'pomega', 
	'f', 'P', 'RHill', 'spacing', 'period_ratio', 
	]

moon_ground_truth_labels = [
	r'$M_S$', r'$R_S$', r'$M_S / M_P$', r'$R_S / R_P$', r'$a_S$', r'$a_S / R_{P}$', r'$e_S$', r'$\mathrm{inc}_S$', 
	r'$\varpi$', r'$f_S$', r'$P_S$', r'$R_{\mathrm{Hill}}$', 'spacing', 'period ratio'
	]

#-------------------------------

#### map the model fitting names to the simulation ground truth names
model_to_ground_truth_dict = {}
ground_truth_to_model_dict = {} 

model_to_ground_truth_dict['per_bary'] = 'Pp' ### days
model_to_ground_truth_dict['a_bary'] = 'aRstar' ### a/Rstar
model_to_ground_truth_dict['r_planet'] = 'RpRstar' #### fraction of the stellar radius
model_to_ground_truth_dict['R_star'] = 'Rstar'
model_to_ground_truth_dict['b_bary'] = 'impact' ### dimensionless
model_to_ground_truth_dict['ecc_bary'] = 'e' ### dimensionless [0-1]
#model_to_ground_truth_dict['w_bary'] = 'pomega' ### 0-360
model_to_ground_truth_dict['q1'] = 'q1' #### 0-1
model_to_ground_truth_dict['q2'] = 'q2'
model_to_ground_truth_dict['M_planet'] = 'm'
#model_to_ground_truth_dict['w_bary'] = 
#model_to_ground_truth_dict['t0_bary_offset'] = 
"""
model_to_ground_truth_dict['r_moon'] = 'r'
model_to_ground_truth_dict['per_moon'] = 'P'
model_to_ground_truth_dict['tau_moon'] = 'f' #### careful -- f is 0-360, tau is 0-1.
model_to_ground_truth_dict['Omega_moon'] = 'pomega'
model_to_ground_truth_dict['i_moon'] = 'inc'
model_to_ground_truth_dict['M_moon'] = 'm'
"""

ground_truth_to_model_dict['Pp'] = 'per_bary'
ground_truth_to_model_dict['aRstar'] = 'a_bary'
ground_truth_to_model_dict['RpRstar'] = 'r_planet'
ground_truth_to_model_dict['Rstar'] = 'R_star'
ground_truth_to_model_dict['impact'] = 'b_bary'
ground_truth_to_model_dict['e'] = 'ecc_bary'
ground_truth_to_model_dict['q1'] = 'q1'
ground_truth_to_model_dict['q2'] = 'q2'
ground_truth_to_model_dict['m'] = 'M_planet'

#-------------------------------









def gen_model(model_dict, times, t0_bary=None, model_type=None):

	cadence = int(1 / (times[1] - times[0]))
	print('generated model cadence: ', cadence)

	#planet_moon_names = [
	#'per_bary', 'a_bary', 'r_planet', 'b_bary', 
	#'ecc_bary', 'w_bary', 't0_bary_offset', 'M_planet', 'r_moon',
	#'per_moon', 'tau_moon', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2',
	#]

	u1, u2 = ld_convert(model_dict['q1'], model_dict['q2'])

	model_keys = model_dict.keys()

	params = pandora.model_params()
	#R_sun = 696_342_000 ### FIX THIS
	#print("FIX THE STELLAR RADIUS!!! IT'S HARD CODED RIGHT NOW!")
	#params.R_star = R_sun  # [m]
	params.R_star = model_dict['R_star'] 
	params.u1 = u1
	params.u2 = u2

	params.per_bary = model_dict['per_bary']
	params.a_bary = model_dict['a_bary']
	params.r_planet = model_dict['r_planet']
	params.b_bary = model_dict['b_bary']
	params.t0_bary = t0_bary
	params.t0_bary_offset = model_dict['t0_bary_offset']
	params.w_bary = model_dict['w_bary']
	params.ecc_bary = model_dict['ecc_bary']

	if model_type == 'planet_only':
		params.M_planet = model_dict['M_planet'] #### comes from the simulation! It hasn't been fit!
		params.r_moon = 1e-8 
		params.per_moon = 30. ### same as the modeling run 
		params.tau_moon = 0.
		params.Omega_moon = 0.
		params.i_moon = 0.
		params.e_moon = 0.
		params.w_moon = 0.
		params.M_moon = 1e-8 

	elif model_type == 'planet_moon':
		params.M_planet = model_dict['M_planet']
		params.r_moon = model_dict['r_moon'] 
		params.per_moon = model_dict['per_moon']
		params.tau_moon = model_dict['tau_moon']
		params.Omega_moon = model_dict['Omega_moon']
		params.i_moon = model_dict['i_moon']
		params.e_moon = 0. #### set to zero for the modeling
		params.w_moon = 0. #### set to zero for the modeling  -- irrelevant when ecc=0.
		params.M_moon = model_dict['M_moon']


	params.epochs = np.nanmax(times) - np.nanmin(times) / model_dict['per_bary']  # [int]
	params.epoch_duration = 4  # [days] #### not in the modeling run 
	params.cadences_per_day = cadence  # [int] ### not in the modeling run 
	params.epoch_distance = model_dict['per_bary'] ### same as the modeling run 
	params.supersampling_factor = 1  # [int] ### same as the modeling run 
	params.occult_small_threshold = 0.01  # [0..1] ### same as the modeling run 
	params.hill_sphere_threshold = 1.1 ### same as the modeling run 
	params.numerical_grid = 25  ###same as the modeling run 

	param_attributes = vars(params)
	print(' ')
	print("PANDORA INPUTS FOR MODEL_TYPE ", model_type)
	for key in param_attributes.keys():
		print(key, param_attributes[key])
	print(' ')

	#time = pandora.time(params).grid()

	model = pandora.moon_model(params)

	injected_flux_total, injected_flux_planet, injected_flux_moon = model.light_curve(times)

	return times, injected_flux_total 

#-------------------------------

def hires_time_maker(times, return_original=False):
	#### NOTE -- this function is rewritten from the above using ChatGPT.

	# Compute the median spacing between data points
	spacings = np.diff(times)
	median_spacing = np.nanmedian(spacings)

	# Find the chunk boundaries, and only interpolate
	gaps = np.where(spacings > 10 * median_spacing)[0] + 1
	chunk_starts = np.concatenate(([0], gaps)) 
	chunk_ends = np.concatenate((gaps, [len(times)])) - 1 
	
	# Compute high-resolution times for each chunk
	hires_times = []
	for csi, cei in zip(chunk_starts, chunk_ends):
		#print('csi, cei = ', csi, cei)
		hires_times.append(np.arange(times[csi]-1, times[cei]+1, 0.1*median_spacing))
	
	hires_times = np.sort(np.concatenate(hires_times))

	if return_original == False:
		return hires_times
	else:
		return times


#-------------------------------







#### first thing, let's just go through each model run, plot the light curve with the solution overlaid!

planet_only_posterior_mean_dict = {}
planet_only_posterior_median_dict = {}
planet_only_posterior_sigma_dict = {}

planet_moon_posterior_mean_dict = {}
planet_moon_posterior_median_dict = {}
planet_moon_posterior_sigma_dict = {}

ground_truth_dict = {}
PMval_minus_POval_dict = {}
PMval_minus_ground_truth_dict = {}
POval_minus_ground_truth_dict = {}

planet_only_attributes_dict = {}
planet_moon_attributes_dict = {}

moon_recovery_dict = {}
#### this will take systems as the first key, and the second key will take the values


#-------------------------------

#### these systems are excluded based on visual inspection which indicated strange photometry or fit features.
systems_to_exclude = [
	'july13_nores_157', 
	'july13_nores_752', 
	'july13_nores_784', 
	'july13_res_161', 
	'july13_res_317', 
	'july13_res_676', 
	'july13_res_738'
	]

final_sim_list = np.load(projectdir+'/final_sim_list.npy')
use_final_sim_list = input('Use the final sim list (based on previous cuts)? y/n: ')
if use_final_sim_list == 'y':
	systems_to_include = final_sim_list
else:
	systems_to_include = systems_run


#-------------------------------


##### THIS IS WHERE THE GIANT LOOP BEGINS!!!!! 


for nsystem, system in enumerate(systems_to_include):

	if (skip_previously_run == 'y') and (system in pm_sims_run):
		continue


	### INTIALIZE THESE THINGS UP HERE, THEY WILL GET CHANGED IN EACH ITERATION OF THE LOOP, IF NECESSARY
	planet_only_logz = None ### initialize as this, to be replaced later 
	planet_only_loglike = None
	planet_only_BIC = None 
	planet_moon_logz = None ### initialize as this, to be replaced later
	planet_moon_loglike = None
	planet_moon_BIC = None
	delta_logZ = None 
	bayes_factor = None
	deltaBIC = None 
	planet_model_exists = False 
	moon_model_exists = False 


	print(' ')
	print(' ')
	print(' - - - - - - - - - - ')
	print(' SYSTEM: ', system)
	print(' ')
	moon_recovery_dict[system] = {}
	planet_only_post_medians = {}
	planet_only_post_means = {}
	planet_only_post_stddevs = {}	
	planet_moon_post_medians = {}
	planet_moon_post_means = {}
	planet_moon_post_stddevs = {}


	cat_system_idx = np.where(system == catalogue_systems)[0]
	if len(cat_system_idx) > 1:
		#### use the first one -- there are duplicates!!! 
		cat_system_idx = cat_system_idx[0]
	try:
		cat_SNR1 = float(catalogue_SNR1s[cat_system_idx])
	except:
		cat_SNR1 = np.nan 
	try:
		cat_SNR2 = float(catalogue_SNR2s[cat_system_idx])
	except:
		cat_SNR2 = np.nan 
	try:
		cat_SNR3 = float(catalogue_SNR3s[cat_system_idx])
	except:
		cat_SNR3 = np.nan 
	try:
		cat_SNR4 = float(catalogue_SNR4s[cat_system_idx])
	except:
		cat_SNR4 = np.nan
	try:
		cat_SNR5 = float(catalogue_SNR5s[cat_system_idx])
	except:
		cat_SNR5 = np.nan

	try:
		cat_transit_duration = float(catalogue_planet_transit_durations[cat_system_idx])
	except:
		cat_transit_duration = np.nan 

	if 'july13' not in system:
		continue 
	#### system name is of the form july13_[res_nores]_[sim_number]
	if 'nores' in system:
		res_nores = 'nores'
	else:
		res_nores = 'res'

	system_reversed = system[::-1] ### reverse it, so the REVERSED number is at the end
	system_reversed_sim_number = system_reversed[:system_reversed.find('_')] ### isolate the REVERSED number
	sim_number = system_reversed_sim_number[::-1]  #### reverse it back


	#### THIS IS WHAT YOU'VE BEEN PLOTTING (MODEL)
	ground_truth_lc_filename = 'sim'+str(sim_number)+'_'+res_nores+'_modelLC.txt'
	ground_truth_lc = np.genfromtxt(model_lcdir+'/'+ground_truth_lc_filename, skip_header=1)
	number_of_ground_truth_lcs = ground_truth_lc.shape[1] - 1 ### the first column is times
	ground_truth_times = ground_truth_lc.T[0]
	ground_truth_fluxes = ground_truth_lc.T[1:] #### shape = (n light curves, n data points) 


	### get important attributes from the catalogue
	catalogue_idx = np.where(catalogue.T[0] == system)[0][0]
	system_t0_bary = float(catalogue[catalogue_idx][-1])
	system_nmoons = int(catalogue[catalogue_idx][2])
	system_timing_offset = float(catalogue[catalogue_idx][-3])
	system_period = float(catalogue[catalogue_idx][3])
	
	#### FIND THE SYSTEM TRANSIT TIMES
	system_transit_times = [system_t0_bary]
	while system_transit_times[-1] < np.nanmax(ground_truth_times):
		system_transit_times.append(system_transit_times[-1] + system_period)
	system_transit_times = system_transit_times[::-1] #### reverse it so that you can append
	while system_transit_times[-1] > np.nanmin(ground_truth_times):
		system_transit_times.append(system_transit_times[-1] - system_period)
	system_transit_times = system_transit_times[::-1] ### trase


	### get attributes from the simulation
	simulation_json_file = open(projectdir+'/sim_data/unpacked_details_july13'+'/'+system+'.json')
	simulation_dict = json.load(simulation_json_file)
	#### update the simulation_dictionary to include aRstar and ecc
	simulation_dict['Planet']['aRstar'] = simulation_dict['Planet']['a'] / simulation_dict['Planet']['Rstar']
	simulation_dict['Planet']['e'] = 0. #### set to zero for all simulations@ 

	simulation_planet_keys = list(simulation_dict['Planet'].keys()) # ['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill']
	simulation_moon_keys = list(simulation_dict['I'].keys()) ### each system has at least one moon!!! 
	simulation_Mplanet = simulation_dict['Planet']['m'] #### kg 
	simulation_Rstar = simulation_dict['Planet']['Rstar'] #### meters 


	for spk in simulation_planet_keys:
		if nsystem == 0:
			ground_truth_dict[spk] = [simulation_dict['Planet'][spk]]
		else:
			ground_truth_dict[spk].append(simulation_dict['Planet'][spk])

	if nsystem == 0:
		ground_truth_dict['sim'] = [system]
	else:
		ground_truth_dict['sim'].append(system)


	if nsystem == 0:
		ground_truth_dict['nmoons'] = [len(simulation_dict.keys())-1] ### if simulation dict has Planet and I, that's two keys, and one moon.
	else:
		ground_truth_dict['nmoons'].append(len(simulation_dict.keys())-1)


	for smk in simulation_moon_keys:
		for moon_option in moon_options:
			new_smk = moon_option+'_'+smk ### if the moon is I, and the smk == 'm', the new_smk is "I_m"
			if moon_option in simulation_dict.keys():
				try:
					ground_truth_dict[new_smk].append(simulation_dict[moon_option][smk])
				except:
					ground_truth_dict[new_smk] = [simulation_dict[moon_option][smk]]



	##### load the final_lightcurve (the lightcurve with the injected model)
	final_lightcurve_path = final_lcdir+'/sim'+str(sim_number)+'_'+res_nores+'_final_lightcurve.csv' ### final_lcdir = projectdir+'/final_lightcurves_'+lcgen_name[:-1] ### leave off the final undersco
	final_lc = pandas.read_csv(final_lightcurve_path, header=2)
	final_lc_times = np.array(final_lc['#times']).astype(float)
	final_lc_fluxes = np.array(final_lc['fluxes']).astype(float)
	final_lc_errors = np.array(final_lc['errors']).astype(float)	




	model_rundir = modeling_resultsdir+'/'+system
	planet_only_dir = model_rundir+'/planet_only'
	planet_moon_dir = model_rundir+'/planet_moon'

	#### define the path for the posterior files
	planet_only_posterior_path = planet_only_dir+'/chains/equal_weighted_post.txt'
	planet_moon_posterior_path = planet_moon_dir+'/chains/equal_weighted_post.txt'

	#### define the path for the post_summary files
	planet_only_post_summary_path = planet_only_dir+'/info/post_summary.csv'
	planet_moon_post_summary_path = planet_moon_dir+'/info/post_summary.csv'

	print('planet_only_post_summary_path: ', planet_only_post_summary_path)

	#### define the path for the results.json files
	planet_only_results_path = planet_only_dir+'/info/results.json'
	planet_moon_results_path = planet_moon_dir+'/info/results.json'


	#### plot the planet-only model light curve

	ground_truth_times_corrected = ground_truth_times - ground_truth_times[0] + 352.397003 + system_timing_offset #### THIS OFFSET IS THE FIRST TIMESTAMP OF dummylc.csv 

	system_transit_times = [system_t0_bary]
	while system_transit_times[-1] < np.nanmax(ground_truth_times_corrected):
		system_transit_times.append(system_transit_times[-1] + system_period)
	system_transit_times = system_transit_times[::-1] #### reverse it so that you can append
	while system_transit_times[-1] > np.nanmin(ground_truth_times_corrected):
		system_transit_times.append(system_transit_times[-1] - system_period)
	system_transit_times = system_transit_times[::-1] ### trase


	#### make sure the system TRANSIT TIMES ARE ALL WITHIN THE BASELINE
	if system_transit_times[0] < np.nanmin(ground_truth_times_corrected):
		system_transit_times = system_transit_times[1:]
	if system_transit_times[-1] > np.nanmax(ground_truth_times_corrected):
		system_transit_times = system_transit_times[:-1]
	ntransits = len(system_transit_times)

	if nsystem == 0:
		ground_truth_dict['ntransits'] = [ntransits]
	else:
		ground_truth_dict['ntransits'].append(ntransits)

	if multipanel_lcs == 'n':
		fig, ax = plt.subplots()
		ax.scatter(final_lc_times, final_lc_fluxes, color='#6B9AC4', edgecolor='k', s=20, zorder=0)
		ax.set_xlim(np.nanmin(final_lc_times), np.nanmax(final_lc_times))

		if show_model == 'y':
			missing_flux = np.zeros(shape=len(ground_truth_times))
			for nlc in np.arange(0,number_of_ground_truth_lcs,1):
				missing_flux += 1 - ground_truth_fluxes[nlc]
			ax.plot(ground_truth_times_corrected, 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')


	else:
		#### number of panels should be even
		isodd = ntransits % 2 #### = 1 (ONE) if ntransits is odd 
		if isodd == 1:
			npanels = ntransits-1
		else:
			npanels = ntransits

		#### factor the number
		factors_npanels = factor(npanels)
		#if (len(factors_npanels) == 3):
		if np.sqrt(npanels) in factors_npanels:
			### #it's square, the values should be the same 
			upper_factor = int(np.sqrt(npanels))
			lower_factor = int(np.sqrt(npanels))
			#### means the first and last number are 1 and the number, the middle number is both nwide and ndown
		else:
			upper_factor_idx = int(len(factors_npanels) / 2)
			lower_factor_idx = int(upper_factor_idx-1)
			upper_factor = factors_npanels[upper_factor_idx]
			lower_factor = factors_npanels[lower_factor_idx]
		assert lower_factor * upper_factor == npanels

		nwide, ndown = upper_factor, lower_factor 
		truncated_plot = False 
		if nwide > 5:
			nwide = 5
			truncated_plot = True
		if ndown > 5:
			ndown = 5 
			truncated_plot = True 
		lc_plot_dimensions = ndown * nwide

		if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):

			fig, ax = plt.subplots(ndown, nwide, figsize=(16,9))

			transit_idx = 0 
			transit_duration = cat_transit_duration
			for i in np.arange(0,ndown,1):
				for j in np.arange(0,nwide,1):

					#### FOR EACH PANEL 

					#### grab the times
					system_transit_time = system_transit_times[transit_idx]
					#panel_time_idxs = np.where((final_lc_times > system_transit_time - 0.1*system_period) & (final_lc_times < system_transit_time + 0.1*system_period))[0]
					panel_time_idxs = np.where((final_lc_times > system_transit_time - 1.5*transit_duration) & (final_lc_times < system_transit_time + 1.5*transit_duration))[0]

					if (upper_factor > 2) and (upper_factor != lower_factor):
						ax[i][j].scatter(final_lc_times[panel_time_idxs], final_lc_fluxes[panel_time_idxs], color='#6B9AC4', edgecolor='k', s=20, zorder=0)
						try:
							ax[i][j].set_xlim(np.nanmin(final_lc_times[panel_time_idxs]), np.nanmax(final_lc_times[panel_time_idxs]))
							use_model_lim = False
						except:
							use_model_lim = True
					elif upper_factor == lower_factor:
						try:
							ax[i][j].scatter(final_lc_times[panel_time_idxs], final_lc_fluxes[panel_time_idxs], facecolor='#6B9AC4', edgecolor='k', s=20, zorder=0)
							use_model_lim = False
						except TypeError:
							ax[j].scatter(final_lc_times[panel_time_idxs], final_lc_fluxes[panel_time_idxs], facecolor='#6B9AC4', edgecolor='k', s=20, zorder=0)
							use_model_lim = True
					else:
						ax[i].scatter(final_lc_times[panel_time_idxs], final_lc_fluxes[panel_time_idxs], facecolor='#6B9AC4', edgecolor='k', s=20, zorder=0)
						try:
							ax[i].set_xlim(np.nanmin(final_lc_times[panel_time_idxs]), np.nanmax(final_lc_times[panel_time_idxs]))
							use_model_lim = False 
						except:
							use_model_lim = True 

					if show_model == 'y':
						#gt_panel_time_idxs = np.where((ground_truth_times_corrected > system_transit_time - 0.1*system_period) & (ground_truth_times_corrected < system_transit_time + 0.1*system_period))[0]
						gt_panel_time_idxs = np.where((ground_truth_times_corrected > system_transit_time - 1.5*transit_duration) & (ground_truth_times_corrected < system_transit_time + 1.5*transit_duration))[0]

						missing_flux = np.zeros(shape=len(ground_truth_times[gt_panel_time_idxs]))

						for nlc in np.arange(0,number_of_ground_truth_lcs,1):
							missing_flux += 1 - ground_truth_fluxes[nlc][gt_panel_time_idxs]
							#ax.plot(ground_truth_times_corrected, ground_truth_fluxes[nlc])
						if (upper_factor > 2) and (upper_factor != lower_factor):
							if i == 0 and j == 0:
								try:
									ax[i][j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')
								except TypeError:
									ax[j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')
							else:
								try:
									ax[i][j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--')
								except TypeError:
									ax[j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--')
							if use_model_lim == True:
								try:
									ax[i][j].set_xlim(np.nanmin(ground_truth_times_corrected[gt_panel_time_idxs]), np.nanmax(ground_truth_times_corrected[gt_panel_time_idxs]))
								except TypeError:
									ax[j].set_xlim(np.nanmin(ground_truth_times_corrected[gt_panel_time_idxs]), np.nanmax(ground_truth_times_corrected[gt_panel_time_idxs]))
						elif upper_factor == lower_factor:
							if i == 0 and j == 0:
								try:
									ax[i][j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')
								except TypeError:
									ax[j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')
							else:
								try:
									ax[i][j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--')
								except TypeError:
									ax[j].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--')
							if use_model_lim == True:
								try:
									ax[i][j].set_xlim(np.nanmin(ground_truth_times_corrected[gt_panel_time_idxs]), np.nanmax(ground_truth_times_corrected[gt_panel_time_idxs]))
								except TypeError:
									ax[j].set_xlim(np.nanmin(ground_truth_times_corrected[gt_panel_time_idxs]), np.nanmax(ground_truth_times_corrected[gt_panel_time_idxs]))

						else:
							if i == 0:
								ax[i].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--', label='ground truth')
							else:
								ax[i].plot(ground_truth_times_corrected[gt_panel_time_idxs], 1-missing_flux, color=model_colors[2], linewidth=2, linestyle='--')
							if use_model_lim == True:
								ax[i].set_xlim(np.nanmin(ground_truth_times_corrected[gt_panel_time_idxs]), np.nanmax(ground_truth_times_corrected[gt_panel_time_idxs]))
					transit_idx += 1



	if os.path.exists(planet_only_post_summary_path) and (check_11_10 == 'n'):
		planet_model_exists = True ### SWITCH TO TRUE! 
		print("planet_model_exists: ", planet_model_exists)

		included_runs = [run_name]
		modeling_resultsdirs = [modeling_resultsdir]
		planet_only_dirs = [planet_only_dir] 
		planet_only_posterior_paths = [planet_only_posterior_path]
		planet_only_post_summary_paths = [planet_only_post_summary_path]
		planet_only_results_paths = [planet_only_results_path]


	else: ### either the new model doesn't exist, or we want to check 11_10 
		if os.path.exists(planet_only_post_summary_path):
			planet_model_exists = True

			#### DO THIS FOR THE FOR LOOP BELOW
			included_runs = [run_name]
			modeling_resultsdirs = [modeling_resultsdir]
			planet_only_dirs = [planet_only_dir] 
			planet_only_posterior_paths = [planet_only_posterior_path]
			planet_only_post_summary_paths = [planet_only_post_summary_path]
			planet_only_results_paths = [planet_only_results_path]

		else:
			planet_model_exists = False 
			included_runs = []
			modeling_resultsdirs = []
			planet_only_dirs = []
			planet_only_posterior_paths = []
			planet_only_post_summary_paths = []
			planet_only_results_paths = []

		print("planet_model_exists: ", planet_model_exists)
		#### check for one in the other runs
		valid_runs = []
			#### start by using the run you want

		if (use_earlier_models == 'y') and (check_11_10 == 'n'):
			for previous_run in list_of_runs:
				print("checking ", previous_run)
				##### check all of them
				if draw_from_posteriors == 'y':
					previous_run_modeling_resultsdir = external_projectdir+'/modeling_results_for_download/'+previous_run
				elif draw_from_posteriors == 'n':
					previous_run_modeling_resultsdir = projectdir+'/modeling_summaries_for_download/'+previous_run

				previous_planet_only_dir = previous_run_modeling_resultsdir+'/'+system+'/planet_only'
				previous_planet_only_posterior_path = planet_only_dir+'/chains/equal_weighted_post.txt'
				previous_planet_only_post_summary_path = planet_only_dir+'/info/post_summary.csv'
				previous_planet_only_results_path = planet_only_dir+'/info/results.json'

				if os.path.exists(previous_planet_only_post_summary_path) or os.path.exists(previous_planet_only_posterior_path):
					valid_runs.append(previous_run)
					planet_model_exists = True 

		print('valid runs for this planet: ', valid_runs)

	if planet_model_exists == True:
		sims_with_planet_model.append(system)
	elif planet_model_exists == False:
		sims_without_planet_model.append(system)

	if planet_model_exists == True:
		#### load it
		if (use_earlier_models == 'y') or (check_11_10 == 'y'):
			if check_11_10 == 'n':
				selected_run = 'run7_25'
				#selected_run = run_name
			elif check_11_10 == 'y':
				selected_run = 'run11_10'
			else:
				selected_run = run_name

			if draw_from_posteriors == 'y':
				previous_run_modeling_resultsdir = external_projectdir+'/modeling_results_for_download/'+selected_run
			elif draw_from_posteriors == 'n':
				previous_run_modeling_resultsdir = projectdir+'/modeling_summaries_for_download/'+selected_run

			planet_only_dir = previous_run_modeling_resultsdir+'/'+system+'/planet_only'
			planet_only_posterior_path = planet_only_dir+'/chains/equal_weighted_post.txt'
			planet_only_post_summary_path = planet_only_dir+'/info/post_summary.csv'
			planet_only_results_path = planet_only_dir+'/info/results.json'

			if overplot == 'n':
				#### USE JUST A SINGLE MODEL!!!! 
				included_runs = [selected_run]
				modeling_resultsdirs = [previous_run_modeling_resultsdir]
				planet_only_dirs = [planet_only_dir]
				planet_only_posterior_paths = [planet_only_posterior_path]
				planet_only_post_summary_paths = [planet_only_post_summary_path]
				planet_only_results_paths = [planet_only_results_path]

			elif overplot == 'y':
				#### we're going to use all the models and overplot them 
				included_runs.append(selected_run)
				modeling_resultsdirs.append(previous_run_modeling_resultsdir)
				planet_only_dirs.append(planet_only_dir)
				planet_only_posterior_paths.append(planet_only_posterior_path)
				planet_only_post_summary_paths.append(planet_only_post_summary_path)
				planet_only_results_paths.append(planet_only_results_path)

		else:
			selected_run = run_name 

		planet_only_logzs = []   #### A LIST -- FOR LOOKING AT MULTIPLE RUNS SIMULTANEOUSLY, IF YOU OPT FOR THAT 
		for run_idx in np.arange(0,len(modeling_resultsdirs),1):

			planet_only_loglike = None
			planet_only_BIC = None

			this_run = included_runs[run_idx]
			print('THIS RUN: ', this_run)

			if os.path.exists(planet_only_post_summary_paths[run_idx]) == False:
				continue 

			if draw_from_posteriors == 'n':
				planet_only_post_summary = pandas.read_csv(planet_only_post_summary_paths[run_idx])

				for pon in planet_only_names:
					planet_only_post_medians[pon] = np.array(planet_only_post_summary[pon+'_median'])[0].astype(float) ### array of size 1, but still an array.
					planet_only_post_means[pon] = np.array(planet_only_post_summary[pon+'_mean'])[0].astype(float)
					planet_only_post_stddevs[pon] = np.array(planet_only_post_summary[pon+'_stdev'])[0].astype(float)

					try:
						planet_only_posterior_median_dict[pon].append(np.array(planet_only_post_summary[pon+'_median'])[0].astype(float))
						planet_only_posterior_mean_dict[pon].append(np.array(planet_only_post_summary[pon+'_mean'])[0].astype(float))
						planet_only_posterior_sigma_dict[pon].append(np.array(planet_only_post_summary[pon+'_stdev'])[0].astype(float))
					except:
						planet_only_posterior_median_dict[pon] = np.array(planet_only_post_summary[pon+'_median']).astype(float).tolist()
						planet_only_posterior_mean_dict[pon] = np.array(planet_only_post_summary[pon+'_mean']).astype(float).tolist()
						planet_only_posterior_sigma_dict[pon] = np.array(planet_only_post_summary[pon+'_stdev']).astype(float).tolist()

				#### add the not-fit values to the dictionary
				planet_only_post_medians['M_planet'] = float(simulation_Mplanet)
				planet_only_post_medians['R_star'] = float(simulation_Rstar)

				planet_only_results_file = open(planet_only_results_paths[run_idx])
				planet_only_results_dict = json.load(planet_only_results_file)
				planet_only_logz = planet_only_results_dict['logz'] 
				planet_only_logzs.append(planet_only_logz)

				#### generate the planet only light curve!
				hires_times = hires_time_maker(final_lc_times, return_original=return_original)

				if force_ground_truth_impact == 'y':
					planet_only_post_medians['b_bary'] = ground_truth_dict['impact'][0] ### extra 0 because its natively a list
					print("FORCED PLANET ONLY IMPACT PARAMETER TO THE GROUND TRUTH VALUE: ", ground_truth_dict['impact'][0])
				if force_ground_truth_ecc == 'y':
					planet_only_post_medians['ecc_bary'] = ground_truth_dict['e'][0]
					print("FORCED PLANET ONLY ECCENTRICITY TO THE GROUND TRUTH VALUE: ", ground_truth_dict['e'][0])
				if force_ground_truth_limb_darkening == 'y':
					planet_only_post_medians['q1'] = ground_truth_dict['q1'][0]
					planet_only_post_medians['q2'] = ground_truth_dict['q2'][0]
					print("FORCED PLANET ONLY LIMB DARKENING TO THE GROUND TRUTH VALUES: ", ground_truth_dict['q1'][0], ground_truth_dict['q2'][0])

				planet_only_model_times, planet_only_model_fluxes = gen_model(model_dict=planet_only_post_medians, times=hires_times, t0_bary=system_t0_bary, model_type='planet_only')

				#### COMPUTE THE RESIDUALS
				planet_only_times_for_residuals, planet_only_model_fluxes_for_residuals = gen_model(model_dict=planet_only_post_medians, times=final_lc_times, t0_bary=system_t0_bary, model_type='planet_only')
				planet_only_residuals = final_lc_fluxes - planet_only_model_fluxes_for_residuals ### ought to be centered around zero
				planet_only_residual_std = np.nanstd(planet_only_residuals)
				print("PLANET ONLY RESIDUAL [ppm]: ", planet_only_residual_std * 1e6)
				if planet_only_residual_std > 3 * 10 * 1e-6: ### 30 ppm
					print('RESIDUAL IS > 3SIGMA OFF THE SCATTER! ')				


				#### COMPUTE THE LOGLIKELIHOOD OF THIS 
				planet_only_loglike = -0.5 * np.nansum( ( (final_lc_fluxes - planet_only_model_fluxes_for_residuals) / (10 * 1e-6) )**2 )
				planet_only_BIC = (9 * np.log( len(final_lc_fluxes) ) ) - ( 2 * planet_only_loglike )  ### planet-only model has 9 free parameters 
				print("planet_only_BIC: ", planet_only_BIC)

				print('PRINTING PLANET ONLY POST MEDIANS: ')
				for key in planet_only_post_medians.keys():
					print(key, planet_only_post_medians[key])
				print(' ')

				if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):

					#### make sure the system TRANSIT TIMES ARE ALL WITHIN THE BASELINE
					if system_transit_times[0] < np.nanmin(planet_only_model_times):
						system_transit_times = system_transit_times[1:]
					if system_transit_times[-1] > np.nanmax(planet_only_model_times):
						system_transit_times = system_transit_times[:-1]

					if multipanel_lcs == 'n':
							if show_model == 'y':
								if (use_earlier_models == 'y') or (check_11_10 == 'y'):
									if (show_run8_10 == 'y') or ((show_run8_10 == 'n') and (this_run != 'run8_10')):
										ax.plot(planet_only_model_times, planet_only_model_fluxes, color=model_colors[0], linewidth=2, linestyle='--', label='planet only')
										#ax.plot(planet_only_model_times, planet_only_model_fluxes, color=sat_colors[::-1][run_idx], label='planet only model')
								else:
									ax.plot(planet_only_model_times, planet_only_model_fluxes, color=model_colors[0], linewidth=2, linestyle='--', label='planet only')

					else:

						transit_idx = 0 
						transit_duration = cat_transit_duration

						if (show_run8_10 == 'y') or ((show_run8_10 == 'n') and (this_run != 'run8_10')):			

							for i in np.arange(0,ndown,1):
								for j in np.arange(0,nwide,1):
									#### grab the times
									try:
										system_transit_time = system_transit_times[transit_idx]
									except:
										break 
									#panel_time_idxs = np.where((planet_only_model_times > system_transit_time - 0.1*system_period) & (planet_only_model_times < system_transit_time + 0.1*system_period))[0]
									panel_time_idxs = np.where((planet_only_model_times > system_transit_time - 1.5*transit_duration) & (planet_only_model_times < system_transit_time + 1.5*transit_duration))[0]

									if (upper_factor > 2) and (upper_factor != lower_factor):
										if i == 0 and j == 0:
											if check_11_10 == 'y':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')

											elif check_11_10 == 'n':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')
										else:
											if check_11_10 == 'y':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)

											elif check_11_10 == 'n':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)

									elif upper_factor == lower_factor:
										if i == 0 and j == 0:
											if check_11_10 == 'y':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')

											elif check_11_10 == 'n':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1, label='planet only')

										else:
											if check_11_10 == 'y':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)

											elif check_11_10 == 'n':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)

									else:
										if i == 0:
											if check_11_10 == 'y':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
											elif check_11_10 == 'n':
												try:
													ax[i][j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
												except TypeError:
													ax[j].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)
										else:
											if check_11_10 == 'y':
													ax[i].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=sat_colors[0], linewidth=2, linestyle='--', zorder=1)

											elif check_11_10 == 'n':
												ax[i].plot(planet_only_model_times[panel_time_idxs], planet_only_model_fluxes[panel_time_idxs], color=model_colors[0], linewidth=2, linestyle='--', zorder=1)

									transit_idx += 1


					print("PLANET ONLY MODEL INPUTS (from fit): ")
					for model_key in planet_only_post_medians.keys():
						print('model_key: ', model_key)
						try: 
							POval = planet_only_post_medians[model_key]
							PMval = planet_moon_post_medians[model_key]
							PMval_minus_POval = PMval - POval ### positive means increased, negative means decreased 

							try:
								PMval_minus_POval_dict[model_key].append(PMval_minus_POval)
							except:
								PMval_minus_POval_dict[model_key] = [PMval_minus_POval]
							print(model_key+' PO: '+str(POval)+' PM: '+str(PMval)+', difference: '+str(PMval_minus_POval))

						except:
							POval = planet_only_post_medians[model_key]
							print(model_key+' PO: '+str(POval))
					print(' ')

			elif draw_from_posteriors == 'y':
				hires_times = hires_time_maker(final_lc_times, return_original=return_original)

				planet_only_posterior = np.genfromtxt(planet_only_posterior_paths[run_idx], names=True)
				nlines = len(planet_only_posterior['per_bary'])
				for ndraw, draw in enumerate(np.random.randint(low=0, high=nlines, size=ndraws)):
					planet_only_draw_dict = {}
					planet_only_draw_dict['M_planet'] = float(simulation_Mplanet)
					planet_only_draw_dict['R_star'] = float(simulation_Rstar)
					for pon in planet_only_names:
						planet_only_draw_dict[pon] = planet_only_posterior[pon][draw]


					planet_only_model_times, planet_only_model_fluxes = gen_model(model_dict=planet_only_draw_dict, times=hires_times, t0_bary=system_t0_bary, model_type='planet_only')
					
					if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):
						if ndraw == 0:
							ax.plot(planet_only_model_times, planet_only_model_fluxes, color=model_colors[0], linewidth=2, linestyle='--', label='planet only model', alpha=10 * (1/ndraws))
						else:
							ax.plot(planet_only_model_times, planet_only_model_fluxes, color=model_colors[0], linewidth=2, linestyle='--', alpha=10 * (1/ndraws))

				planet_only_results_file = open(planet_only_results_paths[run_idx])
				planet_only_results_dict = json.load(planet_only_results_file)
				planet_only_logz = planet_only_results_dict['logz'] 
				planet_only_logzs.append(planet_only_logz)  #### REMINDER, THIS IS A LIST FOR THE DIFFERENT RUNS -- IF THERE ARE MORE THAN ONE. 



			if (type(planet_only_logz) != type(None)) and (type(planet_moon_logz) != type(None)):
				#### means both models exist and we can subtract them
				delta_logZ = planet_moon_logz - planet_only_logz
				bayes_factor = np.exp(delta_logZ) ##### REFERENCE HERE: https://johannesbuchner.github.io/UltraNest/example-sine-modelcomparison.html 
				deltaBIC = planet_only_BIC - planet_moon_BIC

				print(this_run+' delta_logZ = ', delta_logZ)
				print(this_run+' Bayes factor =', bayes_factor)
				print(this_run+' deltaBIC = ', deltaBIC)

				if bayes_factor > 3.2:
					print(this_run+' shows substantial evidence favors the planet+moon model.')
				elif (bayes_factor >= 1) and (bayes_factor <= 3.2):
					print(this_run+' shows marginal evidence in favor of the planet+moon model.') 
				else:
					print(this_run+' evidence favors the planet only model.')
				time.sleep(5)
				if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):	
					if multipanel_lcs == 'n':
						if run_idx == 0:
							if truncated_plot == False:
								ax.set_title(r'$N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
							else:
								ax.set_title('[truncated] '+r', $N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
					else:
						if run_idx == 0:
							if truncated_plot == False:
								plt.suptitle(r', $N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
							else:
								plt.suptitle('[truncated] '+r', $N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))

			else:   #### we do not have both models run yet -- therefore we can't compute these 
				delta_logZ = None 
				bayes_factor = None
				deltaBIC = None 

			try: #### if the dictionary has already been established 
				planet_only_attributes_dict['run_name'].append(this_run)
				planet_only_attributes_dict['delta_logz'].append(delta_logZ)
				planet_only_attributes_dict['bayes_factor'].append(bayes_factor)
				planet_only_attributes_dict['sim'].append(system)
				planet_only_attributes_dict['planet_only_logz'].append(planet_only_logz)
				planet_only_attributes_dict['nmoons'].append(system_nmoons)
				planet_only_attributes_dict['ntransits'].append(ntransits)
				planet_only_attributes_dict['res_nores'].append(res_nores)
				planet_only_attributes_dict['I SNR'].append(cat_SNR1)
				planet_only_attributes_dict['II SNR'].append(cat_SNR2)
				planet_only_attributes_dict['III SNR'].append(cat_SNR3)
				planet_only_attributes_dict['IV SNR'].append(cat_SNR4)
				planet_only_attributes_dict['V SNR'].append(cat_SNR5)
				planet_only_attributes_dict['tdur'].append(cat_transit_duration)
				planet_only_attributes_dict['residuals_std'].append(planet_only_residual_std)
				planet_only_attributes_dict['BIC'].append(planet_only_BIC)
				planet_only_attributes_dict['deltaBIC'].append(deltaBIC)
				if system in timed_out_runs:
					planet_only_attributes_dict['moon_timed_out'].append(True)
				else:
					planet_only_attributes_dict['moon_timed_out'].append(False)
				planet_only_attributes_dict['ndata'].append(len(final_lc_times))

				if loaded_poad == False:
					with open(projectdir+'/'+run_name+'_poad.pickle', 'wb') as poad_pickle:
						pickle.dump(planet_only_attributes_dict, poad_pickle, protocol=pickle.HIGHEST_PROTOCOL)


				ground_truth_dict['I SNR'].append(cat_SNR1)
				ground_truth_dict['II SNR'].append(cat_SNR2)
				ground_truth_dict['III SNR'].append(cat_SNR3)
				ground_truth_dict['IV SNR'].append(cat_SNR4)
				ground_truth_dict['V SNR'].append(cat_SNR5)					

				sim_dict_keys = ['Planet', 'I', 'II', 'III', 'IV', 'V']
				planet_keys = ['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'aRstar']
				moon_keys = ['m', 'r', 'msmp', 'rsrp', 'a', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'spacing', 'period_ratio']
				for key in sim_dict_keys:
					if key == 'Planet':
						object_keys = planet_keys
					else:
						object_keys = moon_keys 

					for object_key in object_keys:
						try:
							planet_only_attributes_dict[key][object_key].append(simulation_dict[key][object_key])
						except:
							planet_only_attributes_dict[key][object_key].append(np.nan)


			except: #### the dictionary hasn't been established yet, so establish it! 
				traceback.print_exc()
				time.sleep(5)
				planet_only_attributes_dict['run_name'] = [this_run]
				planet_only_attributes_dict['delta_logz'] = [delta_logZ]
				planet_only_attributes_dict['bayes_factor'] = [bayes_factor]
				planet_only_attributes_dict['sim'] = [system]
				planet_only_attributes_dict['nmoons'] = [system_nmoons]
				planet_only_attributes_dict['planet_only_logz'] = [planet_only_logz]
				planet_only_attributes_dict['ntransits'] = [ntransits]
				planet_only_attributes_dict['res_nores'] = [res_nores]
				planet_only_attributes_dict['I SNR'] = [cat_SNR1]
				planet_only_attributes_dict['II SNR'] = [cat_SNR2]
				planet_only_attributes_dict['III SNR'] = [cat_SNR3]
				planet_only_attributes_dict['IV SNR'] = [cat_SNR4]
				planet_only_attributes_dict['V SNR'] = [cat_SNR5]
				planet_only_attributes_dict['tdur'] = [cat_transit_duration]
				planet_only_attributes_dict['residuals_std'] = [planet_only_residual_std]
				planet_only_attributes_dict['BIC'] = [planet_only_BIC]
				planet_only_attributes_dict['deltaBIC'] = [deltaBIC]
				if system in timed_out_runs:
					planet_only_attributes_dict['moon_timed_out'] = [True]
				else:
					planet_only_attributes_dict['moon_timed_out'] = [False]
				planet_only_attributes_dict['ndata'] = [len(final_lc_times)]

				if loaded_poad == False:
					with open(projectdir+'/'+run_name+'_poad.pickle', 'wb') as poad_pickle:
						pickle.dump(planet_only_attributes_dict, poad_pickle, protocol=pickle.HIGHEST_PROTOCOL)



				ground_truth_dict['I SNR'] = [cat_SNR1]
				ground_truth_dict['II SNR'] = [cat_SNR2]
				ground_truth_dict['III SNR'] = [cat_SNR3]
				ground_truth_dict['IV SNR'] = [cat_SNR4]
				ground_truth_dict['V SNR'] = [cat_SNR5]			

				sim_dict_keys = ['Planet', 'I', 'II', 'III', 'IV', 'V']
				planet_keys = ['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'aRstar']
				moon_keys = ['m', 'r', 'msmp', 'rsrp', 'a', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'spacing', 'period_ratio']
				for key in sim_dict_keys:
					planet_only_attributes_dict[key] = {}
					#if key in moon_options:
					#object_keys = simulation_dict[key].keys()
					if key == 'Planet':
						object_keys = planet_keys
					else:
						object_keys = moon_keys 

					for object_key in object_keys:
						try:
							planet_only_attributes_dict[key][object_key] = [simulation_dict[key][object_key]]
						except:
							planet_only_attributes_dict[key][object_key] = [np.nan]


				print('run_names: ', planet_only_attributes_dict['run_name'])
				print('bayes_factors: ', planet_only_attributes_dict['bayes_factor'])
				time.sleep(2)



	else: #### planet model doesn't exist
		pass 





	#### NOW plot the planet+moon model light curve

	if os.path.exists(planet_moon_post_summary_path):
		moon_model_exists = True  ### switch to true

		#### load it
		if draw_from_posteriors == 'n':
			planet_moon_post_summary = pandas.read_csv(planet_moon_post_summary_path)

			for pmn in planet_moon_names:
				planet_moon_post_medians[pmn] = np.array(planet_moon_post_summary[pmn+'_median'])[0] ### array of size 1, but still an array.
				planet_moon_post_means[pmn] = np.array(planet_moon_post_summary[pmn+'_mean'])[0]
				planet_moon_post_stddevs[pmn] = np.array(planet_moon_post_summary[pmn+'_stdev'])[0]

				try:
					planet_moon_posterior_median_dict[pmn].append(np.array(planet_moon_post_summary[pmn+'_median'])[0].astype(float))
					planet_moon_posterior_mean_dict[pmn].append(np.array(planet_moon_post_summary[pmn+'_mean'])[0].astype(float))
					planet_moon_posterior_sigma_dict[pmn].append(np.array(planet_moon_post_summary[pmn+'_stdev'])[0].astype(float))
				except:
					planet_moon_posterior_median_dict[pmn] = np.array(planet_moon_post_summary[pmn+'_median']).astype(float).tolist()
					planet_moon_posterior_mean_dict[pmn] = np.array(planet_moon_post_summary[pmn+'_mean']).astype(float).tolist()
					planet_moon_posterior_sigma_dict[pmn] = np.array(planet_moon_post_summary[pmn+'_stdev']).astype(float).tolist()

			try:
				planet_moon_posterior_median_dict['sim'].append(system)
				planet_moon_posterior_mean_dict['sim'].append(system)
				planet_moon_posterior_sigma_dict['sim'].append(system)
			except:
				planet_moon_posterior_median_dict['sim'] = [system]
				planet_moon_posterior_mean_dict['sim'] = [system]
				planet_moon_posterior_sigma_dict['sim'] = [system]


			#### add the not-fit values to the dictionary	
			planet_moon_post_medians['R_star'] = float(simulation_Rstar)

			try:
				planet_moon_posterior_median_dict['R_star'] = float(simulation_Rstar)
				planet_moon_posterior_mean_dict['R_star'] = float(simulation_Rstar)
				planet_moon_posterior_sigma_dict['R_star'] = np.nan 
			except:
				planet_moon_posterior_median_dict['Rstar'] = float(simulation_Rstar)
				planet_moon_posterior_mean_dict['Rstar'] = float(simulation_Rstar)
				planet_moon_posterior_sigma_dict['Rstar'] = np.nan 

			planet_moon_results_file = open(planet_moon_results_path)
			planet_moon_results_dict = json.load(planet_moon_results_file)
			planet_moon_logz = planet_moon_results_dict['logz'] 

			#### generate the planet only light curve!
			hires_times = hires_time_maker(final_lc_times, return_original=return_original)

			#### generate the planet only light curve!
			if force_ground_truth_impact == 'y':
				planet_moon_post_medians['b_bary'] = ground_truth_dict['impact'][0] ### extra 0 because its natively a list
				print("FORCED PLANET MOON IMPACT PARAMETER TO THE GROUND TRUTH VALUE: ", ground_truth_dict['impact'][0])
			if force_ground_truth_ecc == 'y':
				planet_moon_post_medians['ecc_bary'] = ground_truth_dict['e'][0]
				print("FORCED PLANET MOON ECCENTRICITY TO THE GROUND TRUTH VALUE: ", ground_truth_dict['e'][0])
			if force_ground_truth_limb_darkening == 'y':
				planet_moon_post_medians['q1'] = ground_truth_dict['q1'][0]
				planet_moon_post_medians['q2'] = ground_truth_dict['q2'][0]
				print("FORCED PLANET MOON LIMB DARKENING TO GROUND TRUTH VALUE: ", ground_truth_dict['q1'][0], ground_truth_dict['q2'][0])

			if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):
				planet_moon_model_times, planet_moon_model_fluxes = gen_model(model_dict=planet_moon_post_medians, times=hires_times, t0_bary=system_t0_bary, model_type='planet_moon')
			
				#### COMPUTE THE RESIDUALS
				planet_moon_times_for_residuals, planet_moon_model_fluxes_for_residuals = gen_model(model_dict=planet_moon_post_medians, times=final_lc_times, t0_bary=system_t0_bary, model_type='planet_only')
				planet_moon_residuals = final_lc_fluxes - planet_moon_model_fluxes_for_residuals ### ought to be centered around zero
				planet_moon_residual_std = np.nanstd(planet_moon_residuals)
				print("PLANET+MOON RESIDUAL [ppm]: ", planet_moon_residual_std * 1e6)
				if planet_moon_residual_std > 3 * 10 * 1e-6: ### 30 ppm
					print('RESIDUAL IS > 3SIGMA OFF THE SCATTER! ')

				#### COMPUTE THE LOGLIKELIHOOD OF THIS 
				planet_moon_loglike = -0.5 * np.nansum(((final_lc_fluxes - planet_moon_model_fluxes_for_residuals) / (10 * 1e-6))**2) ### each light curve has 10ppm errors
				planet_moon_BIC = (16 * np.log(len(final_lc_fluxes))) - (2 * planet_moon_loglike)  ### planet+moon model has 16 free parameters
				print('planet_moon_BIC: ', planet_moon_BIC)

				print('PRINTING PLANET MOON POST MEDIANS: ')
				for key in planet_moon_post_medians.keys():
					print(key, planet_moon_post_medians[key])
				print(' ')

				if multipanel_lcs == 'n':
						if show_model == 'y':
							ax.plot(planet_moon_model_times, planet_moon_model_fluxes, color=model_colors[1], linewidth=2, linestyle='--', label='planet+moon model')

				else:
					transit_idx = 0 
					transit_duration = cat_transit_duration
					for i in np.arange(0,ndown,1):
						for j in np.arange(0,nwide,1):
							#### grab the times
							try:
								system_transit_time = system_transit_times[transit_idx]
							except:
								break 
							#panel_time_idxs = np.where((planet_moon_model_times > system_transit_time - 0.1*system_period) & (planet_moon_model_times < system_transit_time + 0.1*system_period))[0]
							panel_time_idxs = np.where((planet_moon_model_times > system_transit_time - 1.5*transit_duration) & (planet_moon_model_times < system_transit_time + 1.5*transit_duration))[0]							
							if (upper_factor > 2) and (upper_factor != lower_factor):
								if i == 0 and j == 0:
									try:
										ax[i][j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
									except TypeError:
										ax[j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
								else:
									try:
										ax[i][j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1)
									except TypeError:
										ax[j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1)
							elif upper_factor == lower_factor:
								if i == 0 and j == 0:
									try:
										ax[i][j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
									except TypeError:
										ax[j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
								else:
									try:
										ax[i][j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
									except TypeError:
										ax[j].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
							else:
								if i == 0:
									ax[i].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1, label='planet+moon')
								else:
									ax[i].plot(planet_moon_model_times[panel_time_idxs], planet_moon_model_fluxes[panel_time_idxs], color=model_colors[1], linewidth=2, linestyle='--', zorder=1)

							transit_idx += 1


				print("PLANET+MOON MODEL INPUTS (from fit): ")

				for model_key in planet_moon_post_medians.keys():
					try: 
						POval = planet_only_post_medians[model_key]
						PMval = planet_moon_post_medians[model_key]
						PMval_minus_POval = PMval - POval 

						ground_truth_model_key = model_to_ground_truth_dict[model_key] ### get the ground truth version
						this_ground_truth = simulation_dict['Planet'][ground_truth_model_key]
						PM_minus_gt = PMval - this_ground_truth
						PO_minus_gt = POval - this_ground_truth 
						print('ground_truth_model_key: ', ground_truth_model_key)
						print('ground truth: ', this_ground_truth)
						if this_ground_truth != 0.0:
							print('PM minus gt (pct error): '+str(PM_minus_gt)+', ('+str((PM_minus_gt / this_ground_truth)*100)+')')
							print('PO minus gt (pct error): '+str(PO_minus_gt)+', ('+str((PO_minus_gt / this_ground_truth)*100)+')')
						else:
							print('PM minus gt: '+str(PM_minus_gt))
							print('PO minus gt: '+str(PO_minus_gt))

						try:
							PMval_minus_POval_dict[model_key].append(PMval_minus_POval)
						except:
							PMval_minus_POval_dict[model_key] = [PMval_minus_POval]

						print(model_key+' PO: '+str(POval)+' PM: '+str(PMval)+', difference: ', PMval_minus_POval)

					except:
						try:
							PMval = planet_moon_post_medians[model_key]
							print(model_key+' PM: '+str(PMval))
							ground_truth_model_key = model_to_ground_truth_dict[model_key] ### get the ground truth version
							print('ground_truth_model_key: ', ground_truth_model_key)
							this_ground_truth = simulation_dict['Planet'][ground_truth_model_key]
							PM_minus_gt = PMval - this_ground_truth
							print('PM minus gt (pct error): '+str(PM_minus_gt)+', ('+str((PM_minus_gt / this_ground_truth)*100)+')')
							print('')
							try:
								PMval_minus_ground_truth_dict[model_key].append(PM_minus_gt)
							except:
								PMval_minus_ground_truth_dict[model_key] = [PM_minus_gt]									

						except KeyError:
							print('model_key '+model_key+' not available in the model_to_ground_truth_dict.')
							continue 

						except:
							traceback.print_exc()
							continue

					print(' ')
				print(' ')
				print(' ')


		elif draw_from_posteriors == 'y':
			hires_times = hires_time_maker(final_lc_times, return_original=return_original)

			planet_moon_posterior = np.genfromtxt(planet_moon_posterior_path, names=True)
			nlines = len(planet_moon_posterior['per_bary'])

			for ndraw, draw in enumerate(np.random.randint(low=0, high=nlines, size=ndraws)):
				
				planet_moon_draw_dict = {}
				planet_moon_draw_dict['R_star'] = float(simulation_Rstar)
				for pmn in planet_moon_names:
					planet_moon_draw_dict[pmn] = planet_moon_posterior[pmn][draw]

				#### generate the planet only light curve!
				#hires_times = np.linspace(np.nanmin(final_lc_times), np.nanmax(final_lc_times), 100000)
				
				planet_moon_model_times, planet_moon_model_fluxes = gen_model(model_dict=planet_moon_draw_dict, times=hires_times, t0_bary=system_t0_bary, model_type='planet_moon')

				if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):

					if ndraw == 0:
						ax.plot(planet_moon_model_times, planet_moon_model_fluxes, color=model_colors[1], linewidth=2, linestyle='--', label='planet+moon model', alpha=10 * (1/ndraws))
					else:
						ax.plot(planet_moon_model_times, planet_moon_model_fluxes, color=model_colors[1], linewidth=2, linestyle='--', alpha=10 * (1/ndraws))

			planet_moon_results_file = open(planet_moon_results_path)
			planet_moon_results_dict = json.load(planet_moon_results_file)
			planet_moon_logz = planet_moon_results_dict['logz'] 


		for run_idx, this_run in enumerate(included_runs):
			#try:
			planet_only_idx = np.where(planet_only_attributes_dict['sim'] == system)[0]
			if type(planet_only_logz) == type(None):
				try: 
					planet_only_logz = np.array(planet_only_attributes_dict['planet_only_logz'])[planet_only_idx]
					#planet_only_logz = planet_only_results_dict['logz']
					#planet_only_logz = planet_only_attributes_dict['planet_only_logz']
				except:
					planet_only_logz = None 

			if (type(planet_only_logz) != type(None)) and (type(planet_moon_logz) != type(None)):
				#### means both models exist and we can subtract them
				delta_logZ = planet_moon_logz - planet_only_logz
				bayes_factor = np.exp(delta_logZ) #### reference here: https://johannesbuchner.github.io/UltraNest/example-sine-modelcomparison.html#:~:text=The%20Bayes%20factor%20is%3A 
				deltaBIC = planet_only_BIC - planet_moon_BIC

				print(this_run+' delta_logZ = ', delta_logZ)
				print(this_run+' bayes_factor = ', bayes_factor)
				print(this_run+' deltaBIC = ', deltaBIC)

				#if delta_logZ > 0:
				if bayes_factor > 3.2:
					print('['+this_run+'] shows strong evidence favors the planet+moon model.')
				elif (bayes_factor > 1) and (bayes_factor <= 3.2):
					print('['+this_run+'] shows marginal evidence in favor the planet+moon model.')
				else:
					print('['+this_run+'] evidence favors the planet only model.')
				if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):	
					if multipanel_lcs == 'n':
						if truncated_plot == False:
							ax.set_title(r'$N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
						else:
							ax.set_title('[truncated] '+r', $N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
					else:
						if truncated_plot == False:
							plt.suptitle(r'$N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))
						else:
							plt.suptitle('[truncated] '+r', $N = $'+str(system_nmoons)+','+r' $K = $'+str(round(bayes_factor,2)))

				try:
					planet_moon_attributes_dict['run_name'].append(this_run)
					planet_moon_attributes_dict['delta_logz'].append(delta_logZ)	
					planet_moon_attributes_dict['bayes_factor'].append(bayes_factor)
					planet_moon_attributes_dict['sim'].append(system)
					planet_moon_attributes_dict['planet_moon_logz'].append(planet_only_logz)
					planet_moon_attributes_dict['nmoons'].append(system_nmoons)
					planet_moon_attributes_dict['ntransits'].append(ntransits)
					planet_moon_attributes_dict['res_nores'].append(res_nores)
					planet_moon_attributes_dict['I SNR'].append(cat_SNR1)
					planet_moon_attributes_dict['II SNR'].append(cat_SNR2)
					planet_moon_attributes_dict['III SNR'].append(cat_SNR3)
					planet_moon_attributes_dict['IV SNR'].append(cat_SNR4)
					planet_moon_attributes_dict['V SNR'].append(cat_SNR5)	
					planet_moon_attributes_dict['tdur'].append(cat_transit_duration)	
					planet_moon_attributes_dict['residuals_std'].append(planet_moon_residual_std)
					planet_moon_attributes_dict['BIC'].append(planet_moon_BIC)
					planet_moon_attributes_dict['deltaBIC'].append(deltaBIC)
					if system in timed_out_runs:
						planet_moon_attributes_dict['moon_timed_out'].append(True)
					else:
						planet_moon_attributes_dict['moon_timed_out'].append(False)
					planet_moon_attributes_dict['ndata'].append(len(final_lc_times))

					if loaded_pmad == False:
						with open(projectdir+'/'+run_name+'_pmad.pickle', 'wb') as pmad_pickle:
							pickle.dump(planet_moon_attributes_dict, pmad_pickle, protocol=pickle.HIGHEST_PROTOCOL)


					sim_dict_keys = ['Planet', 'I', 'II', 'III', 'IV', 'V']
					planet_keys = ['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'aRstar']
					moon_keys = ['m', 'r', 'msmp', 'rsrp', 'a', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'spacing', 'period_ratio']
					for key in sim_dict_keys:
						if key == 'Planet':
							object_keys = planet_keys
						else:
							object_keys = moon_keys 

						for object_key in object_keys:
							try:
								planet_moon_attributes_dict[key][object_key].append(simulation_dict[key][object_key])
							except:
								planet_moon_attributes_dict[key][object_key].append(np.nan)

				except:
					planet_moon_attributes_dict['run_name'] = [this_run]
					planet_moon_attributes_dict['delta_logz'] = [delta_logZ]
					planet_moon_attributes_dict['bayes_factor'] = [bayes_factor]
					planet_moon_attributes_dict['sim'] = [system]
					planet_moon_attributes_dict['nmoons'] = [system_nmoons]
					planet_moon_attributes_dict['planet_moon_logz'] = [planet_moon_logz]
					planet_moon_attributes_dict['ntransits'] = [ntransits]
					planet_moon_attributes_dict['res_nores'] = [res_nores]
					planet_moon_attributes_dict['I SNR'] = [cat_SNR1]
					planet_moon_attributes_dict['II SNR'] = [cat_SNR2]
					planet_moon_attributes_dict['III SNR'] = [cat_SNR3]
					planet_moon_attributes_dict['IV SNR'] = [cat_SNR4]
					planet_moon_attributes_dict['V SNR'] = [cat_SNR5]
					planet_moon_attributes_dict['tdur'] = [cat_transit_duration]	
					planet_moon_attributes_dict['residuals_std'] = [planet_moon_residual_std]
					planet_moon_attributes_dict['BIC'] = [planet_moon_BIC]
					planet_moon_attributes_dict['deltaBIC'] = [deltaBIC]
					if system in timed_out_runs:
						planet_moon_attributes_dict['moon_timed_out'] = [True]
					else:
						planet_moon_attributes_dict['moon_timed_out'] = [False]
					planet_moon_attributes_dict['ndata'] = [len(final_lc_times)]

					if loaded_pmad == False:
						with open(projectdir+'/'+run_name+'_pmad.pickle', 'wb') as pmad_pickle:
							pickle.dump(planet_moon_attributes_dict, pmad_pickle, protocol=pickle.HIGHEST_PROTOCOL)


					#for key in simulation_dict.keys():
					sim_dict_keys = ['Planet', 'I', 'II', 'III', 'IV', 'V']
					planet_keys = ['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'aRstar']
					moon_keys = ['m', 'r', 'msmp', 'rsrp', 'a', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'spacing', 'period_ratio']
					for key in sim_dict_keys:
						planet_moon_attributes_dict[key] = {}
						#if key in moon_options:
						#object_keys = simulation_dict[key].keys()
						if key == 'Planet':
							object_keys = planet_keys
						else:
							object_keys = moon_keys 
						for object_key in object_keys:
							try:
								planet_moon_attributes_dict[key][object_key] = [simulation_dict[key][object_key]]
							except:
								planet_moon_attributes_dict[key][object_key] = [np.nan]


	else:
		delta_logZ = None 
		bayes_factor = None
		deltaBIC = None 



		if type(planet_only_logz) != type(None):
			print('planet only logZ = ', planet_only_logz)
			if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):
				if multipanel_lcs == 'n':
					truncated_plot = False 
					ax.set_title(r', $N = $'+str(system_nmoons)+','+r' $\log (Z) = $'+str(round(planet_only_logz,2)))
				else:
					plt.suptitle(r', $N = $'+str(system_nmoons)+','+r' $\log (Z) = $'+str(round(planet_only_logz,2)))

		elif type(planet_moon_logz) != type(None):
			print('planet moon logZ = ', planet_moon_logz)
			if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):
				if multipanel_lcs == 'n':
					ax.set_title(r', $N = $'+str(system_nmoons)+','+r' $\log (Z) = $'+str(round(planet_moon_logz,2)))
				else:
					plt.suptitle(r', $N = $'+str(system_nmoons)+','+r' $\log (Z) = $'+str(round(planet_moon_logz,2)))







	if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_lightcurve.png') == False):
		if multipanel_lcs == 'n':
			ax.set_xlabel('Time [days]')
			ax.set_ylabel('Normalized flux')
			ax.legend(loc='lower right')
		
		else:
			if (upper_factor > 2) and (upper_factor != lower_factor):
				for i in np.arange(0,nwide,1):
					for j in np.arange(0,ndown,1):
						if i == 0:
							ax[j][i].set_ylabel('flux')
						else:
							ax[j][i].yaxis.set_tick_params(labelleft=False)
						if j == ndown-1: ### index of the last entry
							ax[j][i].set_xlabel('Time [days]')

			elif upper_factor == lower_factor:
				for i in np.arange(0,nwide,1):
					for j in np.arange(0,ndown,1):
						if i == 0:
							ax[j][i].set_ylabel('flux')
						else:
							ax[j][i].yaxis.set_tick_params(labelleft=False)
						if j == nwide-1: ### index of the last entry
							ax[j][i].set_xlabel('Time [days]')
						else:
							ax[j][i].xaxis.set_tick_params(labelbottom=False)

			else:
				#for i in np.arange(0,nwide,1):
				ax[-1].set_xlabel('Time [days]')
				for j in np.arange(0,nwide,1):
					ax[j].set_ylabel('flux')

		try:
			handles, labels = ax[0][0].get_legend_handles_labels()
		except TypeError:
			if multipanel_lcs == 'y':
				handles, labels = ax[0].get_legend_handles_labels()
			elif multipanel_lcs == 'n':
				handles, labels = ax.get_legend_handles_labels()
		fig.legend(handles, labels, loc='center right')
		plt.savefig(plotdir+'/'+system+'_lightcurve.png', dpi=300)
		plt.savefig(model_rundir+'/'+system+'_lightcurve.png', dpi=300)
		print('system transit times: ', system_transit_times)
		if show_lc_plots == 'y':
			plt.show()
			continue_query = input('Do you want to continue? y/n: ')
			if continue_query != 'y':
				raise Exception('You opted not to continue.')			
		plt.close()





	else:
		print(plotdir+'/'+system+'_lightcurve.png already exists. SKIPPED.')


	#### make corner plots for the planet-only run and the planet+moon runs
	if draw_from_posteriors == 'y':
		try:
			planet_only_posterior = np.genfromtxt(planet_only_posterior_path, skip_header=1)

			planet_only_posterior.T[2] = (planet_only_posterior.T[2] * float(simulation_Rstar)) / R_earth.value 


			if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_planet_only_corner.png') == False):
				figure = corner.corner(
					planet_only_posterior, 
					labels=planet_only_corner_names,
					quantiles=[0.16,0.5,0.84],
					plot_contours=False,
					show_titles=True,
					title_kwargs={'fontsize':8, 'wrap':True, 'multialignment':'left'},
				)
				figure.set_size_inches(16, 16)

				plt.savefig(plotdir+'/'+system+'_planet_only_corner.png', dpi=300)
				plt.savefig(model_rundir+'/'+system+'_planet_only_corner.png', dpi=300)
				if show_lc_plots == 'y':
					plt.show()
				plt.close()

			else:
				print(plotdir+'/'+system+'_planet_only_corner.png already made. SKIPPED.')

		except:
			traceback.print_exc()
			print(' ')
			print('could not make planet only corner plot.')
			print(' ')

		
		try:
			planet_moon_posterior = np.genfromtxt(planet_moon_posterior_path, skip_header=1)


			#### replace M_planet and M_moon with M_earth values
			planet_moon_posterior.T[2] = (planet_moon_posterior.T[2] * float(simulation_Rstar)) / R_earth.value 
			planet_moon_posterior.T[7] = planet_moon_posterior.T[7] / M_earth.value
			planet_moon_posterior.T[8] = (planet_moon_posterior.T[8] * float(simulation_Rstar)) / R_earth.value 
			planet_moon_posterior.T[13] = planet_moon_posterior.T[13] / M_earth.value


			if (skip_already_made == 'n') or (os.path.exists(plotdir+'/'+system+'_planet_moon_corner.png') == False):
				figure = corner.corner(
					planet_moon_posterior, 
					labels=planet_moon_corner_names,
					quantiles=[0.16,0.5,0.84],
					plot_contours=False,
					show_titles=True,
					title_kwargs={'fontsize':6, 'wrap':True, 'multialignment':'left'},
				)

				figure.set_size_inches(8, 8)
				plt.savefig(plotdir+'/'+system+'_planet_moon_corner.png', dpi=300)
				plt.savefig(model_rundir+'/'+system+'_planet_moon_corner.png', dpi=300)
				if show_lc_plots == 'y':
					plt.show()
				plt.close()

			else:
				print(plotdir+'/'+system+'_planet_moon_corner.png already made. SKIPPED.')

		except:
			traceback.print_exc()
			print(' ')
			print('could not make planet moon corner plot.')
			print(' ')


	if moon_model_exists:
		#reasonable_or_not = input('Is the moon fit reasoanble? y/n: ')
		reasonable_or_not = 'y' 
		if reasonable_or_not == 'y':
			reasonable_moon_fits.append(system)
			reasonable_moon_fits.append(system_nmoons)

		elif reasonable_or_not == 'n':
			unreasonable_moon_fits.append(system)
			unreasonable_moon_fit_nmoons.append(system_nmoons)

	if moon_model_exists:
		try:
			sim_with_moon_model.append(system)
		except:
			sim_with_moon_model = [system]
	elif moon_model_exists == False:
		try:
			sim_without_moon_model.append(system)
		except:
			sim_without_moon_model = [system]


	#### LAST THING YOU'RE GOING TO DO IS POPULATE THE 
	moon_options = ['I', 'II', 'III', 'IV', 'V']
	if moon_model_exists == True:
		moon_recovery_dict[system]['rmoon_rstar_recovered'] = planet_moon_post_medians['r_moon'] ### rmoon / rstar (decimal value)
		moon_recovery_dict[system]['mmoon_recovered'] = planet_moon_post_medians['M_moon'] ### kg
		moon_recovery_dict[system]['permoon_recovered'] = planet_moon_post_medians['per_moon'] ### days
		moon_recovery_dict[system]['impact_recovered'] = planet_moon_post_medians['b_bary']
		moon_recovery_dict[system]['eccentricity_recovered'] = planet_moon_post_medians['ecc_bary']

	moon_recovery_dict[system]['rmoon_rstar_gts'] = []
	moon_recovery_dict[system]['mmoon_gts'] = []
	moon_recovery_dict[system]['permoon_gts'] = []
	moon_recovery_dict[system]['impact_gts'] = []
	moon_recovery_dict[system]['eccentricity_gts'] = []
	for moon_option in moon_options:
		if moon_option in simulation_dict.keys():
			moon_recovery_dict[system]['rmoon_rstar_gts'].append(simulation_dict[moon_option]['r'] / simulation_dict['Planet']['Rstar']) ### both these values are in meters, need to divide for Rs/Rstar
			moon_recovery_dict[system]['mmoon_gts'].append(simulation_dict[moon_option]['m']) #### natively in kg
			moon_recovery_dict[system]['permoon_gts'].append(simulation_dict[moon_option]['P'] / (24 * 60 * 60)) ### natively in seconds, have to convert to days
			moon_recovery_dict[system]['impact_gts'].append(simulation_dict['Planet']['impact'])
			moon_recovery_dict[system]['eccentricity_gts'].append(0.) #### all eccentricities were set to zero!
	if moon_model_exists == True:
		#moon_recovery_dict[system]['rmoon_rstar_pcterrors'] = 100 * (np.array(moon_recovery_dict[system]['rmoon_rstar_gts']) - moon_recovery_dict[system]['rmoon_rstar_recovered']) / np.array(moon_recovery_dict[system]['rmoon_rstar_gts'])
		#moon_recovery_dict[system]['mmoon_pcterrors'] = 100 * (np.array(moon_recovery_dict[system]['mmoon_gts']) - moon_recovery_dict[system]['mmoon_recovered']) / np.array(moon_recovery_dict[system]['mmoon_gts'])
		#moon_recovery_dict[system]['permoon_pcterrors'] =100 * (np.array(moon_recovery_dict[system]['permoon_gts']) - moon_recovery_dict[system]['permoon_recovered']) / np.array(moon_recovery_dict[system]['permoon_gts'])
		#moon_recovery_dict[system]['impact_pcterrors'] = 100 * (np.array(moon_recovery_dict[system]['impact_gts']) - moon_recovery_dict[system]['impact_recovered']) / np.array(moon_recovery_dict[system]['impact_gts'])
		#moon_recovery_dict[system]['eccentricity_pcterrors'] = 100 * (np.array(moon_recovery_dict[system]['eccentricity_gts']) - moon_recovery_dict[system]['eccentricity_recovered']) / np.array(moon_recovery_dict[system]['eccentricity_gts'])

		moon_recovery_dict[system]['rmoon_rstar_pcterrors'] = 100 * (moon_recovery_dict[system]['rmoon_rstar_recovered'] - np.array(moon_recovery_dict[system]['rmoon_rstar_gts'])) / np.array(moon_recovery_dict[system]['rmoon_rstar_gts'])
		moon_recovery_dict[system]['mmoon_pcterrors'] = 100 * (moon_recovery_dict[system]['mmoon_recovered'] - np.array(moon_recovery_dict[system]['mmoon_gts'])) / np.array(moon_recovery_dict[system]['mmoon_gts'])
		moon_recovery_dict[system]['permoon_pcterrors'] =100 * (moon_recovery_dict[system]['permoon_recovered'] - np.array(moon_recovery_dict[system]['permoon_gts'])) / np.array(moon_recovery_dict[system]['permoon_gts'])
		moon_recovery_dict[system]['impact_pcterrors'] = 100 * (moon_recovery_dict[system]['impact_recovered'] - np.array(moon_recovery_dict[system]['impact_gts'])) / np.array(moon_recovery_dict[system]['impact_gts'])
		moon_recovery_dict[system]['eccentricity_pcterrors'] = 100 * (moon_recovery_dict[system]['eccentricity_recovered'] - np.array(moon_recovery_dict[system]['eccentricity_gts'])) / np.array(moon_recovery_dict[system]['eccentricity_gts'])







#-------------------------------


satI_rsrstar_pct_errors = []
satII_rsrstar_pct_errors = []
satIII_rsrstar_pct_errors = []
satIV_rsrstar_pct_errors = []
satV_rsrstar_pct_errors = []
best_rsrstar_pct_errors = []

N1_rsrstar_pct_errors = []
N2_rsrstar_pct_errors = []
N3_rsrstar_pct_errors = []
N4_rsrstar_pct_errors = []
N5_rsrstar_pct_errors = []
#best_rsrstar_pct_errors = []

satI_mmoon_pct_errors = []
satII_mmoon_pct_errors = []
satIII_mmoon_pct_errors = []
satIV_mmoon_pct_errors = []
satV_mmoon_pct_errors = []
best_mmoon_pct_errors = []

N1_mmoon_pct_errors = []
N2_mmoon_pct_errors = []
N3_mmoon_pct_errors = []
N4_mmoon_pct_errors = []
N5_mmoon_pct_errors = []
#best_mmoon_pct_errors = []

satI_permoon_pct_errors = []
satII_permoon_pct_errors = []
satIII_permoon_pct_errors = []
satIV_permoon_pct_errors = []
satV_permoon_pct_errors = []
best_permoon_pct_errors = []

N1_permoon_pct_errors = []
N2_permoon_pct_errors = []
N3_permoon_pct_errors = []
N4_permoon_pct_errors = []
N5_permoon_pct_errors = []
#best_permoon_pct_errors = []

satI_impact_pct_errors = []
satII_impact_pct_errors = []
satIII_impact_pct_errors = []
satIV_impact_pct_errors = []
satV_impact_pct_errors = []
best_impact_pct_errors = []

N1_impact_pct_errors = []
N2_impact_pct_errors = []
N3_impact_pct_errors = []
N4_impact_pct_errors = []
N5_impact_pct_errors = []

satI_eccentricity_pct_errors = []
satII_eccentricity_pct_errors = []
satIII_eccentricity_pct_errors = []
satIV_eccentricity_pct_errors = []
satV_eccentricity_pct_errors = []
best_eccentricity_pct_errors = []

N1_eccentricity_pct_errors = []
N2_eccentricity_pct_errors = []
N3_eccentricity_pct_errors = []
N4_eccentricity_pct_errors = []
N5_eccentricity_pct_errors = []


#-------------------------------

pmad = planet_moon_attributes_dict

#### save this dictionary!
if loaded_pmad == False:
	with open(projectdir+'/'+run_name+'_pmad.pickle', 'wb') as pmad_pickle:
		pickle.dump(pmad, pmad_pickle, protocol=pickle.HIGHEST_PROTOCOL)








combined_run_idxs = np.where(np.array(pmad['run_name']) == 'combined_runs')[0]
run8_10_idxs = np.where(np.array(pmad['run_name']) == 'run8_10')[0]
run11_10_idxs = np.where(np.array(pmad['run_name']) == 'run11_10')[0]
timed_out_idxs = np.where(np.array(pmad['moon_timed_out']) == False)[0]

if run_name == 'combined_runs':
	idxs_to_use = combined_run_idxs
else:
	idxs_to_use = run11_10_idxs


for nsystem, system in enumerate(moon_recovery_dict.keys()):
	if system not in pmad['sim']:
		continue 
	else:
		pmad_system_idx = np.where(system == np.array(pmad['sim'])[idxs_to_use])[0]
		system_bayes_factor = np.array(pmad['bayes_factor'])[idxs_to_use][pmad_system_idx]
		if system_bayes_factor < 3.2:
			continue 
	try:
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) >= 1:
			satI_rsrstar_pct_errors.append(moon_recovery_dict[system]['rmoon_rstar_pcterrors'][0])
			satI_mmoon_pct_errors.append(moon_recovery_dict[system]['mmoon_pcterrors'][0])
			satI_permoon_pct_errors.append(moon_recovery_dict[system]['permoon_pcterrors'][0])
			satI_impact_pct_errors.append(moon_recovery_dict[system]['impact_pcterrors'][0])
			satI_eccentricity_pct_errors.append(moon_recovery_dict[system]['eccentricity_pcterrors'][0])
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) >= 2:
			satII_rsrstar_pct_errors.append(moon_recovery_dict[system]['rmoon_rstar_pcterrors'][1])
			satII_mmoon_pct_errors.append(moon_recovery_dict[system]['mmoon_pcterrors'][1])
			satII_permoon_pct_errors.append(moon_recovery_dict[system]['permoon_pcterrors'][1])
			satII_impact_pct_errors.append(moon_recovery_dict[system]['impact_pcterrors'][1])
			satII_eccentricity_pct_errors.append(moon_recovery_dict[system]['eccentricity_pcterrors'][1])
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) >= 3:
			satIII_rsrstar_pct_errors.append(moon_recovery_dict[system]['rmoon_rstar_pcterrors'][2])
			satIII_mmoon_pct_errors.append(moon_recovery_dict[system]['mmoon_pcterrors'][2])
			satIII_permoon_pct_errors.append(moon_recovery_dict[system]['permoon_pcterrors'][2])
			satIII_impact_pct_errors.append(moon_recovery_dict[system]['impact_pcterrors'][2])
			satIII_eccentricity_pct_errors.append(moon_recovery_dict[system]['eccentricity_pcterrors'][2])
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) >= 4:
			satIV_rsrstar_pct_errors.append(moon_recovery_dict[system]['rmoon_rstar_pcterrors'][3])
			satIV_mmoon_pct_errors.append(moon_recovery_dict[system]['mmoon_pcterrors'][3])
			satIV_permoon_pct_errors.append(moon_recovery_dict[system]['permoon_pcterrors'][3])
			satIV_impact_pct_errors.append(moon_recovery_dict[system]['impact_pcterrors'][3])
			satIV_eccentricity_pct_errors.append(moon_recovery_dict[system]['eccentricity_pcterrors'][3])
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) >= 5:
			satV_rsrstar_pct_errors.append(moon_recovery_dict[system]['rmoon_rstar_pcterrors'][4])
			satV_mmoon_pct_errors.append(moon_recovery_dict[system]['mmoon_pcterrors'][4])
			satV_permoon_pct_errors.append(moon_recovery_dict[system]['permoon_pcterrors'][4])
			satV_impact_pct_errors.append(moon_recovery_dict[system]['impact_pcterrors'][4])
			satV_eccentricity_pct_errors.append(moon_recovery_dict[system]['eccentricity_pcterrors'][4])
		#### grab the lowest number in each list and add that to best_list
		best_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
		best_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
		best_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
		best_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
		best_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))		
		#### DO THE SAME AS ABOVE, BUT NOW DO IT FOR THE N=1, N=2, etc, systems -- and grab the best fit for each of em.
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) == 1:
			N1_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
			N1_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
			N1_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
			N1_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
			N1_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) == 2:
			N2_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
			N2_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
			N2_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
			N2_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
			N2_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) == 3:
			N3_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
			N3_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
			N3_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
			N3_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
			N3_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) == 4:
			N4_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
			N4_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
			N4_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
			N4_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
			N4_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))			
		if len(moon_recovery_dict[system]['rmoon_rstar_pcterrors']) == 5:
			N5_rsrstar_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['rmoon_rstar_pcterrors'])))
			N5_mmoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['mmoon_pcterrors'])))
			N5_permoon_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['permoon_pcterrors'])))
			N5_impact_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['impact_pcterrors'])))
			N5_eccentricity_pct_errors.append(np.nanmin(np.array(moon_recovery_dict[system]['eccentricity_pcterrors'])))			
	except KeyError:
		continue



#-------------------------------


#### plot five histograms
#### RSAT / RSTAR --- for each satellite
pct_xlims = (-100,100)
fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(16,8))
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.13, hspace=0.24)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0][0].hist(np.array(satI_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[0], edgecolor='k')
ax[0][0].set_xlabel(r'I $R_S / R_{*}$ pct error')
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[0][1].hist(np.array(satII_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[1], edgecolor='k')
ax[0][1].set_xlabel(r'II $R_S / R_{*}$ pct error')
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[0][2].hist(np.array(satIII_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[2], edgecolor='k')
ax[0][2].set_xlabel(r'III $R_S / R_{*}$ pct error')
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[1][0].hist(np.array(satIV_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[3], edgecolor='k')
ax[1][0].set_xlabel(r'IV $R_S / R_{*}$ pct error')
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[1][1].hist(np.array(satV_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[4], edgecolor='k')
ax[1][1].set_xlabel(r'V $R_S / R_{*}$ pct error')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n6, bins6, edges6 = ax[1][2].hist(np.array(best_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[5], edgecolor='k')
ax[1][2].set_xlabel(r'best $R_S / R_{*}$ pct error')
plt.savefig(plotdir+'/RsRstar_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()



#-------------------------------




##### MSAT -- for each satellite
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.13, hspace=0.24)
#sat_colors = ['Red', 'Orange', 'Green', 'Blue', 'Purple']
n1, bins1, edges1 = ax[0][0].hist(np.array(satI_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[0], edgecolor='k')
ax[0][0].set_xlabel(r'I $M_S$ pct error')
#plt.savefig(plotdir+'/satI_mmoon_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[0][1].hist(np.array(satII_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[1], edgecolor='k')
ax[0][1].set_xlabel(r'II $M_S$ pct error')
#plt.savefig(plotdir+'/satII_mmoon_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[0][2].hist(np.array(satIII_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[2], edgecolor='k')
ax[0][2].set_xlabel(r'III $M_S$ pct error')
#plt.savefig(plotdir+'/satIII_mmoon_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[1][0].hist(np.array(satIV_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[3], edgecolor='k')
ax[1][0].set_xlabel(r'IV $M_S$ pct error')
#plt.savefig(plotdir+'/satIV_mmoon_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[1][1].hist(np.array(satV_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[4], edgecolor='k')
ax[1][1].set_xlabel(r'V $M_S$ pct error')
#plt.savefig(plotdir+'/satV_mmoon_pcterror.png', dpi=300)
#plt.show()
n6, bins6, edges6 = ax[1][2].hist(np.array(best_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[5], edgecolor='k')
ax[1][2].set_xlabel(r'best $M_S$ pct error')
plt.savefig(plotdir+'/MMoon_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()


#-------------------------------


###PSAT -- for each satellite
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.13, hspace=0.24)
#sat_colors = ['Red', 'Orange', 'Green', 'Blue', 'Purple']
n1, bins1, edges1 = ax[0][0].hist(np.array(satI_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[0], edgecolor='k')
ax[0][0].set_xlabel(r'I $P_S$ pct error')
#plt.savefig(plotdir+'/satI_permoon_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[0][1].hist(np.array(satII_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[1], edgecolor='k')
ax[0][1].set_xlabel(r'II $P_S$ pct error')
#plt.savefig(plotdir+'/satII_permoon_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[0][2].hist(np.array(satIII_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[2], edgecolor='k')
ax[0][2].set_xlabel(r'III $P_S$ pct error')
#plt.savefig(plotdir+'/satIII_permoon_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[1][0].hist(np.array(satIV_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[3], edgecolor='k')
ax[1][0].set_xlabel(r'IV $P_S$ pct error')
#plt.savefig(plotdir+'/satIV_permoon_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[1][1].hist(np.array(satV_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[4], edgecolor='k')
ax[1][1].set_xlabel(r'V $P_S$ pct error')
#plt.savefig(plotdir+'/satV_permoon_pcterror.png', dpi=300)
#plt.show()
n6, bins6, edges6 = ax[1][2].hist(np.array(best_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[5], edgecolor='k')
ax[1][2].set_xlabel(r'best $P_S$ pct error')
plt.savefig(plotdir+'/permoon_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()



###IMPACT-- for each satellite
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.13, hspace=0.24)
#sat_colors = ['Red', 'Orange', 'Green', 'Blue', 'Purple']
n1, bins1, edges1 = ax[0][0].hist(np.array(satI_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[0], edgecolor='k')
ax[0][0].set_xlabel(r'I $b_B$ pct error')
#plt.savefig(plotdir+'/satI_permoon_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[0][1].hist(np.array(satII_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[1], edgecolor='k')
ax[0][1].set_xlabel(r'II $b_B$ pct error')
#plt.savefig(plotdir+'/satII_permoon_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[0][2].hist(np.array(satIII_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[2], edgecolor='k')
ax[0][2].set_xlabel(r'III $b_B$ pct error')
#plt.savefig(plotdir+'/satIII_permoon_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[1][0].hist(np.array(satIV_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[3], edgecolor='k')
ax[1][0].set_xlabel(r'IV $b_B$ pct error')
#plt.savefig(plotdir+'/satIV_permoon_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[1][1].hist(np.array(satV_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[4], edgecolor='k')
ax[1][1].set_xlabel(r'V $b_B$ pct error')
#plt.savefig(plotdir+'/satV_permoon_pcterror.png', dpi=300)
#plt.show()
n6, bins6, edges6 = ax[1][2].hist(np.array(best_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[5], edgecolor='k')
ax[1][2].set_xlabel(r'best $b_B$ pct error')
plt.savefig(plotdir+'/impact_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()




system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
system_nmoons_list = []
#### do an injection - recovery here
fig, ax = plt.subplots(figsize=(8,8))
for system in moon_recovery_dict.keys():
	try:
		this_system_nmoons = len(moon_recovery_dict[system]['impact_gts'])
		system_color = system_colors[this_system_nmoons-1] ### if there's one moon, it's index 0, etc
		if this_system_nmoons not in system_nmoons_list:
			#### want to label it
			ax.scatter(moon_recovery_dict[system]['impact_gts'][0], moon_recovery_dict[system]['impact_recovered'], color=system_color, s=30, edgecolor='k', label=r'$N=$'+str(this_system_nmoons))
		else:
			ax.scatter(moon_recovery_dict[system]['impact_gts'][0], moon_recovery_dict[system]['impact_recovered'], color=system_color, s=30, edgecolor='k')
		system_nmoons_list.append(this_system_nmoons)
	except:
		continue
plt.legend()
ax.set_xlabel(r'$b_B$ ground truth')
ax.set_ylabel(r'$b_B$ recovered')
plt.show()






###ECCENTRICITY-- for each satellite
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.13, hspace=0.24)
#sat_colors = ['Red', 'Orange', 'Green', 'Blue', 'Purple']
n1, bins1, edges1 = ax[0][0].hist(np.array(satI_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[0], edgecolor='k')
ax[0][0].set_xlabel(r'I $e_B$ pct error')
#plt.savefig(plotdir+'/satI_permoon_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[0][1].hist(np.array(satII_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[1], edgecolor='k')
ax[0][1].set_xlabel(r'II $e_B$ pct error')
#plt.savefig(plotdir+'/satII_permoon_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[0][2].hist(np.array(satIII_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[2], edgecolor='k')
ax[0][2].set_xlabel(r'III $e_B$ pct error')
#plt.savefig(plotdir+'/satIII_permoon_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[1][0].hist(np.array(satIV_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[3], edgecolor='k')
ax[1][0].set_xlabel(r'IV $e_B$ pct error')
#plt.savefig(plotdir+'/satIV_permoon_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[1][1].hist(np.array(satV_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[4], edgecolor='k')
ax[1][1].set_xlabel(r'V $e_B$ pct error')
#plt.savefig(plotdir+'/satV_permoon_pcterror.png', dpi=300)
#plt.show()
n6, bins6, edges6 = ax[1][2].hist(np.array(best_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=sat_colors[5], edgecolor='k')
ax[1][2].set_xlabel(r'best $e_B$ pct error')
plt.savefig(plotdir+'/eccentricity_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()






#-------------------------------



####  #### RS/Rstar -- FOR N=1,N=2,etc, systes
fig, ax = plt.subplots(nrows=5,ncols=1,figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.15)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0].hist(np.array(satI_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[0], edgecolor='k')
#ax[0].set_xlabel(r' $N = 1$ $R_S / R_{*}$ pct error')
ax[0].set_ylabel(r'$N = 1$')
ax[0].set_xticks([])
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[1].hist(np.array(satII_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[1], edgecolor='k')
#ax[1].set_xlabel(r'$N = 2$ $R_S / R_{*}$ pct error')
ax[1].set_ylabel(r'$N = 2$')
ax[1].set_xticks([])
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[2].hist(np.array(satIII_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[2], edgecolor='k')
#ax[2].set_xlabel(r'$N = 3$ $R_S / R_{*}$ pct error')
ax[2].set_ylabel(r'$N = 3$')
ax[2].set_xticks([])
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[3].hist(np.array(satIV_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[3], edgecolor='k')
#ax[3].set_xlabel(r'$N = 4$ $R_S / R_{*}$ pct error')
ax[3].set_ylabel(r'$N = 4$')
ax[3].set_xticks([])
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[4].hist(np.array(satV_rsrstar_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[4], edgecolor='k')
ax[4].set_xlabel(r'$R_S / R_{*}$ pct error')
ax[4].set_ylabel(r'$N = 5$')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
plt.savefig(plotdir+'/nsats_per_system_RsRstar_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()



#-------------------------------



####  #### MMoon -- FOR N=1,N=2,etc, systes
fig, ax = plt.subplots(nrows=5,ncols=1,figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.15)
#plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.28, hspace=0.47)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0].hist(np.array(satI_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[0], edgecolor='k')
#ax[0].set_xlabel(r' $N = 1$ $M_S$ pct error')
ax[0].set_ylabel(r'$N = 1$')
ax[0].set_xticks([])
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[1].hist(np.array(satII_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[1], edgecolor='k')
#ax[1].set_xlabel(r'$N = 2$ $M_S$ pct error')
ax[1].set_xticks([])
ax[1].set_ylabel(r'$N = 2$')
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[2].hist(np.array(satIII_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[2], edgecolor='k')
#ax[2].set_xlabel(r'$N = 3$ $M_S$ pct error')
ax[2].set_xticks([])
ax[2].set_ylabel(r'$N = 3$')
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[3].hist(np.array(satIV_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[3], edgecolor='k')
#ax[3].set_xlabel(r'$N = 4$ $M_S$ pct error')
ax[3].set_xticks([])
ax[3].set_ylabel(r'$N = 4$')
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[4].hist(np.array(satV_mmoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[4], edgecolor='k')
ax[4].set_xlabel(r'$M_S$ pct error')
ax[4].set_ylabel(r'$N = 5$')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
plt.savefig(plotdir+'/nsats_per_system_mmoon_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()


#-------------------------------



####  #### Psat -- FOR N=1,N=2,etc, systes
fig, ax = plt.subplots(nrows=5,ncols=1,figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.15)
#plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.28, hspace=0.47)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0].hist(np.array(satI_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[0], edgecolor='k')
#ax[0].set_xlabel(r' $N = 1$ $P_S$ pct error')
ax[0].set_xticks([])
ax[0].set_ylabel(r'$N = 1$')
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[1].hist(np.array(satII_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[1], edgecolor='k')
#ax[1].set_xlabel(r'$N = 2$ $P_S$ pct error')
ax[1].set_xticks([])
ax[1].set_ylabel(r'$N = 2$')
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[2].hist(np.array(satIII_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[2], edgecolor='k')
#ax[2].set_xlabel(r'$N = 3$ $P_S$ pct error')
ax[2].set_xticks([])
ax[2].set_ylabel(r'$N = 3$')
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[3].hist(np.array(satIV_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[3], edgecolor='k')
#ax[3].set_xlabel(r'$N = 4$ $P_S$ pct error')
ax[3].set_xticks([])
ax[3].set_ylabel(r'$N = 4$')
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[4].hist(np.array(satV_permoon_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[4], edgecolor='k')
ax[4].set_xlabel(r'$P_S$ pct error')
ax[4].set_ylabel(r'$N = 5$')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
plt.savefig(plotdir+'/nsats_per_system_permoon_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()







####  #### Psat -- FOR N=1,N=2,etc, systes
fig, ax = plt.subplots(nrows=5,ncols=1,figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.15)
#plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.28, hspace=0.47)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0].hist(np.array(satI_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[0], edgecolor='k')
#ax[0].set_xlabel(r' $N = 1$ $P_S$ pct error')
ax[0].set_xticks([])
ax[0].set_ylabel(r'$N = 1$')
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[1].hist(np.array(satII_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[1], edgecolor='k')
#ax[1].set_xlabel(r'$N = 2$ $P_S$ pct error')
ax[1].set_xticks([])
ax[1].set_ylabel(r'$N = 2$')
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[2].hist(np.array(satIII_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[2], edgecolor='k')
#ax[2].set_xlabel(r'$N = 3$ $P_S$ pct error')
ax[2].set_xticks([])
ax[2].set_ylabel(r'$N = 3$')
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[3].hist(np.array(satIV_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[3], edgecolor='k')
#ax[3].set_xlabel(r'$N = 4$ $P_S$ pct error')
ax[3].set_xticks([])
ax[3].set_ylabel(r'$N = 4$')
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[4].hist(np.array(satV_impact_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[4], edgecolor='k')
ax[4].set_xlabel(r'$P_S$ pct error')
ax[4].set_ylabel(r'$N = 5$')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
plt.savefig(plotdir+'/nsats_per_system_impact_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()








####  #### ECCENTRICITY-- FOR N=1,N=2,etc, systes
fig, ax = plt.subplots(nrows=5,ncols=1,figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.15)
#plt.subplots_adjust(left=0.08, bottom=0.14, right=0.93, top=0.95, wspace=0.28, hspace=0.47)
#sat_colors = ['#B9F291', '#50BF95', '#5E5959', '#F72349', '#FB845E', '#F6DC85']
system_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']
#sat_colors = [(185,242,145), (80,191,149), (94,89,89), (247,35,73), (251,132,94), (246,220,133)]
#### indices are row number, column number
n1, bins1, edges1 = ax[0].hist(np.array(satI_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[0], edgecolor='k')
#ax[0].set_xlabel(r' $N = 1$ $P_S$ pct error')
ax[0].set_xticks([])
ax[0].set_ylabel(r'$N = 1$')
#plt.savefig(plotdir+'/satI_RsRstar_pcterror.png', dpi=300)
#plt.show()
n2, bins2, edges2 = ax[1].hist(np.array(satII_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[1], edgecolor='k')
#ax[1].set_xlabel(r'$N = 2$ $P_S$ pct error')
ax[1].set_xticks([])
ax[1].set_ylabel(r'$N = 2$')
#plt.savefig(plotdir+'/satII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n3, bins3, edges3 = ax[2].hist(np.array(satIII_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[2], edgecolor='k')
#ax[2].set_xlabel(r'$N = 3$ $P_S$ pct error')
ax[2].set_xticks([])
ax[2].set_ylabel(r'$N = 3$')
#plt.savefig(plotdir+'/satIII_RsRstar_pcterror.png', dpi=300)
#plt.show()
n4, bins4, edges4 = ax[3].hist(np.array(satIV_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[3], edgecolor='k')
#ax[3].set_xlabel(r'$N = 4$ $P_S$ pct error')
ax[3].set_xticks([])
ax[3].set_ylabel(r'$N = 4$')
#plt.savefig(plotdir+'/satIV_RsRstar_pcterror.png', dpi=300)
#plt.show()
n5, bins5, edges5 = ax[4].hist(np.array(satV_eccentricity_pct_errors).clip(min=-105, max=105), bins=np.arange(-105,115,10), facecolor=system_colors[4], edgecolor='k')
ax[4].set_xlabel(r'$P_S$ pct error')
ax[4].set_ylabel(r'$N = 5$')
#plt.savefig(plotdir+'/satV_RsRstar_pcterror.png', dpi=300)
#plt.show()
plt.savefig(plotdir+'/nsats_per_system_eccentricity_pcterror_histograms.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()




#-------------------------------



#### FINAL RESULTS PLOTTER
#planet_only_posterior_mean_dict = {}
#planet_only_posterior_median_dict = {}
#planet_only_posterior_sigma_dict = {}
#planet_moon_posterior_mean_dict = {}
#planet_moon_posterior_median_dict = {}
#planet_moon_posterior_sigma_dict = {}


pmkeys = planet_moon_posterior_median_dict.keys()


#### map the ground_truth_dict.keys() to the pmkeys 


for pmkey,pmlabel in zip(pmkeys,planet_moon_names_labels):
	print('pmkey, pmlabel', pmkey, pmlabel)
	#### make a histogram of these results
	if pmkey in planet_only_posterior_median_dict.keys():
	#### IF THIS PARAMETER IS IN BOTH PLANET AND MOON FITS
		try:
			ground_truth_pmkey = model_to_ground_truth_dict[pmkey] ### get the ground truth version
			ground_truths = np.array(ground_truth_dict[ground_truth_pmkey])
			print('len(ground_truths): ', len(ground_truths))
			print('ground_truths: ', ground_truths)
			if pmkey == 'ecc_bary':
				fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
				plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
				ground_truth_available = False
			else:
				fig, ax = plt.subplots(3, sharex=True, figsize=(6,8))
				plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
				ground_truth_available = True
		except:
			traceback.print_exc() 
			fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
			plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
			ground_truth_available = False 
		##### means that the key exists in both
		#fig, ax = plt.subplots(2, sharex=True)
		### compute min and max values
		if 'impact' in pmlabel:
			bin_min, bin_max = 0., 1.
		else:
			bin_min = np.nanmin((np.nanmin(ground_truths), np.nanmin(planet_only_posterior_median_dict[pmkey]), np.nanmin(planet_moon_posterior_median_dict[pmkey])))
			bin_max = np.nanmax((np.nanmax(ground_truths), np.nanmax(planet_only_posterior_median_dict[pmkey]), np.nanmax(planet_moon_posterior_median_dict[pmkey])))
		nbins = 50 
		if (bin_max - bin_min) > 1e3:
			histbins = np.logspace(np.log10(bin_min), np.log10(bin_max), nbins)
			ax[0].set_xscale('log')
			ax[1].set_xscale('log')
		else:
			histbins = np.linspace(bin_min, bin_max, nbins)
		ax[0].hist(planet_only_posterior_median_dict[pmkey], bins=histbins, facecolor=model_colors[0], edgecolor='k', label='planet only')
		ax[1].hist(planet_moon_posterior_median_dict[pmkey], bins=histbins, facecolor=model_colors[1], edgecolor='k', label='planet+moon')
		ax[0].set_ylabel('planet only')
		ax[1].set_ylabel('planet+moon')
		if (ground_truth_available == True) and (pmkey != 'ecc_bary'):
			ax[2].hist(ground_truths, bins=histbins, facecolor=model_colors[2], edgecolor='k', label='ground truth')
			ax[2].set_ylabel('ground truth')
			ax[2].set_xlabel(pmlabel)
		else:
			ax[1].set_xlabel(pmlabel)
		plt.savefig(plotdir+'/'+str(pmkey)+'_histogram.png', dpi=300)
		if show_final_plots == 'y':
			plt.show()
		plt.close()
	else:
		### IF THIS A MOON-ONLY PARAMETER 
		try:
			ground_truth_pmkey = np.array(model_to_ground_truth_dict)[pmkey] ### get the ground truth version
			ground_truths = ground_truth_dict[ground_truth_pmkey]
			fig, ax = plt.subplots(2, sharex=True, figsize=(6,8))
			ground_truth_available = True
			nbins = 50
			bin_min = np.nanmin(planet_moon_posterior_median_dict[pmkey])
			bin_max = np.nanmax(planet_moon_posterior_median_dict[pmkey])
			if (bin_max - bin_min) > 1e3:
				histbins = np.logspace(np.log10(bin_min), np.log10(bin_max), nbins)
				ax[0].set_xscale('log')
				ax[1].set_xscale('log')
			else:
				histbins = np.linspace(bin_min, bin_max, nbins)
			ax[0].hist(planet_moon_posterior_median_dict[pmkey], bins=histbins, facecolor=model_colors[1], edgecolor='k', label='planet+moon')
			ax[0].set_ylabel('planet+moon')
			ax[1].hist(ground_truths, bins=histbins, facecolor=model_colors[2], edgecolor='k', label='ground truth')
			ax[1].set_ylabel('ground truth')
			ax[1].set_xlabel(pmlabel)
			if show_final_plots == 'y':
				plt.show()
			plt.close()
		except:
			traceback.print_exc() 
			fig, ax = plt.subplots(figsize=(6,6))
			ground_truth_available = False 
			#fig, ax = plt.subplots()
			ax.hist(planet_moon_posterior_median_dict[pmkey], bins=50, facecolor=model_colors[1], edgecolor='k', label='planet+moon')
			ax.set_ylabel('planet+moon')
			ax.set_xlabel(pmlabel)
			if show_final_plots == 'y':
				plt.show()
			plt.close()
	print(' ')
	print(' ')
	if ('impact' in pmkey) or ('impact' in pmlabel):
		break


#-------------------------------


pmad = planet_moon_attributes_dict
pmad_labels = [
	r'run', '$\Delta \log Z$', r'$K$', 'simulation', '# of moons', r'planet-moon $\log Z$', '# of transits', 'res_nores', 'Sat I SNR', 'Sat II SNR', 
	'Sat III SNR', 'Sat IV SNR', 'Sat V SNR', r'$T_{\mathrm{dur}}$', 'Planet', 'I', 'II', 'III', 'IV', 'V', 'moon timed out'
	]

dict_keys = [
'run_name', 'delta_logz', 'bayes_factor', 'sim', 'nmoons', 'planet_moon_logz', 'ntransits', 'res_nores', 'I SNR', 'II SNR', 'III SNR', 'IV SNR', 'V SNR', 
'tdur', 'Planet', 'I', 'II', 'III', 'IV', 'V', 'moon_timed_out'
]

combined_run_idxs = np.where(np.array(pmad['run_name']) == 'combined_runs')[0]
run8_10_idxs = np.where(np.array(pmad['run_name']) == 'run8_10')[0]
if run_name == 'run8_10': 
	idxs_to_use = np.where(np.array(pmad['run_name']) == 'run11_10')[0] ### for historical reasons, to replace the planet runs of run8_10
else:
	idxs_to_use = np.where(np.array(pmad['run_name']) == run_name)[0]
timed_out_idxs = np.where(np.array(pmad['moon_timed_out']) == False)[0]


#-------------------------------


xlabel_list = []
xvals_list = []

for key in pmad.keys():
	if type(pmad[key]) == dict:
		objectkeys = pmad[key].keys()
		if key in ['I', 'II', 'III', 'IV', 'V']:
			#objectkeys = pmad[key].keys()
			moon_param = True 
		else:
			#objectkeys = ['not_a_moon']
			moon_param = False 

		for objectkey in objectkeys:
			#if moon_param == True:
			xval_label = str(key)+' '+str(objectkey)
			#xvals = np.array(pmad[key][objectkey])[::2] #### doing this because we have two runs, with identical values for each. So skip one
			#f key in ['II', 'III', 'IV', 'V']:
				#xvals = np.array(pmad[key][objectkey])[::2] #### doing this because we have two runs, with identical values for each. So skip one
			#else:
			xvals = np.array(pmad[key][objectkey])[idxs_to_use]

			xlabel_list.append(xval_label)
			xvals_list.append(xvals)

	elif type(pmad[key]) == list:
			xval_label = key 
			xvals = np.array(pmad[key])[idxs_to_use]

			if 'SNR' in key:
				xvals = np.array(xvals)[np.isfinite(np.array(xvals))].tolist()

			xlabel_list.append(xval_label)
			xvals_list.append(xvals)


#-------------------------------



#### make a histogram of all the delta_logz
#plotted_value = np.sign(pmad['delta_logz']) * np.log10(np.abs(pmad['delta_logz']))
plotted_value = np.array(pmad['bayes_factor'])[idxs_to_use]
#fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,6))
fig, ax = plt.subplots(figsize=(6,6))
#histbins = np.arange(-1,4.2,0.2)
histbins = np.logspace(-1,5,40)
n1, bins1, edges1 = ax.hist(plotted_value, bins=histbins, facecolor='LightSkyBlue', edgecolor='k', zorder=1)
#xlims = [np.nanmin(plotted_value), np.nanmax(plotted_value)]
xlims = [1e-1, 1e5]
ylims = [0, 1.1*np.nanmax(n1)]
ax.set_xlim(xlims[0], xlims[1])
ax.set_ylim(ylims[0], ylims[1])
ax.fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(xlims[0], xlims[0], 100), x2=np.linspace(1,1,100), color='red', alpha=0.5, zorder=0)
ax.fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(1, 1, 100), x2=np.linspace(3.2,3.2,100), color='yellow', alpha=0.5, zorder=0)
ax.fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(3.2,3.2, 100), x2=np.linspace(xlims[1],xlims[1],100), color='green', alpha=0.5, zorder=0)
ax.plot(np.linspace(1,1,100), np.linspace(ylims[0], ylims[1], 100), color='k', linestyle='--')
ax.plot(np.linspace(3.2,3.2,100), np.linspace(ylims[0], ylims[1], 100), color='k', linestyle='--')
#ax.set_xlabel(r'$\log_{10} \Delta \log Z$')
ax.set_xlabel(r'$K$')
ax.set_xscale('log')
plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
#### BELOW IT, PLOT THE SAME THING, NOW AS LOG10(K)
"""
histbins2 = np.linspace(np.nanmin(np.log10(plotted_value)),np.nanmax(np.log10(plotted_value)),40)
n2, bins2, edges2 = ax[1].hist(np.log10(plotted_value), bins=histbins2, facecolor='LightSkyBlue', edgecolor='k', zorder=1)
xlims2 = [np.nanmin(histbins2), np.nanmax(histbins2)]
ylims2 = [0, 1.1*np.nanmax(n2)] #### this is a histogram
ax[1].fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(xlims[0], xlims[0], 100), x2=np.linspace(0,0,100), color='red', alpha=0.5, zorder=0)
ax[1].fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(0, 0, 100), x2=np.linspace(0.5,0.5,100), color='yellow', alpha=0.5, zorder=0)
ax[1].fill_betweenx(y=np.linspace(ylims[0], ylims[1], 100), x1=np.linspace(0.5,0.5, 100), x2=np.linspace(xlims[1],xlims[1],100), color='green', alpha=0.5, zorder=0)
ax[1].plot(np.linspace(0,0,100), np.linspace(ylims[0], ylims[1], 100), color='k', linestyle='--')
ax[1].plot(np.linspace(0.5,0.5,100), np.linspace(ylims[0], ylims[1], 100), color='k', linestyle='--')
"""
#plt.savefig(plotdir+'/all_delta_logZ_histogram.png', dpi=300)
plt.savefig(plotdir+'/all_bayes_factor_histogram.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()

infinite_K_idxs = np.where(np.isfinite(np.array(pmad['bayes_factor'])[idxs_to_use]) == False)[0]
infinite_K_sims = np.array(pmad['sim'])[idxs_to_use][infinite_K_idxs]
high_K_idxs = np.where(np.array(pmad['bayes_factor'])[idxs_to_use] > 1e6)[0]
high_K_sims = np.array(pmad['sim'])[idxs_to_use][high_K_idxs]
print('infinite_K_sims: ', infinite_K_sims)

"""
for infK_sim in infinite_K_sims:
	print('opening '+infK_sim+' lightcurve...')
	os.system('open '+plotdir+'/'+infK_sim+'_lightcurve.png')

for highK_sim in high_K_sims:
	print('opening '+highK_sim+' lightcurve...')
	os.system('open '+plotdir+'/'+highK_sim+'_lightcurve.png')
"""

"""
infinite_K_sims = np.array(['july13_nores_268', 'july13_res_247', 'july13_res_20',
       'july13_nores_201', 'july13_res_424', 'july13_nores_898',
       'july13_res_701'], dtype='<U16')
"""

#-------------------------------


for xval_label, xvals in zip(xlabel_list, xvals_list):
	bayes_factor_idx = int(np.where(np.array(xlabel_list) == 'bayes_factor')[0])
	print('xval_label: ', xval_label)
	if (type(xvals[0]) == str) or (type(xvals[0])) == np.str_:
		continue 
	elif type(xvals[0]) == type(None):
		continue 
	#nmoons_list = np.array(xvals_list[3]) #### now it's 3, because we added Bayes factor 
	nmoons_list = np.array(pmad['nmoons'])[idxs_to_use]
	print('nmoons_list: ', nmoons_list)
	if xval_label.startswith('Planet '):
		good_indices = np.arange(0,len(xvals),1) #### all of em
	elif xval_label.startswith('I '):
		good_indices = np.where(nmoons_list >= 1)[0]
	elif xval_label.startswith('II '):
		good_indices = np.where(nmoons_list >= 2)[0]
	elif xval_label.startswith('III '):
		good_indices = np.where(nmoons_list >= 3)[0]
	elif xval_label.startswith('IV '):
		good_indices = np.where(nmoons_list >= 4)[0]
	elif xval_label.startswith('V '):
		good_indices = np.where(nmoons_list >= 5)[0]
	else:
		good_indices = np.arange(0,len(xvals),1)
	print('good_indices: ', good_indices)
	try:
		if xval_label.startswith('Planet '):
			xval_key = xval_label[7:]
			xvals = np.array(pmad['Planet'][xval_key])[idxs_to_use]
		elif (xval_label.startswith('I ')) and ('SNR' not in xval_label):
			xval_key = xval_label[2:]
			xvals = np.array(pmad['I'][xval_key])[idxs_to_use]
		elif (xval_label.startswith('II ')) and ('SNR' not in xval_label):
			xval_key = xval_label[3:]
			xvals = np.array(pmad['II'][xval_key])[idxs_to_use]
		elif (xval_label.startswith('III ')) and ('SNR' not in xval_label):
			xval_key = xval_label[4:]
			xvals = np.array(pmad['III'][xval_key])[idxs_to_use]
		elif (xval_label.startswith('IV ')) and ('SNR' not in xval_label):
			xval_key = xval_label[3:]
			xvals = np.array(pmad['IV'][xval_key])[idxs_to_use]
		elif (xval_label.startswith('V ')) and ('SNR' not in xval_label):
			xval_key = xval_label[2:]
			xvals = np.array(pmad['V'][xval_key])[idxs_to_use]
		else:
			xvals = np.array(pmad[xval_label])[idxs_to_use]
		yvals = np.array(pmad['bayes_factor'])[idxs_to_use]
		res_nores_list = np.array(planet_moon_attributes_dict['res_nores'])[idxs_to_use][good_indices]
		fig, ax = plt.subplots(figsize=(6,8))
		if ((np.nanmax(xvals) - np.nanmin(xvals)) > 1e2) or ('snr' in xval_label.lower()):
			#### make it log!
			xscale_log = True
			new_xvals = np.array(np.sign(xvals) * np.log10(np.abs(xvals)))
			ax.set_xlabel(r'$\log_{10}($'+str(xval_label)+r'$)$')	
		else:
			xscale_log = False 
			new_xvals = np.array(xvals) ### leave it untouched
			if xval_label.lower() == 'planet impact':
				ax.set_xlabel('impact parameter')
			else:
				ax.set_xlabel(xval_label)
		new_yvals = yvals 
		highK_xvals = new_xvals[high_K_idxs]
		highK_yvals = new_yvals[high_K_idxs]
		#### make sure they're all finite! sort on the yvals -- means you have to modify new_yvals last!
		#new_xvals = new_xvals[np.isfinite(new_yvals)]
		#new_yvals = new_yvals[np.isfinite(new_yvals)]
		### get rid of the crazy outliers
		#new_xvals, new_yvals = new_xvals[(new_xvals < 1e6) & (new_yvals < 1e6)], new_yvals[(new_xvals < 1e6) & (new_yvals < 1e6)]
		planet_favored_idxs = np.where(new_yvals <= 1)[0]
		marginal_moon_idxs = np.where((new_yvals > 1) & (new_yvals < 3.2))[0]
		moon_favored_idxs = np.where(new_yvals >= 3.2)[0]
		res_idxs = np.where(res_nores_list == 'res')[0]
		nores_idxs = np.where(res_nores_list == 'nores')[0]
		##### DOUBLE INDENT BELOW THIS
		ax.set_ylabel(r'$K$')
		if xval_label == 'nmoons':
			xlimits = (0.,6.)
		else:
			#xlimits = (np.nanmin(new_xvals[np.isfinite(new_xvals)]), np.nanmax(new_xvals[np.isfinite(new_xvals)]))
			xlimits = (np.nanmin(new_xvals[np.isfinite(new_xvals)]), np.nanmax(new_xvals[np.isfinite(new_xvals)]))
		#ylimits = (np.nanmin(new_yvals[np.isfinite(new_yvals)]), np.nanmax(new_yvals[np.isfinite(new_yvals)]))
		ylimits = (1e-1,1e6)
		#### make four categories, based on overalp
		moon_favored_res_idxs = np.intersect1d(moon_favored_idxs, res_idxs)
		moon_favored_nores_idxs = np.intersect1d(moon_favored_idxs, nores_idxs)
		marginal_res_idxs = np.intersect1d(marginal_moon_idxs, res_idxs)
		marginal_nores_idxs = np.intersect1d(marginal_moon_idxs, nores_idxs)
		moon_disfavored_res_idxs = np.intersect1d(planet_favored_idxs, res_idxs)
		moon_disfavored_nores_idxs = np.intersect1d(planet_favored_idxs, nores_idxs)
		plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
		ax.scatter(new_xvals[moon_favored_res_idxs], new_yvals[moon_favored_res_idxs], facecolor='DodgerBlue', marker='o', edgecolor='k', s=30, label='resonant', zorder=1)	
		ax.scatter(new_xvals[moon_favored_nores_idxs], new_yvals[moon_favored_nores_idxs], facecolor='DodgerBlue', marker='D', edgecolor='k', s=30, label='non-resonant', zorder=1)
		#xline = np.linspace(np.nanmin(new_xvals), np.nanmax(new_xvals),100)
		xline = np.linspace(xlimits[0], xlimits[1], 100)
		yline1 = np.linspace(1,1,100)
		yline2 = np.linspace(3.2,3.2,100)
		ax.plot(xline, yline1, linestyle='--', color='k')
		ax.plot(xline, yline2, linestyle='--', color='k')
		ax.fill_between(xline, y1=np.linspace(ylimits[0],ylimits[0],100), y2=yline1, color='red', alpha=0.5, zorder=0)
		ax.fill_between(xline, y1=yline1, y2=yline2, color='yellow', alpha=0.5, zorder=0)
		ax.fill_between(xline, y1=yline2, y2=np.linspace(ylimits[1], ylimits[1], 100), color='green', alpha=0.5, zorder=0)
		ax.scatter(new_xvals[marginal_res_idxs], new_yvals[marginal_res_idxs], facecolor='DodgerBlue', marker='o', edgecolor='k', s=30, zorder=1)	
		#ax.scatter(new_xvals[marginal_nores_idxs], new_yvals[marginal_nores_idxs], facecolor='k', marker='D', edgecolor='k', s=30, label='marginal evidence, non-resonant', zorder=0)
		ax.scatter(new_xvals[marginal_nores_idxs], new_yvals[marginal_nores_idxs], facecolor='DodgerBlue', marker='D', edgecolor='k', s=30, zorder=1)
		ax.scatter(new_xvals[moon_disfavored_res_idxs], new_yvals[moon_disfavored_res_idxs], facecolor='DodgerBlue', marker='o', edgecolor='k', s=30, zorder=1)	
		#ax.scatter(new_xvals[moon_disfavored_nores_idxs], new_yvals[moon_disfavored_nores_idxs], facecolor=model_colors[0], marker='D', edgecolor='k', s=30, label='moon disfavored, non-resonant', zorder=0)
		ax.scatter(new_xvals[moon_disfavored_nores_idxs], new_yvals[moon_disfavored_nores_idxs], facecolor='DodgerBlue', marker='D', edgecolor='k', s=30, zorder=1)		
		for highKx, highKy in zip(highK_xvals, highK_yvals):
			#### let the width be 1/20 the distance between the xlimits, and the 
			arrow_width = (xlimits[1] - xlimits[0]) / 200
			arrow_head_length = 2e5
			ax.arrow(x=highKx, y=(ylimits[1]-0.9*ylimits[1]), dx=0, dy=(0.9*ylimits[1]), width=arrow_width, head_length=arrow_head_length, length_includes_head=True, color='red') #### it's a log scale, so 90% off 10^5 = down to 10% of 10^5, which is 10^4. 
		ax.set_xlim(xlimits[0], xlimits[1])
		ax.set_ylim(ylimits[0], ylimits[1])
		#### do 20 different iterations of the polynomial fit -- leaving out some of the data points at random
		if fit_line_to_scatter == 'y':
			for i in np.arange(0,20,1):
				random_idxs = np.random.randint(low=0,high=len(new_xvals), size=int(0.9 * len(new_xvals)))
				poly_xvals = np.linspace(np.nanmin(new_xvals), np.nanmax(new_xvals), 100)
				#####
				deg1_polycoeffs = np.polyfit(new_xvals[random_idxs], np.log(new_yvals[random_idxs]), deg=1) #### xvals are already logged, or they're not! Either way, don't want to double log it.
				deg1_polyfunc = np.polyval(deg1_polycoeffs, np.log(poly_xvals))
				deg1_poly_yvals = np.exp(deg1_polyfunc)
				######
				if i == 0:
					ax.plot(poly_xvals, deg1_poly_yvals, color='k', linestyle='--', alpha=0.3, zorder=1)
					poly_yvals_stack = deg1_poly_yvals
				else:
					ax.plot(poly_xvals, deg1_poly_yvals, color='k', linestyle='--', alpha=0.3, zorder=1)
					poly_yvals_stack = np.vstack((poly_yvals_stack, deg1_poly_yvals))
			ax.plot(poly_xvals, np.nanmedian(poly_yvals_stack, axis=0), color='red', linestyle='--', label='median fit', linewidth=3, alpha=1, zorder=2)
		ax.set_yscale('log')
		#########
		plt.legend()
		plt.savefig(plotdir+'/bayes_factor_vs_'+str(xval_label)+'.png', dpi=300)
		if show_final_plots == 'y':
			plt.show()
		plt.close()
		##########
	except:
		plt.close()
		traceback.print_exc()
		print(' ')
		continue_query = input('An exception was raised. Do you want to continue? ')
		if continue_query != 'y':
			raise Exception('you opted not to continue.')



#-------------------------------


#### DOUBLE INDENT ABOVE THIS LINE 






##### SCATTER PLOT OF TWO VALUES, COLOR-CODED BY DELTA-LOGZ
for i in np.arange(0,len(xlabel_list),1):
	for j in np.arange(0,len(xlabel_list),1):
		#if (i != j):
		if (i != j) and (xlabel_list[i].startswith('Planet ')) and (xlabel_list[j].startswith('Planet ')): 	

			try:
				xval_label, xvals = xlabel_list[i], xvals_list[i]
				yval_label, yvals = xlabel_list[j], xvals_list[j]

				if (type(xvals[0]) == type(None)) or (type(yvals[0]) == type(None)):
					continue

				if xval_label.startswith('Planet '):
					xval_key = xval_label[7:]
					xvals = np.array(pmad['Planet'][xval_key])[idxs_to_use]
				elif xval_label.startswith('I '):
					xval_key = xval_label[2:]
					xvals = np.array(pmad['I'][xval_key])[idxs_to_use]
				elif xval_label.startswith('II '):
					xval_key = xval_label[3:]
					xvals = np.array(pmad['II'][xval_key])[idxs_to_use]
				elif xval_label.startswith('III '):
					xval_key = xval_label[4:]
					xvals = np.array(pmad['III'][xval_key])[idxs_to_use]
				elif xval_label.startswith('IV '):
					xval_key = xval_label[3:]
					xvals = np.array(pmad['IV'][xval_key])[idxs_to_use]
				elif xval_label.startswith('V '):
					xval_key = xval_label[2:]
					xvals = np.array(pmad['V'][xval_key])[idxs_to_use]
				else:
					xvals = np.array(pmad[xval_label])[idxs_to_use]

				if yval_label.startswith('Planet '):
					yval_key = yval_label[7:]
					yvals = np.array(pmad['Planet'][yval_key])[idxs_to_use]
				elif yval_label.startswith('I '):
					yval_key = yval_label[2:]
					yvals = np.array(pmad['I'][yval_key])[idxs_to_use]
				elif yval_label.startswith('II '):
					yval_key = yval_label[3:]
					yvals = np.array(pmad['II'][yval_key])[idxs_to_use]
				elif yval_label.startswith('III '):
					yval_key = yval_label[4:]
					yvals = np.array(pmad['III'][yval_key])[idxs_to_use]
				elif yval_label.startswith('IV '):
					yval_key = yval_label[3:]
					yvals = np.array(pmad['IV'][yval_key])[idxs_to_use]
				elif yval_label.startswith('V '):
					yval_key = yval_label[2:]
					yvals = np.array(pmad['V'][yval_key])[idxs_to_use]
				else:
					yvals = np.array(pmad[yval_label])[idxs_to_use]

				zvals = np.array(pmad['bayes_factor'])[idxs_to_use]


				print('xval_label: ', xval_label)
				print('yval_label: ', yval_label)
				nmoons_list = np.array(xvals_list[3]) ### had to update this when we added 

				res_nores_list = np.array(planet_moon_attributes_dict['res_nores'])
				timed_out_list = np.array(planet_moon_attributes_dict['moon_timed_out'])

				fig, ax = plt.subplots(figsize=(6,8))

				if np.nanmax(xvals) - np.nanmin(xvals) > 1e2:
					#### make it log!
					new_xvals = np.array(np.sign(xvals) * np.log10(np.abs(xvals)))
					ax.set_xlabel(r'$\log_{10}($'+str(xval_label)+r'$)$')	
				else:
					new_xvals = np.array(xvals) ### leave it untouched
					ax.set_xlabel(xval_label)

				if np.nanmax(yvals) - np.nanmin(yvals) > 1e2:
					new_yvals = np.array(np.sign(yvals) * np.log10(np.abs(yvals))) #### logarithmic, but allow negative values 
					ax.set_ylabel(r'$\log_{10}($'+str(yval_label)+r'$)$')
				else:
					new_yvals = np.array(yvals)
					ax.set_ylabel(yval_label)


				new_zvals = zvals

				res_idxs = np.where(res_nores_list[idxs_to_use] == 'res')[0]
				nores_idxs = np.where(res_nores_list[idxs_to_use] == 'nores')[0]
				timed_out_idxs = np.where(timed_out_list[idxs_to_use] == True)[0]
				not_timed_out_idxs = np.where(timed_out_list[idxs_to_use] == False)[0]


				cm = plt.cm.get_cmap('RdYlGn')
				plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.1, hspace=0.08)
				im = ax.scatter(new_xvals[res_idxs], new_yvals[res_idxs], c=np.log10(new_zvals[res_idxs]), vmin=-2, vmax=2, cmap=cm, marker='o', edgecolor='k', s=30, label='resonant', zorder=0)	
				ax.scatter(new_xvals[nores_idxs], new_yvals[nores_idxs], c=np.log10(new_zvals[nores_idxs]), vmin=-2, vmax=2, cmap=cm, marker='D', edgecolor='k', s=30, label='non-resonant', zorder=0)	
				fig.colorbar(im, ax=ax, label=r'$\log_{10} K$')

				ax.set_xlim(np.nanmin(new_xvals), np.nanmax(new_xvals))
				ax.set_ylim(np.nanmin(new_yvals), np.nanmax(new_yvals))
				#### do 20 different iterations of the polynomial fit -- leaving out some of the data points at random

				if fit_line_to_scatter == 'y':
					for i in np.arange(0,20,1):
						random_idxs = np.random.randint(low=0,high=len(new_xvals), size=int(0.9 * len(new_xvals)))
						deg1_polycoeffs = np.polyfit(new_xvals[random_idxs], new_yvals[random_idxs], deg=1)
						deg1_polyfunc = np.poly1d(deg1_polycoeffs)
						poly_xvals = np.linspace(np.nanmin(new_xvals), np.nanmax(new_xvals), 100)
						deg1_poly_yvals = deg1_polyfunc(np.array(poly_xvals))
						if i == 0:
							ax.plot(poly_xvals, deg1_poly_yvals, color='k', linestyle='--', alpha=0.3, zorder=1)
							poly_yvals_stack = deg1_poly_yvals
						else:
							ax.plot(poly_xvals, deg1_poly_yvals, color='k', linestyle='--', alpha=0.3, zorder=1)
							poly_yvals_stack = np.vstack((poly_yvals_stack, deg1_poly_yvals))

					#plt.plot(poly_xvals, deg2_poly_yvals, color='green', label='quadratic')
					plt.plot(poly_xvals, np.nanmedian(poly_yvals_stack, axis=0), color='red', linestyle='--', label='median fit', linewidth=3, alpha=1, zorder=2)
				plt.legend()
				plt.savefig(plotdir+'/'+str(yval_label)+'_vs_'+str(xval_label)+'.png', dpi=300)
				if show_final_plots == 'y':
					plt.show()
				plt.close()


			except:
				plt.close()
				traceback.print_exc()
				continue_query = input('An Exception was raised. Do you want to continue? y/n: ')
				if continue_query != 'y':
					raise Exception('You opted not to continue.')





#-------------------------------

show_complete_incomplete = input('Do you want to stack complete and incomplete histograms? y/n: ')


complete_sims, complete_sim_idxs = [], []
incomplete_sims, incomplete_sim_idxs = [], []
for ngtsim, gtsim in enumerate(ground_truth_dict['sim']):
	if gtsim in pmad['sim']:
		complete_sims.append(gtsim)
		complete_sim_idxs.append(ngtsim)
	else:
		incomplete_sims.append(gtsim)
		incomplete_sim_idxs.append(ngtsim)


if use_final_sim_list != 'y':
	for gtlabel, gtname in zip(planet_ground_truth_labels, planet_ground_truth_names):
		try:
			gtvalues = ground_truth_dict[gtname]
			if gtname == 'm':
				### convert to earth mass
				gtvalues = gtvalues / M_earth.value 
				gtlabel = gtlabel+r' $[M_{\oplus}]$'
			elif gtname == 'Rp':
				gtvalues = gtvalues / R_earth.value 
				gtlabel = gtlabel+r' $[R_{\oplus}]$'
			elif gtname == 'Rstar':
				gtvalues = gtvalues / R_sun.value 
				gtlabel = gtlabel+r' $[R_{\odot}]$'
			elif gtname == 'Pp':
				gtlabel = gtlabel+' [days]'
			elif gtname == 'rhostar':
				gtlabel = gtlabel+r' [kg / m$^3$]'
			elif gtname == 'a':
				gtvalues = gtvalues / au.value 
				gtlabel = gtlabel+r' [AU]'
			elif gtname == 'rhoplan':
				gtlabel = gtlabel+r' [kg / m$^3$]'
			fig, ax = plt.subplots(figsize=(6,6))
			if np.nanmax(gtvalues) - np.nanmin(gtvalues) > 100:
				ax.set_xscale('log')
				histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
			else:
				histbins = np.linspace(np.nanmin(gtvalues), np.nanmax(gtvalues), 30)
			complete_gtvals = np.array(gtvalues)[complete_sim_idxs]
			incomplete_gtvals = np.array(gtvalues)[incomplete_sim_idxs]
			if show_complete_incomplete == 'y':
				gtvalue_arrays = (complete_gtvals, incomplete_gtvals)		
				complete_incomplete_colors = ('#f4b31a', '#143d59')
				n, bins, edges = ax.hist(gtvalue_arrays, color=complete_incomplete_colors, edgecolor='k', bins=histbins, label=('complete', 'incomplete'), stacked=True)
				plt.legend()
			else:
				gtvalue_arrays = np.concatenate((complete_gtvals, incomplete_gtvals))
				n, bins, edges = ax.hist(gtvalue_arrays, facecolor=model_colors[2], edgecolor='k', bins=histbins)
			ax.set_xlabel(gtlabel, fontsize=20)
			if show_complete_incomplete == 'y':
				plt.savefig(plotdir+'/'+gtname+'_COMPLETE_INCOMPLETE_sample_histogram.png', dpi=300)
			else:
				plt.savefig(plotdir+'/'+gtname+'_sample_histogram.png', dpi=300)
			if show_final_plots == 'y':
				plt.show()
			plt.close()
		except:
			traceback.print_exc()
			plt.close()
			continue


#-------------------------------


#### now do it for moons:
if use_final_sim_list != 'y':
	for gtlabel, gtname in zip(moon_ground_truth_labels, moon_ground_truth_names):
		try:
			for nmoon, moon_option in enumerate(moon_options):
				actual_gtname = moon_option+'_'+gtname
				new_gtlabel = 'Sat '+moon_option+' '+gtlabel  
				gtvalues = ground_truth_dict[actual_gtname]
				if gtname == 'm':
					### convert to earth mass
					gtvalues = gtvalues / M_earth.value 
					new_gtlabel = new_gtlabel+r' $[M_{\oplus}]$'
				elif gtname == 'Rp':
					gtvalues = gtvalues / R_earth.value 
					new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
				elif gtname == 'Rstar':
					gtvalues = gtvalues / R_sun.value 
					new_gtlabel = new_gtlabel+r' $[R_{\odot}]$'
				elif gtname == 'Pp':
					new_gtlabel = new_gtlabel+' [days]'
				elif gtname == 'rhostar':
					new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
				elif gtname == 'a':
					gtvalues = np.array(gtvalues) / np.array(ground_truth_dict['Rp'])
					new_gtlabel = new_gtlabel+r' $[R_P]$'
				elif gtname == 'rhoplan':
					new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
				fig, ax = plt.subplots(figsize=(6,6))
				if np.nanmax(gtvalues) - np.nanmin(gtvalues) > 100:
					ax.set_xscale('log')
					histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
				elif gtname == 'm':
					ax.set_xscale('log')
					histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
				else:
					histbins = np.linspace(np.nanmin(gtvalues), np.nanmax(gtvalues), 30)
				n, bins, edges = ax.hist(gtvalues, facecolor=model_colors[2], edgecolor='k', bins=histbins)
				ax.set_xlabel(new_gtlabel, fontsize=20)
				plt.savefig(plotdir+'/sat'+moon_option+'_'+gtname+'_sample_histogram.png', dpi=300)
				if show_final_plots == 'y':
					plt.show()
				plt.close()
		except:
			traceback.print_exc()
			plt.close()
			continue


#-------------------------------

if use_final_sim_list != 'y':
	#### now do it for moons:
	for gtlabel, gtname in zip(moon_ground_truth_labels, moon_ground_truth_names):
		try:
			print("gtlabel, gtname = ", gtlabel, gtname)
			for moon_option in moon_options:
				actual_gtname = moon_option+'_'+gtname
				new_gtlabel = 'All sat '+gtlabel  
				if moon_option == 'I':
					gtvalues = ground_truth_dict[actual_gtname]
				else:
					gtvalues = np.concatenate((gtvalues, ground_truth_dict[actual_gtname]))
			if gtname == 'm':
				### convert to earth mass
				gtvalues = gtvalues / M_earth.value 
				new_gtlabel = new_gtlabel+r' $[M_{\oplus}]$'
			elif gtname == 'r':
				gtvalues = gtvalues / R_earth.value 
				new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
			elif gtname == 'Rp':
				gtvalues = gtvalues / R_earth.value 
				new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
			elif gtname == 'Rs':
				gtvalues = gtvalues / R_earth.value 
				new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
			elif gtname == 'Rstar':
				gtvalues = gtvalues / R_sun.value 
				new_gtlabel = new_gtlabel+r' $[R_{\odot}]$'
			elif gtname == 'Pp':
				new_gtlabel = new_gtlabel+' [days]'
			elif gtname == 'P':
				new_gtlabel = new_gtlabel+' [days]'
				gtvalues = gtvalues / (24 * 60 * 60) ### convert seconds into days 
			elif gtname == 'rhostar':
				new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
			elif gtname == 'a':
				gtvalues = gtvalues / ground_truth_dict['Rp']  
				new_gtlabel = new_gtlabel+r' $[R_P]$'
			elif gtname == 'rhoplan':
				new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
			else:
				if key.startswith('I '):
					SNR_values = np.array(pmad_values)
					label = 'All sat SNR'
				elif key.startswith('II ') or key.startswith('III ') or key.startswith('IV ') or key.startswith('V '):
					SNR_values = np.concatenate((SNR_values, np.array(pmad_values)))
					label = 'All sat SNR'
			fig, ax = plt.subplots(figsize=(6,6))
			if np.nanmax(gtvalues) - np.nanmin(gtvalues) > 100:
				ax.set_xscale('log')
				histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
			elif gtname == 'm':
				ax.set_xscale('log')
				histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
			elif (gtname == 'r') or (gtname == 'Rp') or (gtname == 'Rs'):
				ax.set_xscale('log')
				histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
			elif (gtname == 'P'):
				ax.set_xscale('log')
				histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
			else:
				histbins = np.linspace(np.nanmin(gtvalues), np.nanmax(gtvalues), 30)
			complete_gtvals = gtvalues[complete_sim_idxs]
			incomplete_gtvals = gtvalues[incomplete_sim_idxs]
			if show_complete_incomplete == 'y':
				gtvalue_arrays = (complete_gtvals, incomplete_gtvals)
				complete_incomplete_colors = ('#f4b31a', '#143d59')				
				n, bins, edges = ax.hist(gtvalue_arrays, color=complete_incomplete_colors, edgecolor='k', bins=histbins, label=('complete', 'incomplete'), stacked=True)
				plt.legend()
			else:
				gtvalue_arrays = np.concatenate((complete_gtvals, incomplete_gtvals))
				n, bins, edges = ax.hist(gtvalue_arrays, facecolor=model_colors[2], edgecolor='k', bins=histbins)
			ax.set_xlabel(new_gtlabel, fontsize=20)
			if show_complete_incomplete == 'y':
				plt.savefig(plotdir+'/allsat_'+gtname+'COMPLETE_INCOMPLETE_sample_histogram.png', dpi=300)
			else:
				plt.savefig(plotdir+'/allsat_'+gtname+'_sample_histogram.png', dpi=300)
			if show_final_plots == 'y':
				plt.show()
			plt.close()
		except:
			traceback.print_exc()
			plt.close()
			continue






#### now do it for moons:
"""
for gtlabel, gtname in zip(moon_ground_truth_labels, moon_ground_truth_names):
	try:
		print("gtlabel, gtname = ", gtlabel, gtname)
		for moon_option in moon_options:
			actual_gtname = moon_option+'_'+gtname
			new_gtlabel = 'All sat '+gtlabel  
			if moon_option == 'I':
				gtvalues = ground_truth_dict[actual_gtname]
			else:
				gtvalues = np.concatenate((gtvalues, ground_truth_dict[actual_gtname]))
		if gtname == 'm':
			### convert to earth mass
			gtvalues = gtvalues / M_earth.value 
			new_gtlabel = new_gtlabel+r' $[M_{\oplus}]$'
		elif gtname == 'r':
			gtvalues = gtvalues / R_earth.value 
			new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
		elif gtname == 'Rp':
			gtvalues = gtvalues / R_earth.value 
			new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
		elif gtname == 'Rs':
			gtvalues = gtvalues / R_earth.value 
			new_gtlabel = new_gtlabel+r' $[R_{\oplus}]$'
		elif gtname == 'Rstar':
			gtvalues = gtvalues / R_sun.value 
			new_gtlabel = new_gtlabel+r' $[R_{\odot}]$'
		elif gtname == 'Pp':
			new_gtlabel = new_gtlabel+' [days]'
		elif gtname == 'P':
			new_gtlabel = new_gtlabel+' [days]'
			gtvalues = gtvalues / (24 * 60 * 60) ### convert seconds into days 
		elif gtname == 'rhostar':
			new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
		elif gtname == 'a':
			gtvalues = gtvalues / ground_truth_dict['Rp']  
			new_gtlabel = new_gtlabel+r' $[R_P]$'
		elif gtname == 'rhoplan':
			new_gtlabel = new_gtlabel+r' [kg / m$^3$]'
		fig, ax = plt.subplots(figsize=(6,6))
		if np.nanmax(gtvalues) - np.nanmin(gtvalues) > 100:
			ax.set_xscale('log')
			histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
		elif gtname == 'm':
			ax.set_xscale('log')
			histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
		elif (gtname == 'r') or (gtname == 'Rp') or (gtname == 'Rs'):
			ax.set_xscale('log')
			histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
		elif (gtname == 'P'):
			ax.set_xscale('log')
			histbins = np.logspace(np.log10(np.nanmin(gtvalues)), np.log10(np.nanmax(gtvalues)), 30)
		else:
			histbins = np.linspace(np.nanmin(gtvalues), np.nanmax(gtvalues), 30)
		complete_gtvals = gtvalues[complete_sim_idxs]
		incomplete_gtvals = gtvalues[incomplete_sim_idxs]
		gtvalue_arrays = (complete_gtvals, incomplete_gtvals)
		complete_incomplete_colors = ('#f4b31a', '#143d59')
		n, bins, edges = ax.hist(gtvalue_arrays, color=complete_incomplete_colors, edgecolor='k', stacked=True, bins=histbins, label=('complete', 'incomplete'))
		plt.legend()
		ax.set_xlabel(new_gtlabel, fontsize=20)
		plt.savefig(plotdir+'/allsat_'+gtname+'_COMPLETE_INCOMPLETE_sample_histogram.png', dpi=300)
		if show_final_plots == 'y':
			plt.show()
		plt.close()
	except:
		traceback.print_exc()
		plt.close()
		continue
"""











#-------------------------------



if use_final_sim_list != 'y':
	for key, label in zip(pmad.keys(), pmad_labels):
		try:
			if type(pmad[key]) == list:
				pmad_values = np.array(pmad[key])[idxs_to_use]
				print(key, ' : ', pmad_values)
				pmad_values = pmad_values[np.isfinite(pmad_values)]
				if key.startswith('I '):
					SNR_values = np.array(pmad_values)
					label = 'All sat SNR'
				elif key.startswith('II ') or key.startswith('III ') or key.startswith('IV ') or key.startswith('V '):
					SNR_values = np.concatenate((SNR_values, np.array(pmad_values)))
					label = 'All sat SNR'
				else:
					#### make the histogram
					fig, ax = plt.subplots(figsize=(6,6))
					if np.nanmax(pmad_values) - np.nanmin(pmad_values) > 100:
						ax.set_xscale('log')
						histbins = np.logspace(np.log10(np.nanmin(pmad_values)), np.log10(np.nanmax(pmad_values)), 40)
					elif gtname == 'm':
						ax.set_xscale('log')
						histbins = np.logspace(np.log10(np.nanmin(pmad_values)), np.log10(np.nanmax(pmad_values)), 40)
					else:
						histbins = np.linspace(np.nanmin(pmad_values), np.nanmax(pmad_values), 50)
					n, bins, edges = ax.hist(pmad_values, facecolor=model_colors[2], edgecolor='k', bins=histbins)
					ax.set_xlabel(label, fontsize=20)
					plt.savefig(plotdir+'/allsat_'+label+'_sample_histogram.png', dpi=300)
					if show_final_plots == 'y':
						plt.show()
					plt.close()
		except:
			plt.close()
			traceback.print_exc()
			print(' ')
			print(' ')
			continue 


if use_final_sim_list != 'y':
	#### make the histogram
	SNR_values = np.array(SNR_values)
	SNR_values = SNR_values[np.isfinite(SNR_values)]
	fig, ax = plt.subplots(figsize=(6,6))
	if np.nanmax(SNR_values) - np.nanmin(SNR_values) > 100:
		ax.set_xscale('log')
		histbins = np.logspace(np.log10(np.nanmin(SNR_values)), np.log10(np.nanmax(SNR_values)), 40)
	else:
		histbins = np.linspace(np.nanmin(SNR_values), np.nanmax(SNR_values), 40)
	n, bins, edges = ax.hist(SNR_values, facecolor=model_colors[2], edgecolor='k', bins=histbins)
	ax.set_xlabel('All Sat SNR', fontsize=20)
	plt.savefig(plotdir+'/allsat_SNR_sample_histogram.png', dpi=300)
	if show_final_plots == 'y':
		plt.show()
	plt.close()








poad = planet_only_attributes_dict

if loaded_poad == False:
	with open(projectdir+'/'+run_name+'_poad.pickle', 'wb') as poad_pickle:
		pickle.dump(poad, poad_pickle, protocol=pickle.HIGHEST_PROTOCOL)



complete_SNRs = []
incomplete_SNRs = []

if use_final_sim_list != 'y':
	for key, label in zip(poad.keys(), poad.keys()):
		try:
			if type(poad[key]) == list:
				if run_name == 'combined_runs':
					poad_values = np.array(poad[key])
				else:
					poad_values = np.array(poad[key])[::2]
				print(key, ' : ', poad_values)
				#poad_values = poad_values[np.isfinite(poad_values)]
				if key.startswith('I '):
					if run_name == 'combined_runs':
						SNR_values = np.array(poad_values)
						complete_SNR_values = np.array(poad_values)[complete_sim_idxs]
						incomplete_SNR_values = np.array(poad_values)[incomplete_sim_idxs] ### don't need the [::2], because all the runs are combined.
					else:
						SNR_values = np.array(poad_values)[::2]
						complete_SNR_values = np.array(poad_values)[complete_sim_idxs][::2]
						incomplete_SNR_values = np.array(poad_values)[incomplete_sim_idxs][::2]

					label = 'All sat SNR'
				elif key.startswith('II ') or key.startswith('III ') or key.startswith('IV ') or key.startswith('V '):
					SNR_values = np.concatenate((SNR_values, np.array(poad_values)))

					if run_name == 'combined_runs':
						SNR_values = np.concatenate((SNR_values, np.array(poad_values)[::2]))
						complete_SNR_values = np.concatenate((complete_SNR_values, np.array(poad_values)[complete_sim_idxs]))
						incomplete_SNR_values = np.concatenate((incomplete_SNR_values, np.array(poad_values)[incomplete_sim_idxs]))
					else:
						SNR_values = np.concatenate((SNR_values, np.array(poad_values)[::2]))
						complete_SNR_values = np.concatenate((complete_SNR_values, np.array(poad_values)[complete_sim_idxs][::2]))
						incomplete_SNR_values = np.concatenate((incomplete_SNR_values, np.array(poad_values)[incomplete_sim_idxs][::2]))
					label = 'All sat SNR'
				elif key.startswith('ntransits'):
					label = '# of transits'
					#### make the histogram
				elif key.startswith('nmoons'):
					label = '# of moons'
				if 'SNR' not in label:
					fig, ax = plt.subplots(figsize=(6,6))
					if np.nanmax(poad_values) - np.nanmin(poad_values) > 100:
						ax.set_xscale('log')
						histbins = np.logspace(np.log10(np.nanmin(poad_values)), np.log10(np.nanmax(poad_values)), 40)
					elif gtname == 'm':
						ax.set_xscale('log')
						histbins = np.logspace(np.log10(np.nanmin(poad_values)), np.log10(np.nanmax(poad_values)), 40)
					else:
						histbins = np.linspace(np.nanmin(poad_values), np.nanmax(poad_values), 50)
						complete_gtvals = poad_values[complete_sim_idxs]
						incomplete_gtvals = poad_values[incomplete_sim_idxs]
					if show_complete_incomplete == 'y':
						gtvalue_arrays = (complete_gtvals, incomplete_gtvals)
						complete_incomplete_colors = ('#f4b31a', '#143d59')				
						n, bins, edges = ax.hist(gtvalue_arrays, color=complete_incomplete_colors, edgecolor='k', bins=histbins, label=('complete', 'incomplete'), stacked=True)
						plt.legend()
					elif show_complete_incomplete == 'n':
						gtvalue_arrays = np.concatenate((complete_gtvals, incomplete_gtvals))
						n, bins, edges = ax.hist(gtvalue_arrays, facecolor=model_colors[2], edgecolor='k', bins=histbins)
					ax.set_xlabel(label, fontsize=20)
					if show_complete_incomplete == 'y':
						plt.savefig(plotdir+'/allsat_'+label+'_POAD_COMPLETE_INCOMPLETE_sample_histogram.png', dpi=300)
					elif show_complete_incomplete == 'n':
						plt.savefig(plotdir+'/allsat_'+label+'_POAD_sample_histogram.png', dpi=300)
					if show_final_plots == 'y':
						plt.show()
					plt.close()
		except:
			plt.close()
			traceback.print_exc()
			print(' ')
			print(' ')
			continue 


if use_final_sim_list != 'y':
	#### make the histogram
	#SNR_values = np.array(SNR_values)
	#SNR_values = SNR_values[np.isfinite(SNR_values)]
	fig, ax = plt.subplots(figsize=(6,6))
	#f np.nanmax(SNR_values) - np.nanmin(SNR_values) > 100:
	ax.set_xscale('log')
	histbins = np.logspace(np.log10(np.nanmin(SNR_values)), np.log10(np.nanmax(SNR_values)), 40)
	#else:
	#histbins = np.linspace(np.nanmin(SNR_values), np.nanmax(SNR_values), 40)
	if show_complete_incomplete == 'y':
		gtvalue_arrays = (complete_SNR_values[np.isfinite(complete_SNR_values)], incomplete_SNR_values[np.isfinite(incomplete_SNR_values)])
		complete_incomplete_colors = ('#f4b31a', '#143d59')				
		n, bins, edges = ax.hist(gtvalue_arrays, color=complete_incomplete_colors, edgecolor='k', bins=histbins, label=('complete', 'incomplete'), stacked=True)
		plt.legend()
	else:
		gtvalue_arrays = np.concatenate((complete_SNR_values[np.isfinite(complete_SNR_values)], incomplete_SNR_values[np.isfinite(incomplete_SNR_values)]))
		n, bins, edges = ax.hist(gtvalue_arrays, facecolor=model_colors[2], edgecolor='k', bins=histbins)
	ax.set_xlabel('All Sat SNR', fontsize=20)
	if show_complete_incomplete == 'y':
		plt.savefig(plotdir+'/allsat_SNR_COMPLETE_INCOMPLETE_sample_histogram.png', dpi=300)
	else:
		plt.savefig(plotdir+'/allsat_SNR_sample_histogram.png', dpi=300)
	if show_final_plots == 'y':
		plt.show()
	plt.close()








#-------------------------------


#### make the histogram
if use_final_sim_list != 'y':
	SNR_values = np.array(SNR_values)
	SNR_values = SNR_values[np.isfinite(SNR_values)]
	fig, ax = plt.subplots(figsize=(6,6))
	if np.nanmax(SNR_values) - np.nanmin(SNR_values) > 100:
		ax.set_xscale('log')
		histbins = np.logspace(np.log10(np.nanmin(SNR_values)), np.log10(np.nanmax(SNR_values)), 40)
	else:
		histbins = np.linspace(np.nanmin(SNR_values), np.nanmax(SNR_values), 40)
	n, bins, edges = ax.hist(SNR_values, facecolor=model_colors[2], edgecolor='k', bins=histbins)
	ax.set_xlabel('All Sat SNR', fontsize=20)
	plt.savefig(plotdir+'/allsat_SNR_sample_histogram.png', dpi=300)
	if show_final_plots == 'y':
		plt.show()
	plt.close()



#-------------------------------

"""
>>> pmad.keys()
dict_keys(['delta_logz', 'sim', 'nmoons', 'planet_moon_logz', 'ntransits', 'res_nores', 'I SNR', 'II SNR', 'III SNR', 'IV SNR', 'V SNR', 'Planet', 'I', 'II', 'III', 'IV', 'V'])
>>> planet_moon_posterior_median_dict.keys()
dict_keys(['per_bary', 'a_bary', 'r_planet', 'b_bary', 'ecc_bary', 'w_bary', 't0_bary_offset', 'M_planet', 'r_moon', 'per_moon', 'tau_moon', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2', 'sim'])
>>> ground_truth_dict.keys()
dict_keys(['m', 'Rp', 'Rstar', 'RpRstar', 'rhostar', 'impact', 'q1', 'q2', 'a', 'Pp', 'rhoplan', 'aRp', 'e', 'inc', 'pomega', 'f', 'P', 'RHill', 'aRstar', 'sim', 'I_m', 'II_m', 'III_m', 'IV_m', 
	'I_r', 'II_r', 'III_r', 'IV_r', 'I_msmp', 'II_msmp', 'III_msmp', 'IV_msmp', 'I_rsrp', 'II_rsrp', 'III_rsrp', 'IV_rsrp', 'I_a', 'II_a', 'III_a', 'IV_a', 'I_aRp', 'II_aRp', 'III_aRp', 'IV_aRp', 
	'I_e', 'II_e', 'III_e', 'IV_e', 'I_inc', 'II_inc', 'III_inc', 'IV_inc', 'I_pomega', 'II_pomega', 'III_pomega', 'IV_pomega', 'I_f', 'II_f', 'III_f', 'IV_f', 'I_P', 'II_P', 'III_P', 'IV_P', 
	'I_RHill', 'II_RHill', 'III_RHill', 'IV_RHill', 'I_spacing', 'II_spacing', 'III_spacing', 'IV_spacing', 'I_period_ratio', 'II_period_ratio', 'III_period_ratio', 'IV_period_ratio', 'V_m', 
	'V_r', 'V_msmp', 'V_rsrp', 'V_a', 'V_aRp', 'V_e', 'V_inc', 'V_pomega', 'V_f', 'V_P', 'V_RHill', 'V_spacing', 'V_period_ratio'])
"""

world_density_labels = np.array(['Jupiter', 'Saturn', 'Neptune', 'Earth', 'Moon', 'Titan'])
world_densities = np.array([1326.2, 687.1, 1638., 5515, 3344, 1880]) #### kg/m^3
density_colors = ['#F60000', '#FF8C00', '#FFEE00', '#4DE94C', '#3783FF', '#4815AA']
wd_argsort = np.argsort(world_densities)
world_density_labels = world_density_labels[wd_argsort]
world_densities = world_densities[wd_argsort]

posterior_rmoon_rearths = []
posterior_mmoon_mearths = []
posterior_moon_densities = []
for nsim, sim in enumerate(np.array(pmad['sim'])[idxs_to_use]):
	if np.array(pmad['bayes_factor'])[idxs_to_use][nsim] < 3.2:
		continue 
	posterior_dict_idx = int(np.where(np.array(planet_moon_posterior_median_dict['sim']) == sim)[0])
	ground_truth_dict_idx = int(np.where(np.array(ground_truth_dict['sim']) == sim)[0])
	print('posterior_dict_idx: ', posterior_dict_idx)
	print('ground_truth_dict_idx: ', ground_truth_dict_idx)
	posterior_rmoon_rstar = planet_moon_posterior_median_dict['r_moon'][posterior_dict_idx]
	ground_truth_rstar_meters = ground_truth_dict['Rstar'][ground_truth_dict_idx]
	posterior_rmoon_meters = posterior_rmoon_rstar * ground_truth_rstar_meters 
	posterior_rmoon_rearth = posterior_rmoon_meters / R_earth.value
	posterior_rmoon_rearths.append(posterior_rmoon_rearth)
	posterior_mmoon_kg = planet_moon_posterior_median_dict['M_moon'][posterior_dict_idx] ### natively kg 
	posterior_mmoon_mearth = posterior_mmoon_kg / M_earth.value 
	posterior_mmoon_mearths.append(posterior_mmoon_mearth)
	posterior_moon_density_kgpm3 = posterior_mmoon_kg / ((4/3) * np.pi * posterior_rmoon_meters**3) ### kg / m^3 
	posterior_moon_densities.append(posterior_moon_density_kgpm3)
posterior_rmoon_rearths = np.array(posterior_rmoon_rearths)
posterior_mmoon_mearths = np.array(posterior_mmoon_mearths)
posterior_moon_densities = np.array(posterior_moon_densities)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6,8))
plt.subplots_adjust(left=0.125, bottom=0.11, right=0.9, top=0.979, wspace=0.426, hspace=0.355)
#if np.nanmax(posterior_rmoon_rearths) - np.nanmin(posterior_rmoon_rearths) > 100:
ax[0].set_xscale('log')
histbins0 = np.logspace(np.log10(np.nanmin(posterior_rmoon_rearths)), np.log10(np.nanmax(posterior_rmoon_rearths)), 40)
#else:
#histbins0 = np.linspace(np.nanmin(posterior_rmoon_rearths), np.nanmax(posterior_rmoon_rearths), 40)
n0, bins0, edges0 = ax[0].hist(posterior_rmoon_rearths, facecolor=model_colors[1], edgecolor='k', bins=histbins0)
ax[0].set_xlabel(r'$R_S [R_{\oplus}]$')
#
#ax[0].hist(posterior_rmoon_rearths, facecolor=model_colors[1], edgecolor='k', bins=histbins0)
#if np.nanmax(posterior_mmoon_mearths) - np.nanmin(posterior_mmoon_mearths) > 100:
ax[1].set_xscale('log')
histbins1 = np.logspace(np.log10(np.nanmin(posterior_mmoon_mearths)), np.log10(np.nanmax(posterior_mmoon_mearths)), 40)
#else:
#	histbins1 = np.linspace(np.nanmin(posterior_mmoon_mearths), np.nanmax(posterior_mmoon_mearths), 40)
n1, bins1, edges1 = ax[1].hist(posterior_mmoon_mearths, facecolor=model_colors[1], edgecolor='k', bins=histbins1)
ax[1].set_xlabel(r'$M_S [M_{\oplus}]$')
#
if np.nanmax(posterior_moon_densities) - np.nanmin(posterior_moon_densities) > 100:
	ax[2].set_xscale('log')
	histbins2 = np.logspace(np.log10(np.nanmin(posterior_moon_densities)), np.log10(np.nanmax(posterior_moon_densities)), 40)
else:
	histbins2 = np.linspace(np.nanmin(posterior_moon_densities), np.nanmax(posterior_moon_densities), 40)
n2, bins2, edges2 = ax[2].hist(posterior_moon_densities, facecolor=model_colors[1], edgecolor='k', bins=histbins2)
wdidx = 0
for wdl,wd in zip(world_density_labels, world_densities):
	ax[2].plot(np.linspace(wd,wd,100), np.linspace(0,1.1*np.nanmax(n2),100), linestyle='--', color=density_colors[wdidx], label=wdl)
	wdidx += 1
ax[2].set_ylim(0, 1.1*np.nanmax(n2))
#ax[2].legend()
#box = ax[2].get_position()
#ax[2].set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5),
 #         ncol=1, fancybox=True, shadow=True)
ax[2].set_xlabel(r'$\rho_S$ [kg / m$^{3}]$')
plt.savefig(plotdir+'/derived_moon_densities.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()


#-------------------------------


#### PLOT THE LIMB DARKENING SOLUTIONS 
claret_path = '/Users/hal9000/Documents/Projects/TESS_CVZ_project_2023REDUX/reference_files/Claret_table5.txt'
claret = np.genfromtxt(claret_path)
claret_shape = claret.shape ### = (574, 10)
claret_columns = ['logg','Teff','Z', 'L/HP', 'a', 'b', 'mu', 'chi2', 'od', 'Sys']

### ldcs are idx=4,5
### useful params are logg and teff (idxs=0,1)

claret_loggs = claret.T[0]
claret_teffs = claret.T[1]
claret_ldc_as = claret.T[4]
claret_ldc_bs = claret.T[5]

claret_q1s, claret_q2s = ld_invert(claret_ldc_as, claret_ldc_bs)

#### plot these in 2D space with the third dimension colorcoded 
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
cm1 = plt.cm.get_cmap('Spectral')
im1 = ax[0].scatter(claret_ldc_as, claret_ldc_bs, c=claret_loggs, cmap=cm1, marker='o', edgecolor='k', s=30, zorder=0)
ax[0].set_xlabel('LDC a')
ax[0].set_ylabel('LDC b')
fig.colorbar(im1, ax=ax[0], label=r'$\log g$')

cm2 = plt.cm.get_cmap('bwr')
im2 = ax[1].scatter(claret_ldc_as, claret_ldc_bs, c=claret_teffs, cmap=cm2, marker='o', edgecolor='k', s=30, zorder=0)	
ax[1].set_xlabel("LDC a")
ax[1].set_ylabel("LDC b")
fig.colorbar(im2, ax=ax[1], label=r'$T_{\mathrm{eff}}$')
plt.subplots_adjust(left=0.164, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
plt.savefig(plotdir+'/Claret_quadratic_limb_darkening_a_and_b.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()


#-------------------------------


### LDC experiment
ldc_model_dict = {}
ldc_model_dict['R_star'] = R_sun.value
ldc_model_dict['per_bary'] = 365.25
ldc_model_dict['a_bary'] = au.value / R_sun.value 
ldc_model_dict['r_planet'] = R_jup.value / R_sun.value 
ldc_model_dict['b_bary'] = 0.
ldc_model_dict['t0_bary'] = 100
ldc_model_dict['t0_bary_offset'] = 0.
ldc_model_dict['w_bary'] = 0.
ldc_model_dict['ecc_bary'] = 0.
ldc_model_dict['M_planet'] = M_jup.value 
ldc_model_dict['r_moon'] = 1e-8
ldc_model_dict['per_moon'] = 30.
ldc_model_dict['tau_moon'] = 0.
ldc_model_dict['Omega_moon'] = 0.
ldc_model_dict['i_moon'] = 0.
ldc_model_dict['e_moon'] = 0.
ldc_model_dict['w_moon'] = 0.
ldc_model_dict['M_moon'] = 1e-8 

ldc_experiment_q1s = np.linspace(np.nanmin(claret_q1s),np.nanmax(claret_q1s),2)
ldc_experiment_q2s = np.linspace(np.nanmin(claret_q2s),np.nanmax(claret_q2s),2)
ldc_experiment_times = np.arange(99.5,100.5,(1/48))

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
for nldc1, ldc1 in enumerate(ldc_experiment_q1s):
	for nldc2, ldc2 in enumerate(ldc_experiment_q2s):
		ldc_model_dict['q1'] = ldc1
		ldc_model_dict['q2'] = ldc2 
		ldc_experiment_times, ldc_experiment_fluxes = gen_model(model_dict=ldc_model_dict, times=ldc_experiment_times, t0_bary=100, model_type='planet_only')
		ax[nldc1][nldc2].plot(ldc_experiment_times, ldc_experiment_fluxes, label=r'$q_1 = $'+str(round(ldc1,2))+', '+r'$q_2 = $'+str(round(ldc2,2)))
		#ax[nldc1][nldc2].legend()
		ax[nldc1][nldc2].set_title(r'$q_1 = $'+str(round(ldc1,2))+', '+r'$q_2 = $'+str(round(ldc2,2)))
		#ax[nldc1][nldc2].set_xlabel('q1')
		#ax[nldc1][nldc2].set_ylabel('q2')
		#ax[nldc1][nldc2].set_title(r'$q_1 = $'+str(ldc1)+r', $q_2$ = '+str(ldc2))
#plt.legend()
plt.show()


lc_colors = ['#4477AA', '#228833', '#EE6677', '#BBBBBB']
#### do the same thing, just overplotting
fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(6,8))
color_number = 0
for nldc1, ldc1 in enumerate(ldc_experiment_q1s):
	for nldc2, ldc2 in enumerate(ldc_experiment_q2s):
		ldc_model_dict['q1'] = ldc1
		ldc_model_dict['q2'] = ldc2 
		ldc_model_dict['b_bary'] = 0.
		ldc_experiment_times, ldc_experiment_fluxes = gen_model(model_dict=ldc_model_dict, times=ldc_experiment_times, t0_bary=100, model_type='planet_only')
		#ax[0].plot(ldc_experiment_times, ldc_experiment_fluxes, label=r'$(q_1,q_2) = ($'+str(round(ldc1,2))+','+str(round(ldc2,2))+')', linestyle='--', linewidth=2, color=lc_colors[color_number])
		ax[0].plot(ldc_experiment_times, ldc_experiment_fluxes, label='('+str(round(ldc1,2))+', '+str(round(ldc2,2))+')', linestyle='--', linewidth=2, color=lc_colors[color_number])
		#ax[nldc1][nldc2].legend()
		color_number += 1

color_number = 0
for nldc1, ldc1 in enumerate(ldc_experiment_q1s):
	for nldc2, ldc2 in enumerate(ldc_experiment_q2s):
		ldc_model_dict['q1'] = ldc1
		ldc_model_dict['q2'] = ldc2 
		ldc_model_dict['b_bary'] = 0.9
		ldc_experiment_times, ldc_experiment_fluxes = gen_model(model_dict=ldc_model_dict, times=ldc_experiment_times, t0_bary=100, model_type='planet_only')
		ax[1].plot(ldc_experiment_times, ldc_experiment_fluxes, label='('+str(round(ldc1,2))+', '+str(round(ldc2,2))+')', linestyle='--', linewidth=2, color=lc_colors[color_number])
		#ax[nldc1][nldc2].legend()
		color_number += 1
plt.legend()
plt.show()






#-------------------------------



### compare ground truth and simulation results
gt_q1s = np.array(ground_truth_dict['q1'])
gt_q2s = np.array(ground_truth_dict['q2'])
gt_sims = np.array(ground_truth_dict['sim'])

PM_posterior_q1s = planet_moon_posterior_median_dict['q1']
PM_posterior_q2s = planet_moon_posterior_median_dict['q2']
PM_posterior_sims = planet_moon_posterior_median_dict['sim']

#pmad_q1_minus_gt_q1_list = []
#pmad_q2_minus_gt_q2_list = []

#for ngtsim, gtsim in enumerate(gt_sims):
post_q1s, post_q2s, final_gt_q1s, final_gt_q2s, delta_q1s, delta_q2s = [], [], [], [], [], []
for npostsim, postsim in enumerate(PM_posterior_sims):
	this_post_q1 = PM_posterior_q1s[npostsim]
	this_post_q2 = PM_posterior_q2s[npostsim]
	try:
		gt_idx = np.where(postsim == gt_sims)[0][0]
	except:
		continue
	gt_q1 = gt_q1s[gt_idx]
	gt_q2 = gt_q2s[gt_idx]
	delta_q1 = this_post_q1 - gt_q1
	delta_q2 = this_post_q2 - gt_q2
	final_gt_q1s.append(gt_q1)
	final_gt_q2s.append(gt_q2)
	post_q1s.append(this_post_q1)
	post_q2s.append(this_post_q2)
	delta_q1s.append(delta_q1)
	delta_q2s.append(delta_q2)
final_gt_q1s = np.array(final_gt_q1s)
final_gt_q2s = np.array(final_gt_q2s)
post_q1s = np.array(post_q1s)
post_q2s = np.array(post_q2s)
delta_q1s = np.array(delta_q1s)
delta_q2s = np.array(delta_q2s)

	#### if pmad is increasing from gt, this should be positive. Start with gt, and use this as your deltax. 

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))
cm1 = plt.cm.get_cmap('Spectral')
im1 = ax[0].scatter(claret_q1s, claret_q2s, c=claret_loggs, cmap=cm1, marker='o', s=30, zorder=0)
for gtq1, gtq2, deltaq1, deltaq2 in zip(final_gt_q1s, final_gt_q2s, delta_q1s, delta_q2s):
	ax[0].arrow(x=gtq1, y=gtq2, dx=deltaq1, dy=deltaq2, color='k', alpha=0.5, head_width=0.01, head_length=0.005, zorder=1)
ax[0].set_xlabel(r'$q_1$')
ax[0].set_ylabel(r'$q_2$')
fig.colorbar(im1, ax=ax[0], label=r'$\log g$')
cm2 = plt.cm.get_cmap('RdYlBu')
im2 = ax[1].scatter(claret_q1s, claret_q2s, c=claret_teffs, cmap=cm2, marker='o', s=30, zorder=0)	
for gtq1, gtq2, deltaq1, deltaq2 in zip(final_gt_q1s, final_gt_q2s, delta_q1s, delta_q2s):
	ax[1].arrow(x=gtq1, y=gtq2, dx=deltaq1, dy=deltaq2, color='k', alpha=0.5, head_width=0.01, head_length=0.005, zorder=1)
ax[1].set_xlabel(r'$q_1$')
ax[1].set_ylabel(r'$q_2$')
fig.colorbar(im2, ax=ax[1], label=r'$T_{\mathrm{eff}}$ (K)')
plt.subplots_adjust(left=0.164, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
plt.savefig(plotdir+'/Claret_quadratic_limb_darkening_q1_and_q2.png', dpi=300)
if show_final_plots == 'y':
	plt.show()
plt.close()


#-------------------------------


#### these are the simulations that have K >= 3.2
good_sims = np.array(['july13_res_974', 'july13_nores_338', 'july13_res_838',
       'july13_res_848', 'july13_res_2', 'july13_nores_268',
       'july13_nores_497', 'july13_nores_832', 'july13_res_247',
       'july13_nores_376', 'july13_res_555', 'july13_nores_741',
       'july13_res_52', 'july13_res_39', 'july13_nores_749',
       'july13_nores_771', 'july13_res_544', 'july13_res_74',
       'july13_nores_68', 'july13_res_20', 'july13_res_542',
       'july13_res_292', 'july13_res_659', 'july13_res_603',
       'july13_nores_678', 'july13_nores_201', 'july13_res_289',
       'july13_nores_609', 'july13_res_424', 'july13_res_629',
       'july13_res_243', 'july13_nores_253', 'july13_res_226',
       'july13_nores_898', 'july13_res_645', 'july13_nores_787',
       'july13_nores_917', 'july13_res_701', 'july13_nores_789',
       'july13_nores_716', 'july13_nores_42', 'july13_res_731',
       'july13_nores_146'], dtype='<U16')


