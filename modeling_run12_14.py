import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [6, 4]
import numpy as np

import pandoramoon as pandora
from pandoramoon.helpers import ld_convert, ld_invert
import ultranest
import ultranest.stepsampler
from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot
from scipy.stats import norm,loguniform,beta,truncnorm

import pickle
import math
import random
import os
import csv
import sys
import json
import pandas
import socket 


#### NEED TO CHANGE THIS FOR EACH RUN! (ALSO IT SHOULD HAVE BEEN 5_05 because it's May, not April!!! Oh well.)
run_date = '12_14'
lcgen_date = 'may16_2023_'

if run_date not in __file__: ### test that run_date matches the modeling_run name
	#### change the run date!
	run_date_start_idx = __file__.find('run')+3
	run_date_end_idx = __file__.find('.py')
	run_date = __file__[run_date_start_idx:run_date_end_idx]


local_system = socket.gethostname()

if (local_system == 'xl') or (local_system == 'tiara'):
	garvitdir = '/tiara/home/ateachey/multimoon_modeling'
	mmdir = '/tiara/home/ateachey/multimoon_modeling'
	model_run_savedir = '/tiara/home/ateachey/data/multimoon_modeling_results/run'+run_date

elif local_system == 'Alexs-MacBook-Pro.local':
	garvitdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'
	mmdir = '/Users/hal9000/Documents/Projects/multimoon_modeling'
	model_run_savedir = mmdir+'/run'+run_date

else:
	#### cover your bases
	garvitdir = '/tiara/home/ateachey/multimoon_modeling'
	mmdir = '/tiara/home/ateachey/multimoon_modeling'
	model_run_savedir = '/tiara/home/ateachey/data/multimoon_modeling_results/run'+run_date


sim_directories = ['13july2022_nores_stable_j2_tides_sim_model_settings', '13July2022_RES_stable_j2_tides_sim_model_settings']



def transform_uniform(x,lower,valrange):
	return lower + (valrange)*x

def transform_normal(x,mu,sigma):
	return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,lower,upper):
	return beta.ppf(x,lower,upper)

def transform_truncated_normal(x,mu,sigma,lower=0.,upper=1.):
	ar, br = (lower - mu) / sigma, (upper - mu) / sigma
	return truncnorm.ppf(x,ar,br,loc=mu,scale=sigma)




def prior_transform_planet_only(cube):
	p = np.empty_like(cube)

	#### period -- TRUNCNORMAL 
	period_mu, period_sigma = per_bary + (per_bary * perbary_shift), 0.01 * per_bary 
	#p[0] = truncnorm.ppf(q=cube[0], a=per_bary-1, b=per_bary+1, loc=per_bary, scale=period_sigma) ### truncnorm period 
	p[0] = norm.ppf(q=cube[0], loc=period_mu, scale=period_sigma)
	
	#### semi-major axis -- TRUNCNORMAL 
	a_bary_mu, a_bary_sigma = a_bary + (a_bary*abary_shift), 0.1 * a_bary
	#p[1] = truncnorm.ppf(q=cube[1], a=2, b=2*a_bary, loc=a_bary, scale=a_bary_sigma) #### truncnorm a_bary
	p[1] = norm.ppf(q=cube[1], loc=a_bary_mu, scale=a_bary_sigma)
	
	#### PLANET RADIUS -- TRUNCNORMAL
	r_planet_mu, r_planet_sigma = r_planet + (r_planet * rp_shift), 0.1 * r_planet
	#p[2] = truncnorm.ppf(q=cube[2], a=0., b=2*r_planet, loc=r_planet_mu, scale=r_planet_sigma) ### truncnorm r_planet 
	p[2] = norm.ppf(q=cube[2], loc=r_planet_mu, scale=r_planet_sigma)

	### impact parameter -- uniform 
	b_lower, b_upper = 0, 1
	b_range = b_upper - b_lower 
	p[3] = b_lower + b_range * cube[3] ### uniform b_bary 0-1

	#### eccentricity -- uniform 
	ecc_lower, ecc_upper = 0., 0.99
	ecc_range = ecc_upper - ecc_lower 
	p[4] = ecc_lower + ecc_range * cube[4] #### uniform ecc_bary 0-1

	#omega -- uniform 
	w_lower, w_upper = 0., 360.,
	w_range = w_upper - w_lower
	p[5] = w_lower + w_range * cube[5] #### uniform w_bary 0-360

	#### offset -- uniform 
	##### run8_10 VERSION
	"""
	offset_lower, offset_upper = -1., 1.
	offset_range = offset_upper - offset_lower
	p[6] = offset_lower + offset_range * cube[6] ### uniform t0_bary_offset 
	"""

	#### IDENTICAL TO THE PLANET+MOON VERSION BELOW 
	#### t0 offset -- gaussian around 0 offset.
	offset_lower, offset_upper = -0.1, 0.1
	offset_range = offset_upper - offset_lower
	#p[6] = offset_lower + offset_range * cube[6] ### uniform t0_bary_offset 
	p[6] = norm.ppf(q=cube[6], loc=0, scale=offset_upper)


	#### limb darkening coefficients -- uniform 
	q_lower, q_upper = 0., 1.
	q_range = q_upper - q_lower
	p[7] = q_lower + q_range * cube[7] #### uniform q1 0-1
	p[8] = q_lower + q_range * cube[8]#### uniform q2 0-1

	return p 


def prior_transform_planet_moon(cube):
	p = np.empty_like(cube)

	### PERIOD - TRUNCNORMAL
	period_mu, period_sigma = per_bary + (per_bary * perbary_shift), 0.01 * per_bary 
	#p[0] = truncnorm.ppf(q=cube[0], a=per_bary-1, b=per_bary+1, loc=period_mu, scale=period_sigma) ### truncnorm period 
	p[0] = norm.ppf(q=cube[0], loc=period_mu, scale=period_sigma)

	#### semi-major axis - TRUNCNORMAL
	a_bary_mu, a_bary_sigma = a_bary + (a_bary*abary_shift), 0.1 * a_bary
	#p[1] = truncnorm.ppf(q=cube[1], a=2, b=2*a_bary, loc=a_bary_mu, scale=a_bary_sigma) #### truncnorm a_bary
	p[1] = norm.ppf(q=cube[1], loc=a_bary_mu, scale=a_bary_sigma)
	
	#### r_planet - TRUNCNORMAL 
	r_planet_mu, r_planet_sigma = r_planet + (r_planet * rp_shift), 0.1 * r_planet
	#p[2] = truncnorm.ppf(q=cube[2], a=0., b=2*r_planet, loc=r_planet_mu, scale=r_planet_sigma) ### truncnorm r_planet 
	p[2] = norm.ppf(q=cube[2], loc=r_planet_mu, scale=r_planet_sigma)

	
	#### impact parameter -- UNIFORM 
	b_lower, b_upper = 0, 1
	b_range = b_upper - b_lower 
	p[3] = b_lower + b_range * cube[3] ### uniform b_bary 0-1


	#### eccentricity -- UNIFORM 
	ecc_lower, ecc_upper = 0., 0.99
	ecc_range = ecc_upper - ecc_lower 
	p[4] = ecc_lower + ecc_range * cube[4] #### uniform ecc_bary 0-1


	#### omega -- UNIFORM 
	w_lower, w_upper = 0., 360.,
	w_range = w_upper - w_lower
	p[5] = w_lower + w_range * cube[5] #### uniform w_bary 0-360


	#### t0 offset -- gaussian around 0 offset.
	offset_lower, offset_upper = -0.1, 0.1
	offset_range = offset_upper - offset_lower
	#p[6] = offset_lower + offset_range * cube[6] ### uniform t0_bary_offset 
	p[6] = norm.ppf(q=cube[6], loc=0, scale=offset_upper)


	#### planet mass -- NORMAL
	m_planet_mu, m_planet_sigma = M_planet + (M_planet * mplanet_shift), 0.05 * M_planet
	m_planet_lower, m_planet_upper = 0.5*M_planet, 2.*M_planet 
	#p[7] = truncnorm.ppf(q=cube[7], a=m_planet_lower, b=m_planet_upper, loc=m_planet_mu, scale=m_planet_sigma)
	p[7] = norm.ppf(q=cube[7], loc=m_planet_mu, scale=m_planet_sigma)

	#### moon radius -- LOG UNIFORM 
	log10_rmoon_lowerlim = np.log10(r_planet*1e-4) #### much smaller than (R_Io) / (R_Jup)
	log10_rmoon_upperlim = np.log10(0.9*r_planet)
	log10_rmoon_range = log10_rmoon_upperlim - log10_rmoon_lowerlim 
	p[8] = 10**(log10_rmoon_lowerlim + log10_rmoon_range * cube[8]) ### see here: https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81 

	#### moon period -- LOG UNIFORM 
	per_hill = (((r_hill)**3 * 4 * np.pi**2)/(6.674e-11 * M_planet))**.5
	per_hill /= (3600*24)
	per_lowerlim = ((( 2*systems_details['Planet']['Rp'])**3 * 4 * np.pi**2)/(6.674e-11 * M_planet))**.5
	per_lowerlim /= (3600*24)
	log10_per_lowerlim = np.log10(np.nanmin((1, per_lowerlim))) ### days
	log10_per_upperlim = np.log10(np.nanmin((per_hill, 100))) ### not to exceed 100 days
	log10_per_range = log10_per_upperlim - log10_per_lowerlim 
	p[9] = 10**(log10_per_lowerlim + log10_per_range * cube[9]) ### LOGUNIFORM -- https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81

	#### tau -- UNIFORM
	tau_lower, tau_upper = 0., 1.
	tau_range = tau_upper - tau_lower 
	p[10] = tau_lower + tau_range * cube[10] ### UNIFORM https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81 

	#### Omega -- UNIFORM 
	Omega_lower, Omega_upper = 0., 360.
	Omega_range = Omega_upper - Omega_lower  
	p[11] = Omega_lower + Omega_range * cube[11]  #### UNIFORM https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81 
	

	#### INCLINATION -- TRUNCNORMAL
	inclination_lower, inclination_upper = 0., 180.
	inclination_mu, inclination_sigma = 90., 5.,
	#p[12] = truncnorm.ppf(q=cube[12], a=inclination_lower, b=inclination_upper, loc=inclination_mu, scale=inclination_sigma) ### TRUNCNORM https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81 
	p[12] = norm.ppf(q=cube[12], loc=inclination_mu, scale=inclination_sigma)

	#### moon mass -- log uniform 
	#### minimum mass ratio is where m2/m1 = [(2pi * sma_plan) / (RHill_sat)] * (1 sec / Pplan)
	minimum_mass_ratio = ((2*np.pi*a_bary_meters) / r_hill) * (1/per_bary_seconds)
	Mmoon_lowerlim = minimum_mass_ratio * M_planet 
	Mmoon_upperlim = 0.9 * M_planet 
	log10_Mmoon_lowerlim, log10_Mmoon_upperlim = np.log10(Mmoon_lowerlim), np.log10(Mmoon_upperlim)
	log10_Mmoon_range = log10_Mmoon_upperlim - log10_Mmoon_lowerlim 
	p[13] = 10**(log10_Mmoon_lowerlim + log10_Mmoon_range * cube[13]) #### loguniform, see here: https://johannesbuchner.github.io/UltraNest/priors.html?highlight=transform#:~:text=the%20prior%20distribution.-,Specifying%20priors,-%EF%83%81

	#### limb darkening coefficients -- uniform 
	q_lower, q_upper = 0., 1.
	q_range = q_upper - q_lower
	p[14] = q_lower + q_range * cube[14] #### uniform q1 0-1
	p[15] = q_lower + q_range * cube[15]#### uniform q2 0-1

	return p 




def log_likelihood_planet_only(p):
	# Convert q priors to u LDs (Kipping 2013)
	q1 = p[7]
	q2 = p[8]
	u1, u2 = ld_convert(q1, q2)

	# Calculate pandora model with trial parameters
	_, _, flux_trial_total, _, _, _, _ = pandora.pandora(
		R_star = systems_details['Planet']['Rstar'],
		u1 = u1,
		u2 = u2,

		# Planet parameters
		per_bary = p[0],
		a_bary = p[1],
		r_planet = p[2],
		b_bary = p[3],
		ecc_bary = p[4],
		w_bary = p[5],
		t0_bary = float(system_t0_bary),
		t0_bary_offset = p[6],   

		# Moon (fixed) parameters
		M_planet = systems_details['Planet']['m'], #### don't model it, fix it! 
		r_moon = 1e-8,  # negligible moon size
		per_moon = 30,  # other moon params do not matter
		tau_moon = 0,
		Omega_moon = 0,
		i_moon = 0,
		ecc_moon = 0,
		w_moon = 0,
		M_moon = 1e-8,  # negligible moon mass

		# Other model parameters
		epoch_distance = systems_details['Planet']['Pp'],
		supersampling_factor = 1,
		occult_small_threshold = 0.01,
		hill_sphere_threshold=1.1,
		numerical_grid=25,
		time=timestamps,
		#cache=cache  # Can't use cache because free LDs
	)
	loglike = -0.5 * np.nansum(((flux_trial_total - flux) / noise)**2)

	return loglike


def log_likelihood_planet_moon(p):
	# Convert q priors to u LDs (Kipping 2013)
	q1 = p[14]
	q2 = p[15]
	u1, u2 = ld_convert(q1, q2)

	# Calculate pandora model with trial parameters
	_, _, flux_trial_total, _, _, _, _ = pandora.pandora(
		R_star =  systems_details['Planet']['Rstar'], #p[0], #### changed on May 22nd, 2023 
		u1 = u1,
		u2 = u2,

		# Planet parameters
		per_bary = p[0],
		a_bary = p[1],
		r_planet = p[2],
		b_bary = p[3],
		ecc_bary = p[4],
		w_bary = p[5],
		t0_bary = float(system_t0_bary),
		t0_bary_offset = p[6],   

		# Moon fit parameters
		M_planet = p[7],
		r_moon = p[8], 
		per_moon = p[9], 
		tau_moon = p[10],
		Omega_moon = p[11],
		i_moon = p[12],
		M_moon = p[13],  

		#### Moon fixed parameters 
		ecc_moon = 0,
		w_moon = 0,

		# Other model parameters
		epoch_distance = systems_details['Planet']['Pp'],
		supersampling_factor = 1,
		occult_small_threshold = 0.01,
		hill_sphere_threshold=1.1,
		numerical_grid=25,
		time=timestamps,
		#cache=cache  # Can't use cache because free LDs
	)
	loglike = -0.5 * np.nansum(((flux_trial_total - flux) / noise)**2)

	return loglike




#index = int(sys.argv[1]) - 1
index = int(sys.argv[1]) 
modeling_type = str(sys.argv[2])
sim_name_from_qsub = sys.argv[4] ### of the form sim974_res_planet_moon 
print('sim_name_from_qsub: ', sim_name_from_qsub)
sim_number_from_qsub = sim_name_from_qsub[3:sim_name_from_qsub.find('_')]
print('sim_number_from_qsub: ', sim_number_from_qsub)

if 'no' in sim_name_from_qsub:
	res_nores_from_qsub = 'nores'
else:
	res_nores_from_qsub = 'res'
print('res_nores_from_qsub: ', res_nores_from_qsub)

if 'moon' in sim_name_from_qsub:
	model_type_from_qsub = 'planet_moon'
else:
	model_type_from_qsub = 'planet_only'
print('model_type_from_qsub: ', model_type_from_qsub)



if(modeling_type == ""): modeling_type = 'planet_only'

assert modeling_type == model_type_from_qsub 
print('assert mmodeling_type == model_type_from_qsub PASSED.')
### open the catalogue file
catalogue = np.genfromtxt(garvitdir+'/'+lcgen_date+'catalogue.txt', skip_header=1, dtype='str')
sims = catalogue.T[0]
planet_periods = catalogue.T[3]
t0_barys = catalogue.T[18]
#sim_indices = np.arange(0,len(sims),1)+3 ### the first sim is #3 due to the way Garvit wrote his code.
sim_indices = np.arange(0,len(sims),1) ### got rid of that NOSPACE nonsense.


#catalogue = open('/tiara/home/gagarwal/files_and_data/catalogue.txt', 'r').read().split('\n') ### ORIGINAL
#catalogue = open(garvitdir+'/'+lcgen_date+'catalogue.txt', 'r').read().split('\n')
#system_line = catalogue[index].split('\t')



#system = system_line[0]
system = sims[index] #### ABSOLUTELY NEED TO VERIFY THAT THIS IS THE SAME SYSTEM YOU'RE RUNNING 
system_period = planet_periods[index]
system_t0_bary = t0_barys[index]
system_reversed = system[::-1] ### reverse the string, so that the sim number is at the beginning... remember, it's reversed now!
sim_number = system_reversed[:system_reversed.find('_')][::-1] ### turn it back around!

assert str(sim_number) == str(sim_number_from_qsub) 
print("assert sim_number == sim_number_from_qsub PASSED.")


if 'nores' in system:
	res_nores = 'nores'
else:
	res_nores = 'res'

json_filename = 'july13_'+res_nores+"_"+str(sim_number)+'.json'

#systems_details = json.load(open('/tiara/home/gagarwal/files_and_data/sim_data/final_unpacked_details/{}.json'.format(system), 'r')) ### ORIGINAL 
systems_details = json.load(open(garvitdir+'/sim_data/unpacked_details_july13/'+json_filename, 'r'))


if run_date == '3_30':
	#### the transit depths have been set to a standard 0.01. That's equal to Rp/Rstar
	### (Rp / Rstar)^2 == 0.01, so
	#### Rp / Rstar == sqrt(0.01)
	#### Rp = sqrt(0.01) * Rstar
	#### Rstar = Rp / sqrt(0.01)
	systems_details['Planet']['Rstar'] = systems_details['Planet']['Rp'] / np.sqrt(0.01) 



#lline = open('/tiara/home/gagarwal/files_and_data/most_detectable_systems_final.txt', 'r').read().split('\n')[:-1][index-2] ### ORIGINAL
lline = open(garvitdir+'/'+lcgen_date+'most_detectable_systems_july13.txt', 'r').read().split('\n')[:-1][index-1]

#filename = 'final_lightcurve_{}.csv'.format(system)
filename = ''
#ijlc = open("/tiara/home/gagarwal/files_and_data/final_lightcurves/{}".format(filename), 'r')
ijlc = open(garvitdir+'/final_lightcurves_'+lcgen_date[:-1]+'/sim'+str(sim_number)+'_'+res_nores+"_final_lightcurve.csv", 'r')
injected_lightcurve = ijlc.readlines()
ijlc.close()

kepler_file = injected_lightcurve[1].split(':')[1].strip()

#N = len(injected_lightcurve) - 2
#flux = np.ndarray(N)
#timestamps = np.ndarray(N)

#for j in range(N):
#	line = injected_lightcurve[j+2].split(',')
#	timestamps[j] = float(line[0])
#	flux[j] = float(line[1])

ijlc = pandas.read_csv(garvitdir+'/final_lightcurves_'+lcgen_date[:-1]+'/sim'+str(sim_number)+'_'+res_nores+'_final_lightcurve.csv', header=2)
timestamps = np.array(ijlc['#times']).astype(float)
flux = np.array(ijlc['fluxes']).astype(float)
noise_array = np.array(ijlc['errors']).astype(float)
noise = np.mean(noise_array)


'''
f = open("/tiara/home/gagarwal/files_and_data/good_kepler_data/{}".format(kepler_file), 'r')
kepler_data = f.readlines()
f.close()

N = len(kepler_data) -1
flux_error = np.ndarray((N))
kepler_flux = np.ndarray((N))
kepler_timestamps = np.ndarray((N))
for j in range(N):
	line = kepler_data[j+1].split('\t')
	kepler_timestamps[j] = float(line[0])
	kepler_flux[j] = float(line[1])
	flux_error[j] = float(line[2])/float(line[1])
	
noise_arr = np.ndarray((timestamps.shape[0]))
for j in range(noise_arr.shape[0]):
	noise_arr[j] = flux_error[np.where(kepler_timestamps == timestamps[j])]

frac = float(sys.argv[3])
new_N = int(timestamps.shape[0] * frac)
timestamps = timestamps[:new_N]
flux = flux[:new_N]
noise_arr = noise_arr[:new_N]

noise = np.mean(noise_arr)
'''

#noise = float(open("/tiara/home/gagarwal/files_and_data/uncertainties.txt", 'r').readlines()[index])



"""
##### ELIMINATED MAY 22nd 2023 -- was this causing problems for the moon fits?
if(modeling_type == 'planet_moon'): 
	R_star = float(systems_details['Planet']['Rstar'])
	#print("R_star: {}\n".format(R_star))
"""

per_bary = systems_details['Planet']['Pp'] ### days
per_bary_seconds = per_bary * 24 * 60 * 60
a_bary = systems_details['Planet']['a']/systems_details['Planet']['Rstar']
a_bary_meters = systems_details['Planet']['a']
r_planet = systems_details['Planet']['RpRstar'] ### r_planet is really Rp / Rstar -- will be between 0 and 1!!! 
w_bary = float(lline.split(' ')[3])
omega_moon = float(lline.split(' ')[4])

if(modeling_type == 'planet_only'):
	print('per_bary: {}\na_bary: {}\nr_planet: {}\nb_bary: {}\necc_bary: 0.0\nw_bary: {}\nq1: {}\nq2: {}'.format(per_bary, a_bary, r_planet, systems_details['Planet']['impact'],w_bary, systems_details['Planet']['q1'], systems_details['Planet']['q2']))

	parameters = ['per_bary','a_bary','r_planet','b_bary','ecc_bary', 'w_bary','t0_bary_offset','q1','q2']
	wrapped_params = [False,False,False,False,False, True,False,False,False]

elif(modeling_type == 'planet_moon'):
	print('per_bary: {}\na_bary: {}\nr_planet: {}\nb_bary: {}\necc_bary: 0.0\nw_bary: {}\n'.format(per_bary, a_bary, r_planet, systems_details['Planet']['impact'], w_bary))

	R_star = systems_details['Planet']['Rstar']
	M_planet = systems_details['Planet']['m']
	r_moon = systems_details['I']['r']/R_star ### like r_planet, r_moon is also really Rmoon / Rstar. It's a value between zero and 1. 
	per_moon = systems_details['I']['P']/(24*3600)
	tau_moon = (2*np.pi-systems_details['I']['f'])/(2*np.pi)
	i_moon = (math.degrees(systems_details['I']['inc']) + 90.0)%360
	M_moon =  systems_details['I']['m']
	r_hill = systems_details['Planet']['RHill']
	a_moon = systems_details['I']['a']
	r_roche = systems_details['I']['r'] * (2*M_planet/M_moon)**(1/3)

	print('M_planet: {}\nr_moon: {}\nper_moon: {}\ntau_moon: {}\nomega_moon: {}\ni_moon: {}\nM_moon: {}\nq1: {}\nq2: {}\n'.format(M_planet, r_moon, per_moon, tau_moon,omega_moon, i_moon, M_moon, systems_details['Planet']['q1'], systems_details['Planet']['q2']))

	parameters = ['per_bary','a_bary','r_planet','b_bary','ecc_bary', 'w_bary','t0_bary_offset','M_planet','r_moon', 'per_moon','tau_moon', 'Omega_moon', 'i_moon', 'M_moon','q1','q2']
	wrapped_params = [False,False,False,False,False,True,False,False,False,False,True,True,True,False,False,False]


#log_dir_planet_only="/tiara/home/gagarwal/modeling_results/{}/planet_only/".format(system) ### ORIGINAL 
#log_dir_planet_moon="/tiara/home/gagarwal/modeling_results/{}/planet_moon/".format(system) ### ORIGINAL

if os.path.exists(model_run_savedir+'/'+system) == False:
	os.system('mkdir '+model_run_savedir+'/'+system)

log_dir_planet_only = model_run_savedir+'/'+system+'/planet_only' ### MODIFICATION
log_dir_planet_moon = model_run_savedir+'/'+system+'/planet_moon' ### MODIFICATION

if os.path.exists(log_dir_planet_only) == False:
	os.system('mkdir '+log_dir_planet_only)
if os.path.exists(log_dir_planet_moon) == False:
	os.system('mkdir '+log_dir_planet_moon)


#os.system('mkdir -p /tiara/home/gagarwal/files_and_data/prior_random_shifts/{}/'.format(system)) ### original
os.system('mkdir -p '+garvitdir+'/prior_random_shifts/{}/'.format(system))
"""
if(len(sys.argv)==6 and sys.argv[5]=='resume'):
	resume = 'resume-similar'
	#listt = open('/tiara/home/gagarwal/files_and_data/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'r').readlines()
	listt = open(garvitdir+'/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'r').readlines()
	perbary_shift= float(listt[0])
	abary_shift= float(listt[1])
	rp_shift = float(listt[2])
	mplanet_shift = float(listt[3])

else:
	resume = 'overwrite'
	perbary_shift = np.random.normal(scale=1e-5)
	abary_shift = np.random.normal(scale=1e-2)
	rp_shift = np.random.normal(scale=3* 1e-2)
	mplanet_shift = np.random.normal(scale=1e-2)

	#f = open('/tiara/home/gagarwal/files_and_data/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'w')
	prs = open(garvitdir+'/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'w')
	prs.write('{}\n{}\n{}\n{}'.format(perbary_shift, abary_shift,rp_shift, mplanet_shift))
	prs.close()
"""


#### new -- NEED TO SPECIFY OVERWRITE, OTHERWISE JUST RESUME
if(len(sys.argv)==6 and sys.argv[5]=='overwrite'):
	resume = 'overwrite'
	perbary_shift = np.random.normal(scale=1e-5)
	abary_shift = np.random.normal(scale=1e-2)
	rp_shift = np.random.normal(scale=3* 1e-2)
	mplanet_shift = np.random.normal(scale=1e-2)

	#f = open('/tiara/home/gagarwal/files_and_data/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'w')
	prs = open(garvitdir+'/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'w')
	prs.write('{}\n{}\n{}\n{}'.format(perbary_shift, abary_shift,rp_shift, mplanet_shift))
	prs.close()

else:
	#### assume resume-similar
	resume = 'resume-similar'
	#listt = open('/tiara/home/gagarwal/files_and_data/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'r').readlines()
	listt = open(garvitdir+'/prior_random_shifts/{}/prior_random_shifts_{}.txt'.format(system, modeling_type), 'r').readlines()
	perbary_shift= float(listt[0])
	abary_shift= float(listt[1])
	rp_shift = float(listt[2])
	mplanet_shift = float(listt[3])






if(modeling_type == 'planet_only'):
	sampler = ReactiveNestedSampler(
			parameters,
			log_likelihood_planet_only, 
			prior_transform_planet_only,
			wrapped_params=wrapped_params,
			resume= resume,
			log_dir = log_dir_planet_only
			)
elif(modeling_type == 'planet_moon'):
	sampler = ReactiveNestedSampler(
			parameters,
			log_likelihood_planet_moon, 
			prior_transform_planet_moon,
			wrapped_params=wrapped_params,
			resume = resume,
			log_dir=log_dir_planet_moon
			)


sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(
	nsteps=4000,
	adaptive_nsteps='move-distance',
	)  


result = sampler.run(min_num_live_points=1000)

if(modeling_type == 'planet_only'):
		print('per_bary: {}\na_bary: {}\nr_planet: {}\nb_bary: {}\necc_bary: 0.0\nw_bary: {}\nq1: {}\nq2: {}'.format(per_bary, a_bary, r_planet, systems_details['Planet']['impact'],w_bary, systems_details['Planet']['q1'], systems_details['Planet']['q2']))


elif(modeling_type == 'planet_moon'):
		print('per_bary: {}\na_bary: {}\nr_planet: {}\nb_bary: {}\necc_bary: 0.0\nw_bary: {}\n'.format(per_bary, a_bary, r_planet, systems_details['Planet']['impact'], w_bary))
		print('M_planet: {}\nr_moon: {}\nper_moon: {}\ntau_moon: {}\nomega_moon: {}\ni_moon: {}\nM_moon: {}\nq1: {}\nq2: {}\n'.format(M_planet, r_moon, per_moon, tau_moon,omega_moon, i_moon, M_moon, systems_details['Planet']['q1'], systems_details['Planet']['q2']))

sampler.print_results()
