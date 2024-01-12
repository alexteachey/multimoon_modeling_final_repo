from __future__ import division
import numpy as np
import pandas
import traceback
from scipy.optimize import curve_fit 
import time
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import os


rcParams['font.family'] = 'serif'
rcParams.update({'font.size': 15})

run_name = input('What is the run name? e.g. run8_10 or run11_10: ')
recursive_fit = input('Do you want to fit the modes recursively? y/n: ')
use_peaks_as_modes = input('Do you want to count peaks instead of modes? y/n: ')
if use_peaks_as_modes == 'y':
	modes_filename = 'posterior_peaks.csv'
elif use_peaks_as_modes == 'n':
	modes_filename = 'posterior_gaussian_mode_fits.csv'

use_external = input('Use external? y/n: ')

external_projectdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling'
if use_external == 'y':
	projectdir = external_projectdir
else:
	projectdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'
posteriors_dir = projectdir+'/posteriors'

#catalogue = np.genfromtxt(projectdir+'/catalogue.txt', names=True, usecols=(0,1,2), dtype=(str, str, int))
catalogue = pandas.read_csv(projectdir+'/may16_2023_catalogue.txt', delimiter='\t')

mode_colors = ['#AC92EB', '#4FC1E8', '#A0D567', '#FDCE54', '#ED5564']

planet_cols = [
	'per_bary', 'a_bary', 'r_planet', 'b_bary', 'ecc_bary', 'w_bary', 't0_bary_offset', 'q1', 'q2',
	]

moon_cols = [
	'per_bary', 'a_bary', 'r_planet', 'b_bary', 'ecc_bary', 'w_bary', 't0_bary_offset', 'M_planet', 
	'r_moon', 'per_moon', 'tau_moon', 'Omega_moon', 'i_moon', 'M_moon', 'q1', 'q2',
	]

#moon_cols_labels = [
#	'barycenter period', 'barycenter semimajor axis', r'$R_P / R_{*}$', 'impact parameter', 'barycenter eccentricity', r'barycenter $\omega$', 
#	r'barycenter $t_0$ offset', 'planet mass', 'moon radius', 'moon period', r'moon $\tau$', r'moon $\Omega$', 'moon inclination', 'moon mass', r'$q_1$', r'$q_2$',
#	]

moon_cols_labels = [
r'$P_B$', r'$a_B$', r'$R_P / R_{*}$', r'$b_B$', r'$e_B$', r'$\omega_B$', r'$t_0$ offset', 
r'$M_P$', r'$R_S$', r'$P_S$', r'$\tilde{\phi}$', r'$\omega_S$', r'$i_S$', r'$M_S$', r'$q_1$', r'$q_2$',
]



if os.path.exists(projectdir+'/'+modes_filename) == False:
	### #make it
	modesfile = open(projectdir+'/'+modes_filename, mode='w')
	modesfile.write('system,nmoons')
	for col in moon_cols:
		modesfile.write(','+col)
	modesfile.write('\n')
	modesfile.close()
	systems_run = []

else:
	#### open it and see which systems you've already read
	modesfile_pd = pandas.read_csv(projectdir+'/'+modes_filename)
	systems_run = np.array(modesfile_pd['system'])




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


system_names = np.array(catalogue['system_file']).astype(str)
kepler_files = np.array(catalogue['kepler_file']).astype(str)
nmoons = np.array(catalogue['num_moons']).astype(int)
nmoons_good_sims = []
system_dict = {}





per_bary_nmodes = []
a_bary_nmodes = []
r_planet_nmodes = []
b_bary_nmodes = [] 
ecc_bary_nmodes = [] 
w_bary_nmodes = [] 
t0_bary_offset_nmodes = [] 
M_planet_nmodes = [] 
r_moon_nmodes = [] 
per_moon_nmodes = [] 
tau_moon_nmodes = [] 
Omega_moon_nmodes = [] 
i_moon_nmodes = [] 
M_moon_nmodes = []
q1_nmodes = []
q2_nmodes = []


show_histograms = input('Do you want to show the histograms? y/n: ')

def gaussian(x, mu, sigma, amp):
	numerator = (x-mu)**2
	denominator = 2 * sigma**2
	return amp * np.exp( -(numerator / denominator) )

def onemode(x, mu, sigma, amp):
	output = gaussian(x=x, mu=mu, sigma=sigma, amp=amp)
	return output

def twomodes(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
	output = gaussian(x=x, mu=mu1, sigma=sigma1, amp=amp1) + gaussian(x=x, mu=mu2, sigma=sigma2, amp=amp2) 
	return output 

def threemodes(x, mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3):
	output = gaussian(x=x, mu=mu1, sigma=sigma1, amp=amp1) + gaussian(x=x, mu=mu2, sigma=sigma2, amp=amp2) + gaussian(x=x, mu=mu3, sigma=sigma3, amp=amp3)
	return output 

def fourmodes(x, mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3, mu4, sigma4, amp4):
	output = gaussian(x=x, mu=mu1, sigma=sigma1, amp=amp1) + gaussian(x=x, mu=mu2, sigma=sigma2, amp=amp2) + gaussian(x=x, mu=mu3, sigma=sigma3, amp=amp3) + gaussian(x=x, mu=mu4, sigma=sigma4, amp=amp4)
	return output 

def fivemodes(x, mu1, sigma1, amp1, mu2, sigma2, amp2, mu3, sigma3, amp3, mu4, sigma4, amp4, mu5, sigma5, amp5):
	output = gaussian(x=x, mu=mu1, sigma=sigma1, amp=amp1) + gaussian(x=x, mu=mu2, sigma=sigma2, amp=amp2) + gaussian(x=x, mu=mu3, sigma=sigma3, amp=amp3) + gaussian(x=x, mu=mu4, sigma=sigma4, amp=amp4) + gaussian(x=x, mu=mu5, sigma=sigma5, amp=amp5) 
	return output 


def peak_finder(yvals, return_idxs=True, width=3):
	#### return the indices of peaks
	peak_indices = []
	for idx,yval in enumerate(yvals):
		if (idx < width) or (idx > len(yvals)-width):
			continue
		else:
			neighbor_yvals = np.concatenate((yvals[idx-width:idx], yvals[idx+1:idx+width+1])) 
			if np.all(neighbor_yvals < yval):
				ispeak = True
				peak_indices.append(idx)
			else:
				ispeak = False 
	return np.array(peak_indices)


continue_query = input('Continue?: y/n: ')
if continue_query == 'n':
	raise Exception('you opted not to continue.')


#### for each system, see if we can identify multiple modes in the posteriors
#for nsystem, system in enumerate(system_names):
for nsystem, system in enumerate(good_sims):
	print('system: ', system)
	catalogue_idx = np.where(system == system_names)[0]
	number_of_moons = int(nmoons[catalogue_idx])
	nmoons_good_sims.append(number_of_moons)


	if system in systems_run:
		continue 

	#planet_posterior = np.genfromtxt(str(posteriors_dir)+'/'+str(system)+'_planet_only.txt', names=True) #### ORIGINAL
	planet_posterior = np.genfromtxt(external_projectdir+'/modeling_results_for_download/'+run_name+'/'+system+'/planet_only/chains/equal_weighted_post.txt', names=True)
	#moon_posterior = np.genfromtxt(str(posteriors_dir)+'/'+str(system)+'_planet_moon.txt', names=True)
	moon_posterior = np.genfromtxt(external_projectdir+'/modeling_results_for_download/'+run_name+'/'+system+'/planet_moon/chains/equal_weighted_post.txt', names=True)

	#system_nmoons = nmoons[nsystem]
	system_nmoons = number_of_moons


	modesfile = open(projectdir+'/'+modes_filename, mode='a')
	modesfile.write(str(system)+','+str(system_nmoons))
	

	#### for each parameter in the moon_posteriors, see if you can find multiple modes
	"""
	STEPS:
	1) generate a histogram
	2) grab the nvals (number per bin)
	3) let that be a function
	4) try to fit one or more gaussians (up to five)
	5) use a delta-BIC test to see which fit is the best, while penalizing the extra complexity
	6) for each parameter, record the best model 
	7) compare to the number of moons in the system

	"""

	for postcol in moon_cols:
		parampost = moon_posterior[postcol]

		### make a histogram
		bounds = [np.nanmin(parampost), np.nanmax(parampost)]
		#print('bounds: ', bounds)
		param_range = bounds[1] - bounds[0]
		#print('param_range = ', param_range)
		if param_range > 1e3:
			#### space the bins logarithmically
			histbins = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), 40)
		else:
			histbins = np.linspace(bounds[0], bounds[1], 40)

		histvals, bins, edges = plt.hist(parampost, bins=histbins, facecolor='CornflowerBlue', edgecolor='k')
		if show_histograms != 'y':
			plt.close()
		else:
			pass

		#rint('histbins: ', histbins)

		bin_centers =[]
		for i in np.arange(0,len(histbins),1):
			if i>=1:
				bin_centers.append((histbins[i] + histbins[i-1]) / 2)

		assert len(bin_centers) == len(histvals)


		#### now fit the line with the various models
		try:
			one_mode_popt, one_mode_pcov = curve_fit(
				f=onemode, 
				xdata=bin_centers, 
				ydata=histvals, bounds=(
					[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
					)
				)

			### generate the gaussian
			one_mode_fit = onemode(bin_centers, *one_mode_popt)

			one_mode_sum_square_error = np.nansum((one_mode_fit - histvals)**2)
			ndatapoints = len(histvals)
			one_mode_nparams = 3
			one_mode_klnn = one_mode_nparams * np.log(ndatapoints)
			one_mode_quasiBIC = one_mode_klnn + one_mode_sum_square_error 

		except:
			one_mode_popt = np.linspace(np.nan, np.nan, 3)
			one_mode_pcov = np.linspace(np.nan, np.nan, 3*3).reshape(3,3)
			one_mode_fit = np.linspace(np.nan, np.nan, len(bin_centers))
			one_mode_sum_square_error = np.nan
			one_mode_nparams = 3
			one_mode_klnn = np.nan
			one_mode_quasiBIC = np.nan 
			traceback.print_exc()
			print(' ')




		try:
			if recursive_fit == 'n':
				two_mode_popt, two_mode_pcov = curve_fit(
					f=twomodes, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0, bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				#### generate the gaussians
				two_mode_fit = twomodes(bin_centers, *two_mode_popt)

			elif recursive_fit == 'y':
				first_mode_popt, first_mode_pcov = curve_fit(
					f=onemode, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				one_mode_fit = onemode(bin_centers, *one_mode_popt)


				### #FIT THE SECOND MODE
				input_data = histvals - one_mode_fit
				second_mode_popt, second_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				two_mode_popt = np.concatenate((first_mode_popt, second_mode_popt))
				two_mode_fit = twomodes(bin_centers, *two_mode_popt)


			two_mode_sum_square_error = np.nansum((two_mode_fit - histvals)**2)
			ndatapoints = len(histvals)
			two_mode_nparams = 6
			two_mode_klnn = two_mode_nparams * np.log(ndatapoints)
			two_mode_quasiBIC = two_mode_klnn + two_mode_sum_square_error 

		except:
			two_mode_popt = np.linspace(np.nan, np.nan, 6)
			two_mode_pcov = np.linspace(np.nan, np.nan, 6*6).reshape(6,6)
			two_mode_fit = np.linspace(np.nan, np.nan, len(bin_centers))
			two_mode_sum_square_error = np.nan
			two_mode_nparams = 6
			two_mode_klnn = np.nan
			two_mode_quasiBIC = np.nan 
			traceback.print_exc()
			print(' ')




		try:
			if recursive_fit == 'n':
				three_mode_popt, three_mode_pcov = curve_fit(
					f=threemodes, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				#### generate the gaussians
				three_mode_fit = threemodes(bin_centers, *three_mode_popt)

			elif recursive_fit == 'y':
				first_mode_popt, first_mode_pcov = curve_fit(
					f=onemode, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				one_mode_fit = onemode(bin_centers, *one_mode_popt)

				#### FIT THE SECOND MODE
				input_data = histvals - one_mode_fit
				second_mode_popt, second_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				two_mode_popt = np.concatenate((first_mode_popt, second_mode_popt))
				two_mode_fit = twomodes(bin_centers, *two_mode_popt)

				#### FIT THE THIRD MODE
				input_data = histvals - two_mode_fit
				third_mode_popt, third_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				three_mode_popt = np.concatenate((two_mode_popt, third_mode_popt))
				three_mode_fit = threemodes(bin_centers, *three_mode_popt)




			three_mode_sum_square_error = np.nansum((three_mode_fit - histvals)**2)
			ndatapoints = len(histvals)
			three_mode_nparams = 9
			three_mode_klnn = three_mode_nparams * np.log(ndatapoints)
			three_mode_quasiBIC = three_mode_klnn + three_mode_sum_square_error 

		except:
			three_mode_popt = np.linspace(np.nan, np.nan, 9)
			three_mode_pcov = np.linspace(np.nan, np.nan, 9*9).reshape(9,9)
			three_mode_fit = np.linspace(np.nan, np.nan, len(bin_centers))
			three_mode_sum_square_error = np.nan
			three_mode_nparams = 9
			three_mode_klnn = np.nan
			three_mode_quasiBIC = np.nan 
			traceback.print_exc()
			print(' ')





		try:
			if recursive_fit == 'n':

				four_mode_popt, four_mode_pcov = curve_fit(
					f=fourmodes, 
					xdata=bin_centers, 
					ydata=histvals,
					maxfev=1000*(len(bin_centers)+1), 
					bounds=(
						[bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				#### generate the gaussians
				four_mode_fit = fourmodes(bin_centers, *four_mode_popt)

			elif recursive_fit == 'y':
				first_mode_popt, first_mode_pcov = curve_fit(
					f=onemode, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				one_mode_fit = onemode(bin_centers, *one_mode_popt)


				#### FIT THE SECOND MODE
				input_data = histvals - one_mode_fit
				second_mode_popt, second_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				two_mode_popt = np.concatenate((first_mode_popt, second_mode_popt))
				two_mode_fit = twomodes(bin_centers, *two_mode_popt)


				#### FIT THE THIRD MODE
				input_data = histvals - two_mode_fit
				third_mode_popt, third_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				three_mode_popt = np.concatenate((two_mode_popt, third_mode_popt))
				three_mode_fit = threemodes(bin_centers, *three_mode_popt)


				#### FIT THE FOURTH MODE
				input_data = histvals - three_mode_fit
				fourth_mode_popt, fourth_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				four_mode_popt = np.concatenate((three_mode_popt, fourth_mode_popt))
				four_mode_fit = fourmodes(bin_centers, *four_mode_popt)


			four_mode_sum_square_error = np.nansum((four_mode_fit - histvals)**2)
			ndatapoints = len(histvals)
			four_mode_nparams = 12
			four_mode_klnn = four_mode_nparams * np.log(ndatapoints)
			four_mode_quasiBIC = four_mode_klnn + four_mode_sum_square_error 

		except:
			four_mode_popt = np.linspace(np.nan, np.nan, 12)
			four_mode_pcov = np.linspace(np.nan, np.nan, 12*12).reshape(12,12)
			four_mode_fit = np.linspace(np.nan, np.nan, len(bin_centers))
			four_mode_sum_square_error = np.nan
			four_mode_nparams = 12
			four_mode_klnn = np.nan
			four_mode_quasiBIC = np.nan 
			traceback.print_exc()
			print(' ')


		try:
			if recursive_fit == 'n':
				five_mode_popt, five_mode_pcov = curve_fit(
					f=fivemodes, 
					xdata=bin_centers, 
					ydata=histvals,
					maxfev=1000*(len(bin_centers)+1), 
					bounds=(
						[bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0, bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals), bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				#### generate the gaussians
				five_mode_fit = fivemodes(bin_centers, *five_mode_popt)

			elif recursive_fit == 'y':
				first_mode_popt, first_mode_pcov = curve_fit(
					f=onemode, 
					xdata=bin_centers, 
					ydata=histvals, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				one_mode_fit = onemode(bin_centers, *one_mode_popt)


				#### FIT THE SECOND MODE
				input_data = histvals - one_mode_fit
				second_mode_popt, second_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				two_mode_popt = np.concatenate((first_mode_popt, second_mode_popt))
				two_mode_fit = twomodes(bin_centers, *two_mode_popt)


				#### FIT THE THIRD MODE
				input_data = histvals - two_mode_fit
				third_mode_popt, third_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				three_mode_popt = np.concatenate((two_mode_popt, third_mode_popt))
				three_mode_fit = threemodes(bin_centers, *three_mode_popt)


				#### FIT THE FOURTH MODE
				input_data = histvals - three_mode_fit
				fourth_mode_popt, fourth_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				four_mode_popt = np.concatenate((three_mode_popt, fourth_mode_popt))
				four_mode_fit = fourmodes(bin_centers, *four_mode_popt)


				#### FIT THE FIFTH MODE
				input_data = histvals - four_mode_fit
				fifth_mode_popt, fifth_mode_pcov = curve_fit(
					f=onemode,
					xdata=bin_centers,
					ydata=input_data, 
					maxfev=1000*(len(bin_centers)+1),
					bounds=(
						[bounds[0], 0, 0], [bounds[1], param_range, np.nanmax(histvals)]
						)
					)
				five_mode_popt = np.concatenate((four_mode_popt, fifth_mode_popt))
				five_mode_fit = fivemodes(bin_centers, *five_mode_popt)



			five_mode_sum_square_error = np.nansum((five_mode_fit - histvals)**2)
			ndatapoints = len(histvals)
			five_mode_nparams = 15
			five_mode_klnn = five_mode_nparams * np.log(ndatapoints)
			five_mode_quasiBIC = five_mode_klnn + five_mode_sum_square_error 

		except:
			five_mode_popt = np.linspace(np.nan, np.nan, 15)
			five_mode_pcov = np.linspace(np.nan, np.nan, 15*15).reshape(15,15)
			five_mode_fit = np.linspace(np.nan, np.nan, len(bin_centers))
			five_mode_sum_square_error = np.nan
			five_mode_nparams = 15
			five_mode_klnn = np.nan
			five_mode_quasiBIC = np.nan 
			traceback.print_exc()
			print(' ')



		### summary
		try:
			print('one mode: ')
			one_mode_nreal = 1
			one_mode_mu = one_mode_popt[0]
			one_mode_sigma = one_mode_popt[1]
			one_mode_amp = one_mode_popt[2]
			print('mu = ', one_mode_mu)
			print('sigma = ', one_mode_sigma)
			print('amp = ', one_mode_amp)
			print('# actual modes = ', one_mode_nreal)
			print(' ')
		except:
			pass
		try:
			print('two modes: ')
			two_mode_nreal = 2
			two_mode_mus = two_mode_popt[::3]
			two_mode_sigmas = two_mode_popt[1::3]
			two_mode_amps = two_mode_popt[2::3]
			two_mode_mus_argsort = np.argsort(two_mode_mus)
			two_mode_mus = two_mode_mus[two_mode_mus_argsort]
			two_mode_sigmas = two_mode_sigmas[two_mode_mus_argsort]
			two_mode_amps = two_mode_amps[two_mode_mus_argsort]			
			mu_diffs = []
			#for i in np.arange(0,len(two_mode_mus),1):
			if ((np.abs(two_mode_mus[1] - two_mode_mus[0]) / np.abs(two_mode_sigmas[0])) < 1) or ((np.abs(two_mode_mus[1] - two_mode_mus[0]) / np.abs(two_mode_sigmas[1])) < 1):
				#### indicates that the two peaks are less than one sigma apart from each other, so let's not count it as two modes.
				two_mode_nreal = two_mode_nreal - 1


			print('mus = ', two_mode_mus)
			print('sigmas = ', two_mode_sigmas)
			print('amps = ', two_mode_amps)
			print('# actual modes = ', two_mode_nreal)			
			print(' ')
		except:
			pass
		try:
			print('three modes: ')
			three_mode_nreal = 3
			three_mode_mus = three_mode_popt[::3]
			three_mode_sigmas = three_mode_popt[1::3]
			three_mode_amps = three_mode_popt[2::3]
			three_mode_mus_argsort = np.argsort(three_mode_mus)
			three_mode_mus = three_mode_mus[three_mode_mus_argsort]
			three_mode_sigmas = three_mode_sigmas[three_mode_mus_argsort]
			three_mode_amps = three_mode_amps[three_mode_mus_argsort]

			for j in np.arange(0,len(three_mode_mus),1):
				i = j+1
				try:
					if ((np.abs(three_mode_mus[i] - three_mode_mus[j]) / np.abs(three_mode_sigmas[i])) < 1) or ((np.abs(three_mode_mus[i] - three_mode_mus[j]) / np.abs(three_mode_sigmas[j])) < 1):
						#### indicates that the two peaks are less than one sigma apart from each other, so let's not count it as two modes.
						three_mode_nreal = three_mode_nreal - 1
				except:
					pass

			print('mus = ', three_mode_mus)
			print('sigmas = ', three_mode_sigmas)
			print('amps = ', three_mode_amps)
			print('# actual modes = ', three_mode_nreal)			
			print(' ')
		except:
			pass
		try:
			print('four modes: ')
			four_mode_nreal = 4
			four_mode_mus = four_mode_popt[::3]
			four_mode_sigmas = four_mode_popt[1::3]
			four_mode_amps = four_mode_popt[2::3]
			four_mode_mus_argsort = np.argsort(four_mode_mus)
			four_mode_mus = four_mode_mus[four_mode_mus_argsort]
			four_mode_sigmas = four_mode_sigmas[four_mode_mus_argsort]
			four_mode_amps = four_mode_amps[four_mode_mus_argsort]

			for j in np.arange(0,len(four_mode_mus),1):
				i = j+1
				try:
					if ((np.abs(four_mode_mus[i] - four_mode_mus[j]) / np.abs(four_mode_sigmas[i])) < 1) or ((np.abs(four_mode_mus[i] - four_mode_mus[j]) / np.abs(four_mode_sigmas[j])) < 1):
						#### indicates that the two peaks are less than one sigma apart from each other, so let's not count it as two modes.
						four_mode_nreal = four_mode_nreal - 1	
				except:
					pass		
			print('mus = ', four_mode_mus)
			print('sigmas = ', four_mode_sigmas)
			print('amps = ', four_mode_amps)
			print('# actual modes = ', four_mode_nreal)			
			print(' ')
		except:
			pass
		try:
			print('four modes: ')
			five_mode_nreal = 5 
			five_mode_mus = five_mode_popt[::3]
			five_mode_sigmas = five_mode_popt[1::3]
			five_mode_amps = five_mode_popt[2::3]
			five_mode_mus_argsort = np.argsort(five_mode_mus)
			five_mode_mus = five_mode_mus[five_mode_mus_argsort]
			five_mode_sigmas = five_mode_sigmas[five_mode_mus_argsort]
			five_mode_amps = five_mode_amps[five_mode_mus_argsort]

			for j in np.arange(0,len(five_mode_mus),1):
				i = j+1
				try:
					if ((np.abs(five_mode_mus[i] - five_mode_mus[j]) / np.abs(five_mode_sigmas[i])) < 1) or ((np.abs(five_mode_mus[i] - five_mode_mus[j]) / np.abs(five_mode_sigmas[j])) < 1):
						#### indicates that the two peaks are less than one sigma apart from each other, so let's not count it as two modes.
						five_mode_nreal = five_mode_nreal - 1
				except:
					pass

			print('mus = ', five_mode_mus)
			print('sigmas = ', five_mode_sigmas)
			print('amps = ', five_mode_amps)
			print('# actual modes = ', five_mode_nreal)			
		except:
			pass

		quasiBICs = np.array([one_mode_quasiBIC, two_mode_quasiBIC, three_mode_quasiBIC, four_mode_quasiBIC, five_mode_quasiBIC])
		min_quasiBIC = np.nanmin(quasiBICs)
		quasiBIC_argsort = np.argsort(quasiBICs)
		excess_quasiBICs = quasiBICs - min_quasiBIC
		modes = np.array([1, 2, 3, 4, 5])
		best_mode = modes[quasiBIC_argsort][0]
		number_of_peaks = len(peak_finder(histvals))

		if use_peaks_as_modes == 'y':
			best_mode = number_of_peaks


		modesfile.write(','+str(best_mode))



		print(' ')
		print('system: ', system)
		print('# of moons: ', number_of_moons)
		print('# of peaks: ', number_of_peaks)				
		print('parameter: ', postcol)
		print('# Gaussian modes (best to worst fit): '+str(modes[quasiBIC_argsort]))
		print('excess quasiBICs: '+str(excess_quasiBICs[quasiBIC_argsort]))

		print(' ')
		print(' ')
		print(' X X X X X ')



		### plot the histogram! 
		if show_histograms == 'y':
			plt.plot(bin_centers, histvals, color='black')
			plt.plot(bin_centers, one_mode_fit, linestyle='--', color=mode_colors[0], label='one mode')
			plt.plot(bin_centers, two_mode_fit, linestyle='--', color=mode_colors[1], label='two modes')
			plt.plot(bin_centers, three_mode_fit, linestyle='--', color=mode_colors[2], label='three modes')
			plt.plot(bin_centers, four_mode_fit, linestyle='--', color=mode_colors[3], label='four modes')
			plt.plot(bin_centers, five_mode_fit, linestyle='--', color=mode_colors[4], label='five modes')
			plt.legend()
			plt.title(postcol)
			plt.show()


		#time.sleep(5)


		if postcol == 'per_bary':
			per_bary_nmodes.append(best_mode)
		elif postcol == 'a_bary':
			a_bary_nmodes.append(best_mode)
		elif postcol == 'r_planet':
			r_planet_nmodes.append(best_mode)
		elif postcol == 'b_bary':
			b_bary_nmodes.append(best_mode)
		elif postcol == 'ecc_bary': 
			ecc_bary_nmodes.append(best_mode)
		elif postcol == 'w_bary': 
			w_bary_nmodes.append(best_mode)
		elif postcol == 't0_bary_offset':
			t0_bary_offset_nmodes.append(best_mode)
		elif postcol == 'M_planet':
			M_planet_nmodes.append(best_mode)
		elif postcol == 'r_moon':
			r_moon_nmodes.append(best_mode)
		elif postcol == 'per_moon':
			per_moon_nmodes.append(best_mode)
		elif postcol == 'tau_moon':
			tau_moon_nmodes.append(best_mode)
		elif postcol == 'Omega_moon':
			Omega_moon_nmodes.append(best_mode)
		elif postcol == 'i_moon':
			i_moon_nmodes.append(best_mode)
		elif postcol == 'M_moon':
			M_moon_nmodes.append(best_mode)
		elif postcol == 'q1':
			q1_nmodes.append(best_mode)
		elif postcol == 'q2':
			q2_nmodes.append(best_mode)


	modesfile.write('\n')
	modesfile.close()

	print(' ')
	print(' ')
	print(' ')
	#time.sleep(5)




system_dict['nmoons'] = nmoons_good_sims
system_dict['per_bary'] = per_bary_nmodes
system_dict['a_bary'] = a_bary_nmodes
system_dict['r_planet'] = r_planet_nmodes
system_dict['b_bary'] = b_bary_nmodes 
system_dict['ecc_bary'] = ecc_bary_nmodes
system_dict['w_bary'] = w_bary_nmodes 
system_dict['t0_bary_offset'] = t0_bary_offset_nmodes
system_dict['M_planet'] = M_planet_nmodes
system_dict['r_moon'] = r_moon_nmodes
system_dict['per_moon'] = per_moon_nmodes
system_dict['tau_moon'] = tau_moon_nmodes
system_dict['Omega_moon'] = Omega_moon_nmodes
system_dict['i_moon'] = i_moon_nmodes
system_dict['M_moon'] = M_moon_nmodes
system_dict['q1'] = q1_nmodes
system_dict['q2'] = q2_nmodes




#### now take everything system, grab the number of moons, 
"""
for key in system_dict.keys():
	system_nmoons = system_dict['nmoons']
	
	if key != 'nmoons':
		plt.scatter(system_nmoons, system_dict[key], facecolor='CornflowerBlue', edgecolor='k', s=30)
		plt.xlabel('# of moons')
		plt.ylabel('# of modes (best fit)')
		plt.title(key)
		plt.show()
"""



### open the document you just made
pgmf = pandas.read_csv(projectdir+'/'+modes_filename)
pgmf_dict = {}
for col in pgmf.columns:
	pgmf_dict[col] = np.array(pgmf[col])





columns = pgmf_dict.keys()


### make a heatmap
heatmap = np.zeros(shape=(5,5))

"""
for ncol, column in enumerate(columns):
	if (column == 'system') or (column == 'nmoons'):
		continue


	for moon_idx in np.arange(0,5,1):
			for mode_idx in np.arange(0,5,1):
				moon_number = moon_idx+1 
				mode_number = mode_idx+1

				matching_idxs = np.where((pgmf_dict['nmoons'] == moon_number) & (pgmf_dict[column] == mode_number))[0]
				n_matching_idxs = len(matching_idxs)

				#### indexing is array[row][column], so it should be
				heatmap[mode_idx][moon_idx] = n_matching_idxs

	fig, ax = plt.subplots()
	im = ax.imshow(heatmap, origin='lower', interpolation=None, cmap='viridis')
	#ax.set_xticks(ticks=np.arange(0,5,1), labels=np.arange(1,6,1))
	#ax.set_yticks(ticks=np.arange(0,5,1), labels=np.arange(1,6,1))
	ax.set_xticks(ticks=np.arange(0,5,1))
	ax.set_xticklabels(np.arange(1,6,1))
	ax.set_yticks(ticks=np.arange(0,5,1))
	ax.set_yticklabels(np.arange(1,6,1))
	ax.set_xlabel('# moons')
	if use_peaks_as_modes == 'y':
		ax.set_ylabel('# of peaks')
	else:
		ax.set_ylabel('# Gaussian modes')
	#ax.set_title(column)
	ax.set_title(moon_cols_labels[ncol-2]) #### subtract two because the first two columns are system name and nmoons, and there are 
	fig.colorbar(im)
	plt.savefig('/Users/hal9000/Documents/Projects/multimoon_modeling/plots/'+column+'_nmodes_vs_nmoons_heatmap.png', dpi=300)
	plt.show()
"""



#### MAKE ONE BIG PLOT
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(16,16))

col=0
row=0
for ncol, column in enumerate(columns):
	if (column == 'system') or (column == 'nmoons'):
		continue
	for moon_idx in np.arange(0,5,1):
			for mode_idx in np.arange(0,5,1):
				moon_number = moon_idx+1 
				mode_number = mode_idx+1
				matching_idxs = np.where((pgmf_dict['nmoons'] == moon_number) & (pgmf_dict[column] == mode_number))[0]
				n_matching_idxs = len(matching_idxs)
				#### indexing is array[row][column], so it should be
				heatmap[mode_idx][moon_idx] = n_matching_idxs
	#fig, ax = plt.subplots()
	im = ax[row][col].imshow(heatmap, origin='lower', interpolation=None, cmap='viridis')
	#ax.set_xticks(ticks=np.arange(0,5,1), labels=np.arange(1,6,1))
	#ax.set_yticks(ticks=np.arange(0,5,1), labels=np.arange(1,6,1))
	if row == 3:
		ax[row][col].set_xticks(ticks=np.arange(0,5,1))
		ax[row][col].set_xticklabels(np.arange(1,6,1))		
		ax[row][col].set_xlabel('# moons')
	else:
		ax[row][col].set_xticks(ticks=np.arange(0,5,1))
		ax[row][col].set_xticklabels([])

	if col == 0:
		ax[row][col].set_yticks(ticks=np.arange(0,5,1))
		ax[row][col].set_yticklabels(np.arange(1,6,1))
	else:
		ax[row][col].set_yticks(ticks=np.arange(0,5,1))
		ax[row][col].set_yticklabels([])		


	if use_peaks_as_modes == 'y':
		if col == 0:
			ax[row][col].set_ylabel('# of peaks')
	else:
		if col == 0:
			ax[row][col].set_ylabel('# Gaussian modes')
	ax[row][col].set_title(moon_cols_labels[ncol-2])
	fig.colorbar(im, ax=ax[row][col])
	#ax.set_title(moon_cols_labels[ncol-2]) #### subtract two because the first two columns are system name and nmoons, and there are 
	if row==0:
		row+=1
	else:
		if row % 3 == 0:
			row = 0
			col += 1
		else:
			row+=1

plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

plt.savefig(projectdir+'/plots/all_columns_nmodes_vs_nmoons_heatmap.png', dpi=300)
plt.show()




	
		











