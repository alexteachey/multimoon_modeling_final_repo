import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import traceback


rcParams['font.family'] = 'serif'
rcParams.update({'font.size': 13})

def DWstat(residuals):
	numerator = np.nansum((residuals[1:] - residuals[:-1])**2) #### starts with the for [0,1,2,3,4] this would be [1,2,3,4] - [0,1,2,3].
	denominator = np.nansum(residuals**2)
	return numerator / denominator 


projectdir = '/Volumes/external2023/Documents/Projects/multimoon_modeling'
plotdir = projectdir+'/plots'
run_name = 'combined_runs'

with open(projectdir+'/'+run_name+'_poad.pickle', 'rb') as poad_pickle:
	poad = pickle.load(poad_pickle)

with open(projectdir+'/'+run_name+'_pmad.pickle', 'rb') as pmad_pickle:
	pmad = pickle.load(pmad_pickle)


good_planet_logz_idxs = np.where(np.abs(np.array(poad['planet_only_logz'])) < 1e4)[0].tolist() ### so we can append
#good_planet_logz_idxs = np.arange(0,len(poad['planet_only_logz']), 1)
planet_ndata = np.array(poad['ndata'])[good_planet_logz_idxs]
planet_logz = np.array(poad['planet_only_logz'])[good_planet_logz_idxs]
planet_sims = np.array(poad['sim'])[good_planet_logz_idxs]
planet_BIC = np.array(poad['BIC'])[good_planet_logz_idxs]
planet_iteration = 0
last_planet_DW_minus_2 = 100 
last_planet_logz_scatter = np.inf
keep_iterating = True


discarded_planet_ndata, discarded_planet_logz = [], []
discarded_moon_ndata, discarded_moon_logz = [], []


while keep_iterating:
	print('planet_iteration: ', planet_iteration)
	planet_logz_coeffs = np.polyfit(x=planet_ndata, y=planet_logz, deg=1)
	planet_logz_func = np.poly1d(planet_logz_coeffs)
	planet_logz_fit = planet_logz_func(planet_ndata)
	planet_logz_residual = planet_logz - planet_logz_fit
	planet_residuals_positive = np.where(planet_logz_residual > 0)[0]
	planet_residuals_negative = np.where(planet_logz_residual <= 0)[0]
	planet_DW = DWstat(planet_logz_residual)
	planet_DW_minus_2 = np.abs(planet_DW - 2) ### as close to zero
	planet_logz_scatter = np.nanstd(planet_logz_residual)

	#if planet_DW_minus_2 < last_planet_DW_minus_2: 
	#if planet_logz_scatter < last_planet_logz_scatter:
	#if np.abs(np.nanmedian(planet_logz_residual)) > np.nanstd(planet_logz_residual):
	if (len(planet_residuals_positive) > 1.05 * len(planet_residuals_negative)) or (len(planet_residuals_negative) > 1.05 * len(planet_residuals_positive)):
	
		planet_improved = True
		### update planet_DW_minus_2 
		#last_planet_DW_minus_2 = planet_DW_minus_2
		last_planet_logz_scatter = planet_logz_scatter

		print('planet_DW: ', planet_DW)
		print('len(planet_ndata): ', len(planet_ndata))
		#### add the worst indice
		worst_planet_idx = np.nanargmax(np.abs(planet_logz_residual))
		discarded_planet_ndata.append(planet_ndata[worst_planet_idx])
		discarded_planet_logz.append(planet_logz[worst_planet_idx])
		### remove it!
		planet_ndata = np.delete(planet_ndata, worst_planet_idx)
		planet_logz = np.delete(planet_logz, worst_planet_idx)
		planet_logz_fit = np.delete(planet_logz_fit, worst_planet_idx)
		planet_sims = np.delete(planet_sims, worst_planet_idx)
		planet_BIC = np.delete(planet_BIC, worst_planet_idx)
		planet_iteration += 1

	else:
		planet_improved = False 
		keep_iterating = False 



good_moon_logz_idxs = np.where(np.abs(np.array(pmad['planet_moon_logz'])) < 1e4)[0].tolist() ### so we can append
#good_moon_logz_idxs = np.arange(0,len(pmad['planet_moon_logz']), 1)
moon_ndata = np.array(pmad['ndata'])[good_moon_logz_idxs]
moon_logz = np.array(pmad['planet_moon_logz'])[good_moon_logz_idxs]
moon_sims = np.array(pmad['sim'])[good_moon_logz_idxs]
moon_BIC = np.array(pmad['BIC'])[good_moon_logz_idxs]
moon_DW = 0
moon_iteration = 0
last_moon_DW_minus_2 = 100
keep_iterating = True 




while keep_iterating:
	print('moon_iteration: ', moon_iteration)
	moon_logz_coeffs = np.polyfit(x=moon_ndata, y=moon_logz, deg=1)
	moon_logz_func = np.poly1d(moon_logz_coeffs)
	moon_logz_fit = planet_logz_func(moon_ndata)
	moon_logz_residual = moon_logz - moon_logz_fit
	moon_residuals_positive = np.where(moon_logz_residual > 0)[0]
	moon_residuals_negative = np.where(moon_logz_residual <=0)[0]
	moon_DW = DWstat(moon_logz_residual) 
	moon_DW_minus_2 = np.abs(moon_DW - 2)

	#if moon_DW_minus_2 < last_moon_DW_minus_2:
	#if np.abs(np.nanmedian(moon_logz_residual)) > np.nanstd(moon_logz_residual):
	if (len(moon_residuals_positive) > 1.05 * len(moon_residuals_negative)) or (len(moon_residuals_negative) > 1.05 * len(moon_residuals_positive)):
		moon_improved = True 
		#### update last_moon_DW_minus_2
		last_moon_DW_minus_2 = moon_DW_minus_2 

		print('moon_DW: ', moon_DW)
		worst_moon_idx = np.nanargmax(np.abs(moon_logz_residual))
		discarded_moon_ndata.append(moon_ndata[worst_moon_idx])
		discarded_moon_logz.append(moon_logz[worst_moon_idx])		
		moon_ndata = np.delete(moon_ndata, worst_moon_idx)
		moon_logz = np.delete(moon_logz, worst_moon_idx)
		moon_logz_fit = np.delete(moon_logz_fit, worst_moon_idx)
		moon_sims = np.delete(moon_sims, worst_moon_idx)
		moon_BIC = np.delete(moon_BIC, worst_moon_idx)
		moon_iteration += 1

	else:
		moon_improved = False 
		keep_iterating = False 


### plot logz against ndata
fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(8,8))
ax[0][0].scatter(planet_ndata, planet_logz, s=20, facecolor='#4059AD', edgecolor='k', alpha=0.5)
ax[0][0].scatter(discarded_planet_ndata, discarded_planet_logz, s=20, facecolor='white', edgecolor='k', alpha=0.5, marker='s')
ax[0][1].scatter(moon_ndata, moon_logz, s=20, facecolor='#97D8C4', edgecolor='k', alpha=0.5)
ax[0][1].scatter(discarded_moon_ndata, discarded_moon_logz, s=20, facecolor='white', edgecolor='k', alpha=0.5, marker='s')


#### line fits
ax[0][0].plot(planet_ndata, planet_logz_fit, color='k', linestyle='--', linewidth=2)
ax[0][1].plot(moon_ndata, moon_logz_fit, color='k', linestyle='--', linewidth=2)
ax[0][0].set_ylabel(r'$\log Z$')
ax[0][0].set_ylim(-6000,0)
ax[0][1].set_ylim(-6000,0)

ax[1][0].scatter(planet_ndata, planet_logz_residual, facecolor='#4059AD', edgecolor='k', alpha=0.5)
ax[1][0].plot(planet_ndata, np.linspace(0,0,len(planet_ndata)), color='k', linestyle='--', linewidth=2)
ax[1][1].scatter(moon_ndata, moon_logz_residual, facecolor='#97D8C4', edgecolor='k', alpha=0.5)
ax[1][1].plot(moon_ndata, np.linspace(0,0,len(moon_ndata)), color='k', linestyle='--', linewidth=2)
ax[1][0].set_ylabel('residual')
ax[1][0].set_xlabel('# data points [planet]')
ax[1][1].set_xlabel('# data points [moon]')
ax[1][0].set_ylim(-750,750)
ax[1][1].set_ylim(-750,750)
plt.savefig(plotdir+'/logz_vs_ndata_and_residual.png', dpi=300)
plt.show()


#### do the same thing with BIC
### plot logz against ndata
fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(8,8))
ax[0].scatter(planet_ndata, planet_BIC, s=20, facecolor='Blue', edgecolor='k', alpha=0.5)
ax[1].scatter(moon_ndata, moon_BIC, s=20, facecolor='Red', edgecolor='k', alpha=0.5)
ax[1].set_xlabel('# data')
ax[0].set_ylabel('BIC [planet]')
ax[1].set_ylabel('BIC [moon]')
plt.show()

#### line fits
"""
ax[0][0].plot(planet_ndata, planet_logz_fit, color='k', linestyle='--', linewidth=2)
ax[0][1].plot(moon_ndata, moon_logz_fit, color='k', linestyle='--', linewidth=2)
ax[0][0].set_ylabel(r'$\log Z$')
ax[0][0].set_ylim(-6000,0)
ax[0][1].set_ylim(-6000,0)

ax[1][0].scatter(planet_ndata, planet_logz_residual, facecolor='Blue', edgecolor='k', alpha=0.5)
ax[1][0].plot(planet_ndata, np.linspace(0,0,len(planet_ndata)), color='k', linestyle='--', linewidth=2)
ax[1][1].scatter(moon_ndata, moon_logz_residual, facecolor='Red', edgecolor='k', alpha=0.5)
ax[1][1].plot(moon_ndata, np.linspace(0,0,len(moon_ndata)), color='k', linestyle='--', linewidth=2)
ax[1][0].set_ylabel('residual')
ax[1][0].set_xlabel('# data [planet]')
ax[1][1].set_xlabel('# data [moon]')
ax[1][0].set_ylim(-750,750)
ax[1][1].set_ylim(-750,750)
plt.show()
"""




print('remaining planet sims: ')
for ps in planet_sims:
	print(ps)
print(' ')
print('remaining moon sims: ')
for ms in moon_sims:
	print(ms)
print(' ')
print('Systems where both survive: ')
if len(planet_sims) < len(moon_sims):
	shorter_list = planet_sims
	longer_list = moon_sims
else:
	shorter_list = moon_sims
	longer_list = planet_sims 

final_sim_list = []
final_BF_list = []
final_nmoons_list = []
for sim in shorter_list:
	if sim in longer_list:
		#### look up the bayes factor
		sim_idx = np.where(np.array(pmad['sim']) == sim)[0]
		sim_bf = np.array(pmad['bayes_factor'])[sim_idx]
		sim_nmoons = np.array(pmad['nmoons'])[sim_idx]
		print(sim+', BF: '+str(sim_bf))
		final_sim_list.append(sim)
		final_BF_list.append(sim_bf[0])
		final_nmoons_list.append(sim_nmoons[0])
np.save(projectdir+'/final_sim_list.npy', np.array(final_sim_list))
final_sim_list = np.array(final_sim_list)
final_BF_list = np.array(final_BF_list)
final_nmoons_list = np.array(final_nmoons_list)

len(np.where(final_BF_list[np.where(final_nmoons_list == 1)[0]] > 3.2)[0])



#### look at the delta-logZ distribution for these systems
#for fs in final_sim_list:




