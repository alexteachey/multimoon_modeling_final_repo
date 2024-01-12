from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandoramoon as pandora 
from pandoramoon.helpers import ld_convert, ld_invert 


rcParams['font.family'] = 'serif'
rcParams.update({'font.size': 13})



claret_path = '/Users/hal9000/Documents/Projects/TESS_CVZ_project_2023REDUX/reference_files/Claret_table5.txt'
projectdir = '/Users/hal9000/Documents/Projects/multimoon_modeling/files_from_Garvit/exomoon_project'
plotdir = projectdir

claret = np.genfromtxt(claret_path)
claret_shape = claret.shape ### = (574, 10)

claret_columns = ['logg','Teff','Z', 'L/HP', 'a', 'b', 'mu', 'chi2', 'od', 'Sys']

### ldcs are idx=4,5
### useful params are logg and teff (idxs=0,1)

loggs = claret.T[0]
teffs = claret.T[1]
ldc_as = claret.T[4]
ldc_bs = claret.T[5]

q1s, q2s = ld_invert(ldc_as, ldc_bs)

#### plot these in 2D space with the third dimension colorcoded 




fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))

cm1 = plt.cm.get_cmap('Spectral')
im1 = ax[0].scatter(ldc_as, ldc_bs, c=loggs, cmap=cm1, marker='o', edgecolor='k', s=30, zorder=0)
ax[0].set_xlabel('LDC a')
ax[0].set_ylabel('LDC b')
fig.colorbar(im1, ax=ax[0], label=r'$\log g$')

cm2 = plt.cm.get_cmap('bwr')
im2 = ax[1].scatter(ldc_as, ldc_bs, c=teffs, cmap=cm2, marker='o', edgecolor='k', s=30, zorder=0)	
ax[1].set_xlabel("LDC a")
ax[1].set_ylabel("LDC b")
fig.colorbar(im2, ax=ax[1], label=r'$T_{\mathrm{eff}}$')
plt.subplots_adjust(left=0.164, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
plt.savefig(plotdir+'/Claret_quadratic_limb_darkening_a_and_b.png', dpi=300)
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,8))

cm1 = plt.cm.get_cmap('Spectral')
im1 = ax[0].scatter(q1s, q2s, c=loggs, cmap=cm1, marker='o', edgecolor='k', s=30, zorder=0)
ax[0].set_xlabel(r'$q_1$')
ax[0].set_ylabel(r'$q_2$')
fig.colorbar(im1, ax=ax[0], label=r'$\log g$')

cm2 = plt.cm.get_cmap('bwr')
im2 = ax[1].scatter(q1s, q2s, c=teffs, cmap=cm2, marker='o', edgecolor='k', s=30, zorder=0)	
ax[1].set_xlabel(r'$q_1$')
ax[1].set_ylabel(r'$q_2$')
fig.colorbar(im2, ax=ax[1], label=r'$T_{\mathrm{eff}}$')
plt.subplots_adjust(left=0.164, bottom=0.11, right=0.9, top=0.95, wspace=0.2, hspace=0.2)
plt.savefig(plotdir+'/Claret_quadratic_limb_darkening_q1_and_q2.png', dpi=300)
plt.show()






