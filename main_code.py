#### 1- Install multitaper PSD, arviz, FitSpectrum

import sys
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt
import buildsignal as bs  #

import nitime.utils as utils
from nitime.viz import winspect
from nitime.viz import plot_spectral_estimate
import arviz as az
from scipy.stats import norm

from fun_calc import * 

plt.ion()
np.random.seed(1000)  # set the random seed.

num_sample_post= 2000

###  Specify the signal
dt = 0.001
f, psd_mt, nu = calc_ken_data_generation(dt)

#### rsf - calc covariance matrix and multitaper calculation 
signal= psd_mt
calc_cov_pre_analysis(signal, dt)


#### rsf - generating syntehtic data
# async config

input_synthetic = {'a': 50,
                   'b': 4,
                   'c': 0,
                   
                   'mu_1': 5,
                   'sigma_1': 1,
                   'w_1': 5,
                   
                   'mu_2': 5,
                   'sigma_2': 1,
                   'w_2': 5,
                   
                   'noise_coef': 0.2, # not have an effect
                   'mean_noise': 0,
                   'var_noise': 1,
                   'ro_noise': 0.4,
                   
                   'num_bumps': 1,
                   'num_taper': 5,
                   
                   'low_fr': 1,
                   'high_fr': 10,
                   'step_fr': 0.5}

psd_data_n, freq_range = synthetic_data(input_synthetic)

plt_=1
if plt_:
    ro_str = "{:.2f}".format(input_synthetic['ro_noise'])
    
    # estimated_cov = (np.cov(psd_data_n))
    # plt.figure()
    # plt.imshow(estimated_cov)
    
    # plt.title('estimated cov matrix - ro = ' + ro_str)
    # plt.colorbar()
    # plt.colormaps()
    # plt.clim(0, 8) 
    # plt.show()
    
    if input_synthetic['num_taper']==10:
        mean_val = np.mean(psd_data_n, axis=0)
        resi_ = psd_data_n - mean_val
        max_val = np.max(np.abs(resi_))

        fig, axs = plt.subplots(5,2)
        fig.suptitle('residual  - rho=' +  ro_str)

        axs[0,0].plot(freq_range, resi_[0, :], 'r.')
        axs[0,0].set_title('1')
        axs[0,0].set_ylim(bottom= -max_val, top=max_val)
        axs[0,1].plot(freq_range, resi_[1, :], 'r.')
        axs[0,1].set_title('2')
        axs[0,1].set_ylim(bottom= -max_val, top=max_val)

        axs[1,0].plot(freq_range, resi_[2, :], 'r.')
        axs[1,0].set_title('3')
        axs[1,0].set_ylim(bottom= -max_val, top=max_val)
        axs[1,1].plot(freq_range, resi_[3, :], 'r.')
        axs[1,1].set_title('4')
        axs[1,1].set_ylim(bottom= -max_val, top=max_val)

        axs[2,0].plot(freq_range, resi_[4, :], 'r.')
        axs[2,0].set_title('5')
        axs[2,0].set_ylim(bottom= -max_val, top=max_val)
        axs[2,1].plot(freq_range, resi_[5, :], 'r.')
        axs[2,1].set_title('6')
        axs[2,1].set_ylim(bottom= -max_val, top=max_val)

        axs[3,0].plot(freq_range, resi_[6, :], 'r.')
        axs[3,0].set_title('7')
        axs[3,0].set_ylim(bottom= -max_val, top=max_val)
        axs[3,1].plot(freq_range, resi_[7, :], 'r.')
        axs[3,1].set_title('8')
        axs[3,1].set_ylim(bottom= -max_val, top=max_val)

        axs[4,0].plot(freq_range, resi_[8, :], 'r.')
        axs[4,0].set_title('9')
        axs[4,0].set_ylim(bottom= -max_val, top=max_val)
        axs[4,0].set_xlabel('freq (Hz)')
        axs[4,1].plot(freq_range, resi_[9, :], 'r.')
        axs[4,1].set_title('10')
        axs[4,1].set_ylim(bottom= -max_val, top=max_val)
        axs[4,1].set_xlabel('freq (Hz)')



k=1
#### run the model to fit using pymc3.
# trace, modelsim = do_fit_psd(f_psd, np.log(psd_mt[theseInds]))
if input_synthetic['num_taper']==1:
    method_solving = '1' 
else: 
    method_solving = 'M' 
# 1 > one-dimensional 
# M > multi-dimensional 

input_model = {
'method_solving': method_solving,
'num_bumps' : input_synthetic['num_bumps'], 
'mu_1': input_synthetic['mu_1'] - 3, 
'sd_1': input_synthetic['sigma_1']**2 + 2
}

if input_synthetic['num_bumps']==2:
    added_dict = { 
        'mu_2': input_synthetic['mu_2'] - 5, 
        'sd_2': input_synthetic['sigma_2']**2
        }
    
    input_model.update(added_dict)
    
if method_solving== 'M':
    added_dict = {'cov_mode': -1}
    
    input_model.update(added_dict)
    
data = psd_data_n
trace, modelsim = do_fit_psd(data, freq_range, input_model, num_sample_post)

#### rsf - visualization 
if method_solving == '1':
    y_fit = np.percentile(trace.gauss, 50, axis=0)
    
    mu_val_ = trace['gauss']
    mu_val = mu_val_[-1, :]
    
    data_fit = mu_val
    
    plt.figure()
    plt.plot(freq_range, data_fit, 'r', marker='+', ls='None', ms=5, mew=1, label='Fitted by model')
    plt.plot(freq_range, data.reshape((data.shape[1], )), label='synthetic data')
    plt.legend()
    # plt.plot(f_psd, np.log(psd_mt[theseInds]))
    plt.show()
    
elif method_solving == 'M':
    summary_mu_info = az.summary(trace, ['mu_'])
    mean_mu_tot = summary_mu_info['mean']
    sd_mu_tot = summary_mu_info['sd']
    
    plt.figure()

    data_plt = np.mean(data, axis=0)
    plt.plot(freq_range, data_plt, label='synthetic data')
    plt.plot(freq_range, mean_mu_tot, 'r', marker='+', ls='None', ms=5, mew=1, label='mean of fitted estimation')

    post_dist_minus = mean_mu_tot- 2*sd_mu_tot
    post_dist_add = mean_mu_tot+ 2*sd_mu_tot


    plt.plot(freq_range, post_dist_minus, 'k--', linewidth=0.5)
    plt.plot(freq_range, post_dist_add, 'k--', linewidth=0.5)
    plt.fill_between(freq_range, post_dist_minus, post_dist_add, color='red', alpha=0.05,  label='2 sd interval')
    plt.legend()

    plt.show()
    
k=1
    


# fig = plt.figure() 
# pm.traceplot(trace)
# plt.show()

# RANDOM_SEED = 100
# with modelsim:
#     ppc = pm.sample_posterior_predictive(
#             trace, var_names=['mu1', 'sig1', 'w1'], random_seed=RANDOM_SEED
#         )

#     az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=modelsim))


## 5- Look at marginal posteriors of each component

x_axis = np.arange(-2, 10, 0.001)

# prior dist
prior_dist = norm.pdf(x_axis, input_model['mu_1'], np.sqrt(input_model['sd_1']))

# posterior dist 
posterior_mu1_sample = trace['mu1']
post_mu, post_sigma = norm.fit(posterior_mu1_sample)
post_dist = norm.pdf(x_axis, post_mu, post_sigma)
post_dist = (post_dist/np.max(post_dist))*np.max(prior_dist)


plt.figure()
plt.plot(x_axis, prior_dist, color='red', label='prior data')
plt.plot(x_axis, post_dist, color='green', label='posterior dist')
# plt.axvline(x= mu_1,  'g--', label='synthetic value')
plt.title('f1- synthetic value =' + str(input_synthetic['mu_1']))
plt.legend()
plt.show()




if input_model['num_bumps'] ==2:
    x_axis = np.arange(0, 30, 0.001)
    prior_dist = norm.pdf(x_axis, input_model['mu_2'], np.sqrt(input_model['sd_2']))

    # posterior dist 
    posterior_mu_sample = trace['mu2']
    post_mu, post_sigma = norm.fit(posterior_mu_sample)
    post_dist = norm.pdf(x_axis, post_mu, post_sigma)
    post_dist = (post_dist/np.max(post_dist))*np.max(prior_dist)


    plt.figure()
    plt.plot(x_axis, prior_dist, color='red', label='prior data')
    plt.plot(x_axis, post_dist, color='green', label='posterior dist')
    # plt.axvline(x= mu_1,  'g--', label='synthetic value')
    plt.title('f2- synthetic value =' + str(input_synthetic['mu_2']))
    plt.legend()

    plt.show()
    
    az.plot_trace(trace, ['mu2'])




# az.plot_trace(trace, ['mu1'])
# az.plot_trace(trace, ['mu1'], kind="rank_bars")
# az.summary(trace, ['mu1'])
# az.plot_trace(trace)

if method_solving == '1':
    az.plot_trace(trace, ['mu1'])
    K=1        
    
elif method_solving == 'M':
    
    plt_post = 0 
    if plt_post: 
        # NUTS: [ro, s_d, mu1, sig1, w1, c, b, a]
        # az.plot_trace(trace, ['mu_'])
        # az.plot_trace(trace, ['sig1', 'w1'])
        # az.plot_trace(trace, ['ro', 's_d'])
        # az.summary(trace, ['ro', 's_d'])
        az.plot_trace(trace, ['ro_cov'])
        az.plot_trace(trace, ['sd_cov'])
        az.plot_trace(trace, ['mu1'])
        az.plot_trace(trace, ['sig1'])
        az.plot_trace(trace, ['w1'])
        az.plot_trace(trace, ['a'])
        az.plot_trace(trace, ['b'])
        az.plot_trace(trace, ['c'])
        
        # az.plot_trace(trace, ['a', 'b', 'c'])



k=1
x_axis = np.arange(-0.5, 1, 0.001)

# prior dist
# prior_dist = norm.pdf(x_axis, input_model['mu_1'], np.sqrt(input_model['sd_1']))

# posterior dist 
posterior_ro_sample = trace['ro_cov']
post_mu, post_sigma = norm.fit(posterior_ro_sample)
post_dist = norm.pdf(x_axis, post_mu, post_sigma)
post_dist = (post_dist/np.max(post_dist))*np.max(prior_dist)


plt.figure()
# plt.plot(x_axis, prior_dist, color='red', label='prior data')
plt.plot(x_axis, post_dist, color='green', label='posterior dist')
rho_str_mean = "{:.2f}".format(post_mu)
rho_str_var = "{:.2f}".format(post_sigma)
rho_str_synt= "{:.2f}".format(input_synthetic['ro_noise'])
# plt.axvline(x= mu_1,  'g--', label='synthetic value')
plt.title('rho - synthetic value =' + rho_str_synt + '  mean=' + rho_str_mean+ '* sigma= '+ rho_str_var)
plt.legend()
plt.show()

# post_dist_minus = post_dist-ro_sd
# post_dist_add = post_dist + ro_sd
# plt.figure()
# plt.plot(x_axis, post_dist, color='green', linewidth=2, label='posterior dist')
# plt.plot(x_axis, post_dist_minus, 'k--', linewidth=0.5, label='1 sd interval')
# plt.plot(x_axis, post_dist_add, 'k--', linewidth=0.5)
# plt.fill_between(x_axis, post_dist_minus, post_dist_add, color='yellow', alpha=0.1)
# plt.legend()
# plt.show()

# az.summary(trace)
# summary_info = az.summary(trace, ['ro__'])

# summary_info = az.summary(trace, ['ro_cov', 'sd_cov', 'mu1', 'sig1', 'w1', 'a', 'b', 'c'])
summary_info = az.summary(trace, ['ro_cov', 'sd_cov', 'mu1', 'sig1', 'w1', 'a', 'b'])
# summary_info = az.summary(trace, ['ro_cov', 'sd_cov', 'mu1', 'sig1', 'w1', 'a'])
# summary_info = az.summary(trace, ['ro_cov', 'sd_cov', 'mu1', 'sig1', 'w1'])
mean_info = summary_info['mean']
sd_info = summary_info['sd']

print('-------- cov - RHO  -------')
print('SYNTHETIC: %.2f'%(input_synthetic['ro_noise']))
ro_mean = mean_info['ro_cov']
ro_sd = sd_info['ro_cov']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(ro_mean, ro_sd))


print('-------- cov - SD  -------')
print('SYNTHETIC: %.2f'%(input_synthetic['var_noise']**2))
sd_mean =mean_info['sd_cov']
sd_sd =sd_info['sd_cov']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(sd_mean, sd_sd))


print('-------- bump = Center Frequency   -------')
print('SYNTHETIC: %.2f'%(input_synthetic['mu_1']))
mu1_mean =mean_info['mu1']
mu1_sd =sd_info['mu1']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(mu1_mean, mu1_sd))


print('-------- bump - Variance   -------')
print('SYNTHETIC: %.2f'%(input_synthetic['sigma_1']))
sig1_mean= mean_info['sig1']
sig1_sd= sd_info['sig1']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(sig1_mean, sig1_sd))


print('-------- bump - Coef   -------')
print('SYNTHETIC: %.2f'%(input_synthetic['w_1']))
w1_meam =mean_info['w1']
w1_sd =sd_info['w1']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(w1_meam, w1_sd))

print('-------- baseline - a   -------')
print('SYNTHETIC: %.2f'%(input_synthetic['a']))
a_mean =mean_info['a']
a_sd =sd_info['a']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(a_mean, a_sd))

print('-------- baseline - b   -------')
print('SYNTHETIC: %.2f'%(input_synthetic['b']))
b_mean =mean_info['b']
b_sd =sd_info['b']
print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(b_mean, b_sd))

# print('-------- baseline - c   -------')
# print('SYNTHETIC: %.2f'% (input_synthetic['c']))
# c_mean =mean_info['c']
# c_sd =sd_info['c']
# print('POSTERIOR: mean=%0.2f - sd=%0.2f'%(c_mean, c_sd))


post_cov_mean = np.zeros(shape=(freq_range.shape[0], freq_range.shape[0]))

summary_cov_info = az.summary(trace, ['cov_'])
mean_cov_info = summary_cov_info['mean']
sd_cov_info = summary_cov_info['sd']
for i in range(freq_range.shape[0]):
    for j in range(freq_range.shape[0]):
        str_cov = 'cov_[%d, %d]'%(i, j)
        cov_temp_mean = mean_cov_info[str_cov]
        cov_temp_sd = sd_cov_info[str_cov]

        post_cov_mean[i, j] = cov_temp_mean

plt.figure()
plt.imshow(post_cov_mean)
plt.colorbar()
plt.title('syntehtic rho= '+ str(input_synthetic['ro_noise']))
plt.show()

k=1