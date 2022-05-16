"""
Script for the output analysis of cerebellar scaffold EBCC simulations
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import os
import sim_output
import scipy.stats
plt.interactive(True)


#Extracting simulation parameters from network configuration (.json) file
with open('mouse_cerebellum_config_healthy.json') as fl:
    all_data = json.load(fl)

first = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['start_first']
n_trials = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['n_trials']
between_start = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['between_start']
last = first + between_start*(n_trials-1)
burst_dur = all_data['simulations']['DCN_update']['devices']['CS']['parameters']['burst_dur']
burst_dur_us = all_data['simulations']['DCN_update']['devices']['US']['parameters']['burst_dur']
burst_dur_cs = burst_dur- burst_dur_us
trials_start = np.arange(first, last+between_start, between_start)

selected_trials = np.linspace(1,100,100).astype(int) #Can specify trials to be analyzed

maf_step = 100 #selected step for moving average filter when computing motor output from DCN SDF

threshold = 3.9 #6th trial of DeOude2020 - 70% CRs, value based on sdf_maf_max_dcn output

# All cell names:
# 'basket', 'dcn', 'gaba', 'glomerulus', 'gly', 'golgi', 'grc', 'io', 'mossy', 'pc', 'stellate'


""" Calculations for SDF, SDF with moving average filter, ISI CV, conditioned response latency, incidence, ..."""


#Compute SDF per one selected trial. Returns SDF firing rate of each cell at each time instant
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - integer indicating the trial.
def sdf(file, cell, trial):
    fname = file + '.hdf5'
    f = h5py.File(fname)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>=trials_start[trial]-50) & (spk[:,1]<trials_start[trial]+burst_dur+50)]
    spk_first[:,1] -= trials_start[trial]-50
    dur = burst_dur+100

    sdf_full = np.empty([len(neurons),int(dur)])
    sdf = []
    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(dur)):
            tau_first = t-spike_times_first
            sdf_full[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)

        sdf.append(sdf_full[neu][50:330])

    return(sdf)


#Compute mean SDF for each trial. SDF values are averaged across cells, returns mean firing rate at each time instant
#of a trial
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - integer indicating the trial.
def sdf_mean(file, cell, trial):
    sdf = sim_output.sdf(file, cell, trial)
    sdf_mean = np.mean(sdf, axis=0)

    return(sdf_mean)


#Compute mean SDF during baseline (outside the CS time window)
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def sdf_baseline(file, cell):
    fname = file + '.hdf5'
    f = h5py.File(fname)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>trials_start[0]+burst_dur) & (spk[:,1]<=trials_start[1])]
    spk_first[:,1] -= trials_start[0]+burst_dur

    sdf = np.empty([len(neurons),int(between_start-burst_dur)])

    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(between_start-burst_dur)):
            tau_first = t-spike_times_first
            sdf[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)
    sdf = np.mean(sdf, axis=1)

    return(sdf)


#Compute mean SDF change during the last 10 trials of the simulation. Mean change is computed by subtracting
#mean firing rate during the first 100 ms of a trial (for PCs) or during baseline (for DCN) from the firing rate
#of each cell during the LTD window (150-200 ms of a trial).
##file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def sdf_change(file, cell):
    sdf_change = []
    if cell == 'dcn':
        base_sdf = np.mean(sim_output.sdf_baseline(file, cell))
    for i in range(91,100):
        sdf = sim_output.sdf(file, cell, i)
        sdf_change_trial = []
        for neuron in range(len(sdf)):
            if cell == 'pc':
                baseline_sdf = sdf[neuron][:100]
                avg_baseline_sdf = np.mean(baseline_sdf)
            elif cell == 'dcn':
                avg_baseline_sdf = base_sdf
            current_sdf_change = np.sum(sdf[neuron][150:200]-avg_baseline_sdf)/50
            sdf_change_trial.append(current_sdf_change)
        sdf_change.append(np.array(sdf_change_trial))

    return(sdf_change)


#Compute SDF with moving average filter.
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - trial - integer indicating the trial,
#step - time step for convolution in ms
def sdf_maf(file, cell, trial, step):
    sdf_maf = np.convolve(sim_output.sdf_mean(file, cell, trial), np.ones(step), 'valid') / step
    return(sdf_maf)


#Compute coefficient of variation of the inter spike interval (ISI CV)
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def cv(file, cell):
    fname = file + '.hdf5'
    f = h5py.File(fname)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])
    spk = spk[(spk[:,1]>trials_start[0]+burst_dur) & (spk[:,1]<=trials_start[1])]
    neurons = np.unique(spk[:,0])

    cvs = []
    for i in range(len(neurons)):
        single_spikes = []
        for j in range(spk.shape[0]):
            if spk[j][0] == neurons[i]:
                single_spikes.append(spk[j][1])
        isi = np.diff(single_spikes)
        mu, std = isi.mean(), isi.std()
        cv = std / mu
        cvs.append(cv)

    return(cvs)


#Extract maximum values of SDF during each trial, and split the array into 10 blocks of 10 trials.
#Used for CR threshold selection.
#file - file name (without '.hdf5')
def sdf_maf_max_dcn(file):
    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio-1)

    baseline = np.mean(sim_output.sdf_baseline(file, 'dcn'))
    sdf_maf_max_all = []
    for j in range(1,n_trials):
        sdf_maf = sim_output.sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline
        sdf_maf = sdf_maf[isi_start:isi_end]
        sdf_maf_max = np.max(sdf_maf)
        sdf_maf_max_all.append(sdf_maf_max)
    sdf_maf_max_all = np.split(np.asarray(sdf_maf_max_all), 10)

    return(sdf_maf_max_all)


#Calculate conditioned responses for each block of 10 trials. Returns 0 if no CR, 1 in presence of CR.
#Criteria for CR: 1) CR threshold is reached no earlier than after the first 100 ms of a trial; 2) after crossing
#the CR threshold, the motor output has to stay above the CR threshold for 75% of the remaining time until US.
#file - file name (without '.hdf5')
def cr(file):

    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio)
    baseline = np.mean(sim_output.sdf_baseline(file, 'dcn'))
    over_threshold = []
    for j in selected_trials:
        sdf_maf = sim_output.sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline

        sdf_maf_pre_cs = sdf_maf[:isi_start]
        sdf_maf_cs = sdf_maf[isi_start:isi_end]

        sdf_maf_pre_cs_over = sdf_maf_pre_cs[sdf_maf_pre_cs >= threshold]
        if len(sdf_maf_pre_cs_over) > 0:
            over_threshold.append(0)
        elif len(sdf_maf_pre_cs_over) == 0:
            sdf_maf_win_over = sdf_maf_cs[sdf_maf_cs >= threshold]
            if len(sdf_maf_win_over) == 0:
                over_threshold.append(0)
            elif len(sdf_maf_win_over) > 0:
                for i in range(len(sdf_maf_cs)):
                    if sdf_maf_cs[i] >= threshold:
                        onset_index = i
                        break
                sdf_maf_cs_onset = sdf_maf_cs[onset_index:]
                if len(sdf_maf_win_over) >= len(sdf_maf_cs_onset)*0.75:
                    over_threshold.append(1)
                else:
                    over_threshold.append(0)

    over_threshold = np.split(np.asarray(over_threshold), 10)
    return(over_threshold)


#CR onset latency. Returns the time points from which the motor output begins to consistently rise until reaching
#the CR threshold. Trials in which no CR was produced are indicated as 0.
#file - file name (without '.hdf5')
def onset_latency(file):
    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio)

    baseline = np.mean(sim_output.sdf_baseline(file, 'dcn'))
    ol_all = []
    for j in range(1,n_trials):
        sdf_maf = sim_output.sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline
        sdf_maf_cs = sdf_maf[isi_start:isi_end]
        sdf_maf_pre = sdf_maf[:isi_start]
        sdf_maf_pre_over = sdf_maf_pre[sdf_maf_pre>=threshold]

        if len(sdf_maf_pre_over) > 0:
            ol_all.append(0)
        elif len(sdf_maf_pre_over) == 0:
            sdf_maf_cs_over = sdf_maf_cs[sdf_maf_cs>=threshold]
            if len(sdf_maf_cs_over) == 0:
                ol_all.append(0)
            elif len(sdf_maf_cs_over) > 0:
                for i in range(len(sdf_maf_cs)):
                    if sdf_maf_cs[i] >= threshold:
                        onset_index = i
                        break
                sdf_maf_cs_onset = sdf_maf_cs[onset_index:]
                if len(sdf_maf_cs_over) < len(sdf_maf_cs_onset)*0.75:
                    ol_all.append(0)
                elif len(sdf_maf_cs_over) >= len(sdf_maf_cs_onset)*0.75:
                    for i in range(len(sdf_maf)):
                        if sdf_maf[i] >= threshold:
                            thr_index = i
                            break
                    sdf_to_thr = sdf_maf[:thr_index]
                    sdf_to_thr_diff = np.diff(sdf_to_thr)
                    for k in range(len(sdf_to_thr_diff)):
                        if sdf_to_thr_diff[k] > 0:
                            sdf_to_thr_diff_k = sdf_to_thr_diff[k:-1]
                            sdf_to_thr_diff_k_positive = sdf_to_thr_diff_k[sdf_to_thr_diff_k>0]
                            if len(sdf_to_thr_diff_k) == len(sdf_to_thr_diff_k_positive):
                                ol_time = k+1
                                ol_all.append(np.round((isi_end-ol_time) / sdf_maf_ratio))
                                break
    ol_all = np.array(ol_all)
    return(ol_all)



""" Plot simulation output: SDF, motor output, SDF change, SDF baseline, ISI CV,
    percentages of conditioned responses, raster plots"""


#Graph colors for main cells to be plotted
color_pc = all_data["cell_types"]["purkinje_cell"]["plotting"]["color"]
color_dcn = all_data["cell_types"]["dcn_cell_glut_large"]["plotting"]["color"]
color_io = all_data["cell_types"]["io_cell"]["plotting"]["color"]


#Plot SDF curves in all trials
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc')
def plot_sdf(file, cell):
    if cell == 'pc':
        clr = color_pc
    elif cell == 'dcn':
        clr = color_dcn
    else:
        clr = 'blue'

    for j in selected_trials:
        sdf = sim_output.sdf_mean(file, cell, j)
        plt.figure(file+' {} SDF'.format(cell.upper()))
        plt.title('{} SDF'.format(cell.upper()))
        plt.ylim([20,190])
        sdf_plot = plt.plot(sdf)
        cc = 0.75-j/(max(selected_trials)*4/3)
        rgb_range = [[0,230/255], [100/255,240/255], [0/255,230/255]]
        rc=rgb_range[0][1]-(j/(max(selected_trials)*4/3))*(rgb_range[0][1]-rgb_range[0][0])
        gc=rgb_range[1][1]-(j/(max(selected_trials)*4/3))*(rgb_range[1][1]-rgb_range[1][0])
        bc=rgb_range[2][1]-(j/(max(selected_trials)*4/3))*(rgb_range[2][1]-rgb_range[2][0])
        plt.setp(sdf_plot, color=[rc,gc,bc])
        #plt.setp(sdf_plot, color=clr, alpha=0.1+0.007*j)
        if j == selected_trials[-1]:
            sdf_baseline = np.mean(sim_output.sdf_baseline(file, cell))
            sdf_baseline = [[0, burst_dur], [sdf_baseline, sdf_baseline]]
            plt.plot(sdf_baseline[0], sdf_baseline[1], color="black", linestyle = "dashed")
            plt.axvline(x=burst_dur_cs, color="red")
            plt.xlabel("Time [ms]")
            plt.ylabel("SDF [Hz]")
        #plt.xlim([50,280])
        plt.savefig(file+"_"+cell+"_SDF.svg")
        plt.show()


#Plot motor output curves in all trials
#file - file name (without '.hdf5')
def plot_motor_output(file):
    clr = color_dcn

    baseline = np.mean(sim_output.sdf_baseline(file, 'dcn'))
    for j in selected_trials:
        sdf_maf = sim_output.sdf_maf(file, 'dcn', j, maf_step)
        sdf_maf -= baseline

        plt.figure(file+" {} SDF + moving average filter".format('dcn'.upper()))
        plt.title("Motor output")
        plt.xlabel("Time [ms]")
        plt.ylabel("Motor output")
        plt.ylim([-11,19])

        sdf_maf_plot = plt.plot(sdf_maf)

        #plt.setp(sdf_maf_plot, color=clr, alpha=0.1+0.007*j)
        cc = 0.75-j/(max(selected_trials)*4/3)
        plt.setp(sdf_maf_plot, color=[cc,cc,cc])
        if j == selected_trials[-1]:
            axis = [[0, burst_dur-maf_step], [0, 0]]
            plt.plot(axis[0], axis[1], color="black", linestyle = "dashed")
            us_start = (burst_dur-burst_dur_us)*((burst_dur-maf_step)/burst_dur)
            plt.axvline(x=us_start, color="red")
            cr_threshold = [[0, burst_dur-maf_step], [threshold, threshold]]
            plt.plot(cr_threshold[0], cr_threshold[1], color="cyan")
        #plt.xlim([33,181])
        plt.savefig(file+"_motor_output_full.svg")
        plt.show()


#Plot CR incidence per 10-trial block.
#file - file name (without '.hdf5'), params - integer from 0 to 3, which indicates CR curve parameters (higher number -
#darker gray color)
def plot_cr(file, params):
    if params == 0:
        curve_params = ['black', "o", "white", "black"]
    elif params == 1:
        curve_params = ['black', "^", "lightgray", "black"]
    elif params == 2:
        curve_params = ['black', "s", "darkgray", "black"]
    elif params == 3:
        curve_params = ['black', "p", "dimgray", "black"]
    plt.figure("CR incidence")
    plt.title("CR incidence")
    plt.ylim([-5,105])
    plt.xlabel("Block")
    plt.ylabel("% CR")
    x = range(1,11)
    crs = sim_output.cr(file)
    y = []
    for i in range(len(crs)):
        crs_over = crs[i][crs[i]>0]
        crs_trial = len(crs_over)*10
        y.append(crs_trial)
    plt.plot(x, y, color=curve_params[0], marker = curve_params[1], mfc=curve_params[2], mec=curve_params[3])
    plt.xticks(x)
    plt.show()


#Colors used for onset latency, SDF baseline, SDF change and ISI CV plots.
colorh = 'white'
colorp1 = 'lightgray'
colorp2 = 'darkgray'
colorp3 = 'dimgray'


#Plot onset latency barplot
#file - file name (without '.hdf5'), clr - bar color, name - x axis label
def plot_onset_latency(file, clr, label):
    plt.figure("CR onset latency")
    plt.title("CR onset latency")
    ol_raw = sim_output.onset_latency(file)
    ol_raw = ol_raw[ol_raw > 0]
    x = label
    y = np.mean(ol_raw)
    err = np.std(ol_raw)
    plt.ylim([-5, 280])
    plt.bar(x, y, yerr = err, color=clr, edgecolor = 'black', width=0.5, capsize = 5)
    plt.ylabel("Time before US [ms]")
    plt.show()


#Plot SDF baseline as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_sdf_baseline(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_baseline")
    plt.title("{} baseline firing rate".format(cell.upper()))
    y = []
    for i in range(len(files)):
        baseline = sim_output.sdf_baseline(files[i], cell)
        mean_baseline = baseline
        y.append(mean_baseline)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sim_output.colorh
    elif len(files) == 2:
        colors = [sim_output.colorh, sim_output.colorp1]
    elif len(files) == 3:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2]
    elif len(files) == 4:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2, sim_output.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([60,110])
    #elif cell == 'dcn':
        #plt.ylim([50,100])
    plt.ylabel("Mean baseline firing rate")
    plt.savefig(files[1]+"_"+cell+"_baseline.svg")
    plt.show()
#E. g., sim_output.plot_sdf_baseline(['healthy', 'pathology1', 'pathology2'], ['Healthy', 'Pathology1', 'Pathology2'], 'pc')


#Plot ISI CV as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_cv(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_cv")
    plt.title("{} ISI CV".format(cell.upper()))
    y = []
    for i in range(len(files)):
        cv = sim_output.cv(files[i], cell)
        mean_cv = cv
        y.append(mean_cv)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sim_output.colorh
    elif len(files) == 2:
        colors = [sim_output.colorh, sim_output.colorp1]
    elif len(files) == 3:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2]
    elif len(files) == 4:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2, sim_output.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([0.1,1.5])
    #elif cell == 'dcn':
        #plt.ylim([0.1,0.7])
    plt.ylabel("Mean ISI CV")
    plt.savefig(files[1]+"_"+cell+"_cv.svg")
    plt.show()
#E. g., sim_output.plot_cv(['healthy', 'pathology1'], ['Healthy', 'Pathology1'], 'pc')


#Plot SDF change as boxplots (up to 4 files). Colors specified above are used by default.
#files - file name(s) (without '.hdf5'), labels - x axis label(s), cell - cell name (e. g., 'pc')
def plot_sdf_change(files, labels, cell):
    plt.figure(files[1]+"_"+cell+"_sdf_change")
    plt.title("{} SDF Change".format(cell.upper()))
    y = []
    for i in range(len(files)):
        sdf_change = sim_output.sdf_change(files[i], cell)
        mean_change = np.mean(sdf_change, axis=1)
        y.append(mean_change)
    medianprops = dict(linewidth = 2, color='firebrick')
    meanprops = dict(linewidth = 2, color='#00aeef', linestyle='-')
    bplot = plt.boxplot(y, labels = labels, patch_artist = True, showmeans=True, meanline = True, medianprops = medianprops, meanprops = meanprops)
    if len(files) == 1:
        colors = sim_output.colorh
    elif len(files) == 2:
        colors = [sim_output.colorh, sim_output.colorp1]
    elif len(files) == 3:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2]
    elif len(files) == 4:
        colors = [sim_output.colorh, sim_output.colorp1, sim_output.colorp2, sim_output.colorp3]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    #if cell == 'pc':
        #plt.ylim([-35,15])
    #elif cell == 'dcn':
        #plt.ylim([-5,15])
    plt.ylabel("Mean SDF change")
    plt.savefig(files[1]+"_"+cell+"_sdf_change.svg")
    plt.show()
#E. g., sim_output.plot_sdf_change('healthy', 'Healthy', 'dcn')


#Raster plot for one trial.
#file - file name (without '.hdf5'), cell - cell name (e. g., 'pc'), trial - trial - integer indicating the trial,
#window: 0 - only the CS time window, 1 - CS time window + pause (until the next trial)
def plot_spikes(file, cell, trial, window):
    if cell == 'pc':
        clr = color_pc
    elif cell == 'dcn':
        clr = color_dcn
    elif cell == 'io':
        clr = color_io
    elif cell == 'glomerulus':
        clr = 'gray'
    elif cell == 'mossy':
        clr = 'gray'

    fname = file + '.hdf5'
    f = h5py.File(fname)
    spk = np.array(f['recorders/soma_spikes/record_{}_spikes'.format(cell)])
    if window == 0:
        spk = spk[(spk[:,1]>trials_start[trial]-100) & (spk[:,1]<=trials_start[trial]+burst_dur+100)]
    elif window == 1:
        spk = spk[(spk[:,1]>trials_start[trial]-100) & (spk[:,1]<=trials_start[trial+1])]

    plt.figure(file +' '+ cell.upper() + ' Raster trial no. ' + str(trial), figsize = (12,6))
    plt.title(cell.upper() + ' Spikes')
    plt.scatter(spk[:,1], spk[:,0], s=5, color=clr)
    plt.axvline(x=trials_start[trial], color="red")
    plt.axvline(x=trials_start[trial]+burst_dur_cs, color="red")
    plt.axvline(x=trials_start[trial]+burst_dur, color="red")
    if window == 0:
        plt.xlim([1000,1480])
    elif window == 1:
        plt.xlim([1000,2100])
    plt.show()
