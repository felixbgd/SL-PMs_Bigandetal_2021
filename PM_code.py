#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on September 06 2021

@author: felixbigand
"""

################################################
#######   CODE FOR EXTRACTING THE PMS    #######
#######   SPONTANEOUS SL MOTION FROM     #######
#######   MOCAP1 CORPUS (PLOS ONE paper) #######
################################################

#%% IMPORT LIBRARIES AND SET PARAMETERS

from PLmocap.viz import *
from PLmocap.preprocessing import *
from PLmocap.classif import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from scipy import signal
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split 
import time
import numpy as np
import os
from pylab import *

# Data used to create segments in stick figures animations (dependent on MOCAP1-v2 corpus)
liaisons = [(0,3),(0,4),(3,4),(3,2),(4,1),(2,15),(2,9),(1,15),(1,9),(7,8),(8,6),\
            (6,5),(5,7),(15,16),(16,17),(16,18),(18,17),(17,19),(19,20),(20,18),\
            (9,10),(10,12),(10,11),(11,12),(12,14),(14,13),(13,11)]

##################### PARAMETERS ##################### 
# Signers of the MOCAP1-v2 corpus to import
signers = ['l2-ma.th','l6-viv','l8-lau','l4-pas','l1-dom','l3-cyr']
marker = ['o','s','+','*','.','x']   # for potential plots

# Number of mocap examples (images described) per signer, fps and number of markers
NB_IM=24
fps = 250
N_SENSORS = 21

# Trimming the mocap examples (DURation and start frame for each ex)
DUR=5
start = [7,5,5,4,5,6.5,4.5,4.5,5,5,5,7,5,6,6.5,5,5,6,6,6.5,6,6.5,5,5,\
         12,9,7,7,6,5.5,6,6,5,7,5,6,5,6,5,4,4,5,4,6,5,5,8,6,\
         5.5,6,5,10,6,6,5,7,6,5,12,9,5,6,12,4,8,7,6,5,5,13,17,6,\
         5,4,5,5.5,4,4,3,4,4,5,4.5,3,3,3.5,3.5,3.5,4,4,3,4,3.5,4,3.5,4,\
         4,3,4,4,9,7,5,5,5,5,4.5,5,5,5,5,5,5,6,6,4,5,5,9.5,6,\
         3,5,4,3,4.5,2.5,2,2,5,3,5,3,4,4,3,3,3,4,4,1,4,4,3,5]


# Create directory
output_dir = os.getcwd() + '/Tache1/PMs'
if not (os.path.exists(output_dir)) : os.mkdir(output_dir)

#%% IMPORT MOCAP DATA


data_signers = np.zeros((len(signers),3*N_SENSORS,NB_IM*fps*DUR));
Nsign=0
for s in signers :    
    folder = os.getcwd() + '/Tache1/' + s + '/mocap/c3d'
    files=os.listdir(folder);      files=[x for i,x in enumerate(files) if (x.endswith(".c3d"))]
    files=sorted(files)
    
    Nim=0
    xyz_average=[]
    for im in files :

        data = np.load(os.getcwd() + '/Tache1/' + s + '/mocap/npy_2/' + im[0:-4] +'_data.npy')
        labels = np.load(os.getcwd() + '/Tache1/' + s + '/mocap/npy_2/' + im[0:-4] +'_labels.npy')
        
        if len(labels) >= 23 :    # If statement to assert that the data includes all wanted markers                       
            # Remove one of the two elbow markers for each arm (does not add info and makes visualization confusing)
            elbowL = data[11,:,:];  elbowR = data[18,:,:]
            data[11,:,:] = elbowL;  data[18,:,:] = elbowR
            labels[11] = 'elbowL';  labels[18] = 'elbowR'
            data=np.delete(data,[9,16],0);   labels=np.delete(labels,[9,16],0) 
            
            # Change name of label for the pelvis marker
            labels[0] = 'pelv'            
            
            # Trim the mocap examples
            t_start=start[Nsign*NB_IM + Nim]
            tStart = int(t_start*250);    tEnd = tStart + int(DUR*250)
            t = data[0,0,tStart:tEnd];     xyz = data[:,1:4,tStart:tEnd]
            joints = [i for i,x in enumerate(labels) if x.find("Face")==-1]
            numFrame = len(data[0,0,:])     
            
            # Define the pelvis marker as the origin of the reference system
            Ox = data[0,1,:];   Oy = data[0,2,:];   Oz = data[0,3,:]
            xyz[:,0,:] -= xyz[0,0,:];   xyz[:,1,:] -= xyz[0,1,:];   xyz[:,2,:] -= xyz[0,2,:]
            data[:,1,:] -= Ox;   data[:,2,:] -= Oy;   data[:,3,:] -= Oz
            
            sz = xyz.shape
            xyz_vec = np.reshape(xyz, (sz[0]*sz[1],sz[2]))
                
            # Delete IM07 of each signer bc unavailable from Signer l8-lau 
            if im[1:3] != '07':
                data_signers[Nsign,:,Nim*fps*DUR:(Nim+1)*fps*DUR]=xyz_vec
                
                Nim+=1
                   
    Nsign+=1
               
#%% MOCAP DATA PROCESSING (SIZE, SHAPE & POSTURE NORMALIZATIONS, GLOBAL TO LOCAL POSITIONS)
    
colors = ['blue','orange','green','red','c','purple']
signers = ['Signer 1', 'Signer 2', 'Signer 3', 'Signer 4','Signer 5','Signer 6']
markers = ['o','s','+','*','.','x']
    
ref_posture_mean = np.mean(data_signers,2)   # average posture

############## NORMALIZATION OF ANTHROPOMETRICS (Federolf et al. 2013) ##############
data_signers_NORM = np.zeros(data_signers.shape)
pmean = np.zeros((len(signers) , data_signers.shape[1]));  dmean=np.zeros((len(signers) , 1))
for i in range(0,len(signers)):
    pmean[i,:] = ref_posture_mean[i,:]
    data_signers_NORM[i,:,:] = data_signers[i,:,:] - np.reshape(pmean[i,:],(-1,1)) 
    dmean[i,:] = np.mean( np.linalg.norm(data_signers[i,:,:],axis=0) )
    data_signers_NORM[i,:,:] = data_signers_NORM[i,:,:] / dmean[i,:]

#%% COMPUTE COMMON PRINCIPAL MOVEMENTS (on the whole dataset with 6 signers)

fig_all = plt.figure(figsize=(8,8))
ax_all = fig_all.gca()
    
# Combine data into a matrix usable for PCA
pos_mat=data_signers_NORM[0,:,:]
for i in range(1,len(signers)):
    pos_mat= np.hstack((pos_mat,data_signers_NORM[i,:,:]))
pos_mat = pos_mat.T

# Apply PCA using Singular Value Decomposition
U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
e=S**2
common_nrj = np.cumsum(e) / np.sum(e);     nbEigen = [i for (i, val) in enumerate(common_nrj) if val>0.95][0];
common_PC_scores = (U*S)
common_eigen_vects = V

if common_nrj[0]!=0 : common_nrj = np.hstack((np.zeros((1)) , common_nrj))        # create a first point (0,0) for potential graphs

# Plot cumulative explained variance by the PCs
fig=plt.figure(figsize=(10,10));    ax=fig.gca()
ax.bar(np.arange(1,16),100*np.diff(common_nrj[:16]), color='tab:blue');
ax.set_ylabel("% of explained variance", fontsize=19);   ax.set_xlabel("Number of PMs",fontsize=19); ax.set_ylim((0,30));   ax.set_xlim(((0,16)))
ax.set_xticks(np.arange(1,16));   ax.set_xticklabels(np.arange(1,16));   ax.tick_params(labelsize=15)
fig.savefig(output_dir + '/common_nrj_bar.eps', dpi=600, bbox_inches='tight'); plt.close()  

###### VALIDITY OF THE PM DECOMPOSITION  ######
###### SPECTRAL ANALYSIS OF THE PMs ######
# Compute Power Spectral Density (PSD) using Welch method
Nwin=250
freqs, psd = signal.welch(common_PC_scores[:,:8],fps,axis=0,nperseg=Nwin,noverlap=3*Nwin//4)

# PLOT PSD
fig = plt.figure(figsize=(12,12));
st = fig.suptitle("Frequency content of the first 8 common PMs", fontweight='bold')
for i in range(8):
    ax = fig.add_subplot(4,2,i+1)
    plt.plot(freqs,psd[:,i],c='tab:blue',label='mean');  
    plt.xticks(np.arange(min(freqs), max(freqs)+1, 5));  ax.set_xlim(0,20); 
    plt.title('PM ' + str(i+1),fontweight='bold')
    if (i+1)%2==1: plt.ylabel('PSD (amp**2/Hz)')
    if i==6 or i == 7 : plt.xlabel('Frequency (Hz)')
fig.tight_layout()
fig.savefig(output_dir + '/spectral_analysis.eps', dpi=600, bbox_inches='tight'); plt.close()  

# FILTER the PC_scores
fc = 6  # Cut-off frequency of the filter
w = fc / (fps / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')
common_PC_scores = signal.filtfilt(b, a, common_PC_scores, axis=0)


###### LOO CROSS-VALIDATION ######
# Compute the PCs when leaving the data of one signer out
common_eigen_vects_cross = np.zeros(((len(signers),) + common_eigen_vects.shape))
for s in range(len(signers)):
    data_signers_crossval = np.delete(data_signers_NORM, s, axis=0).copy()
    
    pos_mat=data_signers_crossval[0,:,:]
    for i in range(1,data_signers_crossval.shape[0]):
        pos_mat= np.hstack((pos_mat,data_signers_crossval[i,:,:]))
    pos_mat = pos_mat.T
    
    U, S, V = np.linalg.svd(pos_mat, full_matrices=False)
    common_eigen_vects_cross[s,:,:] = V

# Compute the angle between each leave-one-out PCk and the all-signers PCk
angles_cross = np.zeros((len(signers) , common_eigen_vects.shape[0]))
for s in range(len(signers)):
    for e in range(common_eigen_vects.shape[0]):
        angle = arccos(np.abs ( clip( np.dot(common_eigen_vects[e,:],common_eigen_vects_cross[s,e,:]) / ( np.linalg.norm(common_eigen_vects[e,:]) * np.linalg.norm(common_eigen_vects_cross[s,e,:]) ) , -1, 1) ) ) 
        angles_cross[s,e] = np.rad2deg(angle)

angles_cross_mean = np.mean(angles_cross,axis=0)

#%% SYNTHESIZE THE COMMON PMs IN VIDEOS ######

video=0
if video==1:
    IMAGE=0     # mocap example (image described) to synthesize
    for SIGNER in range(len(signers)):
        for k in range(8) :  
            print('S' + str(SIGNER+1) + ': PM' + str(k+1))
            # Reconstruct motion data from the PMs, and inverse normalization
            common_eigenmov = np.outer(common_PC_scores[:,k] , common_eigen_vects[k,:]).T
            common_eigenmov[:,(SIGNER*Nim)*DUR*fps:(SIGNER*Nim+24)*DUR*fps] = common_eigenmov[:,(SIGNER*Nim)*DUR*fps:(SIGNER*Nim+24)*DUR*fps] * dmean[i,:]
            common_eigenmov = common_eigenmov + np.reshape(np.mean(pmean,axis=0),(-1,1)) 
            common_eigenmov = common_eigenmov[:,(SIGNER*Nim+IMAGE)*DUR*fps:(SIGNER*Nim+IMAGE+1)*DUR*fps]
            
            # Downsample for video
            if fps != 25 : 
                samps = int(DUR*25)
                common_eigenmov_ds=np.zeros((common_eigenmov.shape[0],samps))  
                for i in range(common_eigenmov_ds.shape[0]): 
                    common_eigenmov_ds[i,:]=np.interp(np.linspace(0.0, 1.0, samps, endpoint=False), np.linspace(0.0, 1.0,  common_eigenmov.shape[1], endpoint=False), common_eigenmov[i,:])
                common_eigenmov = common_eigenmov_ds
            
            common_eigenmov = np.reshape( common_eigenmov , (N_SENSORS, 3, DUR*25 ) )
            
            # Synthesize PL video
            maxXZ = (common_eigenmov[:,[0,2],:]).max()*1.3;
            maxY = 1;
            if not (os.path.exists(output_dir + '/S' + str(SIGNER+1) + '_IM' + str(IMAGE+1) + '_common_eigenmov' + str(k+1) +'_FRONTAL.mp4')) :
                video_PL(common_eigenmov,output_dir + '/S' + str(SIGNER+1) + '_IM' + str(IMAGE+1) + '_common_eigenmov' + str(k+1) +'_FRONTAL.mp4',maxX=maxXZ, maxZ=maxXZ)
            if not (os.path.exists(output_dir + '/S' + str(SIGNER+1) + '_IM' + str(IMAGE+1) + '_common_eigenmov' + str(k+1) +'_SAGITTAL.mp4')) :
                video_PL(common_eigenmov,output_dir + '/S' + str(SIGNER+1) + '_IM' + str(IMAGE+1) + '_common_eigenmov' + str(k+1) +'_SAGITTAL.mp4', plan="YZ", maxY=maxY, maxZ=maxXZ)
    
    
#%% VISUALIZATION OF THE COMMON PMs AS 2D PLOTS ######################
            
pos_viz=1
if pos_viz==1:
    ##### VISUALIZE THE N FIRST PMs (2-post graph with the min and max PM postures across signers) #####
    for k in range(13):
        EXAG = 1
        common_eigenmov = np.outer(EXAG*common_PC_scores[:,k] , common_eigen_vects[k,:]).T        
        for i in range(len(signers)) :
            common_eigenmov[:,(i*Nim)*DUR*fps:(i*Nim+24)*DUR*fps] = common_eigenmov[:,(i*Nim)*DUR*fps:(i*Nim+24)*DUR*fps] * dmean[i,:]
        common_eigenmov = common_eigenmov + np.reshape(np.mean(pmean,axis=0),(-1,1)) 
        
        common_eigenmov = np.reshape( common_eigenmov , sz[:2] + (common_PC_scores.shape[0],) )
        
        # plot postures at min and max of the PM weightings
        i_max = argmax(common_PC_scores[:,k]);   i_min = argmin(common_PC_scores[:,k])
        times=[i_min,i_max]
        
        plot_2frames(common_eigenmov,times,"XZ",liaisons=liaisons,save_dir=output_dir + '/common_PM' + str(k+1) + '_2post_XZ.pdf'); plt.close()
        plot_2frames(common_eigenmov,times,"YZ",liaisons=liaisons,save_dir=output_dir + '/common_PM' + str(k+1) + '_2post_YZ.pdf'); plt.close()
#        plot_2frames(common_eigenmov,times,"XY",liaisons=liaisons,save_dir=output_dir + '/PM' + str(k+1) + '_2post_XY.pdf'); plt.close()


#%% COMPUTE INDIVIDUAL PRINCIPAL MOVEMENTS
    
indiv_PC_scores_list = []; indiv_eigen_vects_list = []; indiv_nbEigen_list = [];
indiv_nrj_list=[]
fig_all = plt.figure(figsize=(8,8))
ax_all = fig_all.gca()
for i in range(len(signers)):
    
    # Data matrix usable for PCA
    indiv_pos_mat=data_signers_NORM[i,:,:].T
    
    # Apply PCA using Singular Value Decomposition
    U, S, V = np.linalg.svd(indiv_pos_mat, full_matrices=False)
    e=S**2
    indiv_nrj = np.cumsum(e) / np.sum(e);     indiv_nbEigen = [i for (i, val) in enumerate(indiv_nrj) if val>0.95][0]; indiv_nbEigen_list.append(indiv_nbEigen)
    indiv_nrj_list.append(indiv_nrj)
    indiv_eigen_vects_list.append( V )
    indiv_PC_scores_list.append( (U*S) )
    
    # Filter the PC scores
    fc = 6  # Cut-off frequency of the filter
    w = fc / (fps / 2) # Normalize the frequency
    b, a = signal.butter(4, w, 'low')
    indiv_PC_scores_list[i] = signal.filtfilt(b, a, indiv_PC_scores_list[i], axis=0)
    
    
    # Plot cumulative explained variance by the PCs for each signer
    fig_ORI = plt.figure(figsize=(8,8))
    ax = fig_ORI.gca();  ax.set_ylabel("Cumulative information of PCs");     ax.set_xlabel("Number of PCs")
    ax.plot(indiv_nrj,'*');   ax.set_ybound(0,1); ax.set_title('Signer ' + str(i+1))
    fig_ORI.savefig(output_dir + '/indiv_nrj_S' + str(i+1) + '.pdf', bbox_inches='tight'); plt.close()
    plt.close()
    
    # Plot cumulative information of PCs of all signers in 1 plot
    ax_all.plot(np.arange(1,2*indiv_nbEigen+1),indiv_nrj[:2*indiv_nbEigen],marker=marker[i], linestyle='-', c=colors[i], label=signers[i], alpha=0.6);
    
ax_all.legend(fontsize=12); ax_all.set_ylabel("Cumulative information of PCs");     ax_all.set_xlabel("Number of PCs")
fig_all.savefig(output_dir + '/indiv_nrj_all.pdf', bbox_inches='tight'); plt.close()   

# Plot cumulative explained variance by the PCs as a mean bar plot
indiv_nrj_list = np.asarray(indiv_nrj_list);        
if indiv_nrj_list[0,0]!=0 : indiv_nrj_list = np.hstack((np.zeros((Nsign,1)) , indiv_nrj_list))        #créer un premier point (0,0) pour les graphes
indiv_nrj_mean = np.mean(indiv_nrj_list,0);   indiv_nrj_std = np.std(indiv_nrj_list,0)

fig=plt.figure(figsize=(10,10));    ax=fig.gca()
stderr=np.std(np.diff(indiv_nrj_list,1),0)/np.sqrt(indiv_nrj_list.shape[0])
ax.bar(np.arange(1,16),100*np.diff(indiv_nrj_mean)[:15],yerr=100*stderr[:15], color='tab:blue',capsize=2);
ax.set_ylabel("% of explained variance", fontsize=19);   ax.set_xlabel("Number of PMs", fontsize=19); ax.set_ylim((0,35));   ax.set_xlim(((0,16)))
ax.set_xticks(np.arange(1,16));   ax.set_xticklabels(np.arange(1,16));  ax.tick_params(labelsize=15)
fig.savefig(output_dir + '/indiv_nrj.eps', dpi=600, bbox_inches='tight'); plt.close() 

#%% SYNTHESIZE THE INDIVIDUAL PMs IN VIDEOS ######

video=0
if video==1:
    IMAGE=0     # mocap example (image described) to synthesize
    for SIGNER in range(len(signers)) :
        indiv_PC_scores = indiv_PC_scores_list[SIGNER]; indiv_eigen_vects = indiv_eigen_vects_list[SIGNER]; indiv_nbEigen = indiv_nbEigen_list[SIGNER]
        
        for k in range(8) :
            print('S' + str(SIGNER+1) + ': PM' + str(k+1))
            # Reconstruct motion data from the PMs, and inverse normalization
            indiv_eigenmov = np.outer(indiv_PC_scores[:,k] , indiv_eigen_vects[k,:]).T  
            indiv_eigenmov = indiv_eigenmov * dmean[SIGNER,:]
            indiv_eigenmov = indiv_eigenmov + np.reshape(np.mean(pmean,axis=0),(-1,1)) 
            indiv_eigenmov = indiv_eigenmov[:,IMAGE*DUR*fps:(IMAGE+1)*DUR*fps]
            
            # Downsample for video
            if fps != 25 : 
                samps = int(DUR*25)
                indiv_eigenmov_ds=np.zeros((indiv_eigenmov.shape[0],samps))  
                for i in range(indiv_eigenmov_ds.shape[0]): 
                    indiv_eigenmov_ds[i,:]=np.interp(np.linspace(0.0, 1.0, samps, endpoint=False), np.linspace(0.0, 1.0,  indiv_eigenmov.shape[1], endpoint=False), indiv_eigenmov[i,:])
                indiv_eigenmov = indiv_eigenmov_ds
            
            indiv_eigenmov = np.reshape( indiv_eigenmov , (N_SENSORS, 3, DUR*25 ) )
            
            # Synthesize the PL Video
            maxXZ = (indiv_eigenmov[:,[0,2],:]).max()*1.3;
            maxY = 1;
            if not (os.path.exists(output_dir + '/S' + str(SIGNER+1) + '/IM' + str(IMAGE+1) + '_eigenmov' + str(k+1) +'_FRONTAL.mp4')) :
                video_PL(indiv_eigenmov,output_dir + '/S' + str(SIGNER+1) + '/IM' + str(IMAGE+1) + '_eigenmov' + str(k+1) +'_FRONTAL.mp4',maxX=maxXZ, maxZ=maxXZ)
            if not (os.path.exists(output_dir + '/S' + str(SIGNER+1) + '/IM' + str(IMAGE+1) + '_eigenmov' + str(k+1) +'_SAGITTAL.mp4')) :
                video_PL(indiv_eigenmov,output_dir + '/S' + str(SIGNER+1) + '/IM' + str(IMAGE+1) + '_eigenmov' + str(k+1) +'_SAGITTAL.mp4', plan="YZ", maxY=maxY, maxZ=maxXZ)
    

#%% VISUALIZATION OF THE INDIVIDUAL PMs AS 2D PLOTS ######################

pos_viz=1
if pos_viz==1:
    ##### VISUALIZE THE N FIRST PMs (2-post graph with the min and max PM postures of the signer) #####
    for SIGNER in range(6):
        IMAGE=0
    
        indiv_PC_scores = indiv_PC_scores_list[SIGNER]; indiv_eigen_vects = indiv_eigen_vects_list[SIGNER]; indiv_nbEigen = indiv_nbEigen_list[SIGNER]
        for k in range(13):
            indiv_eigenmov = np.outer(indiv_PC_scores[:,k] , indiv_eigen_vects[k,:]).T
            indiv_eigenmov = indiv_eigenmov * dmean[SIGNER,:]
            indiv_eigenmov = indiv_eigenmov + np.reshape(np.mean(pmean,axis=0),(-1,1)) 

            indiv_eigenmov = np.reshape( indiv_eigenmov , sz[:2] + (indiv_PC_scores.shape[0],) )
        
            # plot 2 important postures
            i_max = argmax(indiv_PC_scores[:,k]);   i_min = argmin(indiv_PC_scores[:,k])
            
            times=[i_min,i_max]
            plot_2frames(indiv_eigenmov,times,"XZ",liaisons=liaisons,save_dir=output_dir + '/S' + str(SIGNER+1) + '/PM' + str(k+1) + '_2post_XZ.pdf'); plt.close()
            plot_2frames(indiv_eigenmov,times,"YZ",liaisons=liaisons,save_dir=output_dir + '/S' + str(SIGNER+1) + '/PM' + str(k+1) + '_2post_YZ.pdf'); plt.close()
    #        plot_2frames(eigenmov,times,"XY",liaisons=liaisons,save_dir=output_dir + '/PM' + str(k+1) + '_2post_XY.pdf'); plt.close()

#%% ASSESSING THE SIMILARITY BETWEEN INDIV AND COMMON PMs ######################
    
   
######## COSINE SIMILARITY BETWEEN THE INDIVIDUAL AND COMMON PMs ########
sim = np.zeros((len(signers),8,8))
for SIGNER in range(len(signers)):
    for PMi in range(8):
        for PMj in range(8):
            indiv_eigenmov = np.outer(indiv_PC_scores_list[SIGNER][:,PMi] , indiv_eigen_vects_list[SIGNER][PMi,:]).T
            common_eigenmov = np.outer(common_PC_scores[(SIGNER*Nim)*DUR*fps:(SIGNER*Nim+24)*DUR*fps,PMj] , common_eigen_vects[PMj,:]).T

            indiv_eigenmov = np.ndarray.flatten(indiv_eigenmov)
            common_eigenmov = np.ndarray.flatten(common_eigenmov)
            
            sim[SIGNER,PMi,PMj] = np.dot( indiv_eigenmov , common_eigenmov ) / ( np.linalg.norm( indiv_eigenmov ) * np.linalg.norm( common_eigenmov ) )
        
# PLOT SIMILARITY MATRIX #
import seaborn as sn
import pandas as pd
pm_labels = ['PM1','PM2','PM3','PM4','PM5','PM6','PM7','PM8']
for SIGNER in range(len(signers)):
    df_sim = pd.DataFrame(sim[SIGNER,:,:], pm_labels, pm_labels)
    #df_sim=df_sim.replace(0.0, nan)
    fig = plt.figure(figsize=(8,8))
    ax=fig.gca()
    sn.set(font_scale=2)
    annot_labels = np.round(sim[SIGNER,:,:], 2).astype(str)
    annot_labels[sim[SIGNER,:,:] < 0.2] = ""
    
    cmap = sn.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    cbar = sn.heatmap(df_sim, annot=annot_labels, fmt ="s" , square=True, linewidths=.5, annot_kws={"size": 13},vmin=0, vmax=1, cmap = cmap,  cbar_kws={"shrink": .4,'label': 'Cosine similarity'})
    cbar.figure.axes[-1].yaxis.label.set_size(16);    cbar.collections[0].colorbar.ax.tick_params(labelsize=12)
    plt.xticks(rotation=0) 
    cbar.set_xticklabels(cbar.get_xmajorticklabels(), fontsize = 14)
    cbar.set_yticklabels(cbar.get_xmajorticklabels(), fontsize = 14)
    plt.yticks(rotation=0) 
    ax.set_xlabel('Common PMs', labelpad=10, fontsize=19)
    ax.set_ylabel('Individual PMs of Signer ' + str(SIGNER +1), labelpad=10, fontsize=19)
    fig.savefig(output_dir + '/similarity_signer' + str(SIGNER+1) + '.eps', dpi=600, bbox_inches='tight'); plt.close() 


######## CROSS-PROJECTION SIMILARITY BETWEEN INDIVIDUAL PM SUBSPACES ########
nbPM = indiv_eigen_vects_list[0].shape[0]
crossproj_sim = np.zeros((nbPM,len(signers),len(signers)))
for pm in range(nbPM):
    for SIGNER_1 in range(len(signers)):
        for SIGNER_2 in range(len(signers)):
            motion_1 = data_signers_NORM[SIGNER_1,:,:].T
            explained_variance_1 = indiv_nrj_list[SIGNER_1][pm+1]
            
            PC_scores_project_1on2 = np.matmul(motion_1 , indiv_eigen_vects_list[SIGNER_2].T)
            explained_variance_2 = np.sum( np.std(PC_scores_project_1on2[:,:pm+1] , axis=0)**2 / np.sum( np.std(PC_scores_project_1on2,axis=0)**2 ) )
            
            crossproj_sim[pm,SIGNER_1,SIGNER_2] = explained_variance_2 / explained_variance_1

# Compute the mean and SD similarities across signers for each pm added
nondiag = ~np.eye(crossproj_sim[pm,:,:].shape[0],crossproj_sim[pm,:,:].shape[1],dtype=bool)
crossproj_sim_mean = np.zeros((nbPM)); crossproj_sim_sem = np.zeros((nbPM))
for pm in range(nbPM):
    crossproj_sim_mean[pm] = np.mean(crossproj_sim[pm,:,:][nondiag])
    crossproj_sim_sem[pm] = np.std(crossproj_sim[pm,:,:][nondiag]) / np.sqrt(crossproj_sim[pm,:,:][nondiag].shape[0])
    
# Plot cross-projection similarity as a function of PMs added
fig=plt.figure(figsize=(18,10)); ax=fig.gca()
ax.errorbar(np.arange(1,len(crossproj_sim_mean)+1),crossproj_sim_mean,crossproj_sim_sem,linewidth=2.5,capsize=3,capthick=2)
ax.vlines(8, 0, crossproj_sim_mean[7], color='red', linewidth=2.5,linestyle='--', label='First 8 PMs');   ax.set_xlabel("PMs added", fontsize=25);  ax.set_ylabel("Proportion of similarity", fontsize=25)    
ax.set_xlim((0,len(crossproj_sim_mean)+1)); ax.set_ylim((0.6,1.005));ax.tick_params(labelsize=20)
plt.legend(fontsize=17);  
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
fig.savefig(output_dir + '/cross_projection_similarity.eps', dpi=600, bbox_inches='tight'); 
plt.close() 
