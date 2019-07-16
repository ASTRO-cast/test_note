#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:10:57 2019

@author: abb22
"""
#Imports the modules needed, numpy for easy numerical computing, matplotlib to create graphs and glob to find files. 

import numpy as np
from matplotlib import pyplot as plt
import glob
import sys
import os
import pyvo as vo


#The code below searches through the im_note directory for any files ending in 'npy' as these are the numpy aray data files
#with the raw data. This then displays the user with the different regions which are number coded. You can then pick a region to load. 

def which_region():
    num_reg = int(input('Please Pick a number. 2,3,6,7 are invalid choices'))
    return num_reg

            
#The variable region will then be passed into this function. This will load the npy file and split the 2D array into 3
#different 1D arrays containing the time and VCI data. 

def load(name):
    service = vo.dal.TAPService("https://herschel-vos.phys.sussex.ac.uk/__system__/tap/run/tap")

    myquery = """
    SELECT astrocast_time,vci,vci3m
    FROM astrocast.main
    WHERE locationkey = 
    """
    
    resultset = service.search(myquery + str(name))
    T  = resultset["astrocast_time"]# days since 1/1/2000
    VCI  = resultset["vci"]# VCI
    VCI3M = resultset["vci3m"]
    T = np.asarray(T)
    VCI = np.asarray(VCI)
    VCI3M = np.asarray(VCI3M)
    return T, VCI, VCI3M 


#Once the region's data has been loaded it will then be passed through to this function to plot the graph. 
#This function limits the plot to between 01/01/2014 and 01/02/2019. 
#It also creates a marker for when each year has passed and a line at VCI = 35. This line defines when a region is said to be in drought.
#This function is used for two different data sets to produce graphs for weekly VCI and 3 monthly VCI.


def plot_vci(X,y,index):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)
    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.show()
    

#The Following funciton below creates the VCI forecast and error attached to this forecast. The main purpose of this function is to set 
#the variables that need to be used later in the linear autoregression functions below. This also defines hwo far in advance you want
#the forecast to be.

    
def forecast(VCI):
    VCImean=np.nanmean(VCI)                           #This sets the average of the VCI
    VCIz=VCI-VCImean                                  #This sets the difference of the VCI from the mean.  
    nlags0=3                                          #This sets the number of weeks of data that will be collected to train the model
    trainlength=200                                   #This sets the number of iterations the model will train for in latter functions. 
    l=len(VCI)                                        #This sets a variable = to the length of the data array.
    
    VCIpred=np.zeros((9,l))                           #The following few lines create empty arrays for the forecast values and errors
    Forecast=np.zeros(9)
    Sigma=np.zeros(9)
    Forecast[0]=VCI[l-1]                              #This sets the first value in the forecast array to be equal to the final value                                                              measured for VCI                                 
    
    for i in range(0,8):                               #The loop below uses the regression functions below to build a forecast for the
        Y=VCIz[i:]                                     #Next 8 weeks and returns the VCI value with error. 
        X=VCIz[0:l-i]
        ret=astro_predict_one(Y,X,nlags0,trainlength)
        ypred=ret[1]
        VCIpred[i,trainlength+i:]=ypred
        VCIpred[i,:]=VCIpred[i,:]+VCImean
        Forecast[i+1]=ret[3]+VCImean
        Sigma[i+1]=ret[0]
    
    return Forecast, Sigma    


#The following function uses the autoregression function to actually create the forecast for the VCI values. It begins by creating
#empty arrays for the predictions to be stored in. The function then sets the amount of iterations for the 'Training' of the algorithm and
#begins to run the training loop. This calls on the auto regress function to actually perform the calculations. The function also handles
#the appearance of 'nan' within the data. It then outputs the the forecast 


def astro_predict_one(Y,X,nlags,trainlength):

    nobs=len(Y)
    ntests=nobs-trainlength
    
    ypred=np.zeros(ntests)
    u=np.zeros(ntests)
    
    nopredict=0
    
    for k in range(ntests):
        ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
        beta=ret[0]
        
        predictors = np.zeros(nlags)
        for tau in range(nlags): 
            predictors[tau] = X[k+trainlength-tau-1]
                
        ypred[k]=np.dot(predictors,beta)
        u[k]=Y[k+trainlength]-ypred[k]
        if np.isnan(u[k]):
            nopredict=nopredict+1
        
    respredict=np.sqrt(np.nanvar(u))
       
    k=ntests
    ret=astro_regress_one(Y[k:k+trainlength],X[k:k+trainlength],nlags)
    beta=ret[0]
        
    predictors = np.zeros(nlags)
    for tau in range(nlags): 
        predictors[tau] = X[k+trainlength-tau-1]
                
    forecast=np.dot(predictors,beta)     
    
    return respredict, ypred, nopredict, forecast




#The following function handles the actualy calculations of the auto-regression method. 



    
def astro_regress_one(Y,X,nlags):
    
    nobs=len(Y)
    
    Xsegs=[]
    Ysegs=[]
    segstart=0
    nsegs=0
    
    for t in range(nobs-1):                                             
        if not np.isnan(X[t]) and not np.isnan(Y[t]):
            if np.isnan(X[t+1]) or np.isnan(Y[t+1]):
                if t+1-segstart>nlags:
                    Xsegs.append(X[segstart:t+1])
                    Ysegs.append(Y[segstart:t+1])
                    nsegs=nsegs+1
        if np.isnan(X[t]) or np.isnan(Y[t]):
            if not np.isnan(X[t+1]) and not np.isnan(Y[t+1]):
                segstart=t+1
                             
    if not np.isnan(X[nobs-1]) and not np.isnan(Y[nobs-1]):
        if nobs-segstart>nlags:
            Xsegs.append(X[segstart:nobs])
            Ysegs.append(Y[segstart:nobs])
            nsegs=nsegs+1
        
    nobs=0
    for i in range(nsegs):
        nobs=nobs+len(Xsegs[i])
                             
    regressors = np.zeros((nobs-nsegs*nlags,nlags))
    ydep=np.zeros(nobs-nsegs*nlags)
    
    segstart=0
                             
    for i in range(nsegs):  
        XX=Xsegs[i]
        YY=Ysegs[i]
        nobsseg=len(XX)
        ydep[segstart:segstart+nobsseg-nlags] = YY[nlags:]
        for tau in range(nlags): 
            regressors[segstart:segstart+nobsseg-nlags,tau] = XX[nlags-tau-1:nobsseg-tau-1]
        segstart=segstart+nobsseg-nlags
       
    beta=np.zeros(nlags)
    ypred=np.zeros(nobs-nsegs*nlags)
    u=np.zeros(nobs-nsegs*nlags)                          
                             
    regrees = np.linalg.lstsq(regressors,ydep)
    beta=regrees[0]
    ypred = np.dot(regressors,beta)  # keep hold of predicted values
    u = ydep-ypred
    res=np.cov(u)
    
    return beta, u, res, ypred
    
    
    
    
#The function below is just used to output the graph with the forecast. This is limited to between 2014 and 2019 to give a better
#resolution. A dotted line is plotted on the time axis at the date 01/02/19 and the forecast and error bars are plotte after said 
#line in red. This function will aslo print out the VCI forecast for the next 8 weeks.  
    
    
def plot_vci_fc(X,y,Forecast,Sigma,index):
    
    n=len(X)
    nw=len(Forecast)
    x1=np.arange(X[n-1],X[n-1]+7*nw,7)

#    f=np.zeros(4)
#    f[0]=Forecast
#    f[1]=Forecast[1]
#    f[2]=Forecast[3]
#    f[3]=Forecast[5]
#    
#    s=np.zeros(4)
#    s[0]=0
#    s[1]=Sigma[1]
#    s[2]=Sigma[3]
#    s[3]=Sigma[5]

    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue',label = 'data')
    plt.errorbar(x1, Forecast, yerr=Sigma,color='red',lw=3,label='Forecast')
    #plt.fill_between(x1,Forecast-Sigma,Forecast+Sigma, \
    #        color = 'red', label = 'Forecast')
    plt.xlabel('Date', size = 20)
    plt.ylabel(index, size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7200)
    plt.ylim(0,100)

    plt.plot([0,7200],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    plt.legend(prop={'size': 20},loc=1) 
    plt.show()
    
    print('Forecast:')
    for i in range(1,nw):
        print('weeks ahead =',i,',',index,'=',"%.0f" % Forecast[i])
    

