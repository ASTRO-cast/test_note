
#This plotting function is very similar to the Autoregression function. The only difference is the method in which the predictions are calculated.
import sys
import fnmatch
import pyvo as vo
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.contrib.gp as gp
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import os
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)
import glob


#The load function below simply loads a numpy array into X and Y variables so the data can be plotted and also used in the guassian
#processes to forecast the VCI values. For this code there are only two available regeions as a different data set is used for the 
#guassian processes modelling. 




        
def load(name):
    resultset = 0
    service = vo.dal.TAPService("https://herschel-vos.phys.sussex.ac.uk/__system__/tap/run/tap")

    myquery = """

    SELECT *
    FROM astrocast.main
    WHERE locationkey = 

    """
    resultset = service.search(myquery + str(name))
    X  = resultset["astrocast_time"]# days since 1/1/2000
    y  = resultset["vci3m"]# VCI
    X =np.asarray(X[0:1011])
    y =np.asarray(y[0:1011])
    return X, y


#This function simply asks the user to pick a region. This code is flexible as if more data is added in the future this will 
#automatically show it to the user as it scans the folder appropriate data. 


def which_region():

    num_reg = int(input("Please select a region index number: "))
    return num_reg
    
            



#The code below plots a graph of VCI against time using the data collected. IT makes ticks on the x-axis to show where each year is.
#It also shows a solid black line at VCI=35 as this is the cutoff to where the region is to be considered in drought. 


def plot_vci(X,y):
    plt.figure(figsize=(17, 7))
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue')
    plt.xlabel('Date', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [5114,5479,5844,6210,6575,6940]
    plt.xticks(x_ax, ('1/1/2014','1/1/2015', '1/1/2016', '1/1/2017','1/1/2018','1/1/2019'), size = 18)
    plt.xlim(5000,7200)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.show()
    
    
#This following function uses a python module called pyro to perform gaussian process modelling. It essentially finds the distribution
#of all possible functions that could be fitted to the data. This allows for an acurate fit and therefore prediction. 
#It also allows for an error to be given to within a confidence interval. 


#It allows for different kernals to be used. In a gaussian process different kernals can be used based on what the process is being 
#used to model. In this model a Radial basis function kernal has been used. 
#The lengthscale for the calculation is then chosen, this essentially decides how smooth the function will be. 
#(The larger the lengthscale the smoother)
    
    
def run_GP(X,y):
    k1 = gp.kernels.RBF(input_dim=2, lengthscale=torch.tensor(50.0),\
                   variance = torch.tensor(0.5))
    #k1.set_constraint("lengthscale", torch.distributions.constraints.interval(7.,1000.))
    smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
    pyro.enable_validation(True)       # can help with debugging
    optim = Adam({"lr": 0.01}) 
    
    pyro.clear_param_store()

    plus_arr = np.max(X)+np.array([7,14,21,28,35,42,49,56])

    X2 = (torch.from_numpy(X))
    y2 = (torch.from_numpy(y-np.mean(y)))

    Xtest_use = np.append(X,plus_arr)
    Xtest_use2 = (torch.from_numpy(Xtest_use))



    gpr = gp.models.GPRegression(X2, y2,k1, noise=torch.tensor(0.01))

    svi = SVI(gpr.model, gpr.guide, optim, loss=Trace_ELBO())
    losses = []

    num_steps = 10

    for k in range(num_steps):
        losses.append(svi.step())



    with torch.no_grad():
      if type(gpr) == gp.models.VariationalSparseGP:
        mean, cov = gpr(Xtest_use2, full_cov=True)
      else:
        mean, cov = gpr(Xtest_use2, full_cov=False, noiseless=False) 

    sd = cov.sqrt().detach().numpy()
    mean = mean.detach().numpy()+np.mean(y)
    
    return mean, Xtest_use 
    
    
    
    
#The following two functions are just used to plot the data using matplotlib.The first creates a graph of the weekly VCI against time
#with a forecast being plotted after the last data point. The predictions are plotted in red with error bars showing the error for
#each week forecasted. The second function does the same, just this time using the 3 month VCI data and not the weekly data. 
#This leads to much smoother graph due to the lower data resolution. 
    
    
    
def plot_vci_fc(Xtest_use,mean,X,y):

    plt.figure(figsize=(17, 7))
    use = Xtest_use >= np.max(X)
    err = np.std(y)
    eb = err*np.array([0,0.5*0.46,0.46,0.46/2+0.66/2,0.66,0.66/2+0.81/2,0.81,0.91,1.0])
    #plt.fill_between(Xtest_use[use],mean[use]-eb,mean[use]+eb, \
    #        color = 'red', label = 'GP prediction')
    plt.errorbar(Xtest_use[use], mean[use], yerr=eb,color='red',lw=3,label='Forecast')

    
    plt.plot(X,y, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI')
    plt.xlabel('Date', size = 20)
    plt.ylabel('weekly VCI', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7150)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    plt.legend(prop={'size': 20}, loc = 3)
    plt.show()
    fc = Xtest_use - np.max(X)
    use = fc > 0
    fc = fc[use]
    for i in range(np.size(fc)):
        num = mean[use][i]
        print('weeks = ',int(fc[i]/7),',   VCI = ',"%.0f" % num)

        
def plot_vci_fc3M(Xtest_use,mean,X,y):
    yf_new = np.zeros(np.size(mean))
    y_new = np.zeros(np.size(y))
    for i in range(12,np.size(mean)):
        yf_new[i] = np.mean(mean[i-12:i])
    for i in range(12,np.size(y)):
        y_new[i] = np.mean(y[i-12:i]) 
        
    plt.figure(figsize=(17, 7))
    use = Xtest_use >= np.max(X)
    err = np.std(y_new)
    eb = err*np.array([0,0.5*0.08,0.08,0.08/2+0.2/2,0.2,0.2/2+0.35/2,0.35,0.42,0.49])
    #plt.fill_between(Xtest_use[use],yf_new[use]-eb,yf_new[use]+eb, \
    #        color = 'red', label = 'GP prediction')
    plt.errorbar(Xtest_use[use], yf_new[use], yerr=eb,color='red',lw=3,label='Forecast')

    #plt.plot(Xtest_use,mean+3*sd, linestyle = '--', lw = 3, color = 'red')
    #plt.plot(Xtest_use,mean-3*sd, linestyle = '--', lw = 3, color = 'red')


    plt.plot(X,y_new, linestyle = 'solid', lw = 3, color = 'blue',label = 'Landsat VCI3M')

    plt.xlabel('Date', size = 20)
    plt.ylabel('VCI3M', size = 20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    x_ax = [6575,6665,6756,6848,6940,7030,7121]
    plt.xticks(x_ax, ('1/1/2018','1/4/2018','1/7/2018','1/10/2018','1/1/2019','1/4/2019','1/7/2019'), size = 18)
    plt.xlim(6575,7150)
    plt.ylim(0,100)

    plt.plot([0,8000],[35,35],color = 'black', lw = 3)
    plt.plot([np.max(X),np.max(X)],[0,100],linestyle = '--',color = 'black', lw = 3,\
            label = 'day of last observation')
    
    
    plt.legend(prop={'size': 20}, loc = 3)

    plt.show()

    fc = Xtest_use - np.max(X)
    use = fc > 0
    fc = fc[use]

    for i in range(np.size(fc)):
        num = yf_new[use][i]
        print('weeks  = ',int(fc[i]/7),',   VCI3M = ',"%.0f" % num)       