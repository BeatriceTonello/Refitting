import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os
import pickle
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from keras import callbacks
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
from keras import callbacks
from keras import backend as K
import os
import random
import tf_keras as k3
import warnings
np.warnings = warnings

# os.environ['PYTHONHASHSEED'] = '42'
# random.seed(42)
# np.random.seed(42)
# tf.compat.v1.set_random_seed(42)
# tf.random.set_seed(42)

@tf.function(reduce_retracing=True)
def fast_predict(model, x):
    return model(x, training=False)

class GWBSpectrum:


    def __init__(self, pathway_principale='./'):

        self.file_freq = pathway_principale+"freqs6parameters30year.npy"
        self.which_name = pathway_principale+"loss_modified_mae_indipendent/"
        model_load = True
        self.file_model = self.which_name+'model'
        self.cache_params = np.zeros(6)
        self.cache_mean = None
        self.cache_sigma = None

        self.models=[]
        for i in range(2):
            #print(i) 
            cwd = os.getcwd() + "/"
            if model_load:
                #print(cwd+pathway_principale+self.file_model+"_"+str(i)+"/")
                model = k3.models.load_model(pathway_principale+self.file_model+"_"+str(i))
            self.models.append(model)

    def prediction6parameters(self, A, alpha, beta, M0, rho, sigma,

               freq=None,    #se None, uso un file di nome freqs.npy
               scaler=None,  #se None, uso file di nome scaler.pkl dentro nella cartella del modello model_path
               scaler_y=None,
                samples=100,   #quanti campioni samplo dalle distribuzioni?
                          dist="normal"
               ):

        if freq==None:
            freqs=np.load(self.file_freq)
        if scaler==None:
            scaler = pickle.load(open(self.which_name+'/scaler.pkl','rb'))
        if scaler_y==None:
            scaler_y = pickle.load(open(self.which_name+'/scaler_y.pkl','rb'))
        #costruisco il dataset
        df=pd.DataFrame(index=[i for i in range(len(freqs))])
        nomi=["A","alpha","beta","M0","rho","e0"]
        quantities=[A,alpha,beta,M0,rho,e0]
        for nome,quantity in zip(nomi,quantities):
            df[nome]=quantity
        df["freqs"]=freqs


        #costruisco il dataset nuovo
        df2=pd.DataFrame(index=freqs,columns=["values"])

        X=df.values
        X=scaler.transform(X)
        X=tf.convert_to_tensor(X.astype('float32'))

        #predico media, varianza e dev standard

        pred_dict=dict()

        val=[]
        for i in range(2):
            val.append(self.models[i].predict(X))
        i=0
        val=scaler_y.inverse_transform(np.array(val)[:,:,0].T)

        for key in ["mean","std"]:
            if key=="mean":
                pred_dict[key]=val[:,0]
            else:
                pred_dict[key]=val[:,1]
            i=i+1

        #

        #if dist=="normal":
        #gnd_distribution_fn=lambda t: tfd.LogNormal(loc=np.log(t[0]**2 / np.sqrt(t[0]**2 + t[1]**2)),scale=np.sqrt(np.log(1 + (t[1]**2 / t[0]**2))))
        #else:
        gnd_distribution_fn=lambda t: tfd.TruncatedNormal(loc=t[0],scale=t[1],low=low_bound,high=up_bound)
        for i in range(len(freqs)):
            mean=pred_dict["mean"][i]
            std=pred_dict["std"][i]
            dist=gnd_distribution_fn([mean,std])
            values=dist.sample(samples).numpy()
            df2.at[freqs[i],"values"]=values
        df2=df2.reset_index()
        df2=df2.explode("values")

        return df2,pred_dict

    def prediction6parameters_mean_sigma(self, A, alpha, beta, M0, rho, e0,
                freq=None,    #se None, uso un file di nome freqs.npy
                scaler=None,  #se None, uso file di nome scaler.pkl dentro nella cartella del modello model_path
                scaler_y=None
                ):

        pars = np.array([A, alpha, beta, M0, rho, e0])
        if np.all(pars == self.cache_params):
            return self.cache_mean, self.cache_sigma, None

        if freq is None:
            freqs=np.load(self.file_freq)
            freqs = np.sort(freqs)
        else:
            freqs = freq
        if scaler is None:
            scaler = pickle.load(open(self.which_name+'/scaler.pkl','rb'))
        if scaler_y is None:
            scaler_y = pickle.load(open(self.which_name+'/scaler_y.pkl','rb'))
        #costruisco il dataset
        df=pd.DataFrame(index=[i for i in range(len(freqs))])
        nomi=["A","alpha","beta","M0","rho","e0"]
        quantities = [-2., alpha, beta, M0, rho, e0]
        for nome,quantity in zip(nomi,quantities):
            df[nome]=quantity
        df["freqs"]=freqs

        #costruisco il dataset nuovo
        df2=pd.DataFrame(index=freqs,columns=["values"])

        X=df.values
        X=scaler.transform(X)
        X=tf.convert_to_tensor(X.astype('float32'))

        #predico media, varianza e dev standard

        pred_dict=dict()

        val=[]
        for i in range(2):
            val.append(fast_predict(self.models[i], X))
        i=0
        val=scaler_y.inverse_transform(np.array(val)[:,:,0].T)

        for key in ["mean","std"]:
            if key=="mean":
                pred_dict[key]=val[:,0]
            else:
                pred_dict[key]=val[:,1]
            i=i+1

        mean = np.zeros(len(freqs))
        std = np.zeros(len(freqs))
        for i in range(len(freqs)):
            mean[i]=pred_dict["mean"][i]
            std[i]=pred_dict["std"][i]

        mean *= (10**A / 1e-2)**0.5
        std *= (10**A / 1e-2)**0.5

        self.cache_params = pars
        self.cache_mean = mean
        self.cache_sigma = std

        return mean, std, pred_dict
    
    def spectrum_delta(self, freqs, A, alpha, beta, M0, rho, e0):

        spec = self.prediction6parameters_mean_sigma(A, alpha, beta, M0, rho, e0, freq=freqs)



# (d2n / dzdM) = A * (M / 1e7Mo)**(-alpha) * exp(-M/M0)**(beta) * (1 + z)**(gamma) * exp(-z/z0)

# A : amplitude density [-7., -2.] (default = -2.)
# alpha : first mass spectrum power-law index [0., 1.5] (default = 0.8)
# beta : second mass spectrum power-law index [0.5, 2.] (default = 1.)
# log10_(M0/Mo) : log10 characteristic mass of mass distribution [7., 9.] (default = 8.)
# rho : stellar density during stellar hardening [0., 5.] (default = 1.)
# e0 : initial eccentricity [0., 0.99] (default = 0.1)

A = -2.5
alpha = 0.8
beta = 1.
log10_M0 = 9.
rho = 2.
e0 = 0.9

# gwb = GWBSpectrum()

# freq = np.linspace(-9., -7., 100)
# mean,sigma,_= gwb.prediction6parameters_mean_sigma(A, alpha, beta, log10_M0, rho, e0, freq)
# print(mean)
# print(sigma)
# print('--------------------------------------------------')
# mean,sigma,_= gwb.prediction6parameters_mean_sigma(A, alpha, beta, log10_M0, rho, e0, freq)

# print(mean)
# print(sigma)


# freqs = freq
# freqs=np.load(file_freq)
# freqs = np.sort(freqs)

# fig, ax=plt.subplots()

# for _ in range(5):
#     flucts = np.random.multivariate_normal(mean=np.zeros(len(freqs)), cov=np.diag(sigma**2))
#     ax.loglog(10**freqs,mean + flucts)

# ax.fill_between(10**freqs,mean-sigma,mean+sigma,color="tab:blue",alpha=0.3)

# ax.set_ylabel(r"$h_c$")
# ax.set_xlabel(r"$f$ [Hz]")

# plt.show()