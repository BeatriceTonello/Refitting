import numpy as np
import pickle, os, json
from create_pta import make_pta


# load psrs pickle file
pickle_name = '/work/falxa/EPTA/pkl/fake_100_psrs_10yrs_real_pop_NOCGW.pkl'
psrs = pickle.load(open(pickle_name, 'rb'))

noisedict = {}
custom_models = {}
for psr in psrs:
    noisedict.update(psr.noisedict)
    custom_models[psr.name] = {'RN':None, 'DM':None, 'Sv':None}

Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])


# create pta
pta = make_pta(psrs=psrs, crn_components=9, crn_spectrum='spectrum', noisedict=noisedict, custom_models=custom_models)

# get initial sample
x0 = np.hstack([p.sample() for p in pta.params])

# get lnlikelihood
pta.get_lnlikelihood(x0)

# get lnprior
pta.get_lnprior(x0)