import numpy as np
import pickle, os, json
from enterprise_models import model_gw
from enterprise.signals import anis_coefficients as ac
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.deterministic import cw_delay


def make_pta(psrs, crn_components, crn_spectrum, noisedict, custom_models, anis_orf=None):

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])

    # set noise parameters
    bayesephem = False
    inc_psr_term = False
    inc_crn = True
    # crn_psd = 'powerlaw'
    crn_components = crn_components
    crn_psd = crn_spectrum
    add_gwb_var = False
    gwb_var_orf = None
    fvary_crn = False
    # orf = 'legendre_orf'
    orf = None
    anis_knots = 4
    vary_f0 = False
    if vary_f0:
        crn_components = 1
        crn_psd = 'spectrum'
    patx = None
    drop = False
    drop_psr = False
    kde = False
    # log10_fgw = np.linspace(-8., -6.5, 15)
    # nfgw = int(options.nf)
    # log10_F = log10_fgw[nfgw]
    log10_F = [-8.5, -7.5]
    n_cgw = 0
    nharm = 1
    vanilla = False
    J1713_expdip = False
    tm = True
    # tm = True
    # crn_Tspan = (10**8.33)
    nside_knots = 2
    nf_bump = 1
    select = False
    gamma_prior = [0, 7]
    gw_gamma = None
    log10_Agw = [-18, -11]
    # log10_Agw = [None, None]
    dipole_sine = False
    tmparam_origs = None
    rn_psd = 'powerlaw'
    dm_psd = 'powerlaw'
    gwb_fourier = False

    # Free spectrum bin
    crn_Tspan = Tspan
    
    print('crn Tspan', 1/crn_Tspan)

    if gwb_fourier:
        inc_crn = False

    # ANISOTROPY
    if gwb_var_orf == 'anis_orf':
        nside = 32
        lmax = 2
        psrs_locs = np.array([[psr.phi, psr.theta] for psr in psrs])
        psrs_pos = np.array([psr.pos for psr in psrs])
        anis_basis = ac.anis_basis(psrs_locs, lmax=lmax, nside=nside)
    else:
        lmax = 0
        psrs_pos = None
        anis_basis = None
        clm = None

    # psrs = psrs[::2]

    lmax = 2
    anis_knots = int((lmax + 1)**2 - 1)

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    freqs = np.arange(1, crn_components+1) / Tspan

    pta = model_gw(psrs, wn_vary=False, upper_limit=False, prefix='cgw', gwb_fourier=gwb_fourier, crn=inc_crn, custom_models=custom_models, orf=orf, n_cgw=n_cgw, rn_psd=rn_psd, dm_psd=dm_psd, anis_knots=anis_knots,
                    drop=drop, drop_psr=drop_psr, crn_psd=crn_psd, crn_components=crn_components, noisedict=noisedict, tm=tm, add_gwb_var=add_gwb_var, gwb_var_orf=gwb_var_orf, anis_orf=anis_orf,
                    bayesephem=bayesephem, skyloc=None, log10_F=log10_F, ecc=False, J1713_expdip=J1713_expdip, crn_Tspan=crn_Tspan, anis_basis=anis_basis, lmax=lmax, psrs_pos=psrs_pos,
                    psrTerm=inc_psr_term, wideband=False, gamma_prior=gamma_prior, log10_Agw=log10_Agw, tmparam_origs=tmparam_origs, gw_gamma_val=gw_gamma)
    return pta