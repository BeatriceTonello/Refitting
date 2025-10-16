import numpy as np
# import healpy as hp
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
import enterprise.signals.signal_base as base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise import constants as const
from enterprise.signals.selections import Selection
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise_extensions import model_utils
from enterprise_extensions import blocks
from enterprise_extensions import timing
from enterprise_extensions import deterministic
from enterprise_extensions.chromatic import chromatic as chrom
from enterprise_extensions import model_orfs
from enterprise.signals.parameter import function
from enterprise.signals import deterministic_signals, parameter, signal_base
import scipy.interpolate as interpolate

@function
def free_spectrum_cbin(f, log10_rho=None, nbin=0, log10_c=[0., 0.]):
    """
    Free spectral model. PSD  amplitude at each frequency
    is a free parameter. Model is parameterized by
    S(f_i) = \rho_i^2 * T,
    where \rho_i is the free parameter and T is the observation
    length.
    """
    fs = np.repeat(10 ** (2 * np.array(log10_rho)), 2)
    fs[int(2*nbin)] *= 10**(1. * log10_c[0])
    fs[int(2*nbin+1)] *= 10**(1. * log10_c[1])
    # print(10**(2*log10_c[1]))
    return fs

@function
def tm_prior(weights):
    return weights * 1e40

@signal_base.function
def gp_anis_orf(pos1, pos2, knots, anis_orf):
    orf = anis_orf.get_anis_orf(pos1, pos2, knots, alpha=0., sigma=1.)
    return orf

@signal_base.function
def spharm_anis_orf(pos1, pos2, blm, anis_orf):
    orf = anis_orf.get_anis_orf(pos1, pos2, blm)
    return orf

@signal_base.function
def von_mises_fisher_orf(pos1, pos2, knots1, knots2, anis_orf):
    orf = anis_orf.get_anis_orf(pos1, pos2, knots1, knots2)
    return orf

@signal_base.function
def anis_orf_local(pos1, pos2, params, **kwargs):
    """Anisotropic GWB spatial correlation function."""

    anis_basis = kwargs["anis_basis"]
    psrs_pos = kwargs["psrs_pos"]
    lmax = kwargs["lmax"]

    psr1_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos1)][0]
    psr2_index = [ii for ii in range(len(psrs_pos)) if np.all(psrs_pos[ii] == pos2)][0]

    clm = np.zeros((lmax + 1) ** 2)
    clm[0] = 0.
    if lmax > 0:
        clm[1:] = params

    return sum(clm[ii] * basis for ii, basis in enumerate(anis_basis[: (lmax + 1) ** 2, psr1_index, psr2_index]))


def TimingModel(coefficients=False, name="linear_timing_model", use_svd=False, normed=True):
    """Class factory for marginalized linear timing model signals."""

    basis = gp_signals.get_timing_model_basis(use_svd, normed)
    prior = tm_prior()

    BaseClass = gp_signals.BasisGP(prior, basis, coefficients=coefficients, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear timing model"
        signal_id = name + "_svd" if use_svd else name

        if coefficients:

            def _get_coefficient_logprior(self, key, c, **params):
                # MV: probably better to avoid this altogether
                #     than to use 1e40 as in get_phi
                return 0

    return TimingModel

@signal_base.function
def fourier_process(toas, freqs, a=np.zeros(1), nmodes=10, k_drop=1., idx=0., tref=0.):

    Tspan = np.amax(toas) - np.amin(toas)
    F, _ = gpb.createfourierdesignmatrix_red(toas, nmodes=nmodes, Tspan=Tspan)
    sine = np.sum([a_i * F_i for a_i, F_i in zip(a, F.T)], axis=0)
    k = np.rint(k_drop)
    res = k * sine
    return res

def fourier_block(drop=False, nmodes=10, idx=0., tref=0., Tspan=None, name='fourier'):


    if drop:
        k_drop = parameter.Uniform(0, 1)
    else:
        k_drop = 1.

    a_name = "{}_a".format(name)
    a = parameter.Uniform(-1., 1., size=int(2*nmodes))

    # continuous wave signal
    wf = fourier_process(a=a, nmodes=nmodes, k_drop=k_drop, idx=idx, tref=tref)
    cw = deterministic.CWSignal(wf, ecc=False, psrTerm=False)

    return cw

def common_red_noise_block(psd='powerlaw', prior='log-uniform', gamma_prior=[0, 7], astro_params=None,
                           Tspan=None, components=30, combine=True, logf=False, anis_orf=None,
                           log10_A_val=None, gamma_val=None, delta_val=None, clm=None, n_knots=12,
                           logmin=None, logmax=None, anis_basis=None, lmax=0, psrs_pos=None,
                           orf=None, orf_ifreq=0, leg_lmax=3, vary_Tspan=False, chrom_idx=0, nbin_ng=0,
                           name='gw', coefficients=False,
                           pshift=False, pseed=None):

    if orf == 'anis_orf':
        clm_name = '{}_clm'.format(name)
        if lmax > 0:
            clm = parameter.Uniform(-10., 10., size=(lmax+1)**2 - 1)('gw_clm')
        else:
            clm = np.zeros(2)

    # if anis_orf != None:
    #     if orf == 'von_mises_fisher':
            

    orfs = {'crn': None, 'hd': model_orfs.hd_orf(),
            'gw_monopole': model_orfs.gw_monopole_orf(),
            'gw_dipole': model_orfs.gw_dipole_orf(),
            'st': model_orfs.st_orf(),
            'gt': model_orfs.gt_orf(tau=parameter.Uniform(-1.5, 1.5)('tau')),
            'dipole': model_orfs.dipole_orf(),
            'monopole': model_orfs.monopole_orf(),
            'param_hd': model_orfs.param_hd_orf(a=parameter.Uniform(-1.5, 3.0)('gw_orf_param0'),
                                                b=parameter.Uniform(-1.0, 0.5)('gw_orf_param1'),
                                                c=parameter.Uniform(-1.0, 1.0)('gw_orf_param2')),
            'spline_orf': model_orfs.spline_orf(params=parameter.Uniform(-0.9, 0.9, size=7)('gw_orf_spline')),
            'freq_hd': model_orfs.freq_hd(params=[components, orf_ifreq]),
            'von_mises_fisher' : von_mises_fisher_orf(knots1=parameter.Uniform(-0.1, 0.1, size=12)('gauss'), knots2=parameter.Uniform(-0.1, 0.1, size=48)('gauss_fine'), anis_orf=anis_orf)}

    # common red noise parameters
    if psd in ['powerlaw', 'powerlaw_bump', 'turnover', 'turnover_knee', 'broken_powerlaw']:
        amp_name = '{}_log10_A'.format(name)
        if log10_A_val is not None:
            log10_Agw = parameter.Constant(log10_A_val)(amp_name)

        if logmin is not None and logmax is not None:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(logmin, logmax)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)
            else:
                log10_Agw = parameter.Uniform(logmin, logmax)(amp_name)

        else:
            if prior == 'uniform':
                log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
            elif prior == 'log-uniform' and gamma_val is not None:
                if np.abs(gamma_val - 4.33) < 0.1:
                    log10_Agw = parameter.Uniform(-18, -14)(amp_name)
                else:
                    log10_Agw = parameter.Uniform(-18, -11)(amp_name)
            else:
                log10_Agw = parameter.Uniform(-18, -11)(amp_name)

        gam_name = '{}_gamma'.format(name)
        if gamma_val is not None:
            gamma_gw = parameter.Constant(gamma_val)(gam_name)
        else:
            gamma_gw = parameter.Uniform(gamma_prior[0], gamma_prior[1])(gam_name)

        # common red noise PSD
        if psd == 'powerlaw':
            cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        elif psd == 'broken_powerlaw':
            delta_name = '{}_delta'.format(name)
            kappa_name = '{}_kappa'.format(name)
            log10_fb_name = '{}_log10_fb'.format(name)
            kappa_gw = parameter.Uniform(0.01, 0.5)(kappa_name)
            log10_fb_gw = parameter.Uniform(-10, -7)(log10_fb_name)

            if delta_val is not None:
                delta_gw = parameter.Constant(delta_val)(delta_name)
            else:
                delta_gw = parameter.Uniform(0, 7)(delta_name)
            cpl = gpp.broken_powerlaw(log10_A=log10_Agw,
                                      gamma=gamma_gw,
                                      delta=delta_gw,
                                      log10_fb=log10_fb_gw,
                                      kappa=kappa_gw)
        elif psd == 'turnover':
            kappa_name = '{}_kappa'.format(name)
            lf0_name = '{}_log10_fbend'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lf0_gw = parameter.Uniform(-9, -7)(lf0_name)
            cpl = utils.turnover(log10_A=log10_Agw, gamma=gamma_gw,
                                 lf0=lf0_gw, kappa=kappa_gw)
        elif psd == 'turnover_knee':
            kappa_name = '{}_kappa'.format(name)
            lfb_name = '{}_log10_fbend'.format(name)
            delta_name = '{}_delta'.format(name)
            lfk_name = '{}_log10_fknee'.format(name)
            kappa_gw = parameter.Uniform(0, 7)(kappa_name)
            lfb_gw = parameter.Uniform(-9.3, -8)(lfb_name)
            delta_gw = parameter.Uniform(-2, 0)(delta_name)
            lfk_gw = parameter.Uniform(-8, -7)(lfk_name)
            cpl = gpp.turnover_knee(log10_A=log10_Agw, gamma=gamma_gw,
                                    lfb=lfb_gw, lfk=lfk_gw,
                                    kappa=kappa_gw, delta=delta_gw)

    if psd == "spectrum":
        rho_name = "{}_log10_rho".format(name)
        if logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(logmin, logmax, size=components)(
                    rho_name
                )
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(
                    rho_name
                )
        else:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(-10, -4, size=components)(rho_name)
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(-10, -4, size=components)(rho_name)

        cpl = gpp.free_spectrum(log10_rho=log10_rho_gw)

    if psd == "spectrum_cbin":
        rho_name = "{}_log10_rho".format(name)
        if logmin is not None and logmax is not None:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(logmin, logmax, size=components)(
                    rho_name
                )
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(
                    rho_name
                )
        else:
            if prior == "uniform":
                log10_rho_gw = parameter.LinearExp(-10, -4, size=components)(rho_name)
            if prior == "constant":
                log10_rho_gw = parameter.Constant(-7.)(rho_name)
            elif prior == "log-uniform":
                log10_rho_gw = parameter.Uniform(-10, -4, size=components)(rho_name)

        log10_c = parameter.Uniform(0, 1, size=2)('log10_c')
        # nbin = parameter.Uniform(0, 1)('nbin')
        nbin = parameter.Constant(nbin_ng)('nbin')
        cpl = free_spectrum_cbin(log10_rho=log10_rho_gw, nbin=nbin, log10_c=log10_c)


    if psd == 'bpl':
        alpha = parameter.Uniform(-11, -5)('{}_bpl_alpha'.format(name))
        gam = parameter.Uniform(-11, -5)('{}_bpl_gamma'.format(name))
        a = parameter.Uniform(0, 6)('{}_bpl_a'.format(name))
        b = parameter.Uniform(0, 6)('{}_bpl_b'.format(name))
        c = parameter.Uniform(0.1, 6.1)('{}_bpl_c'.format(name))
        cpl = psd_BPL(alpha=alpha, gamma=gam, a=a, b=b, c=c)

    if psd == 'ln':
        alpha = parameter.Uniform(-11, -5)('{}_ln_alpha'.format(name))
        gam = parameter.Uniform(-11, -5)('{}_ln_gamma'.format(name))
        sigma = parameter.Uniform(-2, 1)('{}_ln_sigma'.format(name))
        cpl = psd_LN(alpha=alpha, gamma=gam, sigma=sigma)

    if orf == 'diag_peak_orf':

        rho_name = '{}_log10_rho'.format(name)
        log10_rho_gw = parameter.Uniform(logmin, logmax, size=components)(rho_name)
        costheta = parameter.Uniform(-1, 1)('gw_costheta')
        phi = parameter.Uniform(0., 2*np.pi)('gw_phi')
        k = parameter.Uniform(0., 10.)('gw_k')
        cpl = peak_free_spectrum(log10_rho=log10_rho_gw, costheta=costheta, phi=phi, k=k, antenna_orfs=antenna_orfs, pixels=pixels)
        orf = None

    # if orf is 'point_orf_curn':

    if isinstance(chrom_idx, list):
        chrom_idx = parameter.Uniform(chrom_idx[0], chrom_idx[1])('chrom_idx')
        
    if orf is None:
        crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients, logf=logf,
                                        components=components, Tspan=Tspan,
                                        name=name, pshift=pshift, pseed=pseed)
    elif orf in orfs.keys():
        if orf == 'crn':
            crn = gp_signals.FourierBasisGP(cpl, coefficients=coefficients,
                                            components=components, Tspan=Tspan,
                                            name=name, pshift=pshift, pseed=pseed)
        else:
            crn = gp_signals.FourierBasisCommonGP(cpl, orfs[orf], logf=logf,
                                                  components=components,
                                                  Tspan=Tspan,
                                                  name=name, pshift=pshift,
                                                  pseed=pseed)
    elif isinstance(orf, types.FunctionType):
        crn = gp_signals.FourierBasisCommonGP(cpl, orf, logf=logf,
                                              components=components,
                                              Tspan=Tspan,
                                              name=name, pshift=pshift,
                                              pseed=pseed)
    else:
        raise ValueError('ORF {} not recognized'.format(orf))

    return crn


def model_gw(psrs, wn_vary=False, wn_select='backend', upper_limit=False, prefix='cw', custom_models={}, noisedict=None, n_cgw=0, dipole_sine=False, tm=True, orf=None, drop=False, drop_psr=False, crn_logf=False, fvary_crn=False, nbin_ng=0,
             rn_var=True, rn_psd='powerlaw', dm_var=True, dm_psd='powerlaw', components=30, dm_components=None, J1713_expdip=True, crn_name='gw', nharm=1, noise_logf=False, anis_knots=12, add_gwb_var=True, anis_orf=None,
             crn=False, bayesephem=False, skyloc=None, patch=None, log10_F=None, ecc=False, crn_psd='powerlaw', crn_components=30, crn_Tspan=None, vary_f0=False, add_hd=False, gwb_var_orf=None,
             psrTerm=False, wideband=False, anis_basis=None, lmax=0, psrs_pos=None, clm=None, nf_bump=None, fmargin_crn=False, fmargin_noise=False, custom_specs=None,
             gamma_prior=[0, 7], gw_gamma_val=None, log10_Agw=[None, None], tmparam_origs=None, gwb_chrom_idx=0, cw_chrom_idx=0, gwb_fourier=False):
    
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin
    frequencies = np.arange(1, crn_components+1) / Tspan

    # timing model
    s0 = TimingModel(use_svd=True)

    # crn
    if crn_Tspan is None:
        crn_Tspan = Tspan
    if crn:
        s0 += common_red_noise_block(psd=crn_psd, chrom_idx=gwb_chrom_idx, prior='log-uniform', gamma_val=gw_gamma_val, Tspan=crn_Tspan, components=crn_components, orf=orf, name=crn_name, anis_basis=anis_basis, anis_orf=anis_orf, lmax=lmax, psrs_pos=psrs_pos, nbin_ng=nbin_ng)

    if gwb_fourier:
        s0 += fourier_block(drop=False, nmodes=10, idx=0., tref=0., Tspan=Tspan, name='fourier')

    if n_cgw > 0:
        s0 += deterministic.cw_block_circ(amp_prior=amp_prior, log10_fgw=log10_F,
            skyloc=None, psrTerm=psrTerm, name='cgw')

    models = []
    for p in psrs:

        # adding white-noise, red noise, dm noise, chromatic noise and acting on psr objects
        keys = [key for key in [*noisedict] if p.name in key]
        if np.any(['ecorr' in key for key in keys]):
            ecorr = False
        # elif 'NANOGrav' in p.flags['pta'] and not wideband:
        #     ecorr = True
        else:
            ecorr = False

        # white noise
        s2 = s0 + blocks.white_noise_block(vary=wn_vary, select=wn_select, inc_ecorr=ecorr, tnequad=True, gp_ecorr=ecorr, name='')

        # time correlated noise
        if p.name in [*custom_models]:

            Tspan = p.toas.max() - p.toas.min()

            rn_components = custom_models[p.name]['RN']
            dm_components = custom_models[p.name]['DM']
            sv_components = custom_models[p.name]['Sv']

            # red noise
            if rn_components is not None:
                s2 += blocks.red_noise_block(prior=amp_prior, psd=rn_psd, Tspan=Tspan, components=rn_components, logf=noise_logf)
            # dm noise
            if dm_components is not None:
                s2 += blocks.dm_noise_block(psd=dm_psd, prior=amp_prior, components=dm_components, Tspan=Tspan, gamma_val=None, logf=noise_logf)
            # scattering variation
            if sv_components is not None:
                s2 += blocks.chromatic_noise_block(gp_kernel='diag', psd=rn_psd, prior=amp_prior, idx=4, Tspan=Tspan, components=sv_components, logf=noise_logf)
            if p.name == 'J1713+0747' and J1713_expdip:
                # s2 += chrom.dm_exponential_dip(tmin=54650, tmax=54850, idx=4, sign='negative', name='expd-%s_%s_%s'%(4, int(54650),int(54850)))
                s2 += chrom.dm_exponential_dip(tmin=57490, tmax=57530, idx=1, sign='negative', name='expd-%s_%s_%s'%(1, int(57490),int(57530)))
            
        models.append(s2(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        pta.set_default_params(noisedict)

    return pta
