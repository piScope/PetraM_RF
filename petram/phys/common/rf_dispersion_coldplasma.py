from numba import njit, void, int32, int64, float64, complex128, types

from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array)
import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.phys_const import mu0, epsilon0
from petram.phys.numba_coefficient import NumbaCoefficient
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff
from petram.phys.vtable import VtableElement, Vtable

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data = [('B', VtableElement('bext', type='array',
                                   guilabel='magnetic field',
                                   default="=[0,0,0]",
                                   tip="external magnetic field")),
               ('dens_e', VtableElement('dens_e', type='float',
                                        guilabel='electron density(m-3)',
                                        default="1e19",
                                        tip="electron density")),
               ('temperature', VtableElement('temperature', type='float',
                                             guilabel='electron temp.(eV)',
                                             default="10.",
                                             tip="electron temperature used for collisions")),
               ('dens_i', VtableElement('dens_i', type='array',
                                        guilabel='ion densities(m-3)',
                                        default="0.9e19, 0.1e19",
                                        tip="ion densities")),
               ('mass', VtableElement('mass', type='array',
                                      guilabel='ion masses(/Da)',
                                      default="2, 1",
                                      no_func=True,
                                      tip="mass. normalized by atomic mass unit")),
               ('charge_q', VtableElement('charge_q', type='array',
                                          guilabel='ion charges(/q)',
                                          default="1, 1",
                                          no_func=True,
                                          tip="ion charges normalized by q(=1.60217662e-19 [C])")), ]

stix_options = ("SDP", "SD", "P", "w/o xx", "None")
default_stix_option = "(default) include all"


def build_coefficients(ind_vars, omega, B, dens_e, t_e, dens_i, masses, charges, g_ns, l_ns,
                       terms=default_stix_option):

    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
                                                                   epsilonr_pl_cold_g,
                                                                   epsilonr_pl_cold,
                                                                   epsilonr_pl_cold_generic,)

    Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

    masses = np.array(masses, dtype=np.float64) * Da
    charges = np.array(charges, dtype=np.int32)

    num_ions = len(masses)
    l = l_ns
    g = g_ns

    B_coeff = VCoeff(3, [B], ind_vars, l, g,
                     return_complex=False, return_mfem_constant=True)
    dens_e_coeff = SCoeff([dens_e, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)
    t_e_coeff = SCoeff([t_e, ], ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)
    dens_i_coeff = VCoeff(num_ions, [dens_i, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)

    params = {'omega': omega, 'masses': masses, 'charges': charges, }
    if terms == default_stix_option:

        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold(
                omega, B, dens_i, masses, charges, t_e, dens_e)
            return out

        def sdp(ptx, B, dens_e, t_e, dens_i):
            out = epsilonr_pl_cold_std(
                omega, B, dens_i, masses, charges, t_e, dens_e)
            return out

    else:
        from petram.phys.common.rf_stix_terms_panel import value2int
        terms = value2int(len(charges), terms)
        terms = np.array(terms, dtype=np.int32)
        params["sterms"] = terms

        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold_generic(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms)
            return out

        def sdp(ptx, B, dens_e, t_e, dens_i):
            out = epsilonr_pl_cold_g(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms)
            return out

    def mur(ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    numba_debug = False if myid != 0 else get_numba_debug()

    dependency = (B_coeff, dens_e_coeff, t_e_coeff, dens_i_coeff)
    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.matrix(sdim=3, shape=(3, 3), complex=True, params=params,
                             debug=numba_debug, dependency=dependency)
    mfem_coeff1 = jitter(epsilonr)

    jitter2 = mfem.jit.matrix(sdim=3, shape=(3, 3), complex=True, params=params,
                              debug=numba_debug)
    mfem_coeff2 = jitter2(mur)
    mfem_coeff3 = jitter2(sigma)
    mfem_coeff4 = jitter(sdp)

    coeff1 = NumbaCoefficient(mfem_coeff1)
    coeff2 = NumbaCoefficient(mfem_coeff2)
    coeff3 = NumbaCoefficient(mfem_coeff3)
    coeff4 = NumbaCoefficient(mfem_coeff4)

    return coeff1, coeff2, coeff3, coeff4


'''
(reference original)
def epsilonr_pl_warm(x, y, z):
    return epsilonr_pl_f(x, y, z, temp0=50)

def epsilonr_pl_cold(x, y, z):
    return epsilonr_pl_f(x, y, z, temp0=15)
def SPD(x, y, z, Bnorm, ne, ni, nim, Te, Ti, Tim):
   vTe  = sqrt(2*Te/me)
   vTi  = sqrt(2*Ti/mi)
   vTim = sqrt(2*Tim/mim)
   LAMBDA = 1+12*pi*(e0*Te)**(3./2)/(q**3 * sqrt(ne))

   nu_ei = (qi**2 * qe**2 * ni *
           log(LAMBDA)/(4 * pi*e0**2*me**2)/vTe**3)
   nu_eim = (qim**2 * qe**2 * nim *
           log(LAMBDA)/(4 * pi*e0**2*me**2)/vTe**3)

   #effective electrons mass (to account for collisions)
   me_eff  = (1+1j*nu_ei/w + 1j*nu_eim/w )*me
   mi_eff  = (1+1j*nu_ei/w + 1j*nu_eim/w )*mi
   mim_eff = (1+1j*nu_ei/w + 1j*nu_eim/w )*mim
   # when suppressing collisions entirely.
   #me_eff  = me
   #mi_eff  = mi
   #mim_eff = mim

   wpe2  = ne * q**2/(me_eff*e0)
   wpi2  = ni * q**2/(mi_eff*e0)
   wpim2 = nim * q**2/(mim_eff*e0)

   wce  =  qe * Bnorm/me_eff
   wci  =  qi * Bnorm/mi_eff
   wcim =  qim * Bnorm/mim_eff

   P =(1 - wpe2/w**2
            - wpi2/w**2
            - wpim2/w**2)
   #print("wpe2, wpi2, wpim2", wpe2, wpi2, wpim2)
   S = (1-wpe2/(w**2-wce**2)-wpi2/(w**2-wci**2)-
           wpim2/(w**2-wcim**2))
   D = (wce*wpe2/(w*(w**2-wce**2)) + wci*wpi2/(w*(w**2-wci**2)) +
           wcim*wpim2/(w*(w**2-wcim**2)))

   return S, P, D
'''
