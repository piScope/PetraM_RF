from numba import njit, void, int32, int64, float64, complex128, types

from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array)
import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.phys_const import mu0, epsilon0
from petram.phys.numba_coefficient import NumbaCoefficient
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

stix_options = ("1+SDP", "SDP", "SD", "P", "w/o xx")

def build_coefficients(ind_vars, omega, B, dens_e, t_e, dens_i, masses, charges, g_ns, l_ns,
                       terms="1+SDP", has_e=True, has_i=True):

    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold,
                                                                   epsilonr_pl_cold1,
                                                                   epsilonr_pl_cold2,
                                                                   epsilonr_pl_cold3,
                                                                   epsilonr_pl_cold4,)

    Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

    masses = np.array(masses, dtype=np.float64) * Da
    charges = np.array(charges, dtype=np.int64)

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

    if terms == "1+SDP":
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold(
                omega, B, dens_i, masses, charges, t_e, dens_e, has_e, has_i)
            return out

    elif terms == "SDP":
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold1(
                omega, B, dens_i, masses, charges, t_e, dens_e, has_e, has_i)
            return out

    elif terms == "SD":
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold2(
                omega, B, dens_i, masses, charges, t_e, dens_e, has_e, has_i)
            return out

    elif terms == "P":
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold3(
                omega, B, dens_i, masses, charges, t_e, dens_e, has_e, has_i)
            return out

    elif terms == "w/o xx":
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold4(
                omega, B, dens_i, masses, charges, t_e, dens_e, has_e, has_i)
            return out

    else:
        assert False, "unknown STIX term option"

    def sdp(ptx, B, dens_e, t_e, dens_i):
        out = epsilonr_pl_cold(
            omega, B, dens_i, masses, charges, t_e, dens_e, True, True)
        return out

    def mur(ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    params = {'omega': omega, 'masses': masses, 'charges': charges,
              'has_e': int(has_e), 'has_i': int(has_i)}

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
