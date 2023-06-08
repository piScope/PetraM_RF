from numba import njit, int64, float64, complex128, types

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


# vacuum permittivity
e0 = 8.8541878176e-12

# electron
q_base = 1.60217662e-19
qe = -q_base
me = 9.10938356e-31

iarray_ro = types.Array(int64, 1, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


@njit(complex128[:](float64, float64, float64, float64[:]))
def SPD_el(w, Bnorm, dens, nu_eis):
    mass_eff = (1 + sum(1j*nu_eis/w))*me

    wp2 = dens * q_base**2/(mass_eff*e0)
    wc = qe * Bnorm/mass_eff

    Pterm = -wp2/w**2
    Sterm = -wp2/(w**2-wc**2)
    Dterm = wc*wp2/(w*(w**2-wc**2))

    return array([Sterm, Pterm, Dterm])


@njit(complex128[:](float64, float64, float64, float64, int64, float64[:]))
def SPD_ion(w, Bnorm, dens, mass, charge, nu_eis):
    qi = charge*q_base
    mass_eff = (1 + sum(1j*nu_eis/w))*mass
    wp2 = dens * q_base**2/(mass_eff*e0)
    wc = qi * Bnorm/mass_eff

    Pterm = -wp2/w**2
    Sterm = -wp2/(w**2-wc**2)
    Dterm = wc*wp2/(w*(w**2-wc**2))

    return array([Sterm, Pterm, Dterm])


@njit(float64[:](float64[:], iarray_ro, float64, float64))
def f_collisions(denses, charges, Te, ne):
    '''
    electron-ion collision
    '''
    vt_e = sqrt(2*Te*q_base/me)
    LAMBDA = 1+12*pi*(e0*Te*q_base)**(3./2)/(q_base**3 * sqrt(ne))

    nu_eis = zeros(len(charges))
    for k in range(len(charges)):
        ni = denses[k]
        qi = charges[k]*q_base
        nu_ei = (qi**2 * qe**2 * ni *
                 log(LAMBDA)/(4 * pi*e0**2*me**2)/vt_e**3)
        nu_eis[k] = nu_ei
    return nu_eis


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64))
def epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 1 + 0j
    P = 1 + 0j
    D = 0j

    Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
    S += Se
    P += Pe
    D += De

    for dens, mass, charge in zip(denses, masses, charges):
        Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
        S += Si
        P += Pi
        D += Di

    M = array([[S,   -1j*D, 0],
               [1j*D, S,    0],
               [0,   0,    P]])
    return M


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64))
def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne):
    M = epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    #  B=(0,0,1) -> phi = 0,  th =0
    #  B=(1,0,0) -> phi = 90, th =0
    #  B=(0,1,0) -> phi = 90, th =90

    #  Bx = sin(phi) cos(th)
    #  By = sin(phi) sin(th)
    #  Bz = cos(phi)

    # B = [Bx, By, Bz]
    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))

    ans = dot(R2(th), dot(A, R2(-th)))

    return ans


def build_coefficients(ind_vars, omega, B, dens_e, t_e, dens_i, masses, charges, g_ns, l_ns):

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

    def epsilonr(ptx, B, dens_e, t_e, dens_i):
        out = -epsilon0 * omega * omega*epsilonr_pl_cold(
            omega, B, dens_i, masses, charges, t_e, dens_e)
        return out

    def sdp(ptx, B, dens_e, t_e, dens_i):
        out = epsilonr_pl_cold_std(
            omega, B, dens_i, masses, charges, t_e, dens_e)
        return out

    def mur(ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    params = {'omega': omega, 'masses': masses, 'charges': charges, }
    #                  'epsilonr_pl_cold': epsilonr_pl_cold,}
    numba_debug = False if myid != 0 else get_numba_debug()

    dependency = (B_coeff, dens_e_coeff, t_e_coeff, dens_i_coeff)
    dependency = [(x.mfem_numba_coeff if isinstance(B_coeff, NumbaCoefficient) else x)
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
