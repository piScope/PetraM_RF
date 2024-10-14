from numba import njit, void, int32, int64, float64, complex128, types
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array, ascontiguousarray)
import numpy as np

iarray_ro = types.Array(int32, 1, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


# constant
from petram.phys.phys_const import Da
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import mass_electron as me
from petram.phys.phys_const import epsilon0 as e0
qe = -q_base


@njit(void(complex128[:, :], int64, int64))
def print_mat(mat, r, c):
    '''
    routine for debugging
    '''
    for i in range(r):
        for j in range(c):
            print("mat r="+str(i) + ", z="+str(j) + " :")
            print(mat[i, j])

from petram.helper.numba_ive import ive
from petram.phys.common.plasma_formulae import *
from petram.phys.common.numba_zfunc import zfunc

@njit("float64(float64, float64, float64, float64, float64, float64, float64)")
def IminusIp_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm):
    """
    In -I'n (for ions).
    Modified Bessel function in the dielectric tensor.
    Eq. (10-57) of Stix book, p. 258

    We use I'n = (In-1 + In+1)/2.0
    """
    ll = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)
    In0 = ive(nharm, ll)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    Iprime = (In1 + In2) / 2.0
    return In0 - Iprime

@njit("float64(float64, float64, float64, float64, float64)")
def IminusIp_el(nperp, te_kev, Bmagn, freq, nharm):
    """
    In -I'n (for electrons).
    Modified Bessel function in the dielectric tensor.
    Eq. (10-57) of Stix book, p. 258

    We use I'n = (In-1 + In+1)/2.0
    """
    ll = lam_e(nperp, te_kev, Bmagn, freq)    
    In0 = ive(nharm, ll)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    Iprime = (In1 + In2) / 2.0
    return In0 - Iprime

@njit("float64(float64, float64, float64, float64, float64, float64, float64)")
def nIn_z_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm):
    '''
    evaluate nIn/z needed in xx, yy, xz, zx components

    we use nIn/z = (In-1 - In+1)/2.0
    '''
    ll = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)    
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    return (In1 - In2) / 2.0

@njit("float64(float64, float64, float64, float64, float64)")
def nIn_z_el(nperp, te_kev, Bmagn, freq, nharm):
    '''
    evaluate nIn/z needed in xx, yy, xz, zx components

    we use nIn/z = (In-1 - In+1)/2.0
    '''
    ll = lam_e(nperp, te_kev, Bmagn, freq)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    return (In1 - In2) / 2.0


# Y MATRIX
@njit("float64[:](float64, float64, float64, float64, float64, float64, float64, float64)")
def chi_ions(nperp, npar, ni, A, Z, ti_kev, Bmagn, freq, nharm):
    """
    Chi matrix for ions
    """
    nIn_z = nIn_z_ions(nperp, ti_kev, Bmagn, freq,  A, Z,  nharm)
    I_Ip = IminusIp_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm)
    exp_In = ive(nharm, ll)
    
    w_ci = wci(Bmagn, freq, A, Z)
    vt = vti(ti_kev, A)
    ll = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)
    
    zeta_i = 1.0 / (npar * vt) * (1.0 - nharm * w_ci)
    pl_z = zfunc(zeta_i)

    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_i) * pl_z

    wp2_fac =  wpisq(ne, A, Z, freq) 
    
    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac
    
    XX =  nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2.* ll * I_Ip * wp2An

    XZ = nperp / w_ci / clight * nIn_z * wp2Bn
    YZ = (1j)* nperp / w_ci / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ci) * exp_In * wp2Bn
    
    return np.array([XX, XY, YY, XZ, YZ, ZZ])

@njit("float64[:](float64, float64, float64, float64, float64, float64)")
def chi_el(nperp, npar, ne, te_kev, Bmagn, freq, nharm):
    """
    Chi matrix for electron
    """
    nIn_z = nIn_z_el(nperp, te_kev, Bmagn, freq, nharm)
    I_Ip = IminusIp_el(nperp, te_kev, Bmagn, freq, nharm)
    exp_In = ive(nharm, ll)
    
    w_ce = wce(Bmagn, freq)
    vt = vte(te_kev, A)
    ll = lam_e(nperp, te_kev, Bmagn, freq)

    zeta_e = 1.0 / (npar * vt) * (1.0 - nharm * w_ce)
    pl_z = zfunc(zeta_e)
    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_i) * pl_z
    
    wp2_fac =  wpesq(ne, freq)

    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac
    
    XX =  nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2.* ll * I_Ip * wp2An

    XZ = nperp / w_ce / clight * nIn_z * wp2Bn
    YZ = (1j)* nperp / w_ce / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ce) * exp_In * wp2Bn
    
    return np.array([XX, XY, YY, XZ, YZ, ZZ])



@njit(complex128[:, ::1](float64, float64[:], float64[:], float64[:], float64[:],
                         float64, float64, float64, float64, int32))
def epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara, nperp, nhrms):
    
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    freq = w/2/pi

    chi_terms = np.array([0j, 0j, 0j, 0j, 0j. 0j])
    
    if ne > 0.:
        te_kev = Te/1000.        
        for nh in range(-nhrms, nhrms+1):
            chi_terms +=  chi_el(nperp, npara, ne, te_kev, b_norm, freq, nh)

    for Ti, dens, mass, charge, nu_ei in zip(temps, denses, masses, temps):
        ti_kev = Ti/1000.
        A = mass/Da
        Z = charge/q_base
        if dens <= 0.:
            continue
        
        for nh in range(-nhrms, nhrms+1):
            chi_terms +=  chi_ions(nperp, npara, dens, A, Z, ti_kev, b_norm, freq, nh)

    M = array([[1.+0j,   0., 0.],
               [0.,   1.+0j, 0.j],
               [0.,   0.,    1.+0j]])
    M2 = array([[ chi_terms[0], chi_terms[1], chi_terms[3]],
                [-chi_terms[1], chi_terms[2], chi_terms[4]],
                [chi_terms[3], -chi_terms[4], chi_terms[5]],])

    M += M2
    return M


@njit(complex128[:, :](float64[:], complex128[:, :]))
def rotate_dielectric(B, M):
    #
    #  B : magnetic field.
    #  M : dielectric matrix.
    #
    B = ascontiguousarray(B)
    M = ascontiguousarray(M)

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


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64))
def epsilonr_pl_hot(w, B, denses, masses, charges, Te, ne):
    '''
    hot maxwellian
    '''
    M = epsilonr_pl_hot_std(w, B, denses, masses, charges, Te, ne)
    return rotate_dielectric(B, M)


