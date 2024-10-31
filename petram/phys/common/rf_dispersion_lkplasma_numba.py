'''
 non-relativistic Maxwellian hot plasma dispersion, icluding
    all order in k_perp

 max nherm is 20

  epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara, nperp, nhrms)

      temps : ion temperature (eV)
      masses : ion masses (kg)
      charges : ion charges (C)

 (sample run) 
    >>> nperp,npara,denses,masses,charges,temps,ne,Te,Bmang,w,nhrms=(40.,10.,[5.e19,5.e18],[2*Da,Da],[q_base,q_base],[15e3,15e3],5.e19,10e3,0.5,3e7*2*3.1415926,20)
    >>> from petram.phys.common.rf_dispersion_mxwlplasma_numba import epsilonr_pl_hot_std
    >>> epsilonr_pl_hot_std(w,np.array(B),np.array(temps),np.array(denses),np.array(masses),np.array(charges),Te,ne,npara,nperp,nhrms)


array([[-1.56316016e+03+1.23807373e+01j, -1.12033076e+01-9.84907588e+03j,
        -1.22779569e+01-8.12259897e-01j],
       [ 1.12033076e+01+9.84907588e+03j, -2.06928780e+03+9.04122672e+02j,
         2.66395405e+04-2.17944393e+04j],
       [-1.22779569e+01-8.12259897e-01j, -2.66395405e+04+2.17944393e+04j,
         1.29769916e+06+1.58788920e+06j]])

'''
from petram.helper.numba_ive import ive
from petram.phys.common.numba_zfunc import zfunc
from numpy import pi, sqrt
from petram.phys.phys_const import c_cgs as clight
from petram.phys.phys_const import (mass_electron, mass_proton)
from petram.phys.phys_const import q0_cgs
from petram.phys.phys_const import epsilon0 as e0
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import Da
from numba import njit, void, int32, int64, float64, complex128, types
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array, ascontiguousarray)
import numpy as np

iarray_ro = types.Array(int32, 1, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


# constant

# constants
gausspertesla = 1.E4
meter3_per_cm3 = 1.0E-6
me_gram = mass_electron * 1000.
mp_gram = mass_proton * 1000.
ergperkev = q_base * 1e10
qe = -q_base


@njit("float64(float64)")
def om(freq):
    return 2.0 * pi * freq


@njit("float64(float64, float64)")
def wpesq(ne, freq):
    return 4.0 * pi * ne * meter3_per_cm3 * q0_cgs**2 / me_gram / om(freq)**2


@njit("float64(float64, float64, float64, float64)")
def wpisq(ni, A, Z, freq):
    return 4.0 * pi * ni * meter3_per_cm3 * \
        (q0_cgs * Z)**2 / (A * mp_gram) / om(freq)**2


@njit("float64(float64, float64)")
def wce(Bmagn, freq):
    return (-q0_cgs * Bmagn * gausspertesla) / (me_gram * clight) / om(freq)


@njit("float64(float64, float64, float64, float64)")
def wci(Bmagn, freq, A, Z):
    return (q0_cgs * Z * Bmagn * gausspertesla) / (A * mp_gram * clight) / om(freq)


@njit("float64(float64)")
def vte(te_kev):
    tee = ergperkev * te_kev
    return sqrt(2.0 * tee / me_gram) / clight


@njit("float64(float64, float64)")
def vti(ti_kev, A):
    tii = ergperkev * ti_kev
    return sqrt(2.0 * tii / (A * mp_gram)) / clight


@njit("float64(float64, float64, float64, float64, float64, float64)")
def lam_i(nperp, ti_kev, Bmagn, freq, A, Z):
    """Lambda for ions in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vti(ti_kev, A)**2 / (2.0 * wci(Bmagn, freq, A, Z)**2)


@njit("float64(float64, float64, float64, float64)")
def lam_e(nperp, te_kev, Bmagn, freq):
    """Lambda for electrons in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vte(te_kev)**2 / (2.0 * wce(Bmagn, freq)**2)


@njit(void(complex128[:, :], int64, int64))
def print_mat(mat, r, c):
    '''
    routine for debugging
    '''
    for i in range(r):
        for j in range(c):
            print("mat r="+str(i) + ", z="+str(j) + " :")
            print(mat[i, j])


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
@njit("complex128[:](float64, float64, float64, float64, float64, float64, float64, float64, float64)")
def chi_ions(nperp, npar, ni, A, Z, ti_kev, Bmagn, freq, nharm):
    """
    Chi matrix for ions
    """
    w_ci = wci(Bmagn, freq, A, Z)
    lam = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)

    nIn_z = nIn_z_ions(nperp, ti_kev, Bmagn, freq,  A, Z,  nharm)
    I_Ip = IminusIp_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm)
    exp_In = ive(nharm, lam)

    vt = vti(ti_kev, A)
    zeta_i = 1.0 / (npar * vt) * (1.0 - nharm * w_ci)
    pl_z = zfunc(zeta_i)

    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_i * pl_z)

    wp2_fac = wpisq(ni, A, Z, freq) * om(freq)

    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac

    XX = nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2. * lam * I_Ip * wp2An

    XZ = nperp / w_ci / clight * nIn_z * wp2Bn
    YZ = (1j) * nperp / w_ci / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ci) * exp_In * wp2Bn

    return np.array([XX, XY, YY, XZ, YZ, ZZ])


@njit("complex128[:](float64, float64, float64, float64, float64, float64, float64)")
def chi_el(nperp, npar, ne, te_kev, Bmagn, freq, nharm):
    """
    Chi matrix for electron
    """
    w_ce = wce(Bmagn, freq)
    lam = lam_e(nperp, te_kev, Bmagn, freq)

    nIn_z = nIn_z_el(nperp, te_kev, Bmagn, freq, nharm)
    I_Ip = IminusIp_el(nperp, te_kev, Bmagn, freq, nharm)
    exp_In = ive(nharm, lam)

    vt = vte(te_kev)
    zeta_e = 1.0 / (npar * vt) * (1.0 - nharm * w_ce)
    pl_z = zfunc(zeta_e)
    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_e * pl_z)

    wp2_fac = wpesq(ne, freq) * om(freq)
    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac

    XX = nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2. * lam * I_Ip * wp2An

    XZ = nperp / w_ce / clight * nIn_z * wp2Bn
    YZ = (1j) * nperp / w_ce / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ce) * exp_In * wp2Bn

    return np.array([XX, XY, YY, XZ, YZ, ZZ])


@njit(complex128[:, ::1](float64, float64[:], float64[:], float64[:], float64[:], float64[:],
                         float64, float64, float64, float64, int32))
def epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara, nperp, nhrms):

    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    freq = w/2/pi

    chi_terms = np.array([0j, 0j, 0j, 0j, 0j, 0j])

    if ne > 0.:
        te_kev = Te/1000.
        for nh in range(-nhrms, nhrms+1):
            chi_terms += chi_el(nperp, npara, ne, te_kev, b_norm, freq, nh)

    for Ti, dens, mass, charge in zip(temps, denses, masses, charges):
        ti_kev = Ti/1000.
        A = mass/Da
        Z = charge/q_base
        if dens <= 0.:
            continue
        for nh in range(-nhrms, nhrms+1):
            chi_terms += chi_ions(nperp, npara, dens, A,
                                  Z, ti_kev, b_norm, freq, nh)

    M = array([[1.+0j,   0., 0.],
               [0.,   1.+0j, 0.j],
               [0.,   0.,    1.+0j]])
    M2 = array([[chi_terms[0], chi_terms[1], chi_terms[3]],
                [-chi_terms[1], chi_terms[2], chi_terms[4]],
                [chi_terms[3], -chi_terms[4], chi_terms[5]], ])

    M += M2
    return M


@njit(complex128[:, :](float64[:], float64[:], complex128[:, :]))
def rotate_dielectric(B, K, M):
    #
    #  B : magnetic field.
    #  Kp : kperp 
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


