from numba import njit, void, int32, int64, float64, complex128, types
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array, ascontiguousarray)
import numpy as np

# vacuum permittivity
e0 = 8.8541878176e-12
q_base = 1.60217662e-19
qe = -q_base
me = 9.10938356e-31

iarray_ro = types.Array(int32, 1, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


@njit(void(complex128[:, :], int64, int64))
def print_mat(mat, r, c):
    '''
    routine for debugging
    '''
    for i in range(r):
        for j in range(c):
            print("mat r="+str(i) + ", z="+str(j) + " :")
            print(mat[i, j])


@njit(complex128[:](float64, float64, float64, float64[:]))
def SPD_el(w, Bnorm, dens, nu_eis):

    mass_eff = (1 + sum(1j*nu_eis/w))*me

    wp2 = dens * q_base**2/(mass_eff*e0)
    wc = qe * Bnorm/mass_eff

    Pterm = -wp2/w**2
    Sterm = -wp2/(w**2-wc**2)
    Dterm = wc*wp2/(w*(w**2-wc**2))

    return array([Sterm, Pterm, Dterm])


@njit(complex128[:](float64, float64, float64, float64, int64, float64))
def SPD_ion(w, Bnorm, dens, mass, charge, nu_ei):
    qi = charge*q_base
    mass_eff = (1 + 1j*nu_ei/w)*mass
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
    nu_eis = zeros(len(charges))
    if ne == 0:
        return nu_eis

    vt_e = sqrt(2*Te*q_base/me)
    LAMBDA = 1+12*pi*(e0*Te*q_base)**(3./2)/(q_base**3 * sqrt(ne))

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

    if ne > 0.:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        S += Se
        P += Pe
        D += De

    for dens, mass, charge, nu_ei in zip(denses, masses, charges, nu_eis):
        if dens > 0.:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_ei)
            S += Si
            P += Pi
            D += Di
    M = array([[S,   -1j*D, 0j],
               [1j*D, S,    0j],
               [0j,   0j,   P]])
    return M


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, iarray_ro))
def epsilonr_pl_cold_g(w, B, denses, masses, charges, Te, ne, terms):
    '''
    generalized Stix tensor
    terms defined in rf_dispersion_coldplasma
       stix_options = ("SDP", "SD", "DP", "P", "w/o xx", "None")
    '''

    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    M = array([[1.+0j,   0., 0.],
               [0.,   1.+0j, 0.j],
               [0.,   0.,    1.+0j]])

    if ne > 0.:
        S, P, D = SPD_el(w, b_norm, ne, nu_eis)
        if terms[0] == 0:
            M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0., P]])
        elif terms[0] == 1:
            M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0., 0j]])
        elif terms[0] == 2:
            M2 = array([[0j, -1j*D, 0j], [1j*D, 0j, 0j], [0., 0., P]])
        elif terms[0] == 3:
            M2 = array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0., P]])
        elif terms[0] == 4:
            M2 = array([[0j, -1j*D, 0j], [1j*D, S, 0j], [0j, 0., P]])
        else:
            M2 = array([[0j, 0j, 0j], [0j, 0j, 0j], [0., 0., 0j]])
        M += M2

    kion = 1
    for dens, mass, charge, nu_ei in zip(denses, masses, charges, nu_eis):
        if dens > 0.:
            S, P, D = SPD_ion(w, b_norm, dens, mass, charge, nu_ei)
            if terms[kion] == 0:
                M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0., P]])
            elif terms[kion] == 1:
                M2 = array([[S, -1j*D, 0j], [1j*D, S, 0j], [0., 0., 0j]])
            elif terms[kion] == 2:
                M2 = array([[0j, -1j*D, 0j], [1j*D, 0j, 0j], [0., 0., P]])
            elif terms[kion] == 3:
                M2 = array([[0j, 0j, 0j], [0j, 0j, 0j], [0j, 0., P]])
            elif terms[kion] == 4:
                M2 = array([[0j, -1j*D, 0j], [1j*D, S, 0j], [0j, 0., P]])
            else:
                M2 = array([[0j, 0j, 0j], [0j, 0j, 0j], [0., 0., 0j]])
            M += M2

        kion = kion + 1

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
def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne):
    '''
    standard SPD stix
    '''
    M = epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne)
    return rotate_dielectric(B, M)


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, iarray_ro))
def epsilonr_pl_cold_generic(w, B, denses, masses, charges, Te, ne, terms):
    '''
    standard SPD stix
    '''
    M = epsilonr_pl_cold_g(w, B, denses, masses,
                           charges, Te, ne, terms)

    return rotate_dielectric(B, M)
