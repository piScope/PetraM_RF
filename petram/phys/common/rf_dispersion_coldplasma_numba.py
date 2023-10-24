from numba import njit, void, int32, int64, float64, complex128, types
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array)
import numpy as np

# vacuum permittivity
e0 = 8.8541878176e-12
q_base = 1.60217662e-19
qe = -q_base
me = 9.10938356e-31

iarray_ro = types.Array(int64, 1, 'C', readonly=True)
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


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold_std(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 1 + 0j
    P = 1 + 0j
    D = 0j

    if ne > 0. and has_e:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        S += Se
        P += Pe
        D += De

    for dens, mass, charge in zip(denses, masses, charges):
        if dens > 0. and has_i:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
            S += Si
            P += Pi
            D += Di
    M = array([[S,   -1j*D, 0j],
               [1j*D, S,    0j],
               [0j,   0j,   P]])
    return M


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold_SDP(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 0j
    P = 0j
    D = 0j

    if ne > 0. and has_e:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        S += Se
        P += Pe
        D += De

    for dens, mass, charge in zip(denses, masses, charges):
        if dens > 0. and has_i:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
            S += Si
            P += Pi
            D += Di

    M = array([[S,   -1j*D, 0],
               [1j*D, S,    0],
               [0,   0,    P]])
    return M


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold_SD(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 0j
    P = 0j
    D = 0j

    if ne > 0. and has_e:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        S += Se
        D += De

    for dens, mass, charge in zip(denses, masses, charges):
        if dens > 0. and has_i:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
            S += Si
            D += Di

    M = array([[S,   -1j*D, 0],
               [1j*D, S,    0],
               [0,   0,    P]])
    return M


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold_P(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 0j
    P = 0j
    D = 0j

    if ne > 0. and has_e:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        P += Pe
        # D += De
        # S += Se

    for dens, mass, charge in zip(denses, masses, charges):
        if dens > 0. and has_i:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
            P += Pi
            # D += Di
            # S += Si

    M = array([[S,   -1j*D, 0],
               [1j*D, S,    0],
               [0,   0,    P]])
    return M


@njit(complex128[:, ::1](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold_woxx(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    nu_eis = f_collisions(denses, charges, Te, ne)

    S = 0j
    P = 0j
    D = 0j

    if ne > 0. and has_e:
        Se, Pe, De = SPD_el(w, b_norm, ne, nu_eis)
        S += Se
        P += Pe
        D += De

    for dens, mass, charge in zip(denses, masses, charges):
        if dens > 0. and has_i:
            Si, Pi, Di = SPD_ion(w, b_norm, dens, mass, charge, nu_eis)
            S += Si
            P += Pi
            D += Di

    M = array([[0.,   -1j*D, 0],
               [1j*D, S,    0],
               [0,   0,    P]])

    return M


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    '''
    standard SPD stix
    '''
    M = epsilonr_pl_cold_std(w, B, denses, masses,
                             charges, Te, ne, has_e, has_i)

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
    # print_mat(ans, 3, 3)
    return ans


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold1(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    '''
    stix SDP w/o 1
    '''
    M = epsilonr_pl_cold_SDP(w, B, denses, masses,
                             charges, Te, ne, has_e, has_i)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))

    ans = dot(R2(th), dot(A, R2(-th)))
    # print_mat(ans, 3, 3)
    return ans


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold2(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    '''
    stix SD only
    '''
    M = epsilonr_pl_cold_SD(w, B, denses, masses,
                            charges, Te, ne, has_e, has_i)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))

    ans = dot(R2(th), dot(A, R2(-th)))
    # print_mat(ans, 3, 3)
    return ans


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold3(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    '''
    stix P only
    '''
    M = epsilonr_pl_cold_P(w, B, denses, masses, charges, Te, ne, has_e, has_i)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))

    ans = dot(R2(th), dot(A, R2(-th)))
    # print_mat(ans, 3, 3)
    return ans


@njit(complex128[:, :](float64, float64[:], float64[:], darray_ro, iarray_ro, float64, float64, int32, int32))
def epsilonr_pl_cold4(w, B, denses, masses, charges, Te, ne, has_e, has_i):
    '''
    stix w/o xx
    '''
    M = epsilonr_pl_cold_woxx(w, B, denses, masses,
                              charges, Te, ne, has_e, has_i)

    def R1(ph):
        return array([[cos(ph), 0.,  sin(ph)],
                      [0,       1.,   0],
                      [-sin(ph), 0,  cos(ph)]], dtype=complex128)

    def R2(th):
        return array([[cos(th), -sin(th), 0],
                      [sin(th), cos(th), 0],
                      [0, 0,  1.]], dtype=complex128)

    th = arctan2(B[1], B[0])
    ph = arctan2(B[0]*cos(th)+B[1]*sin(th), B[2])
    A = dot(R1(ph), dot(M, R1(-ph)))

    ans = dot(R2(th), dot(A, R2(-th)))

    return ans
