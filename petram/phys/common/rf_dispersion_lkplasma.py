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

from petram.phys.phys_const import c as speed_of_light

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data0= [('B', VtableElement('bext', type='array',
                                   guilabel='magnetic field',
                                   default="=[0,0,0]",
                                   tip="external magnetic field")),
               ('dens_e', VtableElement('dens_e', type='float',
                                        guilabel='electron density(m-3)',
                                        default="1e19",
                                        tip="electron density")),
               ('temperature_e', VtableElement('temperature_e', type='float',
                                             guilabel='electron temp.(eV)',
                                             default="10.",
                                             tip="electron temperature")),
               ('dens_i', VtableElement('dens_i', type='array',
                                        guilabel='ion densities(m-3)',
                                        default="0.9e19, 0.1e19",
                                        tip="ion densities")),
               ('temperatures_i', VtableElement('temperatures_i', type='array',
                                             guilabel='ion temps.(eV)',
                                             default="100., 100",
                                             tip="ion temperatures")),
               ('temperatures_c', VtableElement('temperatures_c', type='float',
                                             guilabel='Tcol(eV)',
                                             default="100.",
                                             tip="temperature used for collision")),
               ('mass', VtableElement('mass', type='array',
                                      guilabel='ion masses(/Da)',
                                      default="2, 1",
                                      no_func=True,
                                      tip="mass. normalized by atomic mass unit")),
               ('charge_q', VtableElement('charge_q', type='array',
                                          guilabel='ion charges(/q)',
                                          default="1, 1",
                                          no_func=True,
                                          tip="ion charges normalized by q(=1.60217662e-19 [C])")),
               ('kpa_kpe', VtableElement('kpa_kpe', type='array',
                                          guilabel='kpa, kpe',
                                          default="1, 1.",
                                          tip="k_parallel and k_perp for computing dielectric. ")),
               ('kpe_vec', VtableElement('kpe_vec', type='array',
                                          guilabel='kpe dir.',
                                          default="0, 0, 1",
                                          tip="k_perp direction. Adjusted to be normal to the magnetic field.")),]

def make_functions():
    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
                                                                   f_collisions)
    from petram.phys.common.rf_dispersion_lkplasma_numba import (epsilonr_pl_hot_std,
                                                                   rottate_dielectric)
    
    def epsilonr(ptx, B, t_c, dens_e, t_e, dens_i, t_i, kpakpe, kpevec ):        
        e_cold = epsilonr_pl_cold_std(omega, B, dens_i, masses, charges, t_e, dens_e)

        e_hot =  epsilonr_pl_hot_std(omega, B, t_i, dens_i,  masses, charges,
                                     t_e, dens_e,
                                     c*kpakpe[0]/omega, c*kpakpe[1]/omega, nhrms)

        e_colda = (e_cold - e_cold.transpose().conj())/2.0 # anti_hermitian (collisional abs.)


        out = -epsilon0 * omega * omega * (e_colda + e_hot)
        out = rotate_dielectric(B, kpevec, out)
        
        return out

    def sdp(ptx, B, dens_e, t_e, dens_i):        
            out = epsilonr_pl_cold_std(
                omega, B, dens_i, masses, charges, t_e, dens_e)
            return out

    def mur(ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    return epsilonr, sdp, mur, sigma


def make_afunction_variable():
    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
                                                                   f_collisions)
    from petram.phys.common.rf_dispersion_lkplasma_numba import (epsilonr_pl_hot_std,
                                                                   rottate_dielectric)


    def epsilonr(*_ptx, B=None, t_c=None, dens_e=None, t_e=None, dens_i=None, t_i=None, kpakpe=None, kpevec=None):        
        e_cold = epsilonr_pl_cold_std(omega, B, dens_i, masses, charges, t_e, dens_e)

        e_hot =  epsilonr_pl_hot_std(omega, B, t_i, dens_i,  masses, charges,
                                     t_e, dens_e,
                                     c*kpakpe[0]/omega, c*kpakpe[1]/omega, nhrms)

        e_colda = (e_cold - e_cold.transpose().conj())/2.0 # anti_hermitian (collisional abs.)


        out = -epsilon0 * omega * omega * (e_colda + e_hot)
        out = rotate_dielectric(B, kpevec, out)
        
        return out

    def sdp(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            out = epsilonr_pl_cold_std(
                omega, B, dens_i, masses, charges, t_e, dens_e)
            return out

    def mur(*_ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(*_ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    def nuei(*_ptx, dens_e=None, t_e=None, dens_i=None):
        # iidx : index of ions
        nuei = f_collisions(dens_i, charges, t_e, dens_e)
        return nuei[iidx]

    def epsilonrac(*_ptx, B=None, t_c=None, dens_e=None, t_e=None, dens_i=None, t_i=None, kpakpe=None, kpevec=None):
        
        e_cold = epsilonr_pl_cold_std(omega, B, dens_i, masses, charges, t_e, dens_e)
        e_colda = (e_cold - e_cold.transpose().conj())/2.0 # anti_hermitian (collisional abs.)
        out = -epsilon0 * omega * omega * e_colda
        out = rotate_dielectric(B, kpevec, out)
        
        return out

    def epsilonrae(*_ptx, B=None, t_c=None, dens_e=None, t_e=None, dens_i=None, t_i=None, kpakpe=None, kpevec=None):
        
        dens_i = np.array([0]*len(dens_i))

        e_hot =  epsilonr_pl_hot_std(omega, B, t_i, dens_i,  masses, charges,
                                     t_e, dens_e,
                                     c*kpakpe[0]/omega, c*kpakpe[1]/omega, nhrms)

        e_hota = (e_hot - e_hot.transpose().conj())/2.0 
        out = -epsilon0 * omega * omega * e_hota
        out = rotate_dielectric(B, kpevec, out)
        
        return out

    def epsilonrai(*_ptx, B=None, t_c=None, dens_e=None, t_e=None, dens_i=None, t_i=None, kpakpe=None, kpevec=None):
                          
        dens_e = 0.0

        ret = np.zeros((len(masses), 3, 3), dtype=np.complex128)
        for i in range(masses):
            dens_i2 = np.zeros(len(masses))
            dens_i2[i] = dens_i[i]              
            e_hot =  epsilonr_pl_hot_std(omega, B, t_i, dens_i2,  masses, charges,
                                     t_e, dens_e,
                                     c*kpakpe[0]/omega, c*kpakpe[1]/omega, nhrms)

            e_hota = (e_hot - e_hot.transpose().conj())/2.0 # anti_hermitian (collisional abs.)
            out = -epsilon0 * omega * omega * e_hota
            out = rotate_dielectric(B, kpevec, out)
                          
            ret[i, :, :] = out
        return ret

    return epsilonr, sdp, mur, sigma, nuei, epsilonrac, epsilonrae, epsilonrai


def build_coefficients(ind_vars, omega, B, t_c, dens_e, t_e, dens_i, t_i,
                       masses, charges, kpakpe, kpevec, g_ns, l_ns,
                       sdim=3):



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
    t_c_coeff = SCoeff([t_c, ], ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)
    dens_i_coeff = VCoeff(num_ions, [dens_i, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)
    t_i_coeff = VCoeff(num_ions, [t_i, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)
    kpakpe_coeff = VCoeff(2, kpakpe, ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)
    kpevec_coeff = VCoeff(2, kpevec, ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)


    params = {'omega': omega, 'masses': masses, 'charges': charges, 'nhrms': 20,
              'c':speed_of_light}

    epsilonr, sdp, mur, sigma = make_functions()
    
    numba_debug = False if myid != 0 else get_numba_debug()

    dependency = (B_coeff, t_c_coeff, dens_e_coeff, t_e_coeff,
                  dens_i_coeff, t_i_coeff, kpakpe_coeff, kpevec_coeff)
    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.matrix(sdim=sdim, shape=(3, 3), complex=True, params=params,
                             debug=numba_debug, dependency=dependency)
    mfem_coeff1 = jitter(epsilonr)

    jitter2 = mfem.jit.matrix(sdim=sdim, shape=(3, 3), complex=True, params=params,
                              debug=numba_debug)
    mfem_coeff2 = jitter2(mur)
    mfem_coeff3 = jitter2(sigma)
    mfem_coeff4 = jitter(sdp)

    coeff1 = NumbaCoefficient(mfem_coeff1)
    coeff2 = NumbaCoefficient(mfem_coeff2)
    coeff3 = NumbaCoefficient(mfem_coeff3)
    coeff4 = NumbaCoefficient(mfem_coeff4)

    return coeff1, coeff2, coeff3, coeff4


def build_variables(solvar, ss, ind_vars, omega, B, t_c, dens_e, t_e, dens_i, t_i,
                    masses, charges, kpakpe, kpevec, g_ns, l_ns, sdim=3):

    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
                                                                   epsilonr_pl_cold,
                                                                   f_collisions)
    from petram.phys.common.rf_dispersion_lkplasma_numba import (epsilonr_pl_hot_std,
                                                                   epsilonr_pl_hot)


    Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

    masses = np.array(masses, dtype=np.float64) * Da
    charges = np.array(charges, dtype=np.int32)

    num_ions = len(masses)
    l = l_ns
    g = g_ns

    def func(x, dens_smooth=None):
        return dens_smooth

    from petram.helper.variables import (variable,
                                         Constant,
                                         ExpressionVariable,
                                         NumbaCoefficientVariable,
                                         PyFunctionVariable)
    d1 = variable.jit.float(dependency=("dens_smooth",))(func)

    def make_variable(x):
        if isinstance(x, str):
            ind_vars_split = [x.strip() for x in ind_vars.split(',')]
            d1 = ExpressionVariable(x, ind_vars_split)
        else:
            d1 = Constant(x)
        return d1

    B_var = make_variable(B)
    te_var = make_variable(t_e)
    ti_var = make_variable(t_i)
    tc_var = make_variable(t_c)                          
    dense_var = make_variable(dens_e)
    densi_var = make_variable(dens_i)

    kpakpe_var = make_variable(kpakpe)
    kpevec_var = make_variable(kpevec)

    params = {'omega': omega, 'masses': masses, 'charges': charges, 'nhrms': 20,
              'c':speed_of_light}

    epsilonr, sdp, mur, sigma, nuei, epsilonrac, epsilonrae, epsilonrai = make_functions_variable()    


    solvar["B_"+ss] = B_var
    solvar["tc_"+ss] = tc_var                                                    
    solvar["ne_"+ss] = dense_var
    solvar["te_"+ss] = te_var
    solvar["ni_"+ss] = densi_var
    solvar["ti_"+ss] = te_var
                          
    dependency = ("B_"+ss, "tc_"+ss, "ne_"+ss, "te_"+ss, "ni_"+ss,  "ti_"+ss,
                  "kpakpe_"+ss, "kpevec_"+ss)

    var1 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(epsilonr)
    var2 = variable.array(complex=True, shape=(3, 3),
                          params=params)(mur)
    var3 = variable.array(complex=True, shape=(3, 3),
                          params=params)(sigma)
    var4 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(sdp)
    var5 = variable.array(complex=True, shape=(len(masses),),
                          dependency=dependency, params=params)(nuei)

    var6 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(epsilonrac)
    var7 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(epsilonrae)

    var8 = variable.array(complex=True, shape=(len(masses), 3, 3),
                          dependency=dependency, params=params)(epsilonrai)


    return var1, var2, var3, var4, var5, var6, var7, var8