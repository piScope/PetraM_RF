'''
   cold plasma.
'''
import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff
from petram.phys.numba_coefficient import NumbaCoefficient

from petram.phys.phys_model import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain
from petram.phys.em1d.em1d_vac import EM1D_Vac

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_ColdPlasma')

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

data = (('B', VtableElement('bext', type='array',
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
                               tip="mass. use  m_h, m_d, m_t, or u")),
        ('charge_q', VtableElement('charge_q', type='array',
                                   guilabel='ion charges(/q)',
                                   default="1, 1",
                                   no_func=True,
                                   tip="ion charges normalized by q(=1.60217662e-19 [C])")),
         ('ky', VtableElement('ky', type='int',
                                     guilabel = 'ky',
                                     default = 0. ,
                                     no_func = True, 
                                     tip = "wave number in the y direction" )),
         ('kz', VtableElement('kz', type='int',
                                     guilabel = 'kz',
                                     default = 0.0,
                                     no_func = True,                               
                                     tip = "wave number in the z direction" )),)


def domain_constraints():
    return [EM1D_ColdPlasma]


class EM1D_ColdPlasma(EM1D_Vac):
    allow_custom_intorder = False
    vt = Vtable(data)
    #nlterms = ['epsilonr']

    def get_possible_child(self):
        return []
    
    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def get_coeffs(self, real=True, return_stix=False):
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, ky, kz = self.vt.make_value_or_expression(
            self)

        Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)        
        masses = np.array(masses, dtype=np.float64) * Da
        charges = np.array(charges, dtype=np.int64)

        num_ions = len(masses)
        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

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
            out = epsilonr_pl_cold_std(omega, B, dens_i, masses, charges, t_e, dens_e)
            return out
        
        def mur(ptx):
            return mu0*np.eye(3, dtype=np.complex128)
        
        def sigma(ptx):
            return - 1j*omega * np.zeros((3,3), dtype=np.complex128)
            

        from .dispersion_cold import epsilonr_pl_cold, epsilonr_pl_cold_std

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
        
        coeff1 = NumbaCoefficient(mfem_coeff1)
        coeff2 = NumbaCoefficient(mfem_coeff2)
        coeff3 = NumbaCoefficient(mfem_coeff3)

        if return_stix:
            mfem_coeff4 = jitter(sdp)
            coeff4 = NumbaCoefficient(mfem_coeff4)
            return coeff1, coeff2, coeff3, coeff4
        else:
            return coeff1, coeff2, coeff3

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        coeff1, _coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)

        #if self.has_pml():
        #    coeff1 = self.make_PML_coeff(coeff1)
        #    coeff2 = self.make_PML_coeff(coeff2)
        #    coeff3 = self.make_PML_coeff(coeff3)
        #coeff2 = coeff2.inv()
        ec = coeff1[kfes, kfes]
        sc = coeff3[kfes, kfes]        

        super(EM1D_Anisotropic, self).add_bf_contribution(engine, a, real=real,
                                                          kfes=kfes, ecsc=(ec,sc))

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
       
        coeff1, _coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)

        super(EM1D_Anisotropic, self).add_mix_contribution(engine, mbf, r, c, is_trans,
                                                           real = real)
        ec = coeff1[r, c]        
        sc = coeff3[r, c]

        self.add_integrator(engine, 'epsilonr', ec, mbf.AddDomainIntegrator,
                            mfem.MixedScalarMassIntegrator)
        self.add_integrator(engine, 'sigma', sc, mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)

    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli=None):
        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import (NativeCoefficientGenBase,
                                             NumbaCoefficientVariable)

        
        if len(self._sel_index) == 0:
            return
    
        coeff1, coeff2, coeff3, coeff4 = self.get_coeffs(return_stix=True)

        c1 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3,3))
        c2 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3,3))
        c3 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3,3))
        c4 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3,3))        

        ss = str(hash(self.fullname()))
        v["_e_"+ss] = c1
        v["_m_"+ss] = c2
        v["_s_"+ss] = c3
        v["_spd_"+ss] = c4

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', ["_e_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', ["_m_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', ["_s_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'Sstix', ["_spd_"+ss+"[0,0]"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'Dstix', ["1j*_spd_"+ss+"[0,1]"])         
        self.do_add_matrix_expr(v, suffix, ind_vars, 'Pstix', ["_spd_"+ss+"[2,2]"])        
        
        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        

