'''
   cold plasma.
'''
import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
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
                               tip="mass. normalized by atomic mass unit")),
        ('charge_q', VtableElement('charge_q', type='array',
                                   guilabel='ion charges(/q)',
                                   default="1, 1",
                                   no_func=True,
                                   tip="ion charges normalized by q(=1.60217662e-19 [C])")),
        ('ky', VtableElement('ky', type='float',
                             guilabel='ky',
                             default=0.,
                             no_func=True,
                             tip="wave number in the y direction")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.0,
                             no_func=True,
                             tip="wave number in the z direction")),)


def domain_constraints():
    return [EM1D_ColdPlasma]


class EM1D_ColdPlasma(EM1D_Vac):
    allow_custom_intorder = False
    vt = Vtable(data)

    def __init__(self, **kargs):
        super(EM1D_ColdPlasma, self).__init__(**kargs)
        self._jited_coeff = None

    def get_possible_child(self):
        return []

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, ky, kz = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars

        from petram.phys.common.rf_dispersion_coldplasma import build_coefficients
        coeff1, coeff2, coeff3, coeff4 = build_coefficients(ind_vars, omega, B, dens_e, t_e,
                                                            dens_i, masses, charges,
                                                            self._global_ns, self._local_ns,)
        return coeff1, coeff2, coeff3, coeff4, ky, kz

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        coeff1, coeff2, coeff3, coeff4, ky, kz = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        # if self.has_pml():
        #    coeff1 = self.make_PML_coeff(coeff1)
        #    coeff2 = self.make_PML_coeff(coeff2)
        #    coeff3 = self.make_PML_coeff(coeff3)

        ec = coeff1[kfes, kfes]
        sc = coeff3[kfes, kfes]

        self.add_integrator(engine, 'epsilonr', ec, a.AddDomainIntegrator,
                            mfem.MassIntegrator)
        self.add_integrator(engine, 'sigma', sc, a.AddDomainIntegrator,
                            mfem.MassIntegrator)

        coeff4 = 1./coeff2[0, 0]
        #
        #
        #
        if kfes == 0:  # Ex
            imu = coeff4*(ky**2 + kz**2)
            self.add_integrator(engine, 'mur', imu, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        elif kfes == 1 or kfes == 2:  # Ey and Ez
            imu = coeff4
            self.add_integrator(engine, 'mur1', imu, a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)

            if kfes == 1:
                fac = kz*kz
            if kfes == 2:
                fac = ky*ky

            imu = coeff4*fac

            self.add_integrator(engine, 'mur2', imu, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real=True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))

        coeff1, coeff2, coeff3, coeff4, ky, kz = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        # super(EM1D_ColdPlasma, self).add_mix_contribution(engine, mbf, r, c, is_trans,
        #                                                   real = real)
        ec = coeff1[r, c]
        sc = coeff3[r, c]

        self.add_integrator(engine, 'epsilonr', ec, mbf.AddDomainIntegrator,
                            mfem.MixedScalarMassIntegrator)
        self.add_integrator(engine, 'sigma', sc, mbf.AddDomainIntegrator,
                            mfem.MixedScalarMassIntegrator)

        coeff4 = 1./coeff2[0, 0]
        if r == 0 and c == 1:
            imu = coeff4*(1j*ky)
            itg = mfem.MixedScalarDerivativeIntegrator
        elif r == 0 and c == 2:
            imu = coeff4*(1j*kz)
            itg = mfem.MixedScalarDerivativeIntegrator
        elif r == 1 and c == 0:
            imu = coeff4*(1j*ky)
            itg = mfem.MixedScalarWeakDerivativeIntegrator
        elif r == 1 and c == 2:
            imu = coeff4*(-ky*kz)
            itg = mfem.MixedScalarMassIntegrator
        elif r == 2 and c == 0:
            imu = coeff4*(1j*kz)
            itg = mfem.MixedScalarWeakDerivativeIntegrator
        elif r == 2 and c == 1:
            imu = coeff4*(-ky*kz)
            itg = mfem.MixedScalarMassIntegrator
        else:
            assert False, "Something is wrong..if it comes here;D"

        self.add_integrator(engine, 'mur', imu,
                            mbf.AddDomainIntegrator, itg)

    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli=None):
        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import (NativeCoefficientGenBase,
                                             NumbaCoefficientVariable)

        if len(self._sel_index) == 0:
            return

        coeff1, coeff2, coeff3, coeff4, ky, kz = self.jited_coeff

        c1 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3, 3))
        c2 = NumbaCoefficientVariable(coeff2, complex=True, shape=(3, 3))
        c3 = NumbaCoefficientVariable(coeff3, complex=True, shape=(3, 3))
        c4 = NumbaCoefficientVariable(coeff4, complex=True, shape=(3, 3))

        ss = str(abs(hash(self.fullname())))
        v["_e_"+ss] = c1
        v["_m_"+ss] = c2
        v["_s_"+ss] = c3
        v["_spd_"+ss] = c4

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', ["_e_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', ["_m_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', ["_s_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Sstix', ["_spd_"+ss+"[0,0]"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'Dstix', [
                                "1j*_spd_"+ss+"[0,1]"])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Pstix', ["_spd_"+ss+"[2,2]"])

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        
        add_constant(v, 'ky', suffix, np.float64(kz),
                     domains = self._sel_index,
                     gdomain = self._global_ns)
        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains = self._sel_index,
                     gdomain = self._global_ns)
        
