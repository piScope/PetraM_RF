'''
   local-K plasma.
'''
from petram.phys.common.rf_dispersion_lkplasma import (vtable_data0,
                                                       default_kpe_option,
                                                       kpe_options)

import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.numba_coefficient import NumbaCoefficient

from petram.phys.phys_model import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain
from petram.phys.em1d.em1d_vac import EM1D_Vac

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_LocalKPlasma')

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data = vtable_data0.copy()
vtable_data.extend(
    [('ky', VtableElement('ky', type='float',
                          guilabel='ky',
                          default=0.,
                          no_func=True,
                          tip="wave number in the y direction")),
     ('kz', VtableElement('kz', type='float',
                          guilabel='kz',
                          default=0.0,
                          no_func=True,
                          tip="wave number in the z direction"))])

kpe_alg_options = ["std", "em1d"]


def domain_constraints():
    return [EM1D_LocalKPlasma]


class EM1D_LocalKPlasma(EM1D_Vac):
    allow_custom_intorder = False
    vt = Vtable(vtable_data)

    def __init__(self, **kargs):
        super(EM1D_LocalKPlasma, self).__init__(**kargs)
        self._jited_coeff = None

    def get_possible_child(self):
        return []

    def attribute_set(self, v):
        EM1D_Vac.attribute_set(self, v)
        v["kpe_mode"] = default_kpe_option
        v["kpe_alg"] = kpe_alg_options[0]
        return v

    def panel1_param(self):
        panels = super(EM1D_LocalKPlasma, self).panel1_param()
        panels.append(["kpe mode", None, 1, {"values": kpe_options}])
        panels.append(["kpe alg.", None, 1, {"values": kpe_alg_options}])

        return panels

    def get_panel1_value(self):
        values = super(EM1D_LocalKPlasma, self).get_panel1_value()
        values.extend([self.kpe_mode, self.kpe_alg])
        return values

    def import_panel1_value(self, v):
        check = super(EM1D_LocalKPlasma, self).import_panel1_value(v[:-2])
        self.kpe_mode = v[-2]
        self.kpe_alg = v[-1]
        return check

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, kpakpe, kpevec, ky, kz = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars
        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        from petram.phys.common.rf_dispersion_coldplasma import build_coefficients

        coeff1, coeff2, coeff3, coeff4 = build_coefficients(ind_vars, omega, B, dens_e, t_e,
                                                            dens_i, masses, charges, kpakpe, kpevec,
                                                            kpe_mode,
                                                            self._global_ns, self._local_ns,
                                                            kpe_alg=kpe_alg,
                                                            sdim=1, terms=self.stix_terms)
        return coeff1, coeff2, coeff3, coeff4, ky, kz

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        coeff1, coeff2, coeff3, coeff4, _coeff_nuei,  ky, kz = self.jited_coeff
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

        coeff1, coeff2, coeff3, coeff4, _coeff_nuei, ky, kz = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        # super(EM1D_LocalKPlasma, self).add_mix_contribution(engine, mbf, r, c, is_trans,
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

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_expression, add_constant

        if len(self._sel_index) == 0:
            return

        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, kpakpe, kpevec, ky, kz = self.vt.make_value_or_expression(
            self)

        ind_vars = self.get_root_phys().ind_vars
        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        add_constant(v, 'ky', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)
        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

        from petram.phys.common.rf_dispersion_lkplasma import build_variables

        ss = self.parent.parent.name()+'_'+self.name()  # phys module name + name
        ret = build_variables(v, ss, ind_vars,
                              omega, B, dens_e, t_e,
                              dens_i, masses, charges,
                              kpakpe, kpevec, kpe_mode, kpe_alg,
                              self._global_ns, self._local_ns,
                              sdim=1)

        v["_e_"+ss] = ret[0]
        v["_m_"+ss] = ret[1]
        v["_s_"+ss] = ret[2]
        v["_spd_"+ss] = ret[3]
        v["_nuei_"+ss] = ret[4]
        v["_eac_"+ss] = ret[5]
        v["_eae_"+ss] = ret[6]
        v["_eai_"+ss] = ret[7]
        v["_nref_"+ss] = ret[8]

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', [
                                "_e_"+ss + "/(-omega*omega*e0)"], ["omega"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonrac', [
                                "_eac_"+ss + "/(-omega*omega*e0)"], ["omega"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonrae', [
                                "_eae_"+ss + "/(-omega*omega*e0)"], ["omega"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonrai', [
                                "_eai_"+ss + "/(-omega*omega*e0)"], ["omega"])

        add_expression(v, 'Pcol', suffix, ind_vars,
                       "omega*conj(E).dot(epsilonrac.dot(E))/2j*e0", ['E', 'epsilonrac', 'omega'])
        add_expression(v, 'Pabsi1', suffix, ind_vars,
                       "omega*conj(E).dot(epsilonrai[0].dot(E))/2j*e0", ['E', 'epsilonrac', 'omega'])
        add_expression(v, 'Pabsi2', suffix, ind_vars,
                       "omega*conj(E).dot(epsilonrai[1].dot(E))/2j*e0", ['E', 'epsilonrac', 'omega'])
        add_expression(v, 'Pabsi3', suffix, ind_vars,
                       "omega*conj(E).dot(epsilonrai[2].dot(E))/2j*e0", ['E', 'epsilonrac', 'omega'])

        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'mur', ["_m_"+ss + "/mu0"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', [
                                "_s_"+ss + "/(-1j*omega)"], ["omega"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'nuei', ["_nuei_"+ss])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Sstix', ["_spd_"+ss+"[0,0]"])
        self.do_add_matrix_expr(v, suffix, ind_vars, 'Dstix', [
                                "1j*_spd_"+ss+"[0,1]"])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Pstix', ["_spd_"+ss+"[2,2]"])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Rstix', ["_spd_"+ss+"[0,0] + 1j*_spd_"+ss+"[0,1]"])
        self.do_add_matrix_expr(v, suffix, ind_vars,
                                'Lstix', ["_spd_"+ss+"[0,0] - 1j*_spd_"+ss+"[0,1]"])
        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        return
