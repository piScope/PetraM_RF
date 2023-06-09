'''
   Anistropic media:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma

    Expansion of matrix is as follows

               [e_xy  e_12 ][Exy ]
[Wxy, Wz]   =  [           ][    ] = Wrz e_xy Erz + Wxy e_12 Ez
               [e_21  e_zz][Ez  ]

                                   + Wz e_21 Exy + Wz*e_zz*Exy


  Exy = Ex e_x + Ey e_y
  Ez =  Ez e_z

'''
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain, EM2D_Domain_helper
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_ColdPlasma')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

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
        ('kz', VtableElement('kz', type='int',
                             guilabel='kz',
                             default=0.0,
                             no_func=True,
                             tip="out-of-plane wave number")),)


def domain_constraints():
    return [EM2D_ColdPlasma]


class EM2D_ColdPlasma(EM2D_Domain, EM2D_Domain_helper):
    vt = Vtable(data)
    #nlterms = ['epsilonr']

    def get_possible_child(self):
        from .em2d_pml import EM2D_LinearPML
        return [EM2D_LinearPML]

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        elif kfes == 1:
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(0, 1, 1, 1), (1, 0, 1, 1), ]  # (0, 1, -1, 1)]

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, kz = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars

        from petram.phys.rf_dispersion_coldplasma import build_coefficients
        coeff1, coeff2, coeff3, coeff4 = build_coefficients(ind_vars, omega, B, dens_e, t_e,
                                                            dens_i, masses, charges,
                                                            self._global_ns, self._local_ns,)
        return coeff1, coeff2, coeff3, coeff4, kz

    def get_coeffs_2(self):
        # e, m, s
        coeff1, coeff2, coeff3, coeff_stix, kz = self.jited_coeff
        '''
        coeff4 = ComplexMatrixSum(coeff1, coeff3)      # -> coeff4 = coeff1 + coeff3
        '''
        coeff4 = coeff1 + coeff3

        if self.has_pml():
            coeff4 = self.make_PML_coeff(coeff4)
            coeff2 = self.make_PML_coeff(coeff2)

        eps11 = coeff4[[0, 1], [0, 1]]
        eps21 = coeff4[[0, 1], 2]
        eps12 = coeff4[2, [0, 1]]
        eps22 = coeff4[2, 2]

        eps = [eps11, eps12, eps21, eps22]

        tmp = coeff2.inv()

        mu11 = tmp[[0, 1], [0, 1]]
        mu11 = mu11.adj()
        mu22 = tmp[2, 2]
        k2_over_mu11 = mu11*(kz*kz)
        ik_over_mu11 = mu11*(1j*kz)

        mu = [mu11, mu22, k2_over_mu11, ik_over_mu11]

        return eps, mu, kz

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        #freq, omega = self.get_root_phys().get_freq_omega()
        eps, mu, kz = self.get_coeffs_2()

        self.set_integrator_realimag_mode(real)

        if real:
            if kfes == 0:
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            elif kfes == 1:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
        else:
            if kfes == 0:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
            elif kfes == 1:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))

        self.call_bf_add_integrator(eps,  mu, kz,  engine, a, kfes)

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real=True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))

        #freq, omega = self.get_root_phys().get_freq_omega()
        eps, mu, kz = self.get_coeffs_2()

        self.set_integrator_realimag_mode(real)
        self.call_mix_add_integrator(eps, mu, engine, mbf, r, c, is_trans)

    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli=None):

        from petram.helper.variables import add_expression, add_constant

        coeff1, coeff2, coeff3, coeff_stix, kz = self.jited_coeff

        c1 = NumbaCoefficientVariable(coeff1, complex=True, shape=(3, 3))
        c2 = NumbaCoefficientVariable(coeff2, complex=True, shape=(3, 3))
        c3 = NumbaCoefficientVariable(coeff3, complex=True, shape=(3, 3))
        c4 = NumbaCoefficientVariable(coeff_stix, complex=True, shape=(3, 3))

        ss = str(hash(self.fullname()))
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

        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

        '''
        e, m, s, kz = self.vt.make_value_or_expression(self)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', m)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        
        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains = self._sel_index,
                     gdomain = self._global_ns)

        '''
