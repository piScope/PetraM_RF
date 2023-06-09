'''
   Cold plasma:
'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient, VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient, Coefficient_Evaluator
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_ColdPlasma')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable
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
        ('t_mode', VtableElement('t_mode', type="int",
                                 guilabel = 'm',
                                 default = 0.0, 
                                 tip = "mode number" )),)

from petram.phys.numba_coefficient import (func_to_numba_coeff_scalar,
                                           func_to_numba_coeff_vector,
                                           func_to_numba_coeff_matrix)
from petram.phys.phys_const import mu0, epsilon0

'''
Expansion of matrix is as follows

               [e_rz  e_12 ][Erz ]
[Wrz, Wphx] =  [           ][    ] = Wrz e_rz Erz + Wrz e_12 Ephi 
               [e_21  e_phi][Ephx]

                                + Wphi e_21 Erz + Wphi*e_phi*Ephi


  Erz = Er e_r + Ez e_z
  Ephx = - rho Ephi
'''

def domain_constraints():
   return [EM2Da_ColdPlasma]
       
class EM2Da_ColdPlasma(EM2Da_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']
    
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        elif kfes == 1: return True        
        else: return False
        
    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(0, 1, 1, 1), (1, 0, 1, 1),]#(0, 1, -1, 1)]
     
    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges, tmode = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars

        from petram.phys.rf_dispersion_coldplasma import build_coefficients
        coeff1, coeff2, coeff3, coeff4 = build_coefficients(ind_vars, omega, B, dens_e, t_e,
                                                            dens_i, masses, charges,
                                                            self._global_ns, self._local_ns,)
        return coeff1, coeff2, coeff3, coeff4, tmode
        
    def add_bf_contribution(self, engine, a, real = True, kfes=0):

        coeff1, coeff2, coeff3, coeff_stix, kz = self.jited_coeff
        invmu = coeff2.inv()
        coeff4 = coeff1 + coeff3

        eps11 = coeff4[[0, 2], [0, 2]]
        eps22 = coeff4[1, 1]

        def invmu_x_r(ptx, invmu):
            return invmu[0,0]*ptx[0]
        def invmu_o_r(ptx, invmu_in):
            return invmu[0,0]/ptx[0]
        def invmu_o_r_tt(ptx, invmu_in):
            return invmu[0,0]/ptx[0]*tmode*tmode

         
        def eps11_x_r(ptx, eps11):         
            return eps11*ptx[0]
        def eps22_o_r(ptx, eps22):
            return eps22/ptx[0]

        if kfes == 0: ## ND element (Epoloidal)
            if real:       
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))


            se_x_r_rz = func_to_numba_coeff_matrix(eps11_x_r, shape=(2,2),
                                                   complex=True,
                                                   depencency=(eps11,))
            imu_x_r = func_to_numba_coeff_scalar(invmu_x_r, 
                                                 complex=True,
                                                 depencency=(invmu,))
            
            self.add_integrator(engine, 'mur', imu_x_r,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'sigma_epsilonr', se_x_r_rz,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            
            if tmode != 0:
                imu_o_r_2 = func_to_numba_coeff_scalar(invmu_o_r_tt, 
                                                       complex=True,
                                                        depencency=(invmu,))
                
                self.add_integrator(engine, 'mur', imu_o_r_2,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
                
        elif kfes == 1: ## H1 element (Etoroidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))

                
            se_o_r = func_to_numba_coeff_scalar(eps22_o_r, 
                                                complex=True,
                                                depencency=(eps22,))
            imv_o_r_1 = func_to_numba_coeff_scalar(invmu_o_r,
                                                   complex=True,
                                                   depencency=(invmu,))
            
            self.add_integrator(engine, 'mur', imv_o_r_1,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', se_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
         
        '''
        if kfes == 0: ## ND element (Epoloidal)
            if real:       
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
            imu_x_r = InvMu_x_r(m,  self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real)
            s_x_r = Sigma_x_r_rz(2, s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)
            e_x_r = Epsilon_x_r_rz(2, e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            
            self.add_integrator(engine, 'mur', imu_x_r,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'epsilonr', e_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            self.add_integrator(engine, 'sigma', s_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            
            if tmode != 0:
                imu_o_r_2 = InvMu_m2_o_r(m,  self.get_root_phys().ind_vars,
                                      self._local_ns, self._global_ns,
                                      real = real, tmode = tmode)
                self.add_integrator(engine, 'mur', imu_o_r_2,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
                
        elif kfes == 1: ## H1 element (Etoroidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))
            imv_o_r_1 = InvMu_o_r(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real)
            e_o_r = Epsilon_o_r_phi(e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            s_o_r = Sigma_o_r_phi(s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)
            
            self.add_integrator(engine, 'mur', imv_o_r_1,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', e_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
            self.add_integrator(engine, 'sigma', s_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
        '''
        
    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
       
        coeff1, coeff2, coeff3, coeff_stix, kz = self.jited_coeff
        invmu = coeff2.inv()
        coeff4 = coeff1 + coeff3

        eps21 = coeff4[[0, 2], 1]
        eps12 = coeff4[1, [0, 2]]

        def iinvmu_o_r_t(ptx, invmu):
            return 1j*invmu[0,0]/ptx[0]*tmode
         
        se_21 = - esp21
        se_12 = - esp12
        
        if r == 1 and c == 0:
            imv_o_r_3 = func_to_numba_coeff_scalar(iinvmu_o_r_t
                                                   complex=True,
                                                   depencency=(invmu,))

            #if  is_trans:
            # (a_vec dot u_vec, v_scalar)                        
            itg = mfem.MixedDotProductIntegrator
            self.add_integrator(engine, 'sigma_epsilon_21', se21,
                                mbf.AddDomainIntegrator, itg)
            # (-a u_vec, div v_scalar)            
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)

        else:
            itg = mfem.MixedVectorProductIntegrator
            self.add_integrator(engine, 'sigma_epsilon_12', se12,
                             mbf.AddDomainIntegrator, itg)

            # (a grad u_scalar, v_vec)

            itg =  mfem.MixedVectorGradientIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                             mbf.AddDomainIntegrator, itg)
           
        '''
        imv_o_r_3 = iInvMu_m_o_r(m,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, tmode = -tmode)
        if r == 1 and c == 0:        
            e = Epsilon_21(2, e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            s = Sigma_21(2, s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)
            #if  is_trans:
            # (a_vec dot u_vec, v_scalar)                        
            itg = mfem.MixedDotProductIntegrator
            self.add_integrator(engine, 'epsilon', e,
                                mbf.AddDomainIntegrator, itg)
            self.add_integrator(engine, 'sigma', s,
                                mbf.AddDomainIntegrator, itg)
            # (-a u_vec, div v_scalar)            
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)
            #print r, c, mbf
        else:
            #if is_trans:
            #    pass
            #else:               
            e = Epsilon_12(2, e, self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real = real, omega = omega)
            s = Sigma_12(2, s,  self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns,
                           real = real, omega = omega)

            itg = mfem.MixedVectorProductIntegrator
            self.add_integrator(engine, 'epsilon', e,
                             mbf.AddDomainIntegrator, itg)
            self.add_integrator(engine, 'sigma', s,
                             mbf.AddDomainIntegrator, itg)
            # (a grad u_scalar, v_vec)

            itg =  mfem.MixedVectorGradientIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                             mbf.AddDomainIntegrator, itg)
        '''
        
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):

        from petram.helper.variables import add_expression, add_constant

        coeff1, coeff2, coeff3, coeff_stix, t_mode = self.jited_coeff

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

        add_constant(v, 't_mode', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)




    
