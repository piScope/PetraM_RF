'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient, PhysConstant
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_Vac')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     default = 1.0, 
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     default = 1.0, 
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     default = 0.0, 
                                     tip = "contuctivity" )),
         ('kz', VtableElement('Nz', type='int',
                                     guilabel = 'kz',
                                     default = 0.0,
                                     no_func=True,
                                     tip = "out-of-plane wave number" )),)

from petram.phys.coefficient import SCoeff

from petram.phys.coefficient import PyComplexScalarInvCoefficient as ComplexScalarInv
from petram.phys.coefficient import PyComplexProductCoefficient as ComplexProduct

from petram.phys.phys_const import mu0, epsilon0

def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega       
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)


def Sigma_Coeff(exprs, ind_vars, l, g, omega):
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)

def Mu_Coeff(exprs, ind_vars, l, g, omega):
    # v = mu * v
    fac = mu0
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
       
class EM2D_Vac(EM2D_Domain):
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
        return [(0, 1, 1, 1), (1, 0, 1, 1)]
     
    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, kz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff([e], ind_vars, l, g, omega)
        coeff2 = Mu_Coeff([m], ind_vars, l, g, omega)
        coeff3 = Sigma_Coeff([s], ind_vars, l, g, omega)

        #dprint1("epsr, mur, sigma " + str(coeff1) + " " + str(coeff2) + " " + str(coeff3))
        return coeff1, coeff2, coeff3, kz

    def get_coeffs_2(self, real = True):
        # e, m, s
        coeff1, coeff2, coeff3, kz = self.get_coeffs()

        if self.has_pml():
            # not yet implemented!
            assert False, "Not Yet Implemetned"
            coeff1 = self.make_PML_epsilon(coeff1r, coeff1i, real)
            coeff2 = self.make_PML_invmu(coeff2r, coeff2i, real)
            coeff3 = self.make_PML_sigma(coeff3r, coeff3i, real)            
        else:
            neg_w2eps = coeff1
            tmp = ComplexScalarInv(coeff2)
            one_over_mu = tmp
            k2_over_mu =   ComplexProduct(tmp, kz*kz)
            ik_over_mu =   ComplexProduct(tmp, 1j*kz)
            neg_iwsigma = coeff3
            
        return  neg_w2eps, one_over_mu, k2_over_mu, ik_over_mu, neg_iwsigma, kz
            
    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        neg_w2eps, one_over_mu, k2_over_mu, ik_over_mu, neg_iwsigma, kz= self.get_coeffs_2(real)

        self.set_integrator_realimag_mode(real)
      
        if kfes == 0: ## ND element (Epoloidal)
            if real:       
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
                
            self.add_integrator(engine, 'mur', one_over_mu,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'epsilonr', neg_w2eps,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            self.add_integrator(engine, 'sigma', neg_iwsigma,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)

            if kz != 0:
                self.add_integrator(engine, 'mur', k2_over_mu,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
                
        elif kfes == 1: ## ND element (Epoloidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))

            self.add_integrator(engine, 'mur_2', one_over_mu,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', neg_w2eps,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
            self.add_integrator(engine, 'sigma', neg_iwsigma,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
        
    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
       
        neg_w2eps, one_over_mu, k2_over_mu, ik_over_mu, neg_iwsigma, kz = self.get_coeffs_2(real)
        
        self.set_integrator_realimag_mode(real)
        
        if r == 1 and c == 0:
            # (-a u_vec, div v_scalar)           
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
        else:
            itg =  mfem.MixedVectorGradientIntegrator

        self.add_integrator(engine, 'mur', ik_over_mu,
                                mbf.AddDomainIntegrator, itg)
        

    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant

        if len(self._sel_index) == 0: return
        var, f_name = self.eval_phys_expr(self.epsilonr, 'epsilonr')
        if callable(var):
            add_expression(v, 'epsilonr', suffix, ind_vars, f_name,
                           [], domains = self._sel_index, 
                           gdomain = self._global_ns)            
        else:
            add_constant(v, 'epsilonr', suffix, var,
                         domains = self._sel_index,
                         gdomain = self._global_ns)

        var, f_name = self.eval_phys_expr(self.mur, 'mur')
        if callable(var):
            add_expression(v, 'mur', suffix, ind_vars, f_name,
                           [], domains = self._sel_index,
                           gdomain = self._global_ns)            
        else:
            add_constant(v, 'mur', suffix, var,
                         domains = self._sel_index,
                         gdomain = self._global_ns)                        

        var, f_name = self.eval_phys_expr(self.sigma, 'sigma')
        if callable(var):
            add_expression(v, 'sigma', suffix, ind_vars, f_name,
                           [], domains = self._sel_index, 
                           gdomain = self._global_ns)            
        else:
            add_constant(v, 'sigma', suffix, var,
                         domains = self._sel_index,
                         gdomain = self._global_ns)


    
