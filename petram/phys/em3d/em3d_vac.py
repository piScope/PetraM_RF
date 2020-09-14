'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient, PhysConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Vac')

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
                                     tip = "contuctivity" )),)

from petram.phys.coefficient import SCoeff
from petram.phys.coefficient import PyComplexPowCoefficient as ComplexPow

from petram.phys.phys_const import mu0, epsilon0

def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega       
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Sigma_Coeff(exprs, ind_vars, l, g, omega):
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Mu_Coeff(exprs, ind_vars, l, g, omega):
    # v = mu * v
    fac = mu0
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

 
'''
class Epsilon(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0
       v = super(Epsilon, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega
       if self.real:  return v.real
       else: return v.imag
       
class Sigma(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0
       v = super(Sigma, self).EvalValue(x)
       v = - 1j * self.omega * v
       if self.real:  return v.real
       else: return v.imag
'''
class InvMu(PhysCoefficient):
   #   1./mu0/mur
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)      
       super(InvMu, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0      
       v = super(InvMu, self).EvalValue(x)
       v = 1/mu0/v
       if self.real:  return v.real
       else: return v.imag

class EM3D_Vac(EM3D_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']
    
    def get_possible_child(self):
        from .em3d_pml      import EM3D_LinearPML
        return [EM3D_LinearPML]
     
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff([e], ind_vars, l, g, omega)
        coeff2 = Mu_Coeff([m], ind_vars, l, g, omega)
        coeff3 = Sigma_Coeff([s], ind_vars, l, g, omega)

        '''
        if isinstance(e, str):
           coeff1 = Epsilon(e,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           eps =  - e*epsilon0*omega*omega
           eps = eps.real if real else eps.imag           
           if eps == 0:
              coeff1 = None
           else:
              coeff1 = PhysConstant(eps)
        '''
        '''
        if isinstance(m, str):
           coeff2 = InvMu(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           mur = 1./mu0/m
           mur = mur.real if real else mur.imag
           if mur == 0:
               coeff2 = None
           else:
               coeff2 = PhysConstant(mur)
        '''
        '''
        if isinstance(s, str):
           coeff3 = Sigma(s,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           sigma = -1j *omega * s
           sigma = sigma.real if real else sigma.imag           
           if sigma == 0:
              coeff3 = None
           else:
              coeff3 = PhysConstant(sigma)
        '''       
        dprint1("epsr, mur, sigma " + str(coeff1) + " " + str(coeff2) + " " + str(coeff3))

        return coeff1, coeff2, coeff3

    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        if kfes != 0: return
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        # e, m, s
        coeff1, coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)
        
        if self.has_pml():
            coeff1 = self.make_PML_epsilon(coeff1)
            coeff2 = self.make_PML_invmu(coeff2)
            coeff3 = self.make_PML_sigma(coeff3)
        else:
            coeff2 = ComplexPow(coeff2, -1)

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        self.add_integrator(engine, 'mur', coeff2,
                            a.AddDomainIntegrator,
                            mfem.CurlCurlIntegrator)
        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import NativeCoefficientGenBase
        
        if len(self._sel_index) == 0: return
        
        e, m, s = self.vt.make_value_or_expression(self)

        self.do_add_scalar_expr(v, suffix, ind_vars, 'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        
        '''
        def add_sigma_epsilonr_mur(name, f_name):
            if isinstance(f_name, NativeCoefficientGenBase):
                pass   
            elif isinstance(f_name, str):      
                add_expression(v, name, suffix, ind_vars, f_name,
                           [], domains = self._sel_index, 
                           gdomain = self._global_ns)            
            else:
                add_constant(v, name, suffix, f_name,
                         domains = self._sel_index,
                         gdomain = self._global_ns)

        add_sigma_epsilonr_mur('epsilonr', e)
        add_sigma_epsilonr_mur('mur', m)
        add_sigma_epsilonr_mur('sigma', s)                           
        '''

