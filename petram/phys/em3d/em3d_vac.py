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

from petram.phys.weakform import SCoeff
from .em3d_const import mu0, epsilon0

def Epsilon_Coeff(exprs, ind_vars, l, g, omega, real):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega       
    coeff = SCoeff(exprs, ind_vars, l, g, real=real, scale=fac)
    return coeff

def Sigma_Coeff(exprs, ind_vars, l, g, omega, real): 
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    coeff = SCoeff(exprs, ind_vars, l, g, real=real, scale=fac)
    return coeff

def InvMu_Coeff(exprs, ind_vars, l, g, omega, real):
    # v = - 1j * self.omega * v
    fac = mu0
    coeff = SCoeff(exprs, ind_vars, l, g, real=real, scale=fac)
    if coeff is None: return None

    c2 = mfem.PowerCoefficient(coeff, -1)
    c2._coeff = coeff
    return c2
   
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

class InvMu(PhysCoefficient):
   '''
      1./mu0/mur
   '''
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
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False

    def get_coeffs(self, real = True):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff([e], ind_vars, l, g, omega, real)
        coeff2 = InvMu_Coeff([m], ind_vars, l, g, omega, real)                
        coeff3 = Sigma_Coeff([s], ind_vars, l, g, omega, real)

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
        coeff1, coeff2, coeff3 = self.get_coeffs(real = real)

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        self.add_integrator(engine, 'mur', coeff2,
                            a.AddDomainIntegrator,
                            mfem.CurlCurlIntegrator)
        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        '''
        if coeff1 is not None:    
            coeff1 = self.restrict_coeff(coeff1, engine)           
            a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff1))
        if coeff2 is not None:
            coeff2 = self.restrict_coeff(coeff2, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No cotrinbution from curlcurl")
        if coeff3 is not None:
            coeff3 = self.restrict_coeff(coeff3, engine)        
            a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff3))
        '''
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


    
