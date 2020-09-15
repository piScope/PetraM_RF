'''
   anistropic region:
   note that MFEM3.3 does not support matrix form of coefficeint
   for curl-curl. Needs next update.
'''
import numpy as np
from scipy.linalg import inv
from petram.phys.phys_model  import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Anisotropic')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable   
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.eye(3, 3),
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.eye(3, 3),
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.zeros((3, 3)),
                                     tip = "contuctivity" )),)

from petram.phys.coefficient import MCoeff
from petram.phys.coefficient import PyComplexMatrixInvCoefficient as ComplexMatrixInv
from petram.phys.em3d.em3d_const import mu0, epsilon0

def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega       
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Sigma_Coeff(exprs, ind_vars, l, g, omega): 
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

''' 
def InvMu_Coeff(exprs, ind_vars, l, g, omega, real):
    fac = mu0
    coeff = MCoeff(3, exprs, ind_vars, l, g, real=real, scale=fac)
    if coeff is None: return None
    if not real: return None
    c2 = mfem.InverseMatrixCoefficient(coeff)
    c2._coeff = coeff
    return c2
''' 
def Mu_Coeff(exprs, ind_vars, l, g, omega):
    fac = mu0
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff
 
'''
class Epsilon(MatrixPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0      
       v = super(Epsilon, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega
       if self.real:  return v.real
       else: return v.imag
    
class Sigma(MatrixPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0      
       v = super(Sigma, self).EvalValue(x)
       v =  - 1j*self.omega * v       
       if self.real:  return v.real
       else: return v.imag

class InvMu(MatrixPhysCoefficient):
   #   1./mu0/mur
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)      
       super(InvMu, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em3d_const import mu0, epsilon0      
       v = super(InvMu, self).EvalValue(x)
       v = 1/mu0*inv(v)
       if self.real:  return v.real
       else: return v.imag
'''

class EM3D_Anisotropic(EM3D_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']
    def get_possible_child(self):
        from .em3d_pml      import EM3D_LinearPML
        return [EM3D_LinearPML]
    
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False
        
    def get_coeffs(self, real = True):
        from .em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s = self.vt.make_value_or_expression(self)
        
        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff(e, ind_vars, l, g, omega)
        coeff2 = Mu_Coeff(m, ind_vars, l, g, omega)
        coeff3 = Sigma_Coeff(s, ind_vars, l, g, omega)

        return coeff1, coeff2, coeff3
                 
    def add_bf_contribution(self, engine, a, real = True, kfes = 0):
        if kfes != 0: return
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))
       
        freq, omega = self.get_root_phys().get_freq_omega()

        coeff1, coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)

        if self.has_pml():
            coeff1 = self.make_PML_epsilon(coeff1)
            coeff2 = self.make_PML_invmu(coeff2)
            coeff3 = self.make_PML_sigma(coeff3)
        else:
            coeff2 = ComplexMatrixInv(coeff2)

        self.add_integrator(engine, 'epsilonr', coeff1,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
                  
        if coeff2 is not None:
            self.add_integrator(engine, 'mur', coeff2,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            #coeff2 = self.restrict_coeff(coeff2, engine)
            #a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No contrinbution from curlcurl")

        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import NativeCoefficientGenBase
        
        if len(self._sel_index) == 0: return

        e, m, s = self.vt.make_value_or_expression(self)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', m)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        

