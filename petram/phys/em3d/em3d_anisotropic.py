'''
   anistropic region:
   note that MFEM3.3 does not support matrix form of coefficeint
   for curl-curl. Needs next update.
'''
import numpy as np

from petram.phys.phys_model  import MatrixPhysCoefficient, PhysCoefficient, PhysConstant
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
   
class Epsilon(MatrixPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0      
       v = super(Epsilon, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega
       if self.real:  return v.real
       else: return v.imag
    
class Sigma(MatrixPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0      
       v = super(Sigma, self).EvalValue(x)
       v =  - 1j*self.omega * v       
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
       from em3d_const import mu0, epsilon0      
       v = super(InvMu, self).EvalValue(x)
       v = 1/mu0/v
       if self.real:  return v.real
       else: return v.imag
       

class EM3D_Anisotropic(EM3D_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']    
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False
        
    def get_coeffs(self, real = True):
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s = self.vt.make_value_or_expression(self)

        coeff1 = Epsilon(3, e,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        
        if isinstance(m[0], str):
           coeff2 = InvMu(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           mur = 1./mu0/m[0]
           mur = mur.real if real else mur.imag
           if mur == 0:
               coeff2 = None
           else:
               coeff2 = PhysConstant(mur)
     
        coeff3 = Sigma(3, s,  self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real, omega = omega)
              
        dprint1("epsr, mur, sigma " + str(coeff1) + " " + str(coeff2) + " " + str(coeff3))

        return coeff1, coeff2, coeff3
                 
    def add_bf_contribution(self, engine, a, real = True, kfes = 0):
        if kfes != 0: return
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))
       
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()

        coeff1, coeff2, coeff3 = self.get_coeffs(real = real)        

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        if coeff2 is not None:
            coeff2 = self.restrict_coeff(coeff2, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No cotrinbution from curlcurl")
        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        '''
        coeff1 = self.restrict_coeff(coeff1, engine, matrix = True)
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff1))
        
        if coeff2 is not None:
            coeff2 = self.restrict_coeff(coeff2, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No cotrinbution from curlcurl")

        coeff3 = self.restrict_coeff(coeff3, engine, matrix = True)
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff3))
        '''
        
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        if len(self._sel_index) == 0: return

        e, m, s = self.vt.make_value_or_expression(self)        
        def add_sigma_epsilonr_mur(name, f_name):
           if len(f_name) == 1:
               if not isinstance(f_name[0], str): expr  = f_name[0].__repr__()
               else: expr = f_name[0]
               add_expression(v, name, suffix, ind_vars, expr, 
                              [], domains = self._sel_index,
                              gdomain = self._global_ns)
           else:  # elemental format
               expr_txt = [x.__repr__() if not isinstance(x, str) else x for x in f_name]
               a = '['+','.join(expr_txt[:3]) +']'
               b = '['+','.join(expr_txt[3:6])+']'
               c = '['+','.join(expr_txt[6:]) +']'
               expr = '[' + ','.join((a,b,c)) + ']'
               add_expression(v, name, suffix, ind_vars, expr, 
                              [], domains = self._sel_index,
                              gdomain = self._global_ns)

        add_sigma_epsilonr_mur('epsilonr', e)
        add_sigma_epsilonr_mur('mur', m)
        add_sigma_epsilonr_mur('sigma', s)                           

