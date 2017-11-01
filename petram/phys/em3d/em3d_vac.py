'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
from petram.model import Domain
from petram.phys.phys_model  import Phys, PhysCoefficient
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Vac')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

class Epsilon(PhysCoefficient):
   '''
    -1j * omega^2 * epsilon0 * epsilonr
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0
       v = super(Epsilon, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega
       if self.real:  return v.real
       else: return v.imag
       
class Sigma(PhysCoefficient):
   '''
    -1j * omega * sigma
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0
       v = super(Sigma, self).EvalValue(x)
       v = - 1j * self.omega * v
       if self.real:  return v.real
       else: return v.imag

class Mur(PhysCoefficient):
   '''
      1./mu0/mur
   '''
   def __init__(self, *args, **kwargs):
       super(Mur, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0      
       v = super(Mur, self).EvalValue(x)
       v = 1/mu0/x
       if self.real:  return v.real
       else: return v.imag
       
class EM3D_Vac(Domain, Phys):
    def __init__(self, **kwargs):
        super(EM3D_Vac, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_Vac, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['epsilonr'] = '1.0'
        v['mur'] = '1.0'
        v['sigma'] = '0.0'
        return v

    def panel1_param(self):
        return [["epsilonr",   str(self.epsilonr), 0,
                 {'validator': self.check_phys_expr,
                  'validator_param':'epsilonr'}],
                ["mur",    str(self.mur)  ,  0, 
                 {'validator': self.check_phys_expr,
                  'validator_param':'mur'}],
                ["sigma",    str(self.sigma) ,  0, 
                 {'validator': self.check_phys_expr,
                  'validator_param':'sigma'}],]

    def panel1_tip(self):
        return ["relative permitivity",
                "relative permeability",
                "electrical conductivity (S/m)"]

    def get_panel1_value(self):
        return (str(self.epsilonr), str(self.mur), str(self.sigma))

    def preprocess_params(self, engine):
        dprint1('Preprocess Vac')
        self.epsilonr =  str(self.epsilonr)
        self.mur =  str(self.mur)
        self.sigma =  str(self.sigma)
       
    def import_panel1_value(self, v):
        self.epsilonr = str(v[0])
        self.mur = str(v[1])
        self.sigma = str(v[2])

    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False

    def get_sigma_coeff(self, real = True, conj = False):
        # sigma
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        var, f_name = self.eval_phys_expr(self.sigma, 'sigma')
        if callable(var):
           coeff = Sigma(f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           sigma = -1j *omega * var
           if real:
              if sigma.real == 0: return None
              coeff = mfem.ConstantCoefficient(sigma.real)
              dprint1("sigma " + str(sigma.real))              
           else:
              if sigma.imag == 0: return None              
              coeff = mfem.ConstantCoefficient(sigma.imag)              
              dprint1("sigma " + str(sigma.imag))

        return coeff
     
    def get_epsilonr_coeff(self, real = True, conj = False):
        # epsilonr
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        
        var, f_name = self.eval_phys_expr(self.epsilonr, 'epsilonr')
        if callable(var):
           coeff = Epsilon(f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           eps =  -var*epsilon0*omega*omega
           if real:
              if eps.real == 0: return None              
              coeff = mfem.ConstantCoefficient(eps.real)
              dprint1("epsilon " + str(eps.real))              
           else:
              if eps.imag == 0: return None                            
              coeff = mfem.ConstantCoefficient(eps.imag)              
              dprint1("epsilon " + str(eps.imag))
        return coeff

    def get_mur_coeff(self, real = True, conj = False):
        # mur
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()                

        var, f_name = self.eval_phys_expr(self.mur, 'mur')
        if callable(var):
           coeff2 = Mur(f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        else:
           mur = 1./mu0/var
           if real:
              if mur.real == 0: return None
              coeff2 = mfem.ConstantCoefficient(mur.real)
              dprint1("mur " + str(mur.real))              
           else:
              if mur.imag == 0: return None
              coeff2 = mfem.ConstantCoefficient(mur.imag)              
              dprint1("mur " + str(mur.imag))
        return coeff2
     
    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        if kfes != 0: return
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        #from em3d_const import mu0, epsilon0

        #freq = self.get_root_phys().freq
        #omega = 2*np.pi*freq

        coeff1 = self.get_mur_coeff(real = real)
        if coeff1 is not None:
            coeff1 = self.restrict_coeff(coeff1, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff1))
        else:
            dprint1("No cotrinbution from curlcurl")
            
        coeff2 = self.get_epsilonr_coeff(real=real)
        if coeff2 is not None:    
            coeff2 = self.restrict_coeff(coeff2, engine)           
            a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff2))
        coeff3 = self.get_sigma_coeff(real = real)
        if coeff3 is not None:
            coeff3 = self.restrict_coeff(coeff3, engine)        
            a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff3))
     
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant

        if len(self._sel_index) == 0: return
        var, f_name = self.eval_phys_expr(self.epsilonr, 'epsilonr')
        if callable(var):
            add_expression(v, 'epsilonr', suffix, ind_vars, f_name,
                           [], domains = self._sel_index)
        else:
            add_constant(v, 'epsilonr', suffix, var,
                         domains = self._sel_index)

        var, f_name = self.eval_phys_expr(self.mur, 'mur')
        if callable(var):
            add_expression(v, 'mur', suffix, ind_vars, f_name,
                           [], domains = self._sel_index)
        else:
            add_constant(v, 'mur', suffix, var,
                         domains = self._sel_index)

        var, f_name = self.eval_phys_expr(self.sigma, 'sigma')
        if callable(var):
            add_expression(v, 'sigma', suffix, ind_vars, f_name,
                           [], domains = self._sel_index)
        else:
            add_constant(v, 'sigma', suffix, var,
                         domains = self._sel_index)
            


    
