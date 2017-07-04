'''
   anistropic region:
      However, can have arbitrary matrix epsilon_r, mu_r, sigma


'''
from petram.model import Domain
from petram.phys.phys_model  import Phys, MatrixPhysCoefficient
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Anisotropic')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

name_suffix = ['_xx', '_xy', '_xz', '_yx', '_yy', '_yz',
             '_zx', '_zy', '_zz',]
   
from petram.utils import set_array_attribute

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
    

class EM3D_Anisotropic(Domain, Phys):
    def __init__(self, **kwargs):
        super(EM3D_Anisotropic, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_Anisotropic, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []        
        v = set_array_attribute(v, 'epsilonr',
                                name_suffix,
                                ['1.0', '0.0', '0.0',
                                 '0.0', '1.0', '0.0',
                                 '0.0', '0.0', '1.0'])
        v = set_array_attribute(v, 'mur',
                                name_suffix,
                                ['1.0', '0.0', '0.0',
                                 '0.0', '1.0', '0.0',
                                 '0.0', '0.0', '1.0'])
        v = set_array_attribute(v, 'sigma',
                                name_suffix,
                                ['0.0', '0.0', '0.0',
                                 '0.0', '0.0', '0.0',
                                 '0.0', '0.0', '0.0'])
        v['epsilonr_m'] = '[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]'
        v['mur_m'] = '[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]'
        v['sigma_m'] = '[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]'
        v['use_m_epsilonr'] = False
        v['use_m_sigma'] = False
        v['use_m_mur'] = False        
        return v

    def panel1_param(self):
        names = name_suffix       
        return [self.make_matrix_panel('epsilonr', name_suffix, 3, 3),
                self.make_matrix_panel('mur', name_suffix, 3, 3),
                self.make_matrix_panel('sigma', name_suffix, 3, 3)]
     
    def form_name(self, v):
        if v: return 'Array Form'
        else: return 'Elemental Form'
        
    def get_panel1_value(self):
        names = name_suffix

        a = [self.form_name(self.use_m_epsilonr), 
             [[str(getattr(self, 'epsilonr'+n)) for n in names]],
             [str(self.epsilonr_m)]]
        b = [self.form_name(self.use_m_mur), 
             [[str(getattr(self, 'mur'+n)) for n in names]],
             [str(self.mur_m)]]
        c = [self.form_name(self.use_m_sigma), 
             [[str(getattr(self, 'sigma'+n)) for n in names]],
             [str(self.sigma_m)]]
        return [a,b,c]

    def preprocess_params(self, engine):
        dprint1('Preprocess Anisotropic Media')
        names = name_suffix
        k = 0
        for n in names:
            setattr(self, 'epsilonr'+n, str(getattr(self, 'epsilonr' + n)))
            k = k + 1
        for n in names:
            setattr(self, 'mur'+n, str(getattr(self, 'mur' + n)))
            k = k + 1
        for n in names:
            setattr(self, 'sigma'+n, str(getattr(self, 'sigma' + n)))             
            k = k + 1
         
    def import_panel1_value(self, v):
        names = ['_xx', '_xy', '_xz',
                 '_yx', '_yy', '_yz',                 
                 '_zx', '_zy', '_zz',]

        self.use_m_epsilonr = (str(v[0][0]) == 'Array Form')
        for k, n in enumerate(names):
            setattr(self, 'epsilonr'+n, str(v[0][1][0][k]))
        self.epsilonr_m = str(v[0][2][0])
            
        self.use_m_mur = (str(v[1][0]) == 'Array Form')
        for k, n in enumerate(names):
            setattr(self, 'mur'+n, str(v[1][1][0][k]))
        self.mur_m = str(v[1][2][0])
            
        self.use_m_sigma = (str(v[2][0]) == 'Array Form')
        for k, n in enumerate(names):
            setattr(self, 'sigma'+n, str(v[2][1][0][k]))
        self.sigma_m = str(v[2][2][0])            
                 

    def _make_f_name(self, basename):
        if getattr(self, 'use_m_'+basename):
            names = ['_m']
            eval_expr = self.eval_phys_array_expr
        else:
            names = ['_xx', '_xy', '_xz',
                     '_yx', '_yy', '_yz',                 
                     '_zx', '_zy', '_zz',]
            eval_expr = self.eval_phys_expr         
        f_name = []
        for n in names:
           var, f_name0 = eval_expr(getattr(self, basename+n), basename + n)
           if f_name0 is None:
               f_name.append(var)
           else:
               f_name.append(f_name0)
        
        return f_name
     
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        else: return False
                 
    def add_bf_contribution(self, engine, a, real = True, kfes = 0):
        if kfes != 0: return
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))
       
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        if real:
            mur = np.float(np.real(1./mu0/float(self.mur_xx)))
        else:
            mur = np.float(np.imag(1./mu0/float(self.mur_xx)))
            
        dprint1("mur " + str(mur))
        if mur != 0.0:
            coeff = mfem.ConstantCoefficient(mur)
            coeff = self.restrict_coeff(coeff, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff))
        else:
            dprint1("No cotrinbution from curlcurl")

        f_name = self._make_f_name('epsilonr')
        coeff1 = Epsilon(3, f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        coeff1 = self.restrict_coeff(coeff1, engine, matrix = True)

        f_name = self._make_f_name('sigma')        
        coeff2 = Sigma(3, f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        coeff2 = self.restrict_coeff(coeff2, engine, matrix = True)           
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff1))
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff2))          

    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        if len(self._sel_index) == 0: return
        def add_sigma_epsilonr_mur(name):
           f_name = self._make_f_name(name)
           if len(f_name) == 1:
               if not isinstance(f_name[0], str): expr  = f_name[0].__repr__()
               else: expr = f_name[0]
               add_expression(v, name, suffix, ind_vars, expr, 
                              [], domains = self._sel_index)
           else:  # elemental format
               expr_txt = [x.__repr__() if not isinstance(x, str) else x for x in f_name]
               a = '['+','.join(expr_txt[:3]) +']'
               b = '['+','.join(expr_txt[3:6])+']'
               c = '['+','.join(expr_txt[6:]) +']'
               expr = '[' + ','.join((a,b,c)) + ']'
               add_expression(v, name, suffix, ind_vars, expr, 
                              [], domains = self._sel_index)

        add_sigma_epsilonr_mur('epsilonr')
        add_sigma_epsilonr_mur('mur')
        add_sigma_epsilonr_mur('sigma')                           

