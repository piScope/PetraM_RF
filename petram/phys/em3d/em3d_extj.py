'''
   external current source
'''
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys, VectorPhysCoefficient
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_extJ')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

class Jext(VectorPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Jext, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0      
       v = super(Jext, self).EvalValue(x)
       v = 1j * self.omega * v
       if self.real:  return v.real
       else: return v.imag

class EM3D_ExtJ(Domain, Phys):
    is_secondary_condition = True
    def __init__(self, **kwargs):
        super(EM3D_ExtJ, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_ExtJ, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['jext_x'] = '0.0'
        v['jext_y'] = '0.0'
        v['jext_z'] = '0.0'
        v['jext_m'] = '[0.0, 0.0, 0.0]'        
        v['use_m_jext'] = False                
        return v
    
    def panel1_param(self):
        names = ['_x', '_y', '_z']
        
        a = [ {'validator': self.check_phys_expr,
               'validator_param':'jext'+n} for n in  names]
        
        elp1 = [[None, None, 43, {'row': 3,
                                  'col': 1,
                                  'text_setting': a}],]
        elp2 = [[None, None, 0, {'validator': self.check_phys_array_expr,
                                 'validator_param': 'jext_m'},]]

        l = [[None, None, 34, ({'text': 'Jext  ',
                                'choices': ['Elemental Form', 'Array Form'],
                                'call_fit': False},
                                {'elp': elp1},  
                                {'elp': elp2},),]]
        return l
        
    def form_name(self, v):
        if v: return 'Array Form'
        else: return 'Elemental Form'

    def get_panel1_value(self):
        names = ['_x', '_y', '_z',]
        a = [self.form_name(self.use_m_jext), 
             [[str(getattr(self, 'jext'+n)) for n in names]],
             [str(self.jext_m)]]
        return [a,]

    def preprocess_params(self, engine):
        dprint1('Preprocess Jext')
        names = ['_x', '_y', '_z',]
        k = 0
        for n in names:
            setattr(self, 'jext'+n, str(getattr(self, 'jext' + n)))
            k = k + 1

    def import_panel1_value(self, v):
        names = ['_x', '_y', '_z',]
        self.use_m_jext = (str(v[0][0]) == 'Array Form')
        for k, n in enumerate(names):
            setattr(self, 'jext'+n, str(v[0][1][0][k]))
        self.jext_m = str(v[0][2][0])

    def _make_f_name(self):
        basename = 'jext'
        if getattr(self, 'use_m_'+basename):
            names = ['_m']
            eval_expr = self.eval_phys_array_expr
        else:
            names = ['_x', '_y', '_z',]
            eval_expr = self.eval_phys_expr            
              
        f_name = []
        for n in names:
           var, f_name0 = eval_expr(getattr(self, basename+n), basename + n)
           if f_name0 is None:
               f_name.append(var)
           else:
               f_name.append(f_name0)
        return f_name
    
    def has_lf_contribution(self):
        return True

    def add_lf_contribution(self, engine, b, real = True):
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))
        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        f_name = self._make_f_name()
        coeff1 = Jext(3, f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))

    '''     
    def add_lf_contribution_imag(self, engine, b):
        dprint1("Add BF(imag) contribution" + str(self._sel_index))
        freq = self.get_root_phys().freq
        omega = 2*np.pi*freq
        f_name = self._make_f_name()
        coeff1 = Jext(3, f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = False, omega = omega)
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))
    '''
