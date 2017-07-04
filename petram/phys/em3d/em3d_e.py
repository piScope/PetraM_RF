import numpy as np

from petram.model import Bdry
from petram.phys.phys_model  import Phys, VectorPhysCoefficient
import petram.debug as debug

dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_E')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

class Et(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Et, self).EvalValue(x)
       if self.real:  return v.real
       else: return v.imag
   
   
class EM3D_E(Bdry, Phys):
    has_essential = True
    def __init__(self, **kwargs):
        super(EM3D_E, self).__init__( **kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_E, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['E_x'] = '0.0'
        v['E_y'] = '0.0'
        v['E_z'] = '0.0'
        v['E_m'] = '[0.0, 0.0, 0.0]'        
        v['use_m_E'] = False                
        return v
    
    def panel1_param(self):
        names = ['_x', '_y', '_z']
        
        a = [ {'validator': self.check_phys_expr,
               'validator_param':'E'+n} for n in  names]
        
        elp1 = [[None, None, 43, {'row': 3,
                                  'col': 1,
                                  'text_setting': a}],]
        elp2 = [[None, None, 0, {'validator': self.check_phys_array_expr,
                                 'validator_param': 'E_m'},]]

        l = [[None, None, 34, ({'text': 'E  ',
                                'choices': ['Elemental Form', 'Array Form'],
                                'call_fit': False},
                                {'elp': elp1},  
                                {'elp': elp2},),],]
        return l
        
    def form_name(self, v):
        if v: return 'Array Form'
        else: return 'Elemental Form'

    def get_panel1_value(self):
        names = ['_x', '_y', '_z',]
        a = [self.form_name(self.use_m_E), 
             [[str(getattr(self, 'E'+n)) for n in names]],
             [str(self.E_m)]]
        return [a, ]

    def preprocess_params(self, engine):
        dprint1('Preprocess E')
        names = ['_x', '_y', '_z',]
        k = 0
        for n in names:
            setattr(self, 'E'+n, str(getattr(self, 'E' + n)))
            k = k + 1

    def import_panel1_value(self, v):
        names = ['_x', '_y', '_z',]
        self.use_m_E = (str(v[0][0]) == 'Array Form')
        for k, n in enumerate(names):
            setattr(self, 'E'+n, str(v[0][1][0][k]))
        self.E_m = str(v[0][2][0])
        
    def _make_f_name(self):
        basename = 'E'
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

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes != 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))

        f_name = self._make_f_name()
        coeff1 = Et(3, f_name,  self.get_root_phys().ind_vars,
                    self._local_ns, self._global_ns,
                    real = real)
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)

        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1
        gf.ProjectBdrCoefficientTangent(coeff1, mfem.intArray(bdr_attr))



        
        
        

