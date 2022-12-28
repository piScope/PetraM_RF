import numpy as np


from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable

#from petram.phys.phys_model  import Phys, VectorPhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_E')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

data = (('E', VtableElement('E', type='complex',
                            guilabel='Electric field',
                            suffix=('x', 'y', 'z'),
                            default=np.array([0, 0, 0]),
                            tip="essential BC")),)

'''
class Et(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Et, self).EvalValue(x)
       if self.real:  return v.real
       else: return v.imag
'''   

class EM3D_E(EM3D_Bdry):
    has_essential = True
    vt = Vtable(data)

    def attribute_set(self, v):
        super(EM3D_E, self).attribute_set(v)
        ### this is for backward compabitibility (to perform data transfer once)
        if not hasattr(self, "had_data_transfer"):
            print("not has_data_transferred")
            if hasattr(self, "E_x") and isinstance("E_x", str):
                self.E_x_txt = self.E_x
            if hasattr(self, "E_y") and isinstance("E_y", str):
                self.E_y_txt = self.E_y
            if hasattr(self, "E_z") and isinstance("E_z", str):
                self.E_z_txt = self.E_z
            self.had_data_transfer=True
        else:
            pass
            #print("has_data_transferred")
        v["had_data_transfer"] = True
        return v
    '''
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
    '''
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

            
        Exyz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = VCoeff(3, Exyz, ind_vars, l, g, return_complex=True)
        #f_name = self._make_f_name()
        #coeff1 = Et(3, f_name,  self.get_root_phys().ind_vars,
        #            self._local_ns, self._global_ns,
        #            real = real)
        coeff1 = self.restrict_coeff(coeff1, engine, vec=True)
        
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff1 = coeff1.get_realimag_coefficient(real)
        gf.ProjectBdrCoefficientTangent(coeff1, mfem.intArray(bdr_attr))



        
        
        

