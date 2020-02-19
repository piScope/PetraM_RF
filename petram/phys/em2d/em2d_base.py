import traceback
import numpy as np

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient, PhysCoefficient

# define variable for this BC.
from petram.phys.vtable import VtableElement, Vtable
data =  (('Einit', VtableElement('Einit', type='float',
                                  guilabel = 'E(init)',
                                  suffix =('x', 'y', 'z'),
                                  default = np.array([0,0,0]), 
                                  tip = "initial_E",
                                  chkbox = True)),)

class Einit_xy(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Einit_xy, self).EvalValue(x)
       v = np.array((v[0], v[1]))
       if self.real:  val = v.real
       else: val =  v.imag
       return val
    
class Einit_z(PhysCoefficient):
   def EvalValue(self, x):
       v = super(Einit_z, self).EvalValue(x)
       v = v[2]
       if self.real:  val = v.real
       else: val =  v.imag
       return val

class EM2D_Domain(Domain, Phys):
    has_3rd_panel = True    
    vt3  = Vtable(data)   
    def __init__(self, **kwargs):
        super(EM2D_Domain, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM2D_Domain, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
    
    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes > 2: return
        if not self.use_Einit: return
        
        f_name = self.vt3.make_value_or_expression(self)
        if kfes == 0:
            coeff = Einit_xy(2, f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            return self.restrict_coeff(coeff, engine, vec = True)            
        else:
            coeff = Einit_z(f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            return self.restrict_coeff(coeff, engine)                        


class EM2D_Bdry(Bdry, Phys):
    has_3rd_panel = True        
    vt3  = Vtable(data)   
    def __init__(self, **kwargs):
        super(EM2D_Bdry, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM2D_Bdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes > 2: return
        if not self.use_Einit: return
        
        f_name = self.vt3.make_value_or_expression(self)
        if kfes == 0:
            coeff = Einit_xy(2, f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            return self.restrict_coeff(coeff, engine, vec = True)            
        else:
            coeff = Einit_z(f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            return self.restrict_coeff(coeff, engine)                                    


     

