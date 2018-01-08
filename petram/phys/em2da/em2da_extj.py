'''
   external current source
'''
import numpy as np

from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import PhysCoefficient
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_extJ')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable      
data =  (('jext', VtableElement('jext', type='complex',
                             guilabel = 'External J',
                             suffix =('r', 'phi', 'z'),
                             default = [0,0,0],
                             tip = "volumetric external current" )),)
   
class rJext_p(VectorPhysCoefficient): # i \omega r Jext_t
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(rJext_p, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from petram.phys.em3d.em3d_const import mu0, epsilon0      
       v = super(rJext_p, self).EvalValue(x)
       v = np.array((v[0], v[2]))
       v = 1j * self.omega * v * x[0]
       if self.real:  return v.real
       else: return v.imag
       
class Jext_t(PhysCoefficient):  # i \omega Jext_phi
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Jext_t, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       from petram.phys.em3d.em3d_const import mu0, epsilon0      
       v = super(Jext_t, self).EvalValue(x)
       v = 1j * self.omega * v[1]
       if self.real:  return v.real
       else: return v.imag

class EM2Da_ExtJ(EM2Da_Domain):
    is_secondary_condition = True  # does not count this class for persing "remaining"
    has_3rd_panel = False
    vt  = Vtable(data)
    
    def has_lf_contribution(self, kfes = 0):
        if kfes > 2: return False
        return True

    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes < 2:
            if real:       
                dprint1("Add LF contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add LF contribution(imag)" + str(self._sel_index))
            freq, omega = self.get_root_phys().get_freq_omega()
            f_name = self.vt.make_value_or_expression(self)
            if kfes == 0:
                coeff1 = rJext_p(2, f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        
                self.add_integrator(engine, 'jext', coeff1,
                                    b.AddDomainIntegrator,
                                    mfem.VectorFEDomainLFIntegrator)
            else:
                coeff1 = Jext_t(f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
        
                self.add_integrator(engine, 'jext', coeff1,
                                    b.AddDomainIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
           assert False, "should not come here"

               
            

        '''
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))
        '''

