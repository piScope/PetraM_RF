'''`
   Essential BC
'''
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model  import Phys, PhysCoefficient
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_E')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable      
data =  (('jext', VtableElement('jext', type='complex',
                             guilabel = 'External J',
                             suffix =('x', 'y', 'z'),
                             default = [0,0,0],
                             tip = "volumetric external current" )),)


class Jext(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       #kwargs['isArray'] = True
       self.Jext_idx = kwargs.pop('Eyz_idx')
       PhysCoefficient.__init__(self, *args, **kwargs)
   def EvalValue(self, x):
       v = super(Jext, self).EvalValue(x)
       v = v[self.Jextem_idx]
       if self.real:  return v.real
       else: return v.imag
   
class EM1D_ExtJ(EM1D_Bdry):
    has_essential = True
    vt  = Vtable(data)
    
    def get_essential_idx(self, kfes):
        if kfes > 3: return
        return self._sel_index

    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes == 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        Eyz = self.vt.make_value_or_expression(self)              
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        if kfes == 1:
            coeff1 = Et(Eyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, Eyz_idx=0)
            gf.ProjectBdrCoefficient(coeff1,
                                            mfem.intArray(bdr_attr))
        elif kfes == 2:
            coeff1 = Et(Eyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, Eyz_idx=1)
            gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))


    def has_lf_contribution(self, kfes = 0):
        if kfes == 0: return True
        if kfes == 1: return True
        if kfes == 2: return True
        return False

    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes < 2:
            if real:       
                dprint1("Add LF contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add LF contribution(imag)" + str(self._sel_index))
            freq, omega = self.get_root_phys().get_freq_omega()
            f_name = self.vt.make_value_or_expression(self)
            self.set_integrator_realimag_mode(real)
            
            jwJ = jwJ_Coeff(f_name[0], self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns, omega)

            if kfes == 0:
            coeff1 = Et(Eyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, Eyz_idx=0)

                jwJ01 = ComplexVectorSlice(jwJ, [0,1])
        
                self.add_integrator(engine, 'jext', jwJ01, 
                                    b.AddDomainIntegrator,
                                    mfem.VectorFEDomainLFIntegrator)
            elif kfes == 1:

            else:
                jwJ2 = ComplexVectorSlice(jwJ, [2])               
                self.add_integrator(engine, 'jext', jwJ2,
                                    b.AddDomainIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
           assert False, "should not come here"

               
            



        
        
        

