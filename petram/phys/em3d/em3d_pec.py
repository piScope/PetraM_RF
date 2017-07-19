from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Electric Conductor',
                                  default =   "Et = 0",
                                  tip = "Essential Homogenous BC" )),)

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class EM3D_PEC(EM3D_Bdry):
    has_essential = True
    nlterms = []
    vt  = Vtable(data)          
    
    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, kfes, real = False,
                        **kwargs):
        pass
        


