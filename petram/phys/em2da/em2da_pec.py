from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Electric Conductor',
                                  default =   "Et = 0",
                                  tip = "Essential Homogenous BC" )),)

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class EM2Da_PEC(EM2Da_Bdry):
    has_essential = True
    nlterms = []
    vt  = Vtable(data)          
    
    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        elif kfes == 1:
            return self._sel_index            
        else:
            return []

    def apply_essential(self, engine, gf, kfes, real = False,
                        **kwargs):
        pass
        


