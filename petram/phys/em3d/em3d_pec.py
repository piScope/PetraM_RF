from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class EM3D_PEC(Bdry, Phys):
    has_essential = True
    def __init__(self, **kwargs):
        super(EM3D_PEC, self).__init__( **kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_PEC, self).attribute_set(v)        
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
        
    def panel1_param(self):
        return [['Pefect Electric Conductor',   "Et = 0",  2, {}],]

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass
    
    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, kfes, real = False,
                        **kwargs):
        pass
        


