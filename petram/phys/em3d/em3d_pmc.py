from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class EM3D_PMC(Bdry, Phys):
    is_essential = False
    def __init__(self, **kwargs):
        super(EM3D_PMC, self).__init__( **kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_PMC, self).attribute_set(v)        
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def panel1_param(self):
        return [['Pefect Magnetic Conductor',   "Ht = 0",  2, {}],]

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass
