'''
  continuity is a condition for internail
  boundary. 
  It has no contribution to weak form
'''
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_cont  import PhysContinuity

class EM3D_Continuity(PhysContinuity):
    pass
