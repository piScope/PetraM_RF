from __future__ import print_function
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain
from petram.phys.phys_model import Phys
from petram.model import Bdry
from petram.mfem_config import use_parallel
'''

   Port load. Define an external circuite attached to Port BC using
   external S-matrix

'''

import numpy as np
from numpy import pi

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_PortLoad')


if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    '''
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   '''
else:
    import mfem.ser as mfem
    nicePrint = dprint1

data = (('port_s', VtableElement('port_s', type='complex',
                                  guilabel='external S-mat',
                                  default=1.0,
                                  tip="S-matrix of external circuit attached to the ports")),)

def bdry_constraints():
   return [EM3D_PortLoad]
    
class EM3D_PortLoad(EM3D_Bdry):
    extra_diagnostic_print = True
    vt = Vtable(data)

    def __init__(self):
        super(EM3D_PortLoad, self).__init__()
    
    def extra_DoF_name(self):
        return self.get_root_phys().dep_vars[0] + "_"+self.name().lower()

    def get_probe(self):
        return self.get_root_phys().dep_vars[0] + "_"+self.name().lower()

    def attribute_set(self, v):
        super(EM3D_PortLoad, self).attribute_set(v)
        v['isTimeDependent_RHS'] = True
        return v
        
    def panel1_param(self):
        return self.vt.panel_param(self)
    
    def get_panel1_value(self):
        return self.vt.get_panel_value(self)

    def import_panel1_value(self, v):
        self.vt.import_panel_value(self, v)
    
    def panel4_tip(self):
        return None
        
    def update_param(self):
        self.vt.preprocess_params(self)
        Smat = self.vt.make_value_or_expression(self)        
        
    def has_extra_DoF(self, kfes):
        if kfes != 0:
            return False
        return True

    def get_extra_NDoF(self):
        return len(self._sel_index)

    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name()
        sol_extra[name] = sol.toarray()
        
    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        '''
        if self._update_flag:
            if mode == 'B':
                return self.isTimeDependent_RHS
            if mode == 'M':
                return self.isTimeDependent
        return False

    def _get_coupled_ports(self):
        from petram.phys.em3d.em3d_port import EM3D_Port
        
        root_bdr = self.get_root_phys()['Boundary']

        target = []
        for mm in root_bdr.get_children():
            if not mm.enabled: continue
            if not isinstance(mm, EM3D_Port): continue

            if mm._sel_index[0] in self._sel_index:
                target.append((mm._sel_index[0], mm, mm.extra_DoF_name()))
        return sorted(target)
    
    def add_extra_contribution(self, engine, **kwargs):        
        '''
        Format of extar   (t2 is returnd as vertical(transposed) matrix)
        [M,  t1]   [  ]
        [      ] = [  ]
        [t2, t3]   [t4]
        and it returns if Lagurangian will be saved.
        '''
        dprint1("Add Extra contribution" + str(self._sel_index))
        from mfem.common.chypre import (LF2PyVec,
                                        PyVec2PyMat,
                                        Array2PyVec,
                                        IdentityPyMat,
                                        HStackPyVec)
        
        fes = engine.get_fes(self.get_root_phys(), 0)
        ports = self. _get_coupled_ports()

        vecs = []
        for x in ports:
            mm = x[1]
            C_Et, C_jwHt = mm.get_coeff_cls()
            inc_amp, inc_phase, eps, mur = mm.vt.make_value_or_expression(mm)            
            lf1 = engine.new_lf(fes)
            Ht1 = C_jwHt(3, pi, mm, real=True, eps=eps, mur=mur)
            Ht2 = mm.restrict_coeff(Ht1, engine, vec=True)
            intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht2)
            lf1.AddBoundaryIntegrator(intg)
            lf1.Assemble()
            lf1i = engine.new_lf(fes)
            Ht3 = C_jwHt(3, pi, mm, real=False, eps=eps, mur=mur)
            Ht4 = self.restrict_coeff(Ht3, engine, vec=True)
            intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht4)
            lf1i.AddBoundaryIntegrator(intg)
            lf1i.Assemble()
            v1 = LF2PyVec(lf1, lf1i)        
            vecs.append(v1)

        t1 = HStackPyVec(vecs)
        t3 = IdentityPyMat(len(self._sel_index), diag=-1)
        t4 = Array2PyVec(np.zeros(len(self._sel_index)))       
        #return (v1, v2, t3, t4, True)
        return (t1, None, t3, t4, True)                
    
    def has_extra_coupling(self):
        '''
        True if it define coupling between Lagrange multipliers
        '''
        name1 = self.extra_DoF_name()

        ports = self. _get_coupled_ports()
        names2 = [x[2] for x in ports]

        if len(names2) == 0:
            return False
        return True

    def extra_coupling_names(self):
        '''
        return its own extra_name, and paired (coupled) extra_names
        '''
        name1 = self.extra_DoF_name()

        ports = self. _get_coupled_ports()
        names2 = [x[2] for x in ports]

        return name1, names2

    def get_extra_coupling(self, target_name):
        '''
        [    t2][paired extra]
        [t1    ][extra]
        t1 = (size of extra, size of targert_extra, )
        t2 = (size of target_extra, size of extra, )        
        '''
        tmp = self.vt.make_value_or_expression(self)
        Smat = tmp[0]

        ports = self. _get_coupled_ports()
        names2 = [x[2] for x in ports]

        idx = names2.index(target_name)

        c1 = np.zeros((1, len(names2)))
        c1[0, idx] = -1
        if len(np.atleast_1d(Smat))==1:
            Smat = np.eye(len(names2))*Smat
        c2 = np.atleast_2d(Smat[:, idx])

        from petram.helper.densemat2pymat import Densemat2PyMat
        ret1 = Densemat2PyMat(c1)
        ret2 = Densemat2PyMat(c2)
        
        return ret1, ret2
    
