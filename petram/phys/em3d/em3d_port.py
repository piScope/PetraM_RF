'''

   3D port boundary condition


    2016 5/20  first version only TE modes
'''

import numpy as np

from petram.model import Bdry
from petram.phys.phys_model  import Phys
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

from petram.helper.geom import connect_pairs
from petram.helper.geom import find_circle_center_radius

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Port')

from .em3d_const import epsilon0, mu0

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   '''
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   '''
else:
   import mfem.ser as mfem

'''
  TE Mode
    Expression are based on Microwave Engineering p122 - p123.
    Note that it consists from two terms
       1)  \int dS W \dot n \times iwH (VectorFETangentIntegrator does this)
       2)  an weighting to evaulate mode amplutude from the E field
           on a boundary              
  TEM Mode
       E is parallel to the periodic edge
       Mode number is ignored (of course)

  About sign of port phasing.
       positive phasing means the incoming wave appears later (phase delay)
       at the port
'''
'''
   rectangular wg TE
'''
class C_Et_TE(mfem.VectorPyCoefficient):
   def __init__(self, sdim, bdry, real = True):
       self.real = real
       self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
       self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
       self.m, self.n = bdry.mn[0], bdry.mn[1]
       freq, omega = bdry.get_root_phys().get_freq_omega()
       
       kc = np.sqrt((bdry.mn[0]*np.pi/bdry.a)**2 +
                    (bdry.mn[1]*np.pi/bdry.b)**2)
     
       self.AA = 1.0 ## bdry.omega*bdry.mur*mu0*np.pi/kc/kc  
       mfem.VectorPyCoefficient.__init__(self, sdim)

   def EvalValue(self, x):
       p = np.array(x)
       # x, y is positive (first quadrant)
       x = abs(np.sum((p - self.c)*self.a_vec)) #
       y = abs(np.sum((p - self.c)*self.b_vec)) #

       Ex = self.AA* self.n/self.b*(
                         np.cos(self.m*np.pi*x/self.a)*
                         np.sin(self.n*np.pi*y/self.b))
       Ey = - self.AA* self.m/self.a*(
                         np.sin(self.m*np.pi*x/self.a)*
                         np.cos(self.n*np.pi*y/self.b))


       E = Ex*self.a_vec + Ey*self.b_vec
       if self.real:
            return -E.real
       else:
            return -E.imag
class C_jwHt_TE(mfem.VectorPyCoefficient):
   def __init__(self, sdim, phase, bdry, real = True, amp = 1.0):
       self.real = real
       self.phase = phase  # phase !=0 for incoming wave

       freq, omega = bdry.get_root_phys().get_freq_omega()        

       self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
       self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
       self.m, self.n = bdry.mn[0], bdry.mn[1]

       k = omega*np.sqrt(bdry.epsilonr*epsilon0 * bdry.mur*mu0)
       kc = np.sqrt((bdry.mn[0]*np.pi/bdry.a)**2 +
                         (bdry.mn[1]*np.pi/bdry.b)**2)
       if kc > k:
          raise ValueError('Mode does not propagate')
       beta = np.sqrt(k**2 - kc**2)
       dprint1("propagation constant:" + str(beta))
       AA = omega*bdry.mur*mu0*np.pi/kc/kc*amp
       self.AA = omega*beta*np.pi/kc/kc/AA

       mfem.VectorPyCoefficient.__init__(self, sdim)

   def EvalValue(self, x):
       p = np.array(x)
       # x, y is positive (first quadrant)       
       x = abs(np.sum((p - self.c)*self.a_vec)) #
       y = abs(np.sum((p - self.c)*self.b_vec)) # 

       Hx = 1j*self.AA* self.m/self.a*(
                         np.sin(self.m*np.pi*x/self.a)*
                         np.cos(self.n*np.pi*y/self.b))
       Hy = 1j*self.AA* self.n/self.b*(
                         np.cos(self.m*np.pi*x/self.a)*
                         np.sin(self.n*np.pi*y/self.b))

       H = Hx*self.a_vec + Hy*self.b_vec

       H = H * np.exp(1j*self.phase/180.*np.pi)
   
       if self.real:
            return H.real
       else:
            return H.imag
'''
   rectangular port parallel metal TEM
'''
class C_Et_TEM(mfem.VectorPyCoefficient):
   def __init__(self, sdim, bdry, real = True):
       mfem.VectorPyCoefficient.__init__(self, sdim)
       freq, omega = bdry.get_root_phys().get_freq_omega()       
       self.real = real
       self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
       self.AA = 1.0
       
   def EvalValue(self, x):
       Ex = self.AA
       E = -Ex*self.b_vec
       if self.real:
            return -E.real
       else:
            return -E.imag
class C_jwHt_TEM(mfem.VectorPyCoefficient):
   def __init__(self, sdim, phase, bdry, real = True, amp = 1.0):   
       mfem.VectorPyCoefficient.__init__(self, sdim)
       freq, omega = bdry.get_root_phys().get_freq_omega()
       
       self.real = real
       self.phase = phase  # phase !=0 for incoming wave
       self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec       
       self.AA = omega*np.sqrt(epsilon0*bdry.epsilonr/mu0/bdry.mur)
       
   def EvalValue(self, x):
       Hy = 1j*self.AA
       H = Hy*self.a_vec
       H = H * np.exp(1j*self.phase*np.pi/180.)
       if self.real:
            return H.real
       else:
            return H.imag
'''
   coax port TEM
'''
class C_Et_CoaxTEM(mfem.VectorPyCoefficient):
   def __init__(self, sdim, bdry, real = True):
       mfem.VectorPyCoefficient.__init__(self, sdim)
       freq, omega = bdry.get_root_phys().get_freq_omega()
       
       self.real = real
       self.a = bdry.a
       self.b = bdry.b
       self.ctr = bdry.ctr
       
   def EvalValue(self, x):
       r = (x - self.ctr)
       rho = np.sqrt(np.sum(r**2))
       nr = r/rho
#       E = nr*np.log(self.b/rho)/np.log(self.b/self.a)
       E = nr/np.log(self.b/self.a)/rho
       if self.real:
            return -E.real
       else:
            return -E.imag
class C_jwHt_CoaxTEM(mfem.VectorPyCoefficient):
   def __init__(self, sdim, phase, bdry, real = True, amp = 1.0):   
       mfem.VectorPyCoefficient.__init__(self, sdim)
       freq, omega = bdry.get_root_phys().get_freq_omega()
       
       self.real = real
       self.norm = bdry.norm
       self.a = bdry.a
       self.b = bdry.b
       self.ctr = bdry.ctr
       self.phase = phase  # phase !=0 for incoming wave
       self.AA = omega*np.sqrt(epsilon0*bdry.epsilonr/mu0/bdry.mur)
       
   def EvalValue(self, x):
       r = (x - self.ctr)
       rho = np.sqrt(np.sum(r**2))
       nr = r/rho
       nr = np.cross(nr, self.norm)
#       H = nr*np.log(self.b/rho)/np.log(self.b/self.a)       
       H = nr/np.log(self.b/self.a)/rho       
       H = 1j*self.AA*H
       H = H * np.exp(1j*self.phase*np.pi/180.)
       if self.real:
            return H.real
       else:
            return H.imag

class EM3D_Port(EM3D_Bdry):
    def __init__(self, mode = 'TE', mn = '0,1', inc_amp='1',
                 inc_phase='0', port_idx= 1):
        super(EM3D_Port, self).__init__(mode = mode,
                                        mn = mn,
                                        inc_amp = inc_amp,
                                        inc_phase = inc_phase,
                                        port_idx = port_idx)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_Port, self).attribute_set(v)
        v['port_idx'] = 1
        v['mode'] = 'TE'
        v['mn'] = ['1, 0']
        v['inc_amp_txt'] = '1.0'
        v['inc_phase_txt'] = '0.0'
        v['inc_amp'] = 1.0
        v['inc_phase'] = 0.0
        v['epsilonr'] = 1.0
        v['mur'] = 1.0
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
        
    def panel1_param(self):
        return [["port id", str(self.port_idx), 0, {}],
                ["mode",   self.mode,  4, {"readonly": True,
                 "choices": ["TE", "TEM", "Coax(TEM)"]}],
                ["m/n",    ','.join(str(x) for x in self.mn)  ,  0, {}],
                self.make_phys_param_panel('incoming amp.', self.inc_amp_txt),
                self.make_phys_param_panel('incoming phase.(deg)', self.inc_phase_txt, chk_float =True),
                ["epsilon_r",    self.epsilonr ,  300, {}],
                ["mu_r",    self.mur ,  300, {}]]
                 
    def get_panel1_value(self):
        return (str(self.port_idx), 
                self.mode, ','.join(str(x) for x in self.mn),
                self.inc_amp_txt, self.inc_phase_txt,
                self.epsilonr, self.mur)

    def import_panel1_value(self, v):
        self.port_idx = v[0]       
        self.mode = v[1]
        self.mn = [long(x) for x in v[2].split(',')]
        self.inc_amp_txt = str(v[3])
        self.inc_phase_txt = str(v[4])
        self.epsilonr = float(v[5])
        self.mur = float(v[6])

    def get_exter_NDoF(self):
        return 1  
        ## (in future) must be number of modes on this port...

    def update_param(self):
        self.update_inc_amp_phase()

    def update_inc_amp_phase(self):
        '''
        set parameter
        '''
        try:
            self.inc_amp = self.eval_phys_expr(str(self.inc_amp_txt),  'inc_amp')[0]
            self.inc_phase = self.eval_phys_expr(str(self.inc_phase_txt), 'inc_phase', chk_float = True)[0]
        except:
            raise ValueError("Cannot evaluate amplitude/phase to float number")

    def preprocess_params(self, engine):
        ### find normal (outward) vector...
        mesh = engine.get_mesh(mm = self)
        fespace = engine.fespaces[self.get_root_phys()][0][1]
        nbe = mesh.GetNBE()
        ibe = np.array([i for i in range(nbe)
                         if mesh.GetBdrElement(i).GetAttribute() == 
                            self._sel_index[0]])
        el = mesh.GetBdrElement(ibe[0])
        Tr = fespace.GetBdrElementTransformation(ibe[0])
        rules = mfem.IntegrationRules()
        ir = rules.Get(el.GetGeometryType(), 1)
        Tr.SetIntPoint(ir.IntPoint(0))
        nor = mfem.Vector(3)
        mfem.CalcOrtho(Tr.Jacobian(), nor)

        self.norm = nor.GetDataArray().copy()
        self.norm = self.norm/np.sqrt(np.sum(self.norm**2))
        
        #freq = self._global_ns["freq"]
        #self.omega = freq * 2 * np.pi
        #dprint1("Frequency " + (freq).__repr__())
        dprint1("Normal Vector " + list(self.norm).__repr__())

        ### find rectangular shape
        if str(self.mode).upper().strip() in ['TE', 'TM', 'TEM']:
           edges = np.array([mesh.GetBdrElementEdges(i)[0] for i in ibe]).flatten()
           d = {}
           for x in edges:d[x] = d.has_key(x)
           edges = [x for x in d.keys() if not d[x]]
           ivert = [mesh.GetEdgeVertices(x) for x in edges]
           ivert = connect_pairs(ivert)
           vv = np.vstack([mesh.GetVertexArray(i) for i in ivert])

           self.ctr = (np.max(vv, 0) + np.min(vv, 0))/2.0
           dprint1("Center " + list(self.ctr).__repr__())

           ### rectangular port
           #idx = np.argsort(np.sqrt(np.sum((vv - self.ctr)**2,1)))
               #corners = vv[idx[-4:],:]
           # since vv is cyclic I need to omit last one element here..
           idx = np.argsort(np.sqrt(np.sum((vv[:-1] - self.ctr)**2,1)))
           corners = vv[:-1][idx[-4:],:]
           for i in range(4):
              dprint1("Corner " + list(corners[i]).__repr__())
           tmp = np.sort(np.sqrt(np.sum((corners - corners[0,:])**2, 1)))
           self.b = tmp[1]
           self.a = tmp[2]
           tmp = np.argsort(np.sqrt(np.sum((corners - corners[0,:])**2, 1)))
           self.c = corners[0]  # corner
           self.b_vec = corners[tmp[1]]-corners[0]
           self.a_vec = np.cross(self.b_vec, self.norm)
#            self.a_vec = corners[tmp[2]]-corners[0]
           self.b_vec = self.b_vec/np.sqrt(np.sum(self.b_vec**2))
           self.a_vec = self.a_vec/np.sqrt(np.sum(self.a_vec**2))
           if np.sum(np.cross(self.a_vec, self.b_vec)*self.norm) > 0:
                self.a_vec = -self.a_vec

           if self.mode == 'TEM':
               '''
               special handling
               set a vector along PEC-like edge, regardless the actual
               length of edges
               '''
               for i in range(nbe):
                   if (edges[0] in  mesh.GetBdrElementEdges(i)[0] and
                       self._sel_index[0] != mesh.GetBdrAttribute(i)):
                       dprint1("Checking surface :",mesh.GetBdrAttribute(i))
                       attr = mesh.GetBdrAttribute(i)
                       break
               for node in self.get_root_phys().walk():
                   if not isinstance(node, Bdry): continue
                   if not node.enabled: continue
                   if attr in node._sel_index:
                       break
               from petram.model import Pair
               ivert = mesh.GetEdgeVertices(edges[0])
               vect = mesh.GetVertexArray(ivert[0]) -  mesh.GetVertexArray(ivert[1])
               vect = vect /np.sqrt(np.sum(vect**2))
               do_swap = False
               if (isinstance(node, Pair) and
                   np.abs(np.sum(self.a_vec*vect)) > 0.9):
                  do_swap = True
               if (not isinstance(node, Pair) and
                   np.abs(np.sum(self.a_vec*vect)) < 0.001):
                  do_swap = True
               if do_swap:
                    dprint1("swapping port edges")
                    tmp = self.a_vec
                    self.a_vec = -self.b_vec
                    self.b_vec = tmp
                    # - sign is to keep a \times b direction.
                    tmp = self.a
                    self.a = self.b
                    self.b = tmp
           if self.a_vec[ np.argmax(np.abs(self.a_vec))] < 0:
               self.a_vec = -self.a_vec
               self.b_vec = -self.b_vec
           dprint1("Long Edge  " + self.a.__repr__())
           dprint1("Long Edge Vec." + list(self.a_vec).__repr__())
           dprint1("Short Edge  " +  self.b.__repr__())
           dprint1("Short Edge Vec." + list(self.b_vec).__repr__())
        elif self.mode == 'Coax(TEM)':
           edges = np.array([mesh.GetBdrElementEdges(i)[0] for i in ibe]).flatten()
           d = {}
           for x in edges:d[x] = d.has_key(x)
           edges = [x for x in d.keys() if not d[x]]
           ivert = [mesh.GetEdgeVertices(x) for x in edges]
           iv1, iv2 = connect_pairs(ivert) # index of outer/inner circles
           vv1 = np.vstack([mesh.GetVertexArray(i) for i in iv1])
           vv2 = np.vstack([mesh.GetVertexArray(i) for i in iv2])
           ctr1, a1 = find_circle_center_radius(vv1, self.norm)
           ctr2, b1 = find_circle_center_radius(vv2, self.norm)
           self.ctr = np.mean((ctr1, ctr2), 0)
           self.a = a1 if a1<b1 else b1
           self.b = a1 if a1>b1 else b1
           dprint1("Big R:  " + self.b.__repr__())
           dprint1("Small R: " + self.a.__repr__())
           dprint1("Center:  " +  self.ctr.__repr__())
           vv = vv1 
        C_Et, C_jwHt = self.get_coeff_cls()
        dprint1("E field pattern")
        Et = C_Et(3, self, real = True)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Et.EvalValue(p).__repr__())
        dprint1("H field pattern")
        Ht = C_jwHt(3, 0.0, self, real = False)
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Ht.EvalValue(p).__repr__())
                  
                
    def get_coeff_cls(self):
        if self.mode == 'TEM':
            return C_Et_TEM, C_jwHt_TEM
        elif self.mode == 'TE':
            return C_Et_TE, C_jwHt_TE
        elif self.mode == 'Coax(TEM)':
            return C_Et_CoaxTEM, C_jwHt_CoaxTEM
        else:
            raise NotImplementedError(
             "you must implement this mode")

    def has_lf_contribution(self, kfes):
        if kfes != 0: return False
        return self.inc_amp != 0

    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))
       
        C_Et, C_jwHt = self.get_coeff_cls()

        inc_wave = self.inc_amp*np.exp(-1j*self.inc_phase/180.*np.pi)

        phase = np.angle(inc_wave)*180/np.pi
        amp   = np.abs(inc_wave)

        Ht = C_jwHt(3, phase, self, real = real, amp = amp)
        Ht = self.restrict_coeff(Ht, engine, vec=True)

        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
        b.AddBoundaryIntegrator(intg)

    '''    
    def add_lf_contribution_imag(self, engine, b):
        dprint1("Adding LF(imag) contribution")
        C_Et, C_jwHt = self.get_coeff_cls()        
        Ht = C_jwHt(3, self.inc_phase, self, real = False, amp = self.inc_amp)
        Ht = self.restrict_coeff(Ht, engine, vec=True)
        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
        b.AddBoundaryIntegrator(intg)
    '''
    def has_extra_DoF(self, kfes):
        if kfes != 0: return False       
        return True
     
    def get_exter_NDoF(self):
        return 1
     
    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name()+'_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()
        
    def add_extra_contribution(self, engine, **kwargs):
        dprint1("Add Extra contribution" + str(self._sel_index)) 

        C_Et, C_jwHt = self.get_coeff_cls()
        
        fes = engine.get_fes(self.get_root_phys(), 0)
        
        lf1 = engine.new_lf(fes)
        Ht = C_jwHt(3, 0.0, self, real = True)
        Ht = self.restrict_coeff(Ht, engine, vec=True)
        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
        lf1.AddBoundaryIntegrator(intg)
        lf1.Assemble()
        lf1i = engine.new_lf(fes)
        Ht = C_jwHt(3, 0.0, self, real = False)
        Ht = self.restrict_coeff(Ht, engine, vec=True)
        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
        lf1i.AddBoundaryIntegrator(intg)
        lf1i.Assemble()
        

        lf2 = engine.new_lf(fes)
        Et = C_Et(3, self, real = True)
        Et = self.restrict_coeff(Et, engine, vec=True)
        intg = mfem.VectorFEDomainLFIntegrator(Et)
        lf2.AddBoundaryIntegrator(intg)
        lf2.Assemble()
        
        x = engine.new_gf(fes)
        x.Assign(0.0)
        arr = self.get_restriction_array(engine)
        x.ProjectBdrCoefficientTangent(Et,  arr)

        t4 = np.array([[self.inc_amp*np.exp(1j*self.inc_phase/180.*np.pi)]])


        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat

        v1 = LF2PyVec(lf1, lf1i)
        v1 *= -1
        v2 = LF2PyVec(lf2, None, horizontal = True)
        x  = LF2PyVec(x, None)
        
        # output formats of InnerProduct
        # are slightly different in parallel and serial
        # in serial numpy returns (1,1) array, while in parallel
        # MFEM returns a number. np.sum takes care of this.
        tmp = np.sum(v2.dot(x))
        v2 *= 1./tmp

        v1 = PyVec2PyMat(v1)
        v2 = PyVec2PyMat(v2.transpose()).transpose()
        t4 = Array2PyVec(t4)
        t3 = IdentityPyMat(1)
        return (v1, v2, t3, t4, True)

        '''
        if use_parallel:

            from petram.solver.solver_utils import gather_vector
            from mfem.common.chypre import CHypreVec

            lf1bv = lf1.ParallelAssemble()
            lf1ibv =lf1i.ParallelAssemble()
            lf2v = lf2.ParallelAssemble()
            xbv   = x.ParallelAssemble()
            
            lf1bv  *= -1.0
            lf1ibv *= -1.0

            tmp = mfem.InnerProduct(lf2v, xbv)
            lf2v *= 1./tmp
            
            v1 = CHypreVec(lf1bv, lf1ibv)
            v2 = CHypreVec(lf2v,  None)

            from petram.solver.mumps.hypre_to_mumps import PyZMatrix
            if myid == 0:
                t3  = PyZMatrix(1, 1, 1)
                t3.set_rdata(1.0)
            else:
                t3 = None

            return (v1, v2, t3, t4, True)
         
        else:

            v1 = -(lf1.GetDataArray() + 1j*lf1i.GetDataArray())
            v2 = lf2.GetDataArray()/np.sum(lf2.GetDataArray()*x.GetDataArray())
            
            dprint2("extra matrix nnz before elimination ", len(v1.nonzero()[0]), " ",  len(v2.nonzero()[0]))
            return (v1.reshape(-1,1), v2.reshape(1,-1), np.array(1).reshape(1,1),
                    t4, True)

            return (v1, v2, t3,  t4, True)
        '''             
        #  (0,x), (x, 0), (x, x) , rhs, number of DoF
  

        
