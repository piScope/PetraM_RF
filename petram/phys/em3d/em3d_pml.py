'''
   PML

   * Linear PML
            [S(x)          ]
        S = [     S(y)     ]
            [          S(z)]

        S  = 1 + (Sr + iSi)*((r -Lr)/Lpml)^p

        Lr = the coordinate of boundar between normal and PML
        Lpml = PML thickness
        Sr, Si = PML strech factor
        p = the order of PML stretch

      We replase epsilon and mu in the PML region as follow
         epsilonr -> G * epsilon * S^-1, 
         mur -> G * mur * S^-1,
      where G = det(S) S^-1. det(S) is S(x)*S(y)*S(z) for linear case

      CopyRight (c) 2020-  S. Shiraiwa
''' 
import numpy as np

from petram.phys.phys_model  import Phys, PhysCoefficient, MatrixPhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_PML')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   
data =  (('stretch', VtableElement('stretch', type='complex',
                                     guilabel = 'S(stretch)',
                                     default = 1.0, 
                                     tip = "streach factor" )),
         ('s_order', VtableElement('s_order', type='complex',
                                     guilabel = 'order',
                                     default = 1.0, 
                                     tip = "streach order" )),)

from petram.phys.coefficient import SCoeff
from petram.phys.em3d.em3d_const import mu0, epsilon0

class EM3D_PML(EM3D_Domain):
    def make_PML_epsilon(self, coeffr, coeffi, returnReal):
        # update epsilon        
        raise NotImplementedError(
             "you must specify this method in subclass")

    def make_PML_invmu(self, coeffr, coeffi, returnReal):
        # update mu        
        raise NotImplementedError(
             "you must specify this method in subclass")

    def make_PML_sigma(self, coeffr, coeffi, returnReal):
        # update sigma
        raise NotImplementedError(
             "you must specify this method in subclass")

    def get_parent_selection(self):
        return self.parent.sel_index

class LinearPML(mfem.MatrixPyCoefficient):
   def __init__(self, coeffr, coeffi, S, direction, ref_point, pml_width, order, real, inv):
      self.coeffs = coeffr, coeffi
      self.direction = direction
      self.ref_point = ref_point
      self.pml_width = pml_width
      self.S = S
      self.order = order
      self.real = real
      self.inv = inv
      
   def Eval(self, K, T, ip):
       ptx = mfem.Vector(3)
       T.transform(ip, ptx)
       ptxx = ptx.GetDataArray()

       invS = self.Eval_invS(ptxx)
       detS = self.Eval_detS(ptxx)       
       
       if isinstance(self.coeffs[0], mfem.MatrixCoefficient):
           self.coeffs[0].Eval(K, T, ip)
           K_m  = K.GetDataArray().copy()
           if self.coeffs[1] is not None:
               self.coeffs[1].Eval(K, T, ip)              
               K_m += 1j * K.GetDataArray()
       else:
           K_m  = self.coeffs[0].Eval(T, ip)
           if self.coeffs[1] is not None:
               K_m += 1j * self.coeffs[1].Eval(T, ip)              


       detS_inv_S_x_inv_S = (invS/detS).dot(K_m).dot(invS)

       if self.inv:
           detS_inv_S_x_inv_S = np.linalg.inv(detS_inv_S_x_inv_S)
           
       if self.real:
          return K.Assign(detS_inv_S_x_inv_S.real)
       else:
          return K.Assign(detS_inv_S_x_inv_S.imag)          
       
   def Eval_invS(self, x):
       ret = np.array([1, 1., 1.])
       if self.direction[0]:
           ret[0] = 1 + self.S * ((x[0] - self.ref_point[0])/self.pml_width[0])^self.order
       if self.direction[1]:
           ret[1] = 1 + self.S * ((x[1] - self.ref_point[1])/self.pml_width[1])^self.order
       if self.direction[2]:          
           ret[2] = 1 + self.S * ((x[2] - self.ref_point[2])/self.pml_width[2])^self.order
       return np.diag(1/ret, dtype=complex)

   def Eval_detS(self, x):
       ret = np.array([1, 1., 1.])
       if self.direction[0]:
           ret[0] = 1 + self.S * ((x[0] - self.ref_point[0])/self.pml_width[0])^self.order
       if self.direction[1]:
           ret[1] = 1 + self.S * ((x[1] - self.ref_point[1])/self.pml_width[1])^self.order
       if self.direction[2]:          
           ret[2] = 1 + self.S * ((x[2] - self.ref_point[2])/self.pml_width[2])^self.order
       return ret[0]*ret[1]*ret[2]
    
class EM3D_LinearPML(EM3D_PML):
    has_2nd_panel = False
    has_3rd_panel = False
    _has_4th_panel = False    
    vt  = Vtable(data)

    def attribute_set(self, v):
        v = super(EM3D_LinearPML, self).attribute_set(v)
        v["stretch_dir"] = [False, False, False]
        v["ref_point"] = ""
        return v
    
    def panel1_param(self):
        panels1 = [["Direction", self.stretch_dir, 36,
                    {'col':3, 'labels':['x', 'y', 'z']}],
                   ["Ref. point", self.ref_point, 0, {}],                   
                  ]
        panels2 = super(EM3D_LinearPML, self).panel1_param()
        
        panels = panels1 + panels2
        return panels

    def get_panel1_value(self):
        val =  super(EM3D_LinearPML, self).get_panel1_value()
        val = [self.stretch_dir, self.ref_point] + val
        return val

    def import_panel1_value(self, v):
        print("PML import value", v)
        super(EM3D_LinearPML, self).import_panel1_value(v[2:])
        self.ref_point = str(v[1])
        self.stretch_dir = [vv[1] for vv in v[0]]
    
    def preprocess_params(self, engine):
        base_mesh = engine.meshes[0]

        ec = base_mesh.extended_connectivity
        v2v = ec['vert2vert']

        idx = int(self.ref_point)
        self.ref_point_coord = base_mesh.GetVertexArray(v2v[idx])
        
        dprint1("PML reference point:", self.ref_point_coord)

        s = sum([ec['vol2surf'][k] for k in self.get_parent_selection()], [])
        l = sum([ec['surf2line'][k] for k in s], [])
        l = list(set(l))
        v = sum([ec['line2vert'][k] for k in l], [])
        v = list(set(v))        

        ptx = np.vstack([base_mesh.GetVertexArray(v2v[idx]) for idx in v])

        self.pml_width = [0]*3
        for k in range(3):
           dd = ptx[:,k] - self.ref_point_coord[k]
           ii = np.argmax(np.abs(dd))
           self.pml_width[k] = dd[ii]

        dprint1("PML width ", self.pml_width)
        

    def make_PML_epsilon(self, coeffr, coeffi, returnReal):
        coeff = LinearPML(coeffr, coeffi, self.stretch, self.stretch_dir,
                          self.ref_point, self.pml_width, self.s_order, returnReal, False)
        return coeff

    def make_PML_invmu(self, coeffr, coeffi, returnReal):
        coeff = LinearPML(coeffr, coeffi, self.stretch, self.stretch_dir,
                          self.ref_point, self.pml_width, self.s_order, returnReal, True)
        return coeff

    def make_PML_sigma(self, coeffr, coeffi, returnReal):
        coeff = LinearPML(coeffr, coeffi, self.stretch, self.stretch_dir,
                          self.ref_point, self.pml_width, self.s_order, returnReal, False)
        return coeff


    

