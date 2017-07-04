'''
   Surface Current Boundary Condition

   On external surface    n \times B = J_surf
   On internal surface    n \times (B1 - B2) = J_surf (not tested)

   (note)
    In MKSA,  1/mu curl E = -dB/dt 1/mu= i omega B / mu .
      n \times 1/mu curl E = -dB/dt 1/mu= i omega B / mu = i omega /mu J_surf
    
 
    Therefore, surface integral 
        \int W \dot (n \times 1/mu curl E) d\Omega
    becomes
         \int W \dot (i omega J_surf) d\Omega

    and VectorFEDomainLFIntegrator can be used.


   CopyRight (c) 2016-  S. Shiraiwa
''' 
import numpy as np

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys, VectorPhysCoefficient

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_SurfJ')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

name_suffix = ['_x', '_y', '_z']   

class Jsurf(VectorPhysCoefficient):
   def __init__(self, *args, **kwargs):
       omega = kwargs.pop('omega', 1.0)
       self.mur = kwargs.pop('mur', 1.0)
       from em3d_const import mu0, epsilon0
       self.fac = -1j*omega
       super(Jsurf, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from em3d_const import mu0, epsilon0      
       v = super(Jsurf, self).EvalValue(x)
       v = self.fac * v
       if self.real:  return v.real
       else: return v.imag


class EM3D_SurfJ(Bdry, Phys):
    is_essential = False
    def __init__(self, **kwargs):
        super(EM3D_SurfJ, self).__init__( **kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_SurfJ, self).attribute_set(v)               
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['surfJ_x'] = '0.0'
        v['surfJ_y'] = '0.0'
        v['surfJ_z'] = '0.0'
        v['surfJ_m'] = '[0.0, 0.0, 0.0]'        
        v['use_m_surfJ'] = False

        return v
    
    def panel1_param(self):
        return [self.make_matrix_panel('surfJ', name_suffix, 3, 1)]
        
    def form_name(self, v):
        if v: return 'Array Form'
        else: return 'Elemental Form'

    def get_panel1_value(self):
        a = [self.form_name(self.use_m_surfJ), 
             [[str(getattr(self, 'surfJ'+n)) for n in name_suffix]],
             [str(self.surfJ_m)]]
        return [a, ]

    def preprocess_params(self, engine):
        dprint1('Preprocess SurfJ')
        k = 0
        for n in name_suffix:
            setattr(self, 'surfJ'+n, str(getattr(self, 'surfJ' + n)))
            k = k + 1

    def import_panel1_value(self, v):
        self.use_m_surfJ = (str(v[0][0]) == 'Array Form')
        for k, n in enumerate(name_suffix):
            setattr(self, 'surfJ'+n, str(v[0][1][0][k]))
        self.surfJ_m = str(v[0][2][0])

    def _make_f_name(self):
        basename = 'surfJ'
        if getattr(self, 'use_m_'+basename):
            names = ['_m']
            eval_expr = self.eval_phys_array_expr
        else:
            names = ['_x', '_y', '_z',]
            eval_expr = self.eval_phys_expr
              
        f_name = []
        for n in names:
           var, f_name0 = eval_expr(getattr(self, basename+n), basename + n)
           if f_name0 is None:
               f_name.append(var)
           else:
               f_name.append(f_name0)
        return f_name

        
    def has_lf_contribution(self, kfes = 0):
        if kfes != 0: return False
        return True
    
    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes != 0: return 
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))

        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        f_name = self._make_f_name()

        coeff1 = Jsurf(3, f_name,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega,)
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddBoundaryIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))


