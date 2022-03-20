from __future__ import print_function
from petram.phys.vtable import VtableElement, Vtable
'''
EM1d : 1D Frequency domain Maxwell equation.

    This module is meant to solve 

    (curl v, curl u) + (v, e u) 
           - (v, n x (1/mu curl u)_s = i w (jext, u) 

    in 1D. x is the direction of the wave propagation.
    Ex, Ey, Ez are treated as L2, H1, H1 elements respectively.

  Domain:   
     EM1D_Anisotropic : tensor dielectric
     EM1D_Vac         : scalar dielectric

  Boundary:
     EM1D_PEC         : Perfect electric conductor
     EM1D_PMC         : Perfect magnetic conductor
     EM1D_H           : Mangetic field boundary
     EM1D_Port        : Surface current
     EM1D_E           : Electric field
     EM1D_Continuity  : Continuitiy

  Note: the above modules are plan. Not fully implemented.

'''
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.em1d.em1d_base import EM1D_Bdry
from petram.phys.em1d.em1d_vac import EM1D_Vac

txt_predefined = 'freq, e0, mu0'


data2 = (('label1', VtableElement(None,
                                  guilabel='Default Bdry (PMC)',
                                  default="Ht = 0",
                                  tip="this is a natural BC")),)


class EM1D_DefDomain(EM1D_Vac):
    can_delete = False
    nlterms = []
    #vt  = Vtable(data1)
    # do not use vtable here, since we want to use
    # vtable defined in EM3D_Vac in add_bf_conttribution

    def __init__(self, **kwargs):
        super(EM1D_DefDomain, self).__init__(**kwargs)

    def panel1_param(self):
        return [['Default Domain (Vac)',   "eps_r=1, mu_r=1, sigma=0, ky=0, kz=0",  2, {}], ]

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass

    def panel1_tip(self):
        return None

    def get_possible_domain(self):
        return []


class EM1D_DefBdry(EM1D_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []
    vt = Vtable(data2)

    def __init__(self, **kwargs):
        super(EM1D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM1D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class EM1D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = True

    def __init__(self, **kwargs):
        super(EM1D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM1D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []


class EM1D(PhysModule):
    der_vars_base = ['B']
    geom_dim = 1

    def __init__(self, **kwargs):
        super(EM1D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = EM1D_DefDomain()
        self['Boundary'] = EM1D_DefBdry()
        self['Pair'] = EM1D_DefPair()

    @property
    def dep_vars(self):
        '''
        list of dependent variables, for example.
           [Et, rEf]      
           [Et, rEf, psi]
        '''
        coord = ['x', 'y', 'z']
        ret = self.dep_vars_base + self.dep_vars_suffix
        return [ret + x for x in coord]

    @property
    def der_vars(self):
        coord = ['x', 'y', 'z']
        ret = EM1D.der_vars_base[0] + self.dep_vars_suffix
        return [ret + x for x in coord]

    @property
    def dep_vars0(self):
        coord = ['x', 'y', 'z']
        ret = self.dep_vars_base + self.dep_vars_suffix
        return [ret + x for x in coord]

    @property
    def dep_vars_base(self):
        ret = 'E'
        return ret

    def get_fec_type(self, idx):
        '''
        H1 
        H1v2 (vector dim)
        ND
        RT
        '''
        values = ['L2', 'H1', 'H1']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        return [(v[0], 'L2_FECollection'),
                (v[1], 'H1_FECollection'),
                (v[2], 'H1_FECollection'), ]
    
    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)        
        if idx == 0:
            return self.order -1
        else:
            return self.order            

    def _has_div_constraint(self):
        return False
        #from .em1d_div import EM1D_Div
        # for mm in self['Domain'].iter_enabled():
        #    if isinstance(mm, EM1D_Div): return True
        # return False

    def attribute_set(self, v):
        v = super(EM1D, self).attribute_set(v)
        v["element"] = 'L2_FECollection, H1_FECollection, H1_FECollection'
        v["freq_txt"] = 1.0e9
        v["ndim"] = 1
        v["ind_vars"] = 'x'
        v["dep_vars_suffix"] = ''
        return v

    def panel1_param(self):
        panels = super(EM1D, self).panel1_param()
        panels.extend([self.make_param_panel('freq',  self.freq_txt),
                       ["indpendent vars.", self.ind_vars, 0, {}],
                       ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                       ["dep. vars.", ','.join(self.dep_vars), 2, {}],
                       ["derived vars.", ','.join(self.der_vars), 2, {}],
                       ["predefined ns vars.", txt_predefined, 2, {}]])
        return panels

    def get_panel1_value(self):
        names = ','.join([x for x in self.dep_vars])
        names2 = ','.join([x for x in self.der_vars])
        val = super(EM1D, self).get_panel1_value()
        val.extend([self.freq_txt, self.ind_vars, self.dep_vars_suffix,
                    names, names2, txt_predefined])
        return val

    def attribute_expr(self):
        return ["freq"], [float]

    def get_default_ns(self):
        from petram.phys.em1d.em1d_const import mu0, epsilon0, q0
        ns = {'mu0': mu0,
              'e0': epsilon0,
              'q0': q0}
        return ns

    def attribute_mirror_ns(self):
        return ['freq']

    def import_panel1_value(self, v):
        v = super(EM1D, self).import_panel1_value(v)
        self.freq_txt = str(v[0])
        self.ind_vars = str(v[1])
        self.dep_vars_suffix = str(v[2])

        from petram.phys.em1d.em1d_const import mu0, epsilon0

        self._global_ns['mu0'] = mu0
        self._global_ns['epsilon0'] = epsilon0

    def get_possible_bdry(self):
        from .em1d_pec import EM1D_PEC
        from .em1d_pmc import EM1D_PMC
        #from em1d_h       import EM1D_H
        #from em1d_surfj       import EM1D_SurfJ
        from .em1d_port import EM1D_Port
        from .em1d_e import EM1D_E
        from .em1d_cont import EM1D_Continuity

        bdrs = super(EM1D, self).get_possible_bdry()

        return [EM1D_PEC,
                EM1D_Port,
                EM1D_E,
                EM1D_PMC,
                EM1D_Continuity] + bdrs

    def get_possible_domain(self):
        from .em1d_anisotropic import EM1D_Anisotropic
        from .em1d_vac import EM1D_Vac
        from .em1d_extj import EM1D_ExtJ

        doms = super(EM1D, self).get_possible_domain()
        #from em3d_div       import EM3D_Div

        return [EM1D_Vac, EM1D_Anisotropic, EM1D_ExtJ] + doms

    def get_possible_edge(self):
        return []

    def get_possible_pair(self):
        #from em3d_floquet       import EM3D_Floquet
        return []

    def get_possible_point(self):
        return []

    def is_complex(self):
        return True

    def get_freq_omega(self):
        return self._global_ns['freq'], 2.*np.pi*self._global_ns['freq']

    def add_variables(self, v, name, solr, soli=None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_component_expression as addc_expression
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant

        from petram.phys.em1d.eval_deriv import eval_grad

        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        from petram.helper.variables import TestVariable

        freq, omega = self.get_freq_omega()
        add_constant(v, 'omega', suffix, np.float(omega),)
        add_constant(v, 'freq', suffix, np.float(freq),)
        add_constant(v, 'mu0', '', self._global_ns['mu0'])
        add_constant(v, 'e0', '', self._global_ns['e0'])

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)
        add_scalar(v, name, "", ind_vars, solr, soli)

        if name.startswith('E'):
            if name.endswith('y'):
                add_scalar(v, 'gradEy', suffix, ind_vars, solr, soli,
                           deriv=eval_grad, vars=['E'])
            if name.endswith('z'):
                add_scalar(v, 'gradEz', suffix, ind_vars, solr, soli,
                           deriv=eval_grad, vars=['E'])

        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(1j*ky*Ez - 1j*kz*Ey)',
                        ['ky', 'kz', 'E', 'omega'], 0)
        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(1j*kz*Ex - gradEz)',
                        ['ky', 'kz', 'E', 'omega'], 'y')
        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(gradEy - 1j*ky*Ex)',
                        ['ky', 'kz', 'E', 'omega'], 'z')
        add_expression(v, 'E', suffix, ind_vars,
                       'array([Ex, Ey, Ez])',  ['E'])

        add_expression(v, 'B', suffix, ind_vars,
                       'array([Bx, By, Bz])',  ['B'])

        # Poynting Flux
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ey)*Bz - conj(Ez)*By)/mu0',
                        ['B', 'E'], 0)
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ez)*Bx - conj(Ex)*Bz)/mu0',
                        ['B', 'E'], 'y')
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ex)*By - conj(Ey)*Bx)/mu0',
                        ['B', 'E'], 'z')

        return v

    def get_fes_for_dep(self, unknown_name, soldict):
        keys = soldict.keys()

        print(unknown_name, soldict.keys())
        assert False,  "check if this is called"
        k = unknown_name
        sol = soldict[k]
        solr = sol[0]
        soli = sol[1] if len(sol) > 1 else None
        return solr, soli
