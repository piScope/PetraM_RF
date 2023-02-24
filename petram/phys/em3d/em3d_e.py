from petram.mfem_config import use_parallel
import numpy as np


from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable

#from petram.phys.phys_model  import Phys, VectorPhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_E')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('E', VtableElement('E', type='complex',
                            guilabel='Electric field',
                            suffix=('x', 'y', 'z'),
                            default=np.array([0, 0, 0]),
                            tip="essential BC")),)


class Vtable_mod(Vtable):
    # this is for backword compatibility. previously E_(xyz)_txt was not used.
    def make_value_or_expression(self, obj, keys=None):
        if not hasattr(obj, "_had_data_transfer"):
            if hasattr(obj, "E_x") and isinstance(obj.E_x, str) and obj.E_x_txt != obj.E_x:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! making adjustment")
                obj.E_x_txt = obj.E_x
            if hasattr(self, "E_y") and isinstance(obj.E_y, str) and obj.E_y_txt != obj.E_y:
                obj.E_y_txt = obj.E_y
            if hasattr(obj, "E_z") and isinstance(obj.E_z, str) and obj.E_z_txt != obj.E_z:
                obj.E_z_txt = obj.E_z
            obj._had_data_transfer = True
        return Vtable.make_value_or_expression(self, obj, keys=keys)


def bdry_constraints():
    return [EM3D_E]


class EM3D_E(EM3D_Bdry):
    has_essential = True
    vt = Vtable_mod(data)

    def attribute_set(self, v):
        super(EM3D_E, self).attribute_set(v)
        return v

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))

        Exyz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = VCoeff(3, Exyz, ind_vars, l, g, return_complex=True)
        #f_name = self._make_f_name()
        # coeff1 = Et(3, f_name,  self.get_root_phys().ind_vars,
        #            self._local_ns, self._global_ns,
        #            real = real)
        coeff1 = self.restrict_coeff(coeff1, engine, vec=True)

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff1 = coeff1.get_realimag_coefficient(real)
        gf.ProjectBdrCoefficientTangent(coeff1, mfem.intArray(bdr_attr))
