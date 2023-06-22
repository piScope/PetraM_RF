import numpy as np
from numba import njit,  int32, int64, float64, complex128, types

zfunc_c0 = -1.8545998915445172e-05-1.5661929389884643e-05j
z_func_cd = np.array([(1.5286514261986445e-05+0.0001338198102574673j, 1.716400180773478+1.4077210296411122j),
                      (-0.04535121664037525+0.21770596727006358j, 1.716400180773478-1.4077210296411122j),
                      (8.868862346159413e-05+9.518447335868001e-05j, -1.6671994586793688+1.390541660571995j),
                      (-0.009161315175716632-0.2474253047433214j, -1.6671994586793688-1.390541660571995j),
                      (0.00031896829654753906-0.00012225173013488797j, 0.8085401388105352+1.4737003544066358j),
                      (1.625988208680783-1.3818455149161408j, 0.8085401388105352-1.4737003544066358j),
                      (-0.0002441894449757977+0.0001708062926842305j, -0.7742194394628614+1.4722571763175376j),
                      (1.5404561025083063+1.5857586800922299j, -0.7742194394628614-1.4722571763175376j),
                      (-0.00020828998612896403-0.0003401290008802969j, 0.014033476202404027+1.496503032385095j),
                      (-4.112237248224732-0.17403327041934671j, 0.014033476202404027-1.496503032385095j),])


@njit(complex128(float64))
def zfunc(x):
    '''
    Rational approximation of 1j* np.sqrt(np.pi)*wofz(z) using 10 poles
    over 0<x<10. Use it carefully, the worst error of real value ~ 10^-4 
    '''
    if np.abs(x) > 10.:
        return -1/x - 1/2/x**3 - 3/4/x**5 - 15/8/x**7
    value = zfunc_c0
    for c, d in z_func_cd:
        value = value + c / (x - d)
    return value
