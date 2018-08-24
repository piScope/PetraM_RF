## Petra-M(RF)

### EM3D : Frequency domain Maxwell equation in 3D
  Domain:   
 *    EM3D_Anisotropic : tensor dielectric
 *    EM3D_Vac         : scalar dielectric
 *    EM3D_ExtJ        : external current
 *    EM3D_Div         : div J = 0 constraints (add Lagrange multiplier)

  Boundary:
 *    EM3D_PEC         : Perfect electric conductor
 *    EM3D_PMC         : Perfect magnetic conductor
 *    EM3D_H           : Mangetic field boundary
 *    EM3D_SurfJ       : Surface current
 *    EM3D_Port        : TE, TEM, Coax port
 *    EM3D_E           : Electric field
 *    EM3D_Continuity  : Continuitiy

  Pair:
 *    EM3D_Floquet     : Periodic boundary condition

### EM2Da : Frequency domain Maxwell equation in 2D axissymetric space
