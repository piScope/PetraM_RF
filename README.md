## Petra-M(RF)

### EM3D : Frequency domain Maxwell equation in 3D
  Domain:   
 *    EM3D_Anisotropic : tensor dielectric
 *    EM3D_Vac         : scalar dielectric
 *    EM3D_ExtJ        : external current
 *    EM3D_Div         : div J = 0 constraints (add Lagrange multiplier)
 *    EM3D_PML         : PML cartesian streaching
 
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
  Domain:   
 *    EM2Da_Anisotropic : tensor dielectric
 *    EM2Da_Vac         : scalar dielectric
 *    EM2Da_ExtJ        : external current
 *    EM2Da_Div         : div J = 0 constraints (add Lagrange multiplier)

  Boundary:
 *    EM2Da_PEC         : Perfect electric conductor
 *    EM2Da_PMC         : Perfect magnetic conductor
 *    EM2Da_H           : Mangetic field boundary
 *    EM2Da_SurfJ       : Surface current
 *    EM2Da_Port        : TE, TEM, Coax port
 *    EM2Da_E           : Electric field
 *    EM2Da_Continuity  : Continuitiy

### EM2D : Frequency domain Maxwell equation in 2D space
  Domain:   
 *    EM2D_Anisotropic : tensor dielectric
 *    EM2D_Vac         : scalar dielectric
 *    EM2D_ExtJ        : external current
 *    EM2D_PML         : PML cartesian streaching

  Boundary:
 *    EM2D_PEC         : Perfect electric conductor
 *    EM2D_PMC         : Perfect magnetic conductor
 *    EM2D_H           : Mangetic field boundary (N.I)
 *    EM2D_SurfJ       : Surface current         (N.I)
 *    EM2D_Port        : TE, TEM, Coax port      (N.I)
 *    EM2D_E           : Electric field
 *    EM2D_Continuity  : Continuitiy

### EM1D : Frequency domain Maxwell equation in 1D
  Domain:   
 *    EM1D_Anisotropic : tensor dielectric
 *    EM1D_Vac         : scalar dielectric
 *    EM1D_ExtJ        : external current
 
  Boundary:
 *    EM1D_PEC         : Perfect electric conductor
 *    EM1D_PMC         : Perfect magnetic conductor
 *    EM1D_H           : Mangetic field boundary
 *    EM1D_Port        : Surface current
 *    EM1D_E           : Electric field
 *    EM1D_Continuity  : Continuitiy

