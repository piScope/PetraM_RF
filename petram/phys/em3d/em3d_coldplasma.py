from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys
class coldplasma_parameters(object):
    def __init__(self):
        pass
   
    def __call__(self, *args):
        '''
        returns bx, by, bz, 
               ne(1,2,3), ni(1,2,3), zeff       
        '''
        return self.b(args), self.dens(args), self.temp(args), self.zeff(args)
   
    def dens(self, x, y, z):
        return [1e19,1e19]
   
    def temp(self, x, y, z):
        return  [100,100]
   
    def zeff(self, x, y, z):
        return 1.0

    def b(self, x, y, z):
        return [1, 0, 0]
   

class tokamak_profile(coldplasma_parameters):
   def __init__(self):
       self.zgrid = None
       self.rgrid = None
       self.psirz = None
       self.btrz = None
       self.brrz = None
       self.bzrz = None       

   def import_gfile(self, gfile):
       self.zgrid = gfile.get_contents("table","zgrid")
       self.rgrid = gfile.get_contents("table","rgrid")
       self.psirz = gfile.get_contents("table","psirz")
       self.btrz  = gfile.get_contents("table","btrz")
       self.brrz  = gfile.get_contents("table","brrz")
       self.bzrz  = gfile.get_contents("table","bzrz")
       
   def dens(self, x, y, z):
       return [1e19,1e19]
   
   def temp(self, x, y, z):
       return  [100,100]
   
   def zeff(self, x, y, z):
       return 1.0

   def b(self, x, y, z):
       return [1, 0, 0]
    
class EM3D_ColdPlasma(Domain, Phys):
    def __init__(self,  **kwargs):
        super(EM3D_ColdPlasma, self).__init__(**kwargs)
        self.sel_readonly = False
        self.sel_index = []

        # this values are name of variabls in param
        self.coldplasma_params = kwargs.pop('param', 'coldplasma_params')

    def panel1_param(self):
        # this is a function object
        #    argument is x, y, z
        #    returns bx, by, bz, ne(1,2,3), ni(1,2,3), zeff
        return [["Params",  str(self.coldplasma_params),  0, {}],]
    
    def get_panel1_value(self):
        return str(self.coldplasma_params)
        
    def import_panel1_value(self, v):
        pass
        
    def write_setting(self):
        '''
        read gfile
        read denstiy, temperature
        write all data
        '''
        pass
    def has_bf_contribution(self):
        return True
    
    def add_bf_contribution(self, engine, a):
        dprint1("Add BF(real) contribution" + str(self._sel_index))
        from em3d_const import mu0, epsilon0

        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        mur = np.float(np.real(1./mu0/self.mur))
        dprint1("mur " + str(mur))
        if mur != 0.0:
            coeff = mfem.ConstantCoefficient(mur)
            coeff = self.restrict_coeff(coeff, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff))
        else:
            dprint1("No cotrinbution from curlcurl")
            
        mass = -(self.epsilonr*epsilon0*omega*omega - 1j*omega*self.sigma)
        dprint1("mass " + str(mass))        
#        if mass.real != 0.0:
        coeff = mfem.ConstantCoefficient(mass.real)
        coeff = self.restrict_coeff(coeff, engine)
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff))
#        else:
#            dprint1("No cotrinbution from mass")            

    def add_bf_contribution_imag(self, engine, a):
        dprint1("Add BF(imag) contribution"+ str(self._sel_index))    
        from em3d_const import mu0, epsilon0

        from em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        mur = np.float(np.imag(1./mu0/self.mur))
        dprint1("mur " + str(mur))        
        if mur != 0.0:
            coeff = mfem.ConstantCoefficient(mur)
            coeff = self.restrict_coeff(coeff, engine)
            a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff))
        else:
            dprint1("No cotrinbution from curlcurl")
        mass = -(self.epsilonr*epsilon0*omega*omega - 1j*omega*self.sigma)
        dprint1("mass " + str(mass))                
#        if mass.imag != 0.0:
        coeff = mfem.ConstantCoefficient(mass.imag)
        coeff = self.restrict_coeff(coeff, engine)
        a.AddDomainIntegrator(mfem.VectorFEMassIntegrator(coeff))
#        else:
#            dprint1("No cotrinbution from mass")            


'''
variables
ne_exponential ne_edge*exp(-wd.Dw/sol_lambda_ne)
te_exponential te_edge*exp(-wd.Dw/sol_lambda_te)
ni ne*(1-fast_frac)
nim ne*fast_frac
Ti Te ions temperature
Tim Te
vTe sqrt(2*Te/me) electrons thermal velocity
vTeF vTe*sqrt(Te0F/Te0) fast electrons velocity
vTi sqrt(2*Ti/mi) ions thermal velocity
vTim sqrt(2*Ti/mim)
LAMBDA 1+12*pi*(e0*Te/1[J*F/m])^(3/2)/(q^3/1[C^3]*sqrt(ne*1[m^3])) 1+LAMDA as in log(1+LAMBDA) coulomb logarithm
nu_ei qi^2*qe^2*ni*log(LAMBDA)/(4*pi*e0^2*me^2)/vTe^3 e-i collision frequency
nu_eim qim^2*qe^2*nim*log(LAMBDA)/(4*pi*e0^2*me^2)/vTe^3
me_eff (1+nu_ei/(i*w)+nu_eim/(i*w))*me effective electrons mass (to account for collisions)
mi_eff (1+nu_ei/(i*w)+nu_eim/(i*w))*mi effective ions mass (to account for collisions)
mim_eff (1+nu_ei/(i*w)+nu_eim/(i*w))*mim
wpe sqrt(ne*q^2/(me_eff*e0)) electron plasma freuquency
wpi sqrt(ni*qi^2/(mi_eff*e0)) ion plasma frequency
wpim sqrt(nim*qim^2/(mim_eff*e0))
wce -q*B/me_eff electron cyclotron frequency
wci qi*B/mi_eff ion cyclotron frequency
wcim qim*B/mim_eff
P_rf 1-wpe^2/w^2-wpi^2/w^2-wpim^2/w^2 STIX P
S_rf 1-wpe^2/(w^2-wce^2)-wpi^2/(w^2-wci^2)-wpim^2/(w^2-wcim^2) STIX S
D_rf wce*wpe^2/(w*(w^2-wce^2))+wci*wpi^2/(w*(w^2-wci^2))+wcim*wpim^2/(w*(w^2-wcim^2)) STIX D                                                	                                                                                                                                                                                                                                                                                                  poloidal field
Br -eqpsiz/r X component of poloidal field
Bz eqpsir/r Toroidal magnetic fiedl
Bt fpol_efit/r
Bp sqrt(Br^2+Bz^2)
B sqrt(Bt^2+Bp^2)
th atan2(Bz,Br) angular coordinate
ph atan2(Br*cos(th)+Bz*sin(th),Bt)
S S_rf
P P_rf
D D_rf
Ezeta (emw.Er*Br+emw.Ez*Bz+emw.Ephi*Bt)/B
Eeta (emw.Er*Br+emw.Ez*Bz)/Bp*cos(ph)-emw.Ephi*sin(ph)
Bzeta (emw.Br*Br+emw.Bz*Bz+emw.Bphi*Bt)/B
Beta (emw.Br*Br+emw.Bz*Bz)/Bp*cos(ph)-emw.Bphi*sin(ph)
e_xx -P*sin(ph)^2*sin(th)^2+P*sin(ph)^2+S*sin(ph)^2*sin(th)^2-S*sin(ph)^2+S epsilon_xx
e_xy (1j*D*sin(th)+P*cos(ph)*cos(th)-S*cos(ph)*cos(th))*sin(ph) epsilon_xy
e_xz -1j*D*cos(ph)+P*sin(ph)^2*sin(th)*cos(th)-S*sin(ph)^2*sin(th)*cos(th) epsilon_xz
e_yx -(1j*D*sin(th)-P*cos(ph)*cos(th)+S*cos(ph)*cos(th))*sin(ph) epsilon_yx
e_yy P*cos(ph)^2+S*sin(ph)^2 epsilon_yy
e_yz (1j*D*cos(th)+P*sin(th)*cos(ph)-S*sin(th)*cos(ph))*sin(ph) epsilon_yz
e_zx 1j*D*cos(ph)+P*sin(ph)^2*sin(th)*cos(th)-S*sin(ph)^2*sin(th)*cos(th) epsilon_zx
e_zy -(1j*D*cos(th)-P*sin(th)*cos(ph)+S*sin(th)*cos(ph))*sin(ph) epsilon_zy
e_zz P*sin(ph)^2*sin(th)^2-S*sin(ph)^2*sin(th)^2+S epsilon_zz
Eb_phi Eb*exp(i*pmode*theta(r,z))*cos(ph+e_angle) not used
Eb_z Eb*exp(i*pmode*theta(r,z))*sin(ph+e_angle)*sin(th) not used
Eb_r Eb*exp(i*pmode*theta(r,z))*sin(ph+e_angle)*cos(th) not used
Bt_toric -Bt
Bp_toric s_toric*Bp
Theta atan2(Bp_toric,Bt_toric) Capital theta in Toric
Eeta_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*s_toric*cos(Thetab(r,z))-sin(Thetab(r,z))*(-emw.Ephi)
Beta_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*s_toric*cos(Thetab(r,z))-sin(Thetab(r,z))*(-emw.Bphi)
Ezeta_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*s_toric*sin(Thetab(r,z))+cos(Thetab(r,z))*(-emw.Ephi)
Bzeta_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*s_toric*sin(Thetab(r,z))+cos(Thetab(r,z))*(-emw.Bphi)
Epsi (emw.Er*Br+emw.Ez*Bz)/Bp*sin(ph)+emw.Ephi*cos(ph)
tEr_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*cos(th_efit(r,z))
tEz_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*sin(th_efit(r,z))
tBr_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*cos(th_efit(r,z))
tBz_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*sin(th_efit(r,z))

###const
r0 0.6737237722806181
z0 -0.02697478581931928
pmode 60
Eb0_zeta 0
Eb0_eta 1
Eant 0
amode 10
f_rf 80000000
w f_rf*2*pi
fpol_efit -3.539542
q 1.602E-19[C] eV
mH 1.609E-27[kg] Hidrogen mass
qe -q electron charge
me 9.109E-31[kg] electron mass
Zi 1 ion nuclear charge
Ai 2 ion nuclear charge
mi mH*Ai ion mass
qi q*Zi ion charge
Zim 1 impurity nuclear charge
Aim 1 impurity nuclear mass
mim mH*Aim impurity mass
qim q*Zim impurity charge
Zeff 1 Z effective
c 2.998E8[m/s] speed of light
e0 8.854187817e-12[F/m] vacuum electric permittitivity
u0 4*pi*1e-7[H/m] vacuum magnetic permeability
fast_frac 0.05
ne_edge 4e19[1/m^3] LCFS density
ne_min 0.2e19[1/m^3]
sol_width 0.025
psi_center -0.0558478292
psi_edge 0.013956892219253551
sol_lambda_te 0.1
sol_lambda_ne 0.1
te_edge 30*q
te_min 10*q
s_toric sign(psi_edge-psi_center)
'''
