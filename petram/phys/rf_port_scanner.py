import numpy as np

from petram.solver.parametric_scanner import DefaultParametricScanner
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('PortScanner')
dprint0 = debug.regular_print('PortScanner', True)
format_memory_usage = debug.format_memory_usage

class PortScanner(DefaultParametricScanner):
    ''' 
    Scanner for port BC amplitude

    PortScan([1,2,3], amplitude=1, phase=0.0)

    # user need to load this in the global namelist

    '''
    def __init__(self, *args, **kwargs):
        amplitude = kwargs.pop("amplitude", 1)
        self.phase     = kwargs.pop("phase", 0.0)        

        if len(args) != 1:
            assert False, "port id must be specified"
            
        self.port = [int(x) for x in args[0]]
        data = []
        for i in range(len(self.port)):
            amp = np.array([0]*max(self.port))
            amp[self.port[i]-1] = amplitude
            data.append(amp)

        names = ["port.".join(str(i)) for i in range(len(self.port))]
        self._names = names
                                    
        DefaultParametricScanner.__init__(self, data=data)

    def apply_param(self, data):
        from petram.phys.em3d.em3d_port import EM3D_Port
        #from petram.phys.em2d.em2d_port import EM2D_Port
        from petram.phys.em2da.em2da_port import EM2Da_Port
        from petram.phys.em1d.em1d_port import EM1D_Port                
        
        names = self._names
                            
        dprint1("Port Scanner: Port Amplitude" + str(data))

        for phys in self.target_phys:
            for obj in phys.walk():
                if (isinstance(obj, EM3D_Port) or
                    isinstance(obj, EM2Da_Port) or
                    isinstance(obj, EM1D_Port)):

                    iport = int(obj.port_idx)
                    obj.inc_amp_txt = '0.0'
                    if iport < len(data)+1:
                        obj.inc_amp_txt = str(data[iport-1])
                        obj.inc_phase_txt = str(self.phase)
                    dprint1("Port Scanner: Setting port:", obj, "=", obj.inc_amp_txt)
    @property
    def names(self):
        '''
        suposed to return parameternames
        '''
        return self._names

                 
PortScan = PortScanner
