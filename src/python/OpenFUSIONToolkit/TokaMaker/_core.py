'''! Python interface for TokaMaker Grad-Shafranov functionality

@authors Chris Hansen
@date May 2023
@ingroup doxy_oft_python
'''

#
# Python interface for TokaMaker Grad-Shafranov functionality
import ctypes
# import json
import math
import numpy
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from ..util import *
from ._interface import *


def tokamaker_default_settings():
    '''! Initialize settings object with default values

    @result tokamaker_settings_struct object
    '''
    settings = tokamaker_settings_struct()
    settings.pm = True
    settings.free_boundary = True
    settings.has_plasma = True
    settings.limited_only = False
    settings.maxits = 40
    settings.mode = 1
    settings.urf = 0.2
    settings.nl_tol = 1.E-6
    settings.rmin = 0.0
    settings.lim_zmax = 1.E99
    settings.limiter_file = 'none'.encode()
    return settings


def create_prof_file(self, filename, profile_dict, name):
    '''! Create profile input file to be read by load_profiles()

    @param filename Name of input file, see options in set_profiles()
    @param profile_dict Dictionary object containing profile values ['y'] and sampled locations 
    in normalized Psi ['x']
    @param filename Name of input quantity, see options in set_profiles()
    '''
    file_lines = [profile_dict['type']]
    if profile_dict['type'] == 'flat':
        pass
    elif profile_dict['type'] == 'linterp':
        x = profile_dict.get('x',None)
        if x is None:
            raise KeyError('No array "x" for piecewise linear profile.')
        else:
            x = numpy.array(x.copy())
        y = profile_dict.get('y',None)
        if y is None:
            raise KeyError('No array "y" for piecewise linear profile.')
        else:
            y = numpy.array(y.copy())
        if numpy.min(numpy.diff(x)) < 0.0:
            raise ValueError("psi values in {0} profile must be monotonically increasing".format(name))
        if (x[0] < 0.0) or (x[-1] > 1.0):
            raise ValueError("Invalid psi values in {0} profile ({1}, {2})".format(name, x[0], x[-1]))
        if self.psi_convention == 0:
            x = 1.0 - x
            sort_inds = x.argsort()
            x = x[sort_inds]
            y = y[sort_inds]
        elif self.psi_convention == 1:
            pass
        else:
            raise ValueError('Unknown convention type, must be 0 (tokamak) or 1 (spheromak)')
        file_lines += [
            "{0} {1}".format(x.shape[0]-1, y[0]),
            "{0}".format(" ".join(["{0}".format(val) for val in x[1:]])),
            "{0}".format(" ".join(["{0}".format(val) for val in y[1:]]))
        ]
    else:
        raise KeyError('Invalid profile type ("flat", "linterp")')
    with open(filename, 'w+') as fid:
        fid.write("\n".join(file_lines))

oft_in_template = """&runtime_options
 debug={DEBUG_LEVEL}
/

&mesh_options
 meshname='none'
 {MESH_TYPE}
/

{MESH_DEF}
"""


class TokaMaker():
    '''! TokaMaker G-S solver class'''
    def __init__(self,debug_level=0,nthreads=2):
        '''! Initialize TokaMaker object

        @param debug_level Level of debug printing (0-3)
        '''
        ## Input file settings
        self._oft_in_dict = {'DEBUG_LEVEL': debug_level, 'MESH_TYPE': '', 'MESH_DEF': ''}
        self._update_oft_in()
        oft_init(c_int(nthreads))
        ## Internal Grad-Shafranov object (@ref psi_grad_shaf.gs_eq "gs_eq")
        self.gs_obj = c_void_p()
        tokamaker_alloc(ctypes.byref(self.gs_obj))
        ## General settings object
        self.settings = tokamaker_default_settings()
        ## Conductor definition dictionary
        self._cond_dict = {}
        ## Vacuum definition dictionary
        self._vac_dict = {}
        ## Coil definition dictionary
        self._coil_dict = {}
        ## Coil set definitions, including sub-coils
        self.coil_sets = {}
        ## Vacuum F value
        self._F0 = 0.0
        ## Plasma current target value (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._Ip_target=c_double(-1.0)
        ## Plasma current target ratio I_p(FF') / I_p(P') (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._Ip_ratio_target=c_double(-1.E99)
        ## Axis pressure target value (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._pax_target=c_double(-1.0)
        ## Stored energy target value (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._estore_target=c_double(-1.0)
        ## R0 target value (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._R0_target=c_double(-1.0)
        ## V0 target value (use @ref TokaMaker.TokaMaker.set_targets "set_targets")
        self._V0_target=c_double(-1.E99)
        ## F*F' normalization value [1] (use @ref TokaMaker.TokaMaker.alam "alam" property)
        self._alam = None
        ## Pressure normalization value [1] (use @ref TokaMaker.TokaMaker.pnorm "pnorm" property)
        self._pnorm = None
        ## Location of O-point (magnetic axis) [2]
        self.o_point = None
        ## Limiting point (limter or active X-point) [2]
        self.lim_point = None
        ## Location of X-points [20,2]
        self.x_points = None
        ## Diverted (limited) flag [1] (use @ref TokaMaker.TokaMaker.diverted "diverted" property)
        self._diverted = None
        ## Bounding values for \f$\psi\f$ (\f$\psi_a\f$,\f$\psi_0\f$) [2]
        self.psi_bounds = None
        ## Normalized flux convention (0 -> tokamak, 1 -> spheromak)
        self.psi_convention = 0
        ## Number of regions in mesh
        self.nregs = -1
        ## Number of points in mesh
        self.np = -1
        ## Number of cells in mesh
        self.nc = -1
        ## Mesh vertices [np,3] (last column should be all zeros)
        self.r = None
        ## Mesh triangles [nc,3] 
        self.lc = None
        ## Mesh regions [nc] 
        self.reg = None
        ## Number of vacuum regions in mesh
        self.nvac = 0
        ## Limiting contour
        self.lim_contour = None
    
    def _update_oft_in(self):
        '''! Update input file (`oftpyin`) with current settings'''
        with open('oftpyin','w+') as fid:
            fid.write(oft_in_template.format(**self._oft_in_dict))

    def reset(self):
        '''! Reset G-S object to enable loading a new mesh and coil configuration'''
        cstring = c_char_p(b""*200)
        tokamaker_reset(cstring)
        if cstring.value != b'':
            raise Exception(cstring.value)
        self.nregs = -1
        self.np = -1
        self._oft_in_dict['MESH_TYPE'] = ''
        self._oft_in_dict['MESH_DEF'] = ''
        # Reset defaults
        self.settings = tokamaker_default_settings()
        self._cond_dict = {}
        self._vac_dict = {}
        self._coil_dict = {}
        self._F0 = 0.0
        self._Ip_target=c_double(-1.0)
        self._Ip_ratio_target=c_double(-1.E99)
        self._pax_target=c_double(-1.0)
        self._estore_target=c_double(-1.0)
        self._R0_target=c_double(-1.0)
        self._V0_target=c_double(-1.E99)
        self._alam = None
        self._pnorm = None
        self.o_point = None
        self.lim_point = None
        self.x_points = None
        self._diverted = None
        self.psi_bounds = None
        self.nregs = -1
        self.np = -1
        self.nc = -1
        self.r = None
        self.lc = None
        self.reg = None
        self.nvac = 0
        self.lim_contour = None

    def setup_mesh(self,r=None,lc=None,reg=None,mesh_file=None):
        '''! Setup mesh for static and time-dependent G-S calculations

        A mesh should be specified by passing "r", "lc", and optionally "reg" or using a "mesh_file".
        When a region is specified the following ordering should apply:
          - 1: Plasma region
          - 2: Vacuum/air regions
          - 3+: Conducting regions and coils

        @param r Mesh point list [np,2]
        @param lc Mesh cell list [nc,3] (base one)
        @param reg Mesh region list [nc] (base one)
        @param mesh_file Filename containing mesh to load (native format only)
        '''
        if self.nregs != -1:
            raise ValueError('Mesh already setup, must call "reset" before loading new mesh')
        nregs = c_int()
        if mesh_file is not None:
            ndim = c_int(-1)
            rfake = numpy.ones((1,1),dtype=numpy.float64)
            lcfake = numpy.ones((1,1),dtype=numpy.int32)
            regfake = numpy.ones((1,),dtype=numpy.int32)
            self._oft_in_dict['MESH_TYPE'] = 'cad_type=0'
            self._oft_in_dict['MESH_DEF'] = """&native_mesh_options
 filename='{0}'
/""".format(mesh_file)
            self._update_oft_in()
            oft_setup_smesh(ndim,ndim,rfake,ndim,ndim,lcfake,regfake,ctypes.byref(nregs))
        elif r is not None:
            r = numpy.ascontiguousarray(r, dtype=numpy.float64)
            lc = numpy.ascontiguousarray(lc, dtype=numpy.int32)
            ndim = c_int(r.shape[1])
            np = c_int(r.shape[0])
            npc = c_int(lc.shape[1])
            nc = c_int(lc.shape[0])
            if reg is None:
                reg = numpy.ones((nc.value,),dtype=numpy.int32)
            else:
                reg = numpy.ascontiguousarray(reg, dtype=numpy.int32)
            oft_setup_smesh(ndim,np,r,npc,nc,lc+1,reg,ctypes.byref(nregs))
        else:
            raise ValueError('Mesh filename (native format) or mesh values required')
        self.nregs = nregs.value
    
    def setup_regions(self,cond_dict={},coil_dict={}):
        '''! Define mesh regions (coils and conductors)

        @param cond_dict Dictionary specifying conducting regions
        '''
        xpoint_mask = numpy.zeros((self.nregs,),dtype=numpy.int32)
        xpoint_mask[0] = 1
        eta_vals = -2.0*numpy.ones((self.nregs,),dtype=numpy.float64)
        eta_vals[0] = -1.0
        contig_flag = numpy.ones((self.nregs,),dtype=numpy.int32)
        # Process conductors and vacuum regions
        self._vac_dict = {}
        for key in cond_dict:
            if 'vac_id' in cond_dict[key]:
                self._vac_dict[key] = cond_dict[key]
            else:
                eta_vals[cond_dict[key]['reg_id']-1] = cond_dict[key]['eta']/mu0
                if cond_dict[key].get('noncontinuous',False):
                    contig_flag[cond_dict[key]['reg_id']-1] = 0
            xpoint_mask[cond_dict[key]['reg_id']-1] = int(cond_dict[key].get('allow_xpoints',False))
        # Remove vacuum regions
        for key in self._vac_dict:
            del cond_dict[key]
        self._cond_dict = cond_dict
        # Process coils
        nCoils = 0
        self.coil_sets = {}
        for key in coil_dict:
            xpoint_mask[coil_dict[key]['reg_id']-1] = int(coil_dict[key].get('allow_xpoints',False))
            eta_vals[coil_dict[key]['reg_id']-1] = -1.0
            coil_set = coil_dict[key].get('coil_set',key)
            if coil_set not in self.coil_sets:
                self.coil_sets[coil_set] = {
                    'id': nCoils,
                    'sub_coils': []
                }
                nCoils += 1
            self.coil_sets[coil_set]['sub_coils'].append(coil_dict[key])
        self._coil_dict = coil_dict
        # Mark vacuum regions
        self.nvac = 0
        for i in range(self.nregs):
            if eta_vals[i] < -1.5:
                eta_vals[i] = 1.E10
                self.nvac += 1 
        coil_nturns = numpy.zeros((nCoils, self.nregs))
        for key in self.coil_sets:
            for sub_coil in self.coil_sets[key]['sub_coils']:
                coil_nturns[self.coil_sets[key]['id'],sub_coil['reg_id']-1] = sub_coil.get('nturns',1.0)
        cstring = c_char_p('none'.encode())
        tokamaker_setup_regions(cstring,eta_vals,contig_flag,xpoint_mask,coil_nturns,nCoils)
    
    def setup(self,order=2,F0=0.0,full_domain=False):
        r'''! Setup G-S solver

        @param order Order of FE representation to use
        @param F0 Vacuum \f$F(\psi)\f$ value (B0*R0)
        '''
        if self.np != -1:
            raise ValueError('G-S instance already setup')
        self.update_settings()
        #
        ncoils = c_int()
        cstring = c_char_p(b""*200)
        # filename = c_char_p(input_filename.encode())
        tokamaker_setup(order,full_domain,ctypes.byref(ncoils),cstring)
        if cstring.value != b'':
            raise Exception(cstring.value)
        ## Number of coils in mesh
        self.ncoils = ncoils.value
        ## Isoflux constraint points (use @ref TokaMaker.TokaMaker.set_isoflux "set_isoflux")
        self._isoflux_targets = None
        ## Flux constraint points (use @ref TokaMaker.TokaMaker.set_isoflux "set_flux")
        self._flux_targets = None
        ## Saddle constraint points (use @ref TokaMaker.TokaMaker.set_saddles "set_saddles")
        self._saddle_targets = None
        # Get references to internal variables
        o_loc = c_double_ptr()
        lim_loc = c_double_ptr()
        x_loc = c_double_ptr()
        div_flag_loc = c_bool_ptr()
        bounds_loc = c_double_ptr()
        alam_loc = c_double_ptr()
        pnorm_loc = c_double_ptr()
        tokamaker_get_refs(ctypes.byref(o_loc),ctypes.byref(lim_loc),ctypes.byref(x_loc),ctypes.byref(div_flag_loc),
                    ctypes.byref(bounds_loc),ctypes.byref(alam_loc),ctypes.byref(pnorm_loc))
        self.o_point = numpy.ctypeslib.as_array(o_loc,shape=(2,))
        self.lim_point = numpy.ctypeslib.as_array(lim_loc,shape=(2,))
        self.x_points = numpy.ctypeslib.as_array(x_loc,shape=(20, 2))
        self._diverted = numpy.ctypeslib.as_array(div_flag_loc,shape=(1,))
        self.psi_bounds = numpy.ctypeslib.as_array(bounds_loc,shape=(2,))
        self._alam = numpy.ctypeslib.as_array(alam_loc,shape=(1,))
        self._pnorm = numpy.ctypeslib.as_array(pnorm_loc,shape=(1,))
        # Set default targets
        self.alam = 0.1
        self.pnorm = 0.1
        default_prof={
            'type': 'linterp',
            'x': numpy.array([0.0,0.8,1.0]),
            'y': numpy.array([2.0,1.0,0.0])
        }
        self.set_profiles(ffp_prof=default_prof, foffset=F0, pp_prof=default_prof)
        # Get limiter contour
        npts = c_int()
        r_loc = c_double_ptr()
        tokamaker_get_limiter(ctypes.byref(npts),ctypes.byref(r_loc))
        self.lim_contour = numpy.ctypeslib.as_array(r_loc,shape=(npts.value, 2))
        # Get plotting mesh
        np_loc = c_int()
        nc_loc = c_int()
        r_loc = c_double_ptr()
        lc_loc = c_int_ptr()
        reg_loc = c_int_ptr()
        tokamaker_get_mesh(ctypes.byref(np_loc),ctypes.byref(r_loc),ctypes.byref(nc_loc),ctypes.byref(lc_loc),ctypes.byref(reg_loc))
        ## Number of points in mesh
        self.np = np_loc.value
        ## Number of cells in mesh
        self.nc = nc_loc.value
        ## Mesh vertices [np,3] (last column should be all zeros)
        self.r = numpy.ctypeslib.as_array(r_loc,shape=(self.np, 3))
        ## Mesh triangles [nc,3] 
        self.lc = numpy.ctypeslib.as_array(lc_loc,shape=(self.nc, 3))
        ## Mesh regions [nc] 
        self.reg = numpy.ctypeslib.as_array(reg_loc,shape=(self.nc,))

    @property
    def alam(self):
        if self._alam is not None:
            return self._alam[0]
        else:
            return None
    
    @alam.setter
    def alam(self,value):
        if self._alam is not None:
            self._alam[0] = value
        else:
            raise ValueError('Class must be initialized to set "alam"')
    
    @property
    def pnorm(self):
        if self._pnorm is not None:
            return self._pnorm[0]
        else:
            return None
    
    @pnorm.setter
    def pnorm(self,value):
        if self._pnorm is not None:
            self._pnorm[0] = value
        else:
            raise ValueError('Class must be initialized to set "pnorm"')
    
    @property
    def diverted(self):
        if self._diverted is not None:
            return self._diverted[0]
        else:
            return None
    
    def set_coil_reg(self,reg_mat,reg_targets=None,reg_weights=None):
        '''! Set regularization matrix for coils when isoflux and/or saddle constraints are used

        Can be used to enforce "soft" constraints on coil currents. For hard constraints see
        @ref TokaMaker.TokaMaker.set_coil_bounds "set_coil_bounds".

        @param reg_mat Regularization matrix [nregularize,ncoils+1]
        @param reg_targets Regularization targets [nregularize] (default: 0)
        @param reg_weights Weights for regularization terms [nregularize] (default: 1)
        '''
        if reg_mat.shape[1] != self.ncoils+1:
            raise ValueError('Incorrect shape of "reg_mat", should be [nregularize,ncoils+1]')
        nregularize = reg_mat.shape[0]
        if reg_targets is None:
            reg_targets = numpy.zeros((nregularize,), dtype=numpy.float64)
        if reg_weights is None:
            reg_weights = numpy.ones((nregularize,), dtype=numpy.float64)
        if reg_targets.shape[0] != nregularize:
            raise ValueError('Incorrect shape of "reg_targets", should be [nregularize]')
        if reg_weights.shape[0] != nregularize:
            raise ValueError('Incorrect shape of "reg_weights", should be [nregularize]')
        reg_mat = numpy.ascontiguousarray(reg_mat.transpose(), dtype=numpy.float64)
        reg_targets = numpy.ascontiguousarray(reg_targets, dtype=numpy.float64)
        reg_weights = numpy.ascontiguousarray(reg_weights, dtype=numpy.float64)
        tokamaker_set_coil_regmat(nregularize,reg_mat, reg_targets, reg_weights)

    def set_coil_bounds(self,coil_bounds):
        '''! Set hard constraints on coil currents

        Can be used with or without regularization terms (see
        @ref TokaMaker.TokaMaker.set_coil_reg "set_coil_reg").

        @param coil_bounds Minimum and maximum allowable coil currents [ncoils+1,2]
        '''
        if (coil_bounds.shape[0] != self.ncoils+1) or (coil_bounds.shape[1] != 2):
            raise ValueError('Incorrect shape of "coil_bounds", should be [ncoils+1,2]')
        boucoil_boundsnds = numpy.ascontiguousarray(coil_bounds, dtype=numpy.float64)
        tokamaker_set_coil_bounds(coil_bounds)

    def set_coil_vsc(self,coil_gains):
        '''! Define a vertical stability coil set from one or more coils

        @param coil_gains Gains for each coil (absolute scale is arbitrary)
        '''
        if coil_gains.shape[0] != self.ncoils:
            raise ValueError('Incorrect shape of "coil_gains", should be [ncoils]')
        coil_gains = numpy.ascontiguousarray(coil_gains, dtype=numpy.float64)
        tokamaker_set_coil_vsc(coil_gains)

    def init_psi(self, r0=-1.0, z0=0.0, a=0.0, kappa=0.0, delta=0.0):
        r'''! Initialize \f$\psi\f$ using uniform current distributions

        If r0>0 then a uniform current density inside a surface bounded by
        a curve of the form defined in @ref oftpy.create_isoflux is used.
        If r0<0 then a uniform current density over the entire plasma region is used.

        @param r0 Major radial position for flux surface-based approach
        @param z0 Vertical position for flux surface-based approach
        @param a Minor radius for flux surface-based approach
        @param kappa Elongation for flux surface-based approach
        @param delta Triangularity for flux surface-based approach
        '''
        error_flag = c_int()
        tokamaker_init_psi(c_double(r0),c_double(z0),c_double(a),c_double(kappa),c_double(delta),ctypes.byref(error_flag))
        return error_flag.value

    def load_profiles(self, f_file='f_prof.in', foffset=None, p_file='p_prof.in', eta_file='eta_prof.in', f_NI_file='f_NI_prof.in'):
        r'''! Load flux function profiles (\f$F*F'\f$ and \f$P'\f$) from files

        @param f_file File containing \f$F*F'\f$ (or \f$F'\f$ if `mode=0`) definition
        @param foffset Value of \f$F0=R0*B0\f$
        @param p_file File containing \f$P'\f$ definition
        @param eta_file File containing $\eta$ definition
        @param f_NI_file File containing non-inductive \f$F*F'\f$ definition
        '''
        if foffset is not None:
            self._F0 = foffset
        tokamaker_load_profiles(c_char_p(f_file.encode()),c_double(self._F0),c_char_p(p_file.encode()),c_char_p(eta_file.encode()),c_char_p(f_NI_file.encode()))

    def set_profiles(self, ffp_prof=None, foffset=None, pp_prof=None, ffp_NI_prof=None):
        r'''! Set flux function profiles (\f$F*F'\f$ and \f$P'\f$) using a piecewise linear definition

        @param ffp_prof Dictionary object containing FF' profile ['y'] and sampled locations 
        in normalized Psi ['x']
        @param foffset Value of \f$F0=R0*B0\f$
        @param pp_prof Dictionary object containing P' profile ['y'] and sampled locations 
        in normalized Psi ['x']
        @param ffp_NI_prof Dictionary object containing non-inductive FF' profile ['y'] and sampled locations 
        in normalized Psi ['x']
        '''
        ffp_file = 'none'
        if ffp_prof is not None:
            ffp_file = 'tokamaker_f.prof'
            create_prof_file(self, ffp_file, ffp_prof, "F*F'")
        pp_file = 'none'
        if pp_prof is not None:
            pp_file = 'tokamaker_p.prof'
            create_prof_file(self, pp_file, pp_prof, "P'")
        eta_file = 'none'
        ffp_NI_file = 'none'
        if ffp_NI_prof is not None:
            ffp_NI_file = 'tokamaker_ffp_NI.prof'
            create_prof_file(self, ffp_NI_file, ffp_NI_prof, "ffp_NI")
        if foffset is not None:
            self._F0 = foffset
        self.load_profiles(ffp_file,foffset,pp_file,eta_file,ffp_NI_file)

    def set_resistivity(self, eta_prof=None):
        r'''! Set flux function profile $\eta$ using a piecewise linear definition

        Arrays should have the form array[i,:] = (\f$\hat{\psi}_i\f$, \f$f(\hat{\psi}_i)\f$) and span
        \f$\hat{\psi}_i = [0,1]\f$.

        @param eta_prof Values defining $\eta$ [:,2]
        '''
        ffp_file = 'none'
        pp_file = 'none'
        eta_file = 'none'
        if eta_prof is not None:
            eta_file = 'tokamaker_eta.prof'
            create_prof_file(self, eta_file, eta_prof, "eta'")
        ffp_NI_file = 'none'
        self.load_profiles(ffp_file,None,pp_file,eta_file,ffp_NI_file)

    def solve(self, vacuum=False):
        '''! Solve G-S equation with specified constraints, profiles, etc.'''
        if vacuum:
            raise ValueError('"vacuum=True" no longer supported, use "vac_solve()"')
        error_flag = c_int()
        tokamaker_solve(ctypes.byref(error_flag))
        return error_flag.value
    
    def vac_solve(self,psi=None):
        '''! Solve for vacuum solution (no plasma), with present coil currents
        
        @param psi Boundary values for vacuum solve
        '''
        if psi is None:
            psi = numpy.zeros((self.np,),dtype=numpy.float64)
        else:
            if psi.shape[0] != self.np:
                raise ValueError('Incorrect shape of "psi", should be [np]')
            psi = numpy.ascontiguousarray(psi, dtype=numpy.float64)
        error_flag = c_int()
        tokamaker_vac_solve(psi,ctypes.byref(error_flag))
        return psi, error_flag.value

    def get_stats(self,lcfs_pad=0.01,li_normalization='std'):
        r'''! Get information (Ip, q, kappa, etc.) about current G-S equilbirium

        See eq. 1 for `li_normalization='std'` and eq 2. for `li_normalization='iter'`
        in [Jackson et al.](http://dx.doi.org/10.1088/0029-5515/48/12/125002)

        @param lcfs_pad Padding at LCFS for boundary calculations
        @param li_normalization Form of normalized \f$ l_i \f$ ('std', 'ITER')
        @result Dictionary of equilibrium parameters
        '''
        _,qvals,_,dl,rbounds,zbounds = self.get_q(numpy.r_[1.0-lcfs_pad,0.95,0.02]) # Given backward so last point is LCFS (for dl)
        Ip,centroid,vol,pvol,dflux,tflux,Bp_vol = self.get_globals()
        _,_,_,p,_ = self.get_profiles(numpy.r_[0.001])
        if self.diverted:
            for i in range(self.x_points.shape[0]):
                if self.x_points[i,0] < 0.0:
                    break
                x_active = self.x_points[i,:]
            if x_active[1] < zbounds[0,1]:
                zbounds[0,:] = x_active
            elif x_active[1] > zbounds[1,1]:
                zbounds[1,:] = x_active
        # Compute normalized inductance
        if li_normalization.lower() == 'std':
            li = (Bp_vol/vol)/numpy.power(mu0*Ip/dl,2)
        elif li_normalization.lower() == 'iter':
            li = 2.0*Bp_vol/(numpy.power(mu0*Ip,2)*self.o_point[0])
        else:
            raise ValueError('Invalid "li_normalization"')
        #
        eq_stats = {
            'Ip': Ip,
            'Ip_centroid': centroid,
            'kappa': (zbounds[1,1]-zbounds[0,1])/(rbounds[1,0]-rbounds[0,0]),
            'kappaU': (zbounds[1,1]-self.o_point[1])*2.0/(rbounds[1,0]-rbounds[0,0]),
            'kappaL': (self.o_point[1]-zbounds[0,1])*2.0/(rbounds[1,0]-rbounds[0,0]),
            'delta': ((rbounds[1,0]+rbounds[0,0])/2.0-(zbounds[1,0]+zbounds[0,0])/2.0)*2.0/(rbounds[1,0]-rbounds[0,0]),
            'deltaU': ((rbounds[1,0]+rbounds[0,0])/2.0-zbounds[1,0])*2.0/(rbounds[1,0]-rbounds[0,0]),
            'deltaL': ((rbounds[1,0]+rbounds[0,0])/2.0-zbounds[0,0])*2.0/(rbounds[1,0]-rbounds[0,0]),
            'vol': vol,
            'q_0': qvals[2],
            'q_95': qvals[1],
            'P_ax': p[0],
            'W_MHD': pvol*1.5,
            'beta_pol': 100.0*(2.0*pvol*mu0/vol)/numpy.power(Ip*mu0/dl,2),
            'dflux': dflux,
            'tflux': tflux,
            'l_i': li
        }
        if self._F0 > 0.0:
            eq_stats['beta_tor'] = 100.0*(2.0*pvol*mu0/vol)/(numpy.power(self._F0/centroid[0],2))
        return eq_stats

    def print_info(self,lcfs_pad=0.01,li_normalization='std'):
        '''! Print information (Ip, q, etc.) about current G-S equilbirium
        
        @param lcfs_pad Padding at LCFS for boundary calculations
        @param li_normalization Form of normalized \f$ l_i \f$ ('std', 'ITER')
        '''
        eq_stats = self.get_stats(lcfs_pad=lcfs_pad,li_normalization=li_normalization)
        print("Equilibrium Statistics:")
        if self.diverted:
            print("  Topology                =   Diverted")
        else:
            print("  Topology                =   Limited")
        print("  Toroidal Current [A]    =   {0:11.4E}".format(eq_stats['Ip']))
        print("  Current Centroid [m]    =   {0:6.3F} {1:6.3F}".format(*eq_stats['Ip_centroid']))
        print("  Magnetic Axis [m]       =   {0:6.3F} {1:6.3F}".format(*self.o_point))
        print("  Elongation              =   {0:6.3F} (U: {1:6.3F}, L: {2:6.3F})".format(eq_stats['kappa'],eq_stats['kappaU'],eq_stats['kappaL']))
        print("  Triangularity           =   {0:6.3F} (U: {1:6.3F}, L: {2:6.3F})".format(eq_stats['delta'],eq_stats['deltaU'],eq_stats['deltaL']))
        print("  Plasma Volume [m^3]     =   {0:6.3F}".format(eq_stats['vol']))
        print("  q_0, q_95               =   {0:6.3F} {1:6.3F}".format(eq_stats['q_0'],eq_stats['q_95']))
        print("  Peak Pressure [Pa]      =   {0:11.4E}".format(eq_stats['P_ax']))
        print("  Stored Energy [J]       =   {0:11.4E}".format(eq_stats['W_MHD']))
        print("  <Beta_pol> [%]          =   {0:7.4F}".format(eq_stats['beta_pol']))
        if 'beta_tor' in eq_stats:
            print("  <Beta_tor> [%]          =   {0:7.4F}".format(eq_stats['beta_tor']))
        print("  Diamagnetic flux [Wb]   =   {0:11.4E}".format(eq_stats['dflux']))
        print("  Toroidal flux [Wb]      =   {0:11.4E}".format(eq_stats['tflux']))
        print("  l_i                     =   {0:7.4F}".format(eq_stats['l_i']))
    
    def set_isoflux(self,isoflux,weights=None,grad_wt_lim=-1.0):
        r'''! Set isoflux constraint points (all points lie on a flux surface)

        To constraint points more uniformly in space additional weighting based on
        the gradient of $\psi$ at each point can also be included by setting
        `grad_wt_lim>0`. When set the actual weight will be
        $w_i * min(grad_wt_lim,|\nabla \psi|_{max} / |\nabla \psi|_i)$

        @param isoflux List of points defining constraints [:,2]
        @param weights Weight to be applied to each constraint point [:] (default: 1)
        @param grad_wt_lim Limit on gradient-based weighting (negative to disable)
        '''
        if isoflux is None:
            tokamaker_set_isoflux(numpy.zeros((1,1)),numpy.zeros((1,)),0,grad_wt_lim)
            self._isoflux_targets = None
        else:
            if weights is None:
                weights = numpy.ones((isoflux.shape[0],), dtype=numpy.float64)
            if weights.shape[0] != isoflux.shape[0]:
                raise ValueError('Shape of "weights" does not match first dimension of "isoflux"')
            isoflux = numpy.ascontiguousarray(isoflux, dtype=numpy.float64)
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float64)
            tokamaker_set_isoflux(isoflux,weights,isoflux.shape[0],grad_wt_lim)
            self._isoflux_targets = isoflux.copy()
    
    def set_flux(self,locations,targets,weights=None): #,grad_wt_lim=-1.0):
        r'''! Set explicit flux constraint points \f$ \psi(x_i) \f$

        @param locations List of points defining constraints [:,2]
        @param targets Target \f$ \psi \f$ value at each point [:]
        @param weights Weight to be applied to each constraint point [:] (default: 1)
        '''
        if locations is None:
            tokamaker_set_flux(numpy.zeros((1,1)),numpy.zeros((1,)),numpy.zeros((1,)),0,-1.0)
            self._flux_targets = None
        else:
            if targets.shape[0] != locations.shape[0]:
                raise ValueError('Shape of "targets" does not match first dimension of "locations"')
            if weights is None:
                weights = numpy.ones((locations.shape[0],), dtype=numpy.float64)
            if weights.shape[0] != locations.shape[0]:
                raise ValueError('Shape of "weights" does not match first dimension of "locations"')
            locations = numpy.ascontiguousarray(locations, dtype=numpy.float64)
            targets = numpy.ascontiguousarray(targets, dtype=numpy.float64)
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float64)
            tokamaker_set_flux(locations,targets,weights,locations.shape[0],-1.0)
            self._flux_targets = (locations.copy(), targets.copy())
    
    def set_saddles(self,saddles,weights=None):
        '''! Set saddle constraint points (poloidal field should vanish at each point)

        @param saddles List of points defining constraints [:,2]
        @param weights Weight to be applied to each constraint point [:] (default: 1)
        '''
        if saddles is None:
            tokamaker_set_saddles(numpy.zeros((1,1)),numpy.zeros((1,)),0)
            self._saddle_targets = None
        else:
            if weights is None:
                weights = numpy.ones((saddles.shape[0],), dtype=numpy.float64)
            if weights.shape[0] != saddles.shape[0]:
                raise ValueError('Shape of "weights" does not match first dimension of "saddles"')
            saddles = numpy.ascontiguousarray(saddles, dtype=numpy.float64)
            weights = numpy.ascontiguousarray(weights, dtype=numpy.float64)
            tokamaker_set_saddles(saddles,weights,saddles.shape[0])
            self._saddle_targets = saddles.copy()
    
    def set_targets(self,Ip=None,Ip_ratio=None,pax=None,estore=None,R0=None,V0=None,retain_previous=False):
        r'''! Set global target values

        Once set, values are retained until they are explicitly set to their respective disabled
        values (see below). By default, all targets are disabled so this function should be called
        at least once to set "sane" values for `alam` and `pnorm`.

        @param alam Scale factor for \f$F*F'\f$ term (disabled if `Ip` is set)
        @param pnorm Scale factor for \f$P'\f$ term (disabled if `pax`, `estore`, or `R0` are set)
        @param Ip Target plasma current [A] (disabled if <0)
        @param Ip_ratio Amplitude of net plasma current contribution from FF' compared to P' (disabled if <-1.E98)
        @param pax Target axis pressure [Pa] (disabled if <0 or if `estore` is set)
        @param estore Target sotred energy [J] (disabled if <0)
        @param R0 Target major radius for magnetic axis (disabled if <0 or if `pax` or `estore` are set)
        @param V0 Target vertical position for magnetic axis (disabled if <-1.E98)
        @param retain_previous Keep previously set targets unless explicitly updated? (default: False)
        '''
        # Reset all targets unless specified
        if not retain_previous:
            self._Ip_target.value = -1.E99
            self._estore_target.value = -1.0
            self._pax_target.value = -1.0
            self._Ip_ratio_target.value = -1.E99
            self._R0_target.value = -1.0
            self._V0_target.value = -1.E99
        # Set new targets
        if Ip is not None:
            self._Ip_target.value=Ip
        if estore is not None:
            self._estore_target.value=estore
        if pax is not None:
            self._pax_target.value=pax
        if Ip_ratio is not None:
            self._Ip_ratio_target.value=Ip_ratio
        if R0 is not None:
            self._R0_target.value=R0
        if V0 is not None:
            self._V0_target.value=V0
        tokamaker_set_targets(self._Ip_target,self._Ip_ratio_target,self._pax_target,self._estore_target,self._R0_target,self._V0_target)

    def get_delstar_curr(self,psi):
        r'''! Get toroidal current density from \f$ \psi \f$ through \f$ \Delta^{*} \f$ operator
 
        @param psi \f$ \psi \f$ corresponding to desired current density
        @result \f$ J_{\phi} = \textrm{M}^{-1} \Delta^{*} \psi \f$
        '''
        curr = numpy.copy(psi)
        tokamaker_get_dels_curr(curr)
        return curr/mu0

    def get_psi(self,normalized=True):
        r'''! Get poloidal flux values on node points

        @param normalized Normalize (and offset) poloidal flux
        @result \f$\hat{\psi} = \frac{\psi-\psi_0}{\psi_a-\psi_0} \f$ or \f$\psi\f$
        '''
        psi = numpy.zeros((self.np,),dtype=numpy.float64)
        psi_lim = c_double()
        psi_max = c_double()
        tokamaker_get_psi(psi,ctypes.byref(psi_lim),ctypes.byref(psi_max))
        if normalized:
            psi = (psi-psi_lim.value)/(psi_max.value-psi_lim.value)
            if self.psi_convention == 0:
                psi = 1.0 - psi
        return psi

    def set_psi(self,psi):
        '''! Set poloidal flux values on node points

        @param psi Poloidal flux values (should not be normalized!)
        '''
        if psi.shape[0] != self.np:
            raise ValueError('Incorrect shape of "psi", should be [np]')
        psi = numpy.ascontiguousarray(psi, dtype=numpy.float64)
        tokamaker_set_psi(psi)
    
    def set_psi_dt(self,psi0,dt):
        '''! Set reference poloidal flux and time step for eddy currents in .solve()

        @param psi0 Reference poloidal flux at t-dt (unnormalized)
        @param dt Time since reference poloidal flux
        '''
        if psi0.shape[0] != self.np:
            raise ValueError('Incorrect shape of "psi0", should be [np]')
        psi0 = numpy.ascontiguousarray(psi0, dtype=numpy.float64)
        tokamaker_set_psi_dt(psi0,c_double(dt))
    
    def get_field_eval(self,field_type):
        r'''! Create field interpolator for vector potential

        @param field_type Field to interpolate, must be one of ("B", "psi", "F", or "P")
        @result Field interpolation object
        '''
        #
        mode_map = {'B': 1, 'PSI': 2, 'F': 3, 'P': 4}
        imode = mode_map.get(field_type.upper())
        if imode is None:
            raise ValueError('Invalid field type ("B", "psi", "F", "P")')
        #
        int_obj = c_void_p()
        cstring = c_char_p(b""*200)
        tokamaker_get_field_eval(imode,ctypes.byref(int_obj),cstring)
        if cstring.value != b'':
            raise Exception(cstring.value)
        field_dim = 1
        if imode == 1:
            field_dim = 3
        return TokaMaker_field_interpolator(int_obj,imode,field_dim)
    
    def get_coil_currents(self):
        '''! Get currents in each coil [A] and coil region [A-turns]

        @result Coil currents [ncoils], Coil currents by region [nregs]
        '''
        currents = numpy.zeros((self.ncoils,),dtype=numpy.float64)
        currents_reg = numpy.zeros((self.nregs,),dtype=numpy.float64)
        tokamaker_get_coil_currents(currents, currents_reg)
        return currents, currents_reg

    def get_coil_Lmat(self):
        r'''! Get mutual inductance matrix between coils

        @note This is the inductance in terms of A-turns. To get in terms of
        current in a single of the \f$n\f$ windings you must multiply by \f$n_i*n_j\f$.

        @result L[ncoils+1,ncoils+1]
        '''
        Lmat = numpy.zeros((self.ncoils+1,self.ncoils+1),dtype=numpy.float64)
        tokamaker_get_coil_Lmat(Lmat)
        return Lmat
    
    def trace_surf(self,psi):
        r'''! Trace surface for a given poloidal flux

        @param psi Flux surface to trace \f$\hat{\psi}\f$
        @result \f$r(\hat{\psi})\f$
        '''
        if self.psi_convention == 0:
            psi = 1.0-psi
        npoints = c_int()
        points_loc = c_double_ptr()
        tokamaker_trace_surf(c_double(psi),ctypes.byref(points_loc),ctypes.byref(npoints))
        if npoints.value > 0:
            return numpy.ctypeslib.as_array(points_loc,shape=(npoints.value, 2))
        else:
            return None
    
    def get_q(self,psi=None,psi_pad=0.02,npsi=50):
        r'''! Get q-profile at specified or uniformly spaced points

        @param psi Explicit sampling locations in \f$\hat{\psi}\f$
        @param psi_pad End padding (axis and edge) for uniform sampling (ignored if `psi` is not None)
        @param npsi Number of points for uniform sampling (ignored if `psi` is not None)
        @result \f$\hat{\psi}\f$, \f$q(\hat{\psi})\f$, \f$[<R>,<1/R>]\f$, length of last surface,
        [r(R_min),r(R_max)], [r(z_min),r(z_max)]
        '''
        if psi is None:
            psi = numpy.linspace(psi_pad,1.0-psi_pad,npsi,dtype=numpy.float64)
            if self.psi_convention == 0:
                psi = numpy.ascontiguousarray(numpy.flip(psi), dtype=numpy.float64)
                psi_save = 1.0-psi
        else:
            if self.psi_convention == 0:
                psi_save = numpy.copy(psi)
                psi = numpy.ascontiguousarray(1.0-psi, dtype=numpy.float64)
        qvals = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        ravgs = numpy.zeros((2,psi.shape[0]), dtype=numpy.float64)
        dl = c_double()
        rbounds = numpy.zeros((2,2),dtype=numpy.float64)
        zbounds = numpy.zeros((2,2),dtype=numpy.float64)
        tokamaker_get_q(psi.shape[0],psi,qvals,ravgs,ctypes.byref(dl),rbounds,zbounds)
        if self.psi_convention == 0:
            return psi_save,qvals,ravgs,dl.value,rbounds,zbounds
        else:
            return psi,qvals,ravgs,dl.value,rbounds,zbounds

    def sauter_fc(self,psi=None,psi_pad=0.02,npsi=50):
        r'''! Evaluate Sauter trapped particle fractions at specified or uniformly spaced points

        @param psi Explicit sampling locations in \f$\hat{\psi}\f$
        @param psi_pad End padding (axis and edge) for uniform sampling (ignored if `psi` is not None)
        @param npsi Number of points for uniform sampling (ignored if `psi` is not None)
        @result \f$ f_c \f$, [\f$<R>,<1/R>,<a>\f$], [\f$<|B|>,<|B|^2>\f$]
        '''
        if psi is None:
            psi = numpy.linspace(psi_pad,1.0-psi_pad,npsi,dtype=numpy.float64)
            if self.psi_convention == 0:
                psi = numpy.ascontiguousarray(numpy.flip(psi), dtype=numpy.float64)
                psi_save = 1.0 - psi
        else:
            if self.psi_convention == 0:
                psi_save = numpy.copy(psi)
                psi = numpy.ascontiguousarray(1.0-psi, dtype=numpy.float64)
        fc = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        r_avgs = numpy.zeros((3,psi.shape[0]), dtype=numpy.float64)
        modb_avgs = numpy.zeros((2,psi.shape[0]), dtype=numpy.float64)
        tokamaker_sauter_fc(psi.shape[0],psi,fc,r_avgs,modb_avgs)
        if self.psi_convention == 0:
            return psi_save,fc,r_avgs,modb_avgs
        else:
            return psi,fc,r_avgs,modb_avgs

    def get_globals(self):
        r'''! Get global plasma parameters

        @result Ip, [R_Ip, Z_Ip], \f$\int dV\f$, \f$\int P dV\f$, diamagnetic flux,
        enclosed toroidal flux
        '''
        Ip = c_double()
        centroid = numpy.zeros((2,),dtype=numpy.float64)
        vol = c_double()
        pvol = c_double()
        dflux = c_double()
        tflux = c_double()
        Bp_vol = c_double()
        tokamaker_get_globals(ctypes.byref(Ip),centroid,ctypes.byref(vol),ctypes.byref(pvol),
            ctypes.byref(dflux),ctypes.byref(tflux),ctypes.byref(Bp_vol))
        return Ip.value, centroid, vol.value, pvol.value, dflux.value, tflux.value, Bp_vol.value

    def calc_loopvoltage(self):
        r'''! Get plasma loop voltage

        @param eta Dictionary object containing resistivity profile ['y'] and sampled locations 
        in normalized Psi ['x']
        @param ffp_NI Dictionary object containing non-inductive FF' profile ['y'] and sampled locations 
        in normalized Psi ['x']
        @result Vloop [Volts]
        '''
        V_loop = c_double()

        tokamaker_gs_calc_vloop(ctypes.byref(V_loop))

        if V_loop.value < 0.:
            raise ValueError('eta array not specified')
        else:
            return V_loop.value

    def get_profiles(self,psi=None,psi_pad=1.E-8,npsi=50):
        r'''! Get G-S source profiles

        @param psi Explicit sampling locations in \f$\hat{\psi}\f$
        @param psi_pad End padding (axis and edge) for uniform sampling (ignored if `psi` is not None)
        @param npsi Number of points for uniform sampling (ignored if `psi` is not None)
        @result \f$\hat{\psi}\f$, \f$F(\hat{\psi})\f$, \f$F'(\hat{\psi})\f$,
        \f$P(\hat{\psi})\f$, \f$P'(\hat{\psi})\f$
        '''
        if psi is None:
            psi = numpy.linspace(psi_pad,1.0-psi_pad,npsi,dtype=numpy.float64)
            if self.psi_convention == 0:
                psi = numpy.ascontiguousarray(numpy.flip(psi), dtype=numpy.float64)
                psi_save = 1.0 - psi
        else:
            if self.psi_convention == 0:
                psi_save = numpy.copy(psi)
                psi = numpy.ascontiguousarray(1.0-psi, dtype=numpy.float64)
        f = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        fp = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        p = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        pp = numpy.zeros((psi.shape[0],), dtype=numpy.float64)
        tokamaker_get_profs(psi.shape[0],psi,f,fp,p,pp)
        if self.psi_convention == 0:
            return psi_save,f,fp,p/mu0,pp/mu0
        else:
            return psi,f,fp,p/mu0,pp/mu0
    
    def get_xpoints(self):
        '''! Get X-points

        @result X-points, is diverted?
        '''
        if self.x_points[0,0] < 0.0:
            return None, False
        else:
            for i in range(self.x_points.shape[0]):
                if self.x_points[i,0] < 0.0:
                    break
            return self.x_points[:i,:], self.diverted
    
    def set_coil_currents(self, currents):
        '''! Set coil currents

        @param currents Current in each coil [A]
        '''
        if currents.shape[0] != self.ncoils:
            raise ValueError('Incorrect shape of "currents", should be [ncoils]')
        currents = numpy.ascontiguousarray(currents, dtype=numpy.float64)
        tokamaker_set_coil_currents(currents)

    def update_settings(self):
        '''! Update settings after changes to values in python'''
        tokamaker_set_settings(ctypes.byref(self.settings))
    
    def plot_machine(self,fig,ax,vacuum_color='whitesmoke',cond_color='gray',limiter_color='k',
                     coil_color='gray',coil_colormap=None,coil_symmap=False,coil_scale=1.0,coil_clabel=r'$I_C$ [A]',colorbar=None):
        '''! Plot machine geometry

        @param fig Figure to add to
        @param ax Axis to add to
        @param vacuum_color Color to shade vacuum region (None to disable)
        @param cond_color Color for conducting regions (None to disable)
        @param limiter_color Color for limiter contour (None to disable)
        @param coil_color Color for coil regions (None to disable)
        @param coil_colormap Colormap for coil current values
        @param coil_symmap Make coil current colorscale symmetric
        @param coil_scale Scale for coil currents when plotting
        @param coil_clabel Label for coil current colorbar (None to disable colorbar)
        @param colorbar Colorbar instance to overwrite (None to add)
        @result Colorbar instance for coil colors or None
        '''
        mask_vals = numpy.ones((self.np,))
        # Shade vacuum region
        if vacuum_color is not None:
            mask = numpy.logical_and(self.reg > 1, self.reg <= self.nvac+1)
            if mask.sum() > 0.0:
                ax.tricontourf(self.r[:,0], self.r[:,1], self.lc[mask,:], mask_vals, colors=vacuum_color)
        # Shade coils
        if coil_colormap is not None:
            _, region_currents = self.get_coil_currents()
            mesh_currents = numpy.zeros((self.lc.shape[0],))
            for i in range(self.ncoils):
                mesh_currents = region_currents[self.reg-1]
            mask = (abs(mesh_currents) > 0.0)
            if mask.sum() > 0.0:
                mesh_currents *= coil_scale
                if coil_symmap:
                    max_curr = abs(mesh_currents).max()
                    clf = ax.tripcolor(self.r[:,0], self.r[:,1], self.lc[mask,:], mesh_currents[mask], cmap=coil_colormap, vmin=-max_curr, vmax=max_curr)
                else:
                    clf = ax.tripcolor(self.r[:,0], self.r[:,1], self.lc[mask,:], mesh_currents[mask], cmap=coil_colormap)
                if coil_clabel is not None:
                    cax = None
                    if colorbar is not None:
                        cax = colorbar.ax
                    colorbar = fig.colorbar(clf,ax=ax,cax=cax,label=coil_clabel)
        else:
            for _, coil_reg in self._coil_dict.items():
                mask_tmp = (self.reg == coil_reg['reg_id'])
                ax.tricontourf(self.r[:,0], self.r[:,1], self.lc[mask_tmp,:], mask_vals, colors=coil_color, alpha=1)
        # Shade conductors
        for _, cond_reg in self._cond_dict.items():
            mask_tmp = (self.reg == cond_reg['reg_id'])
            ax.tricontourf(self.r[:,0], self.r[:,1], self.lc[mask_tmp,:], mask_vals, colors=cond_color, alpha=1)
        # Show limiter
        if limiter_color and (self.lim_contour is not None):
            ax.plot(self.lim_contour[:,0],self.lim_contour[:,1],color=limiter_color)
        # Make 1:1 aspect ratio
        ax.set_aspect('equal','box')
        return colorbar

    def plot_constraints(self,fig,ax,isoflux_color='tab:red',isoflux_marker='+',saddle_color='tab:green',saddle_marker='x'):
        '''! Plot geometry constraints

        @param fig Figure to add to
        @param ax Axis to add to
        @param isoflux_color Color of isoflux points (None to disable)
        @param saddle_color Color of saddle points (None to disable)
        '''
        # Plot isoflux constraints
        if (isoflux_color is not None) and (self._isoflux_targets is not None):
            ax.plot(self._isoflux_targets[:,0],self._isoflux_targets[:,1],color=isoflux_color,marker=isoflux_marker,linestyle='none')
        # Plot saddle constraints
        if (saddle_color is not None) and (self._saddle_targets is not None):
            ax.plot(self._saddle_targets[:,0],self._saddle_targets[:,1],color=saddle_color,marker=saddle_marker,linestyle='none')

    def plot_psi(self,fig,ax,psi=None,normalized=True,
                 plasma_color=None,plasma_nlevels=8,plasma_levels=None,plasma_colormap=None,plasma_linestyles=None,
                 vacuum_color='darkgray',vacuum_nlevels=8,vacuum_levels=None,vacuum_colormap=None,vacuum_linestyles=None,
                 xpoint_color='k',xpoint_marker='x',opoint_color='k',opoint_marker='*'):
        r'''! Plot contours of \f$\hat{\psi}\f$

        @param fig Figure to add to
        @param ax Axis to add to
        @param psi Flux values to plot (otherwise `self.get_psi()` is called)
        @param normalized Retreive normalized flux, or assume normalized psi if passed as argument
        @param plasma_color Color for plasma contours
        @param plasma_nlevels Number of plasma contours
        @param plasma_levels Explicit levels for plasma contours
        @param plasma_colormap Colormap for plasma contours (cannot be specified with `plasma_color`)
        @param plasma_linestyles Linestyle for plasma contours
        @param vacuum_color Color for plasma contours
        @param vacuum_nlevels Number of plasma contours
        @param vacuum_levels Explicit levels for plasma contours (cannot be specified with `vacuum_color`)
        @param vacuum_colormap Colormap for plasma contours
        @param vacuum_linestyles Linestyle for vacuum contours
        @param xpoint_color Color for X-point markers (None to disable)
        @param xpoint_marker Colormap for plasma contours
        @param opoint_color Colormap for plasma contours (None to disable)
        @param opoint_marker Colormap for plasma contours
        '''
        # Plot poloidal flux
        if psi is None:
            psi = self.get_psi(normalized)
        if normalized and (self.psi_convention == 0):
            psi = 1.0-psi
        if plasma_levels is None:
            if normalized:
                plasma_levels = numpy.linspace(0.0,1.0,plasma_nlevels)
            else:
                plasma_levels = numpy.linspace(psi.min(),psi.max(),plasma_nlevels)
        else:
            if normalized:
                if self.psi_convention == 0:
                    plasma_levels = sorted(1.0-numpy.array(plasma_levels))
                else:
                    plasma_levels = sorted(numpy.array(plasma_levels))
        if vacuum_levels is None:
            if normalized:
                vacuum_levels1 = numpy.zeros((0,))
                vacuum_levels2 = numpy.zeros((0,))
                if psi.min() < -0.1:
                    vacuum_levels1 = numpy.linspace(psi.min(),0.0,vacuum_nlevels,endpoint=False)
                if psi.max() > 1.1:
                    vacuum_levels2 = numpy.linspace(1.0,psi.max(),vacuum_nlevels,endpoint=False)
                vacuum_levels = numpy.hstack((vacuum_levels1,vacuum_levels2))
        else:
            if normalized:
                if self.psi_convention == 0:
                    vacuum_levels = sorted(1.0-numpy.array(vacuum_levels))
                else:
                    vacuum_levels = sorted(numpy.array(vacuum_levels))
        if (plasma_color is None) and (plasma_colormap is None):
            plasma_colormap='viridis'
        if vacuum_levels is not None:
            ax.tricontour(self.r[:,0],self.r[:,1],self.lc,psi,levels=vacuum_levels,colors=vacuum_color,cmap=vacuum_colormap,linestyles=vacuum_linestyles)
        if plasma_levels is not None:
            ax.tricontour(self.r[:,0],self.r[:,1],self.lc,psi,levels=plasma_levels,colors=plasma_color,cmap=plasma_colormap,linestyles=plasma_linestyles)

        # Plot saddle points
        if xpoint_color is not None:
            x_points, _ = self.get_xpoints()
            if x_points is not None:
                ax.plot(x_points[:,0], x_points[:,1], color=xpoint_color, marker=xpoint_marker, linestyle='none')
        if (opoint_color is not None) and (self.o_point[0] > 0.0):
            ax.plot(self.o_point[0], self.o_point[1], color=opoint_color, marker=opoint_marker)
        # Make 1:1 aspect ratio
        ax.set_aspect('equal','box')
    
    def get_conductor_currents(self,psi,cell_centered=False):
        r'''! Get toroidal current density in conducting regions for a given \f$ \psi \f$

        @param psi Psi corresponding to field with conductor currents (eg. from time-dependent simulation)
        @param cell_centered Get currents at cell centers
        '''
        curr = self.get_delstar_curr(psi)
        if cell_centered:
            mesh_currents = numpy.zeros((self.lc.shape[0],))
        # Loop over conducting regions and get mask/fields
        mask = numpy.zeros((self.lc.shape[0],), dtype=numpy.int32)
        for _, cond_reg in self._cond_dict.items():
            eta = cond_reg.get('eta',-1.0)
            if eta > 0:
                mask_tmp = (self.reg == cond_reg['reg_id'])
                if cell_centered:
                    mesh_currents[mask_tmp] = numpy.sum(curr[self.lc[mask_tmp,:]],axis=1)/3.0
                mask = numpy.logical_or(mask,mask_tmp)
        if cell_centered:
            return mask, mesh_currents
        else:
            return mask, curr
    
    def get_conductor_source(self,dpsi_dt):
        r'''! Get toroidal current density in conducting regions for a \f$ d \psi / dt \f$ source

        @param dpsi_dt dPsi/dt source eddy currents (eg. from linear stability)
        '''
        # Apply 1/R scale (avoiding divide by zero)
        curr = dpsi_dt.copy()
        curr[self.r[:,0]>0.0] /= self.r[self.r[:,0]>0.0,0]
        # Compute cell areas
        have_noncontinuous = False
        for _, cond_reg in self._cond_dict.items():
            if 'noncontinuous' in cond_reg:
                have_noncontinuous = True
                break
        if have_noncontinuous:
            area = numpy.zeros((self.lc.shape[0],))
            for i in range(self.nc):
                v1 = self.r[self.lc[i,1],:]-self.r[self.lc[i,0],:]
                v2 = self.r[self.lc[i,2],:]-self.r[self.lc[i,0],:]
                area[i] = numpy.linalg.norm(numpy.cross(v1,v2))/2.0
        #
        mesh_currents = numpy.zeros((self.lc.shape[0],))
        # Loop over conducting regions and get mask/fields
        mask = numpy.zeros((self.lc.shape[0],), dtype=numpy.int32)
        for _, cond_reg in self._cond_dict.items():
            eta = cond_reg.get('eta',-1.0)
            if eta > 0:
                mask_tmp = (self.reg == cond_reg['reg_id'])
                field_tmp = dpsi_dt/eta
                mesh_currents[mask_tmp] = numpy.sum(field_tmp[self.lc[mask_tmp,:]],axis=1)/3.0
                if cond_reg.get('noncontinuous',False):
                    mesh_currents[mask_tmp] -= (mesh_currents[mask_tmp]*area[mask_tmp]).sum()/area[mask_tmp].sum()
                mask = numpy.logical_or(mask,mask_tmp)
        return mask, mesh_currents
    
    def plot_eddy(self,fig,ax,psi=None,dpsi_dt=None,nlevels=40,colormap='jet',clabel=r'$J_w$ [$A/m^2$]',symmap=False):
        r'''! Plot contours of \f$\hat{\psi}\f$

        @param fig Figure to add to
        @param ax Axis to add to
        @param psi Psi corresponding to eddy currents (eg. from time-dependent simulation)
        @param dpsi_dt dPsi/dt source eddy currents (eg. from linear stability)
        @param nlevels Number contour lines used for shading (with "psi" only)
        @param colormap Colormap to use for shadings
        @param clabel Label for colorbar (None to disable colorbar)
        @result Colorbar object
        '''
        if psi is not None:
            mask, plot_field = self.get_conductor_currents(psi,cell_centered=(nlevels < 0))
        elif dpsi_dt is not None:
            mask, plot_field = self.get_conductor_source(dpsi_dt)
        if plot_field.shape[0] == self.nc:
            if symmap:
                max_curr = abs(plot_field).max()
                clf = ax.tripcolor(self.r[:,0],self.r[:,1],self.lc[mask,:],plot_field[mask],cmap=colormap,vmin=-max_curr,vmax=max_curr)
            else:
                clf = ax.tripcolor(self.r[:,0],self.r[:,1],self.lc[mask],plot_field[mask],cmap=colormap)
        else:
            if symmap:
                max_curr = abs(plot_field[self.lc[mask,:]]).max(axis=None)
                clf = ax.tricontourf(self.r[:,0],self.r[:,1],self.lc[mask],plot_field,nlevels,cmap=colormap,vmin=-max_curr,vmax=max_curr)
            else:
                clf = ax.tricontourf(self.r[:,0],self.r[:,1],self.lc[mask],plot_field,nlevels,cmap=colormap)
        if clabel is not None:
            cb = fig.colorbar(clf,ax=ax)
            cb.set_label(clabel)
        else:
            cb = None
        # Make 1:1 aspect ratio
        ax.set_aspect('equal','box')
        return cb

    def get_vfixed(self):
        '''! Get required vacuum flux values to balance fixed boundary equilibrium

        @result sampling points [:,2], flux values [:]
        '''
        npts = c_int()
        pts_loc = c_double_ptr()
        flux_loc = c_double_ptr()
        tokamaker_get_vfixed(ctypes.byref(npts),ctypes.byref(pts_loc),ctypes.byref(flux_loc))
        return numpy.ctypeslib.as_array(pts_loc,shape=(npts.value, 2)), \
            numpy.ctypeslib.as_array(flux_loc,shape=(npts.value,))

    def save_eqdsk(self,filename,nr=65,nz=65,rbounds=None,zbounds=None,run_info='',lcfs_pad=0.01,meshsearch=0,maxsteps=0,ttol=1e-10):
        '''! Save current equilibrium to gEQDSK format

        @param filename Filename to save equilibrium to
        @param nr Number of radial sampling points
        @param nz Number of vertical sampling points
        @param rbounds Extents of grid in R
        @param zbounds Extents of grid in Z
        @param run_info Run information for EQDSK file (maximum of 36 characters)
        @param lcfs_pad Padding in normalized flux at LCFS
        @param meshsearch Set > 100 if more than ~100 cells between axis and LCFS
        @param maxsteps Field line tracer max number of steps above default (8e4)
        @param ttol Field line tracer tolerance near separatrix 
        '''
        if len(filename) > 80:
            raise ValueError('"filename cannot be longer than 80 characters')
        cfilename = c_char_p(filename.encode())
        if len(run_info) > 36:
            raise ValueError('"run_info" cannot be longer than 36 characters')
        crun_info = c_char_p(run_info.encode())
        if rbounds is None:
            rbounds = numpy.r_[self.lim_contour[:,0].min(), self.lim_contour[:,0].max()]
            dr = rbounds[1]-rbounds[0]
            rbounds += numpy.r_[-1.0,1.0]*dr*0.05
        if zbounds is None:
            zbounds = numpy.r_[self.lim_contour[:,1].min(), self.lim_contour[:,1].max()]
            dr = zbounds[1]-zbounds[0]
            zbounds += numpy.r_[-1.0,1.0]*dr*0.05
        cstring = c_char_p(b""*200)
        tokamaker_save_eqdsk(cfilename,c_int(nr),c_int(nz),rbounds,zbounds,crun_info,c_double(lcfs_pad),cstring,c_int(meshsearch),c_int(maxsteps),c_double(ttol))
        if cstring.value != b'':
            raise Exception(cstring.value)

    def save_GPEC_input(self,filename,npsi=256,psi=None,ntheta=512,lcfs_pad=0.01,meshsearch=0,maxsteps=0,ttol=1e-10,gpow=-2):
        '''! Save current equilibrium to GPEC format used in read_eq_hansen_inverse

        @param filename Filename to save DCON input file to
        @param npsi Number of radial sampling points
        @param psi Explicit samplng locations in normalised \f$\hat{\psi}\f$ (should go from 0.0 to 1.0)
        @param ptheta Number of poloidal angle sampling points
        @param lcfs_pad Padding in normalized flux at LCFS
        @param meshsearch Set > 100 if more than ~100 cells between axis and LCFS
        @param maxsteps Field line tracer max number of steps above default (8e4)
        @param ttol Field line tracer tolerance near separatrix 
        @param gpow Built in fortran options for psi grid sampling (see gs_save_decon function for details)
        '''
        if psi is not None:
            npsi=len(psi)-1
            if gpow != -2:
                print('Warning: Input psi-grid will override psi grid specified by gpow')
            gpow=0
        else:
            #initialising psi grid of zeros (gpow will take precedence)
            psi = numpy.linspace(0.0,0.0,npsi+1,dtype=numpy.float64)
            if gpow == 0: 
                gpow=-2 
        if len(filename) > 80:
            raise ValueError('"filename cannot be longer than 80 characters')
        cfilename = c_char_p(filename.encode())
        cstring = c_char_p(b""*200)
        tokamaker_save_decon(cfilename,c_int(npsi),psi,c_int(ntheta),c_double(lcfs_pad),cstring,c_int(meshsearch),c_int(maxsteps),c_double(ttol),c_int(gpow))
        if cstring.value != b'':
            raise Exception(cstring.value)

    def save_ifile(self,filename,npsi=256,psi=None,ntheta=512,lcfs_pad=0.01,meshsearch=0,maxsteps=0,ttol=1e-10,gpow=-2):
        '''! Save current equilibrium to L. Don Pearlstein's ifile format

        @param filename Filename to save ifile to
        @param npsi Number of radial sampling points
        @param psi Explicit samplng locations in normalised \f$\hat{\psi}\f$ (should go from 0.0 to 1.0)
        @param ptheta Number of poloidal angle sampling points
        @param lcfs_pad Padding in normalized flux at LCFS
        @param meshsearch Set > 100 if more than ~100 cells between axis and LCFS
        @param maxsteps Field line tracer max number of steps above default (8e4)
        @param ttol Field line tracer tolerance near separatrix 
        @param gpow Built in fortran options for psi grid sampling (see gs_save_ifile function for details)
        '''
        if psi is not None:
            npsi=len(psi)-1
            if gpow != -2:
                print('Warning: Input psi-grid will override psi grid specified by gpow')
            gpow=0
        else:
            #initialising psi grid of zeros (gpow will take precedence)
            psi = numpy.linspace(0.0,0.0,npsi+1,dtype=numpy.float64)
            if gpow == 0: 
                gpow=-2
        if len(filename) > 80:
            raise ValueError('"filename cannot be longer than 80 characters')
        cfilename = c_char_p(filename.encode())
        cstring = c_char_p(b""*200)
        tokamaker_save_ifile(cfilename,c_int(npsi),psi,c_int(ntheta),c_double(lcfs_pad),cstring,c_int(meshsearch),c_int(maxsteps),c_double(ttol),c_int(gpow))
        if cstring.value != b'':
            raise Exception(cstring.value)

    def eig_wall(self,neigs=4,pm=False):
        '''! Compute eigenvalues (1 / Tau_L/R) for conducting structures

        @param neigs Number of eigenvalues to compute
        @param pm Print solver statistics and raw eigenvalues?
        @result eigenvalues[neigs], eigenvectors[neigs,:]
        '''
        eig_vals = numpy.zeros((neigs,2),dtype=numpy.float64)
        eig_vecs = numpy.zeros((neigs,self.np),dtype=numpy.float64)
        tokamaker_eig_wall(c_int(neigs),eig_vals,eig_vecs,pm)
        return eig_vals, eig_vecs

    def eig_td(self,omega=-1.E4,neigs=4,include_bounds=True,pm=False):
        '''! Compute eigenvalues for the linearized time-dependent system

        @param omega Growth rate localization point (eigenvalues closest to this value will be found)
        @param neigs Number of eigenvalues to compute
        @param include_bounds Include bounding flux terms for constant normalized profiles?
        @param pm Print solver statistics and raw eigenvalues?
        @result eigenvalues[neigs], eigenvectors[neigs,:]
        '''
        eig_vals = numpy.zeros((neigs,2),dtype=numpy.float64)
        eig_vecs = numpy.zeros((neigs,self.np),dtype=numpy.float64)
        tokamaker_eig_td(c_double(omega),c_int(neigs),eig_vals,eig_vecs,c_bool(include_bounds),pm)
        return eig_vals, eig_vecs

    def setup_td(self,dt,lin_tol,nl_tol,pre_plasma=False):
        '''! Setup the time-dependent G-S solver

        @param dt Starting time step
        @param lin_tol Tolerance for linear solver
        @param nl_tol Tolerance for non-linear solver
        @param pre_plasma Use plasma contributions in preconditioner (default: False)
        '''
        tokamaker_setup_td(c_double(dt),c_double(lin_tol),c_double(nl_tol),c_bool(pre_plasma))
    
    def step_td(self,time,dt):
        '''! Compute eigenvalues for the time-dependent system

        @param time Growth rate enhancement point (should be approximately expected value)
        @param dt Number of eigenvalues to compute
        @result new time, new dt, # of NL iterations, # of linear iterations, # of retries
        '''
        dt = c_double(dt)
        time = c_double(time)
        nl_its = c_int()
        lin_its = c_int()
        nretry = c_int()
        tokamaker_step_td(ctypes.byref(time),ctypes.byref(dt),ctypes.byref(nl_its),ctypes.byref(lin_its),ctypes.byref(nretry))
        return time.value, dt.value, nl_its.value, lin_its.value, nretry.value

def solve_with_bootstrap(self,ne,Te,ni,Ti,inductive_jtor,Zeff,smooth_inputs=True,jBS_scale=1.0,Zis=[1.],max_iterations=6,rescale_ind_Jtor_by_ftr=True,initialize_eq=True,fig=None,ax=None,plot_iteration=False):
    '''! Self-consistently compute bootstrap contribution from H-mode profiles,
    and iterate solution until all functions of Psi converge. 

    @note if using nis and Zis, dnis_dpsi must be specified in sauter_bootstrap() 
    as a list of impurity gradients over Psi. See 
    https://omfit.io/_modules/omfit_classes/utils_fusion.html for more 
    detailed documentation 

    @note if initialize_eq=True, cubic polynomials will be fit to the core of all 
    kinetic profiles in order to flatten the pedestal. This will initialize the G-S 
    solution at an estimated L-mode pressure profile and using the L-mode bootstrap 
    contribution. Initializing the solver in L-mode before raising the pedestal 
    height increases the likelihood that the solver will converge in H-mode.

    @param ne Electron density profile, sampled over psi_norm
    @param Te Electron temperature profile [eV], sampled over psi_norm
    @param ni Ion density profile, sampled over psi_norm
    @param Ti Ion temperature profile [eV], sampled over psi_norm
    @param inductive_jtor Inductive toroidal current, sampled over psi_norm
    @param Zeff Effective Z profile, sampled over psi_norm
    @param scale_jBS Scalar which can scale bootstrap current profile
    @param nis List of impurity density profiles; NOT USED
    @param Zis List of impurity profile atomic numbers; NOT USED. 
    @param max_iterations Maximum number of H-mode mygs.solve() iterations
    @param initialize_eq Initialize equilibrium solve with flattened pedestal. 
    @param return_jBS Return bootstrap profile alongside err_flag
    '''
    try:
        from omfit_classes.utils_fusion import sauter_bootstrap
    except:
        raise ImportError('omfit_classes.utils_fusion not installed')

    kBoltz = eC
    pressure = (kBoltz * ne * Te) + (kBoltz * ni * Ti) # 1.602e-19 * [m^-3] * [eV] = [Pa]

    ### Set new pax target
    self.set_targets(pax=pressure[0],retain_previous=True)

    ### Reconstruct psi_norm and n_psi from input inductive_jtor
    psi_norm = numpy.linspace(0.,1.,len(inductive_jtor))
    n_psi = len(inductive_jtor)

    def profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,init_Lmode=False,rescale_ind_Jtor_by_ftr=True,smooth_inputs=False,include_jBS=True,plot_iteration=False,ax=None,fig=None,iteration=-1,max_iterations=0):

        pprime=get_pprime(psi_norm,pressure,(1.0/(self.psi_bounds[1]-self.psi_bounds[0])))
        #pprime = numpy.gradient(pressure) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))

        ### Get final remaining quantities for Sauter from TokaMaker
        psi,f,fp,_,_ = self.get_profiles(npsi=n_psi)
        _,fc,r_avgs,_ = self.sauter_fc(npsi=n_psi)
        ft = 1 - fc # Trapped particle fraction on each flux surface
        eps = r_avgs[2] / r_avgs[0] # Inverse aspect ratio
        _,qvals,ravgs,_,_,_ = self.get_q(npsi=n_psi)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]

        #option to rescale input inductive current profile by trapped fraction
        # use if input current profile was taken from spitzer resistivity (see Sauter 1999 fig 2)
        if rescale_ind_Jtor_by_ftr: 
            if plot_iteration:
                ax[0].plot(psi_norm,inductive_jtor,label=r"$J_{ind, Spitzer}$",color='tab:orange',alpha=(iteration+2)/(iteration+2))
            ind_Ip_target=cross_section_surf_int(inductive_jtor,r_avgs[2]) #getting total inductive current
            inductive_jtor=inductive_jtor*fc    #reshaping inductive current profile by passing fraction
            inductive_jtor,rscaling_fac=rescale_Jtor(inductive_jtor,ind_Ip_target,r_avgs[2])
        
        if include_jBS:
            ### Calculate flux derivatives for Sauter
            dn_e_dpsi = numpy.gradient(ne) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_e_dpsi = numpy.gradient(Te) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dn_i_dpsi = numpy.gradient(ni) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_i_dpsi = numpy.gradient(Ti) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))

            ### Solve for bootstrap current profile. See https://omfit.io/_modules/omfit_classes/utils_fusion.html for more detailed documentation 
            flux_surf_avg_of_B_timesj_BS = sauter_bootstrap(
                                    psi_N=psi_norm,
                                    Te=Te,
                                    Ti=Ti,
                                    ne=ne,
                                    p=pressure,
                                    nis=[ni,],
                                    Zis=Zis,
                                    Zeff=Zeff,
                                    gEQDSKs=[None],
                                    R0=0., # not used
                                    device=None,
                                    psi_N_efit=None,
                                    psiraw=psi*(self.psi_bounds[1]-self.psi_bounds[0]) + self.psi_bounds[0],
                                    R=R_avg,
                                    eps=eps, 
                                    q=qvals,
                                    fT=ft,
                                    I_psi=f,
                                    nt=1,
                                    version='neo_2021',
                                    debug_plots=False,
                                    return_units=True,
                                    return_package=False,
                                    charge_number_to_use_in_ion_collisionality='Koh',
                                    charge_number_to_use_in_ion_lnLambda='Zavg',
                                    dT_e_dpsi=dT_e_dpsi,
                                    dT_i_dpsi=dT_i_dpsi,
                                    dn_e_dpsi=dn_e_dpsi,
                                    dnis_dpsi=[dn_i_dpsi,],
                                    )[0]
            inductive_jtor[-1] = 0. ### FORCING inductive_jtor TO BE ZERO AT THE EDGE
            j_BS = flux_surf_avg_of_B_timesj_BS*(R_avg / f) ### Convert into [A/m^2] by dividing by <Bt>... should be using <B> instead!!
            j_BS *= jBS_scale ### Scale j_BS by user specified scalar
            j_BS[-1] = 0. ### FORCING j_BS TO BE ZERO AT THE EDGE

            #   We rescale inductive_jtor to match self._Ip_target, since we can't rescale bootstrap current (its given by pressure profile etc.)
            #   If you instead want to respect the inductive_jtor profile, don't set a self._Ip_target.
            if self._Ip_target.value != -1.E99 and not init_Lmode: #    We don't rescale inductive_jtor unless we are getting the proper, full bootstrap current from the pedestal... no init_Lmode here
                total_BS_current=cross_section_surf_int(j_BS,r_avgs[2])
                inductive_jtor,rescling_fac=rescale_Jtor(inductive_jtor,(self._Ip_target.value-total_BS_current),r_avgs[2],verbose=True)
            jtor_total = inductive_jtor + j_BS

            if plot_iteration:
                if iteration==max_iterations:
                    ax[0].plot(psi_norm,inductive_jtor,label=r"$J_{ind, neoclassical}$",color='b',alpha=(iteration+2)/(max_iterations+2))
                    ax[0].plot(psi_norm,j_BS,label=r"$J_{bootstrap}$",color='r',alpha=(iteration+2)/(max_iterations+2))
                    ax[0].plot(psi_norm,jtor_total,label=r"$J_{total}$",color='k',linestyle='dotted',alpha=(iteration+2)/(max_iterations+2))
                else:
                    ax[0].plot(psi_norm,j_BS,color='r',alpha=(iteration+2)/(max_iterations+2))
                    ax[0].plot(psi_norm,jtor_total,color='k',linestyle='dotted',alpha=(iteration+2)/(max_iterations+2))
                    ax[0].plot(psi_norm,inductive_jtor,color='b',alpha=(iteration+2)/(max_iterations+2))
        else:
            j_BS = None
            flux_surf_avg_of_B_timesj_BS = None
            inductive_jtor[-1] = 0. ### FORCING inductive_jtor TO BE ZERO AT THE EDGE
            jtor_total = inductive_jtor
            if plot_iteration:
                ax[0].plot(psi_norm,inductive_jtor,label=f'J {iteration} (L-mode init)')
        
        ffprime = ffprime_from_jtor_pprime(jtor_total, pprime, R_avg, one_over_R_avg)
        if plot_iteration: 
            if iteration==max_iterations:
                ax[1].plot(psi_norm,ffprime,label=r"$FF'$ input",color='r',alpha=(iteration+2)/(max_iterations+2))
                ax[1].plot(psi_norm,f*fp,label=r"$FF'$ output",color='k',linestyle='dotted',alpha=(iteration+2)/(max_iterations+2))
            else:
                ax[1].plot(psi_norm,ffprime,color='r',alpha=(iteration+2)/(max_iterations+2))
                ax[1].plot(psi_norm,f*fp,color='k',linestyle='dotted',alpha=(iteration+2)/(max_iterations+2))
            ax[0].legend(loc='best')
            ax[0].set_ylabel(r"$J_{tor}$")
            ax[1].legend(loc='best')
            ax[1].set_ylabel(r"$FF'$")

        if smooth_inputs:
            newpsi,ffprime=make_smooth(psi_norm,ffprime,npts=450)
            newpsi,pprime=make_smooth(psi_norm,pprime,npts=450)
        else:
            newpsi=psi_norm

        ffp_prof = {
            'type': 'linterp',
            'x': newpsi,
            'y': ffprime / ffprime[0]
        }

        pp_prof = {
            'type': 'linterp',
            'x': newpsi,
            'y': pprime / pprime[0]
        }

        if plot_iteration:
            return pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, fig, ax
        else:
            return pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS

    if initialize_eq:
        x_trimmed = psi_norm.tolist().copy()
        ne_trimmed = ne.tolist().copy()
        Te_trimmed = Te.tolist().copy()
        ni_trimmed = ni.tolist().copy()
        Ti_trimmed = Ti.tolist().copy()

        ### Remove profile values from psi_norm ~0.5 to ~0.99, leaving single value at the edge
        mid_index = int(len(x_trimmed)/2)
        end_index = len(x_trimmed)-1
        del x_trimmed[mid_index:end_index]
        del ne_trimmed[mid_index:end_index]
        del Te_trimmed[mid_index:end_index]
        del ni_trimmed[mid_index:end_index]
        del Ti_trimmed[mid_index:end_index]

        ### Fit cubic polynomials through all core and one edge value
        ne_model = numpy.poly1d(numpy.polyfit(x_trimmed, ne_trimmed, 3))
        Te_model = numpy.poly1d(numpy.polyfit(x_trimmed, Te_trimmed, 3))
        ni_model = numpy.poly1d(numpy.polyfit(x_trimmed, ni_trimmed, 3))
        Ti_model = numpy.poly1d(numpy.polyfit(x_trimmed, Ti_trimmed, 3))

        init_ne = ne_model(psi_norm)
        init_Te = Te_model(psi_norm)
        init_ni = ni_model(psi_norm)
        init_Ti = Ti_model(psi_norm)

        init_pressure = (kBoltz * init_ne * init_Te) + (kBoltz * init_ni * init_Ti)

        ### Initialize equilibirum on L-mode-like P' and inductive j_tor profiles
        print('>>> Initializing equilibrium with pedestal removed:')

        if plot_iteration:
            init_pp_prof, init_ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, fig, ax = profile_iteration(self,init_pressure,init_ne,init_ni,init_Te,init_Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,smooth_inputs=smooth_inputs,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,include_jBS=False,init_Lmode=True,iteration=-1,plot_iteration=plot_iteration,fig=fig,ax=ax,max_iterations=max_iterations)
        else:
            init_pp_prof, init_ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS = profile_iteration(self,init_pressure,init_ne,init_ni,init_Te,init_Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,smooth_inputs=smooth_inputs,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,include_jBS=False,init_Lmode=True,)

        init_pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
        init_ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge

        init_pp_prof['y'] = numpy.nan_to_num(init_pp_prof['y'])
        init_ffp_prof['y'] = numpy.nan_to_num(init_ffp_prof['y'])

        self.set_profiles(ffp_prof=init_ffp_prof,pp_prof=init_pp_prof)

        flag = self.solve()

    if initialize_eq: #Don't rescale by trapped fraction than once... you'll keep peaking the current indefinitely
        rescale_ind_Jtor_by_ftr=False

    if initialize_eq and flag<0:
        print('Warning: H-mode equilibrium solve errored at initialisation')
        print(error_reason(flag))
        return self, flag, None, None, None, None, None, None, None
    ### Specify original H-mode profiles, iterate on bootstrap contribution until reasonably converged
    n = 0
    flag = -1
    print('>>> Iterating on H-mode equilibrium solution:')
    while n <= max_iterations:
        print('> Iteration '+str(n)+':')

        if not plot_iteration: 
            pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS = profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,smooth_inputs=smooth_inputs)
        else: 
            pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, fig, ax = profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,smooth_inputs=smooth_inputs,iteration=n,plot_iteration=plot_iteration,fig=fig,ax=ax,max_iterations=max_iterations)

        pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
        ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge
    
        pp_prof['y'] = numpy.nan_to_num(pp_prof['y']) # Check for any nan's
        ffp_prof['y'] = numpy.nan_to_num(ffp_prof['y']) # Check for any nan's

        self.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)

        flag = self.solve()

        rescale_ind_Jtor_by_ftr=False #Don't rescale by trapped fraction more than once... you'll keep peaking the current indefinitely
        n += 1
        if flag<0:
            print('Warning: H-mode equilibrium solve did not converge, solve failed:')
            print(error_reason(flag))
            return self, flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, inductive_jtor, pp_prof, ffp_prof, pressure[0]
        if (n > max_iterations) and (flag >= 0):
            break
        elif n > max_iterations+1:
            raise TypeError('H-mode equilibrium solve did not converge')
        if plot_iteration:
            self.print_info()
    return self, flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, inductive_jtor, pp_prof, ffp_prof, pressure[0]


def basic_dynamo_w_bootstrap(self,ne,Te,ni,Ti,inductive_jtor,Zeff,smooth_inputs=True,jBS_scale=1.0,Zis=[1.],max_iterations=6,rescale_ind_Jtor_by_ftr=False,initialize_eq=True,fig=None,ax=None,plot_iteration=False):
    '''! 
    Takes equilibrium, iterates current profile shape to achieve on axis above q>1.
    Does this by subtracting current density from inside the q=1 rational surface and depositing it 
    both immediately outside the rational surface (like a heuristic dynamo effect, see 
    https://indico.cern.ch/event/934747/contributions/4514274/attachments/2303331/3962875/Krebs_EFTC_slides.pdf)
    and also in the bulk plasma. Exits once q0>1 is achieved. Function will keep an imposed total current goal 
    if one was previously specified in the targets. Also the neoclassical bootstrap current is included.
    Warning: this is not a simulation of sawtooth dynamics.
    
    @note if using nis and Zis, dnis_dpsi must be specified in sauter_bootstrap() 
    as a list of impurity gradients over Psi. See https://omfit.io/_modules/omfit_classes/utils_fusion.html for more detailed documentation 

    @param ne Electron density profile, sampled over psi_norm
    @param Te Electron temperature profile [eV], sampled over psi_norm
    @param ni Ion density profile, sampled over psi_norm
    @param Ti Ion temperature profile [eV], sampled over psi_norm
    @param inductive_jtor Inductive toroidal current, sampled over psi_norm
    @param Zeff Effective Z profile, sampled over psi_norm
    @param scale_jBS Scalar which can scale bootstrap current profile
    @param nis List of impurity density profiles; NOT USED
    @param Zis List of impurity profile atomic numbers; NOT USED. 
    @param max_iterations Maximum number of H-mode mygs.solve() iterations
    @param initialize_eq Initialize equilibrium solve with flattened pedestal. 
    @param return_jBS Return bootstrap profile alongside err_flag
    '''
    try:
        from omfit_classes.utils_fusion import sauter_bootstrap
    except:
        raise ImportError('omfit_classes.utils_fusion not installed')
    
    #DYNAMO
    q0_vals = []
    q1_psi_surf=[]
    def gaussian(x, mu, sig):
        return (
            1.0 / (numpy.sqrt(2.0 * numpy.pi) * sig) * numpy.exp(-numpy.power((x - mu) / sig, 2.0) / 2)
    )#DYNAMO

    kBoltz = eC
    pressure = (kBoltz * ne * Te) + (kBoltz * ni * Ti) # 1.602e-19 * [m^-3] * [eV] = [Pa]

    ### Set new pax target
    self.set_targets(pax=pressure[0],retain_previous=True)

    ### Reconstruct psi_norm and n_psi from input inductive_jtor
    psi_norm = numpy.linspace(0.,1.,len(inductive_jtor))
    n_psi = len(inductive_jtor)

    def profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,q_1_psi_location=0.2,init_Lmode=False,rescale_ind_Jtor_by_ftr=False,smooth_inputs=False,include_jBS=True,plot_iteration=False,ax=None,fig=None,iteration=-1,max_iterations=0):
        flag_end=False
        pprime=get_pprime(psi_norm,pressure,(1.0/(self.psi_bounds[1]-self.psi_bounds[0])))
        #pprime = numpy.gradient(pressure) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))

        ### Get final remaining quantities for Sauter from TokaMaker
        psi,f,fp,_,_ = self.get_profiles(npsi=n_psi)
        _,fc,r_avgs,_ = self.sauter_fc(npsi=n_psi)
        ft = 1 - fc # Trapped particle fraction on each flux surface
        eps = r_avgs[2] / r_avgs[0] # Inverse aspect ratio
        _,qvals,ravgs,_,_,_ = self.get_q(npsi=n_psi)
        R_avg = ravgs[0]
        one_over_R_avg = ravgs[1]

        #option to rescale input inductive current profile by trapped fraction
        # use if input current profile was taken from spitzer resistivity (see Sauter 1999 fig 2)
        if rescale_ind_Jtor_by_ftr: 
            if plot_iteration:
                ax[0].plot(psi_norm,inductive_jtor,label=r"$J_{ind, Spitzer}$",color='b',alpha=0.5)
            ind_Ip_target=cross_section_surf_int(inductive_jtor,r_avgs[2]) #getting total inductive current
            inductive_jtor=inductive_jtor*fc    #reshaping inductive current profile by passing fraction
            inductive_jtor,rscale_fac=rescale_Jtor(inductive_jtor,ind_Ip_target,r_avgs[2])
        
        if include_jBS:
            ### Calculate flux derivatives for Sauter
            dn_e_dpsi = numpy.gradient(ne) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_e_dpsi = numpy.gradient(Te) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dn_i_dpsi = numpy.gradient(ni) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))
            dT_i_dpsi = numpy.gradient(Ti) / (numpy.gradient(psi_norm) * (self.psi_bounds[1]-self.psi_bounds[0]))

            ### Solve for bootstrap current profile. See https://omfit.io/_modules/omfit_classes/utils_fusion.html for more detailed documentation 
            flux_surf_avg_of_B_timesj_BS = sauter_bootstrap(
                                    psi_N=psi_norm,
                                    Te=Te,
                                    Ti=Ti,
                                    ne=ne,
                                    p=pressure,
                                    nis=[ni,],
                                    Zis=Zis,
                                    Zeff=Zeff,
                                    gEQDSKs=[None],
                                    R0=0., # not used
                                    device=None,
                                    psi_N_efit=None,
                                    psiraw=psi*(self.psi_bounds[1]-self.psi_bounds[0]) + self.psi_bounds[0],
                                    R=R_avg,
                                    eps=eps, 
                                    q=qvals,
                                    fT=ft,
                                    I_psi=f,
                                    nt=1,
                                    version='neo_2021',
                                    debug_plots=False,
                                    return_units=True,
                                    return_package=False,
                                    charge_number_to_use_in_ion_collisionality='Koh',
                                    charge_number_to_use_in_ion_lnLambda='Zavg',
                                    dT_e_dpsi=dT_e_dpsi,
                                    dT_i_dpsi=dT_i_dpsi,
                                    dn_e_dpsi=dn_e_dpsi,
                                    dnis_dpsi=[dn_i_dpsi,],
                                    )[0]
            inductive_jtor[-1] = 0. ### FORCING inductive_jtor TO BE ZERO AT THE EDGE
            j_BS = flux_surf_avg_of_B_timesj_BS*(R_avg / f) ### Convert into [A/m^2] by dividing by <Bt>... should be using <B> instead!!
            j_BS *= jBS_scale ### Scale j_BS by user specified scalar
            j_BS[-1] = 0. ### FORCING j_BS TO BE ZERO AT THE EDGE

            #   We rescale inductive_jtor to match self._Ip_target, since we can't rescale bootstrap current (its given by pressure profile etc.)
            #   If you instead want to respect the inductive_jtor profile, don't set a self._Ip_target.
            if self._Ip_target.value != -1.E99 and not init_Lmode: #    We don't rescale inductive_jtor unless we are getting the proper, full bootstrap current from the pedestal... no init_Lmode here
                total_BS_current=cross_section_surf_int(j_BS,r_avgs[2])
                inductive_jtor,rescling_fac=rescale_Jtor(inductive_jtor,(self._Ip_target.value-total_BS_current),r_avgs[2],verbose=plot_iteration)
                total_ind_current=self._Ip_target.value-total_BS_current #DYNAMO

            #DYNAMO
            q_offset=qvals[0]-1.02
            if q_offset<0.0 or q_offset>0.05: #0.05 #1# broke here 
                qoff_min=0.03  #0.03 #1# broke here
                if q_offset>0:
                    q_offset=q_offset*0.8
                #SETUP:
                jtor_total_temp = inductive_jtor + j_BS
                rho = numpy.sqrt(psi)

                q_1_psi_location_record=q_1_psi_location
                found,q_1_psi_location,i_near = solve_for_psi_crossing(psi,qvals,1.02)

                if found:
                    print("q0 = ",qvals[0],":::: q = 1 psi location = ",q_1_psi_location,"::::::::::::::::::::::::::::::::::::")
                elif not found and q_offset<0:
                    raise "Error, q<1 surface was unable to be located"
                elif not found and q_offset>0:
                    q_1_psi_location=q_1_psi_location_record
                    print("q0 = ",qvals[0]," > 1.1. Reducing q0 with last psi location = ",q_1_psi_location,"::::::::::::::::::::::::::::::::::::")

                outside_rho_step=0.08+0.4*min(q_1_psi_location,0.2)

                total_norm_current=cross_section_surf_int(jtor_total_temp,rho)
                avg_norm_current_density_at_that_rho=total_norm_current/(numpy.pi*(min(numpy.sqrt(q_1_psi_location)+0.15,0.8))**2)
                #print("Comparison of two values:",jtor_total_temp[i_near],avg_norm_current_density_at_that_rho) #How to make sure we adding a reasonable amount:
                
                #ARB SECTION:
                add_factorV=0.5*numpy.array([0.2,0.6,1.0])
                #add_factorV=0.5*np.array(add_factorV)
                    #[0.2,0.8,1.0] worked ish
                    #[0.3,0.7,1.0] worked ish too much add_factor (sensitive to 0.2)
                    #[0.2,0.6,1.0] working well (just reducing the peak)
                width=1.0*min(q_1_psi_location,0.2)+0.1
                width2=0.3
                swidth=min(q_1_psi_location,0.15)+0.1
                add_factor=add_factorV[0]*(-q_offset+qoff_min)*avg_norm_current_density_at_that_rho  
                add_factor2=add_factorV[1]*((5+iteration)/30)*(-q_offset+qoff_min)*avg_norm_current_density_at_that_rho 
                subtract_factor=add_factorV[2]*(q_1_psi_location)*(-q_offset+qoff_min)*avg_norm_current_density_at_that_rho
                    #0.15
                    

                for i in range(len(inductive_jtor)):
                    inductive_jtor[i]-=subtract_factor*gaussian(rho[i],0.0,swidth)
                    inductive_jtor[i]+=add_factor*gaussian(rho[i],numpy.sqrt(q_1_psi_location)+outside_rho_step,width)
                    inductive_jtor[i]+=add_factor2*gaussian(rho[i],0.5,width2) 
                    if found:
                        if i<i_near and inductive_jtor[i]<inductive_jtor[i_near]:
                            inductive_jtor[i]=inductive_jtor[i_near]

                #Approximately monotonic decreasing inductive current near the core...
                damping_fac=0.5
                j_spline = InterpolatedUnivariateSpline(psi, inductive_jtor, k=4)
                j_spline_d = j_spline.derivative()
                cr_pts = j_spline_d.roots()
                inboard_max_pts=[]
                if len(cr_pts)>0:
                    inboard_max_pts = [x for x in cr_pts if (j_spline_d(x,1)<0 and  x<0.2)] #no local maxima near the core... 
                if len(inboard_max_pts)>0:
                    for i in range(len(inductive_jtor)):
                        if psi[i]<inboard_max_pts[0]:
                            inductive_jtor[i]=inductive_jtor[i]-damping_fac*(inductive_jtor[i]-j_spline(inboard_max_pts[0]))

                #Rescale inductive current profile
                if self._Ip_target.value != -1.E99 and not init_Lmode: #    We don't rescale inductive_jtor unless we are getting the proper, full bootstrap current from the pedestal... no init_Lmode here
                    total_BS_current=cross_section_surf_int(j_BS,r_avgs[2])
                    inductive_jtor,rescling_fac=rescale_Jtor(inductive_jtor,(self._Ip_target.value-total_BS_current),r_avgs[2],verbose=plot_iteration)
                    total_ind_current=self._Ip_target.value-total_BS_current
                flag_end=False
            else:
                flag_end=True
                q_1_psi_location=None
            #DYNAMO
                
            jtor_total = inductive_jtor + j_BS

            if plot_iteration:
                if iteration==0:
                    ax[0].plot(psi_norm,inductive_jtor,label=r"$J_{inductive}$",color='b',alpha=0.5)
                    ax[0].plot(psi_norm,j_BS,label=r"$J_{bootstrap}$",color='r',alpha=0.5)
                    ax[0].plot(psi_norm,jtor_total,label=r"$J_{total}$",color='k',linestyle='dotted',alpha=0.5)
                else:
                    ax[0].plot(psi_norm,j_BS,color='r',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
                    ax[0].plot(psi_norm,jtor_total,color='k',linestyle='dotted',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
                    ax[0].plot(psi_norm,inductive_jtor,color='b',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
        else:
            j_BS = None
            flux_surf_avg_of_B_timesj_BS = None
            inductive_jtor[-1] = 0. ### FORCING inductive_jtor TO BE ZERO AT THE EDGE
            jtor_total = inductive_jtor
            if plot_iteration:
                ax[0].plot(psi_norm,inductive_jtor,label=f'J {iteration} (L-mode init)')
        
        ffprime = ffprime_from_jtor_pprime(jtor_total, pprime, R_avg, one_over_R_avg)
        if plot_iteration: 
            if iteration==0:
                ax[1].plot(psi_norm,ffprime,label=r"$FF'$ input",color='r',alpha=0.5)
                ax[1].plot(psi_norm,f*fp,label=r"$FF'$ output",color='k',linestyle='dotted',alpha=0.5)
            else:
                ax[1].plot(psi_norm,ffprime,color='r',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
                ax[1].plot(psi_norm,f*fp,color='k',linestyle='dotted',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
            ax[0].legend(loc='best')
            ax[0].set_ylabel(r"$J_{tor}$")
            ax[1].legend(loc='best')
            ax[1].set_ylabel(r"$FF'$")
            ax[2].set_ylim(bottom=0.0,top=5)
            ax[2].plot(psi,qvals,color='k',alpha=(0.5+0.5*(iteration+2)/(max_iterations+2)))
            ax[2].set_ylabel(r"$q$")
            ax[2].legend(loc='best')

        if smooth_inputs:
            newpsi,ffprime=make_smooth(psi_norm,ffprime,npts=450)
            newpsi,pprime=make_smooth(psi_norm,pprime,npts=450)
        else:
            newpsi=psi_norm

        ffp_prof = {
            'type': 'linterp',
            'x': newpsi,
            'y': ffprime / ffprime[0]
        }

        pp_prof = {
            'type': 'linterp',
            'x': newpsi,
            'y': pprime / pprime[0]
        }

        #DYNAMO
        if plot_iteration:
            return pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end, q_1_psi_location, qvals[0], fig, ax
        else:
            return pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end, q_1_psi_location, qvals[0]

    if initialize_eq:
        x_trimmed = psi_norm.tolist().copy()
        ne_trimmed = ne.tolist().copy()
        Te_trimmed = Te.tolist().copy()
        ni_trimmed = ni.tolist().copy()
        Ti_trimmed = Ti.tolist().copy()

        ### Remove profile values from psi_norm ~0.5 to ~0.99, leaving single value at the edge
        mid_index = int(len(x_trimmed)/2)
        end_index = len(x_trimmed)-1
        del x_trimmed[mid_index:end_index]
        del ne_trimmed[mid_index:end_index]
        del Te_trimmed[mid_index:end_index]
        del ni_trimmed[mid_index:end_index]
        del Ti_trimmed[mid_index:end_index]

        ### Fit cubic polynomials through all core and one edge value
        ne_model = numpy.poly1d(numpy.polyfit(x_trimmed, ne_trimmed, 3))
        Te_model = numpy.poly1d(numpy.polyfit(x_trimmed, Te_trimmed, 3))
        ni_model = numpy.poly1d(numpy.polyfit(x_trimmed, ni_trimmed, 3))
        Ti_model = numpy.poly1d(numpy.polyfit(x_trimmed, Ti_trimmed, 3))

        init_ne = ne_model(psi_norm)
        init_Te = Te_model(psi_norm)
        init_ni = ni_model(psi_norm)
        init_Ti = Ti_model(psi_norm)

        init_pressure = (kBoltz * init_ne * init_Te) + (kBoltz * init_ni * init_Ti)

        ### Initialize equilibirum on L-mode-like P' and inductive j_tor profiles
        print('>>> Initializing equilibrium with pedestal removed:')

        if plot_iteration: #DYNAMO
            init_pp_prof, init_ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end_deprecated, q_1_psi_location, q0, fig, ax = profile_iteration(self,init_pressure,init_ne,init_ni,init_Te,init_Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,smooth_inputs=smooth_inputs,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,include_jBS=False,init_Lmode=True,iteration=-1,plot_iteration=plot_iteration,fig=fig,ax=ax,max_iterations=max_iterations)
        else:
            init_pp_prof, init_ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end_deprecated, q_1_psi_location, q0, = profile_iteration(self,init_pressure,init_ne,init_ni,init_Te,init_Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,smooth_inputs=smooth_inputs,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,include_jBS=False,init_Lmode=True,)

        q0_vals.append(q0) #DYNAMO 
        q1_psi_surf.append(q_1_psi_location)

        init_pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
        init_ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge

        init_pp_prof['y'] = numpy.nan_to_num(init_pp_prof['y'])
        init_ffp_prof['y'] = numpy.nan_to_num(init_ffp_prof['y'])

        self.set_profiles(ffp_prof=init_ffp_prof,pp_prof=init_pp_prof)

        flag = self.solve()

    if initialize_eq: #Don't rescale by trapped fraction than once... you'll keep peaking the current indefinitely
        rescale_ind_Jtor_by_ftr=False
    else:
        q_1_psi_location=0.2 #this will be instantly overwritten

    if initialize_eq and flag<0:
        print('Warning: H-mode equilibrium solve errored at initialisation')
        print(error_reason(flag))
        return self, flag, None, None, None, None, None, None, None, None, None, None, False
    ### Specify original H-mode profiles, iterate on bootstrap contribution until reasonably converged
    n = 0
    flag = -1
    print('>>> Iterating on H-mode equilibrium solution:')
    while n <= max_iterations:
        print('> Iteration '+str(n)+':')

        if not plot_iteration: 
            pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end_deprecated, q_1_psi_location, q0, = profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,q_1_psi_location=q_1_psi_location,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,smooth_inputs=smooth_inputs,iteration=n)
        else: 
            pp_prof, ffp_prof, j_BS, inductive_jtor, jtor_total, flux_surf_avg_of_B_timesj_BS, flag_end_deprecated, q_1_psi_location, q0, fig, ax = profile_iteration(self,pressure,ne,ni,Te,Ti,psi_norm,n_psi,Zeff,inductive_jtor,jBS_scale,Zis,q_1_psi_location=q_1_psi_location,rescale_ind_Jtor_by_ftr=rescale_ind_Jtor_by_ftr,smooth_inputs=smooth_inputs,iteration=n,plot_iteration=plot_iteration,fig=fig,ax=ax,max_iterations=max_iterations)

        q0_vals.append(q0)  #DYNAMO
        q1_psi_surf.append(q_1_psi_location)

        pp_prof['y'][-1] = 0. # Enforce 0.0 at edge
        ffp_prof['y'][-1] = 0. # Enforce 0.0 at edge
    
        pp_prof['y'] = numpy.nan_to_num(pp_prof['y']) # Check for any nan's
        ffp_prof['y'] = numpy.nan_to_num(ffp_prof['y']) # Check for any nan's

        self.set_profiles(ffp_prof=ffp_prof,pp_prof=pp_prof)

        flag = self.solve()

        _,qvals,ravgs,_,_,_ = self.get_q(npsi=len(j_BS)) #DYNAMO
        if qvals[0]>1.02 and qvals[0]<1.07: #flag_end_deprecated checks q0 of the previous cycle, so we use this instead
            flag_end=True
        else:
            flag_end=False

        rescale_ind_Jtor_by_ftr=False #Don't rescale by trapped fraction than once... you'll keep peaking the current indefinitely
        n += 1
        if flag<0:
            print('Warning: H-mode equilibrium solve did not converge, solve failed:')
            print(error_reason(flag))
            return self, flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, inductive_jtor, pp_prof, ffp_prof, q0_vals, q1_psi_surf, qvals, pressure[0], flag_end
        if (flag_end) and (flag >= 0): #DYNAMO
            print('q0 reduced below 1')
            break
        elif (n > max_iterations) and (flag >= 0):
            print('q0 failed to reduce below 1')
            break
        elif n > max_iterations+1:
            raise TypeError('H-mode equilibrium solve did not converge')
        if plot_iteration:
            self.print_info()  
        #DYNAMO
    return self, flag, j_BS, jtor_total, flux_surf_avg_of_B_timesj_BS, inductive_jtor, pp_prof, ffp_prof, q0_vals, q1_psi_surf, qvals, pressure[0], flag_end

def solve_for_psi_crossing(psi,qvals,qmin):
    qcross=qvals-qmin
    qmin_arg=numpy.argmin(qcross)
    qcross_arg=numpy.argmin(abs(qcross))

    if qcross[qmin_arg]>0:
        print("No need to change anything, q already above qmin")
        return False,None,None
    elif qcross[0] > qmin:
        print("q profile is hollow, aborting.")
        return False,None,None
    elif qcross[qcross_arg]==0: #checking special case is necessary, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.roots.html
        psicross=psi[qcross_arg]
    else:
        qcross_spln=CubicSpline(psi,qcross)
        roots=qcross_spln.roots()
        roots= [x for x in roots if x > 0.0 and x < 1.0] 
        if len(roots)>0:
            if len(roots)>3:
                print("q crosses qmin multiple times... (a) aborting.")
                return False,None,None
            if len(roots)==1: #Usual case
                psicross=roots[0]
            elif len(roots)==2:  
                assert qcross[-1]>0 
                if (roots[1]-roots[0])>0.3 and min(qvals)<(qmin-0.1):
                    print(roots)
                    print("q crosses qmin multiple times... (b) aborting.")
                    return False,None,None
                else: #small dip beneath qmin, choose crossing on rhs for reshaping...
                    print("q crosses qmin twice in close proximity.")
                    psicross=max(roots)
            elif len(roots)==3: 
                if (roots[2]-roots[0])>0.3:
                    print("q crosses qmin multiple times... (c) aborting.")
                    return False,None,None
                else:
                    print("q crosses qmin 3 times in close proximity.")
                    psicross=max(roots)
        else:
            print("cannot find case of q crossing qmin, aborting")
            return False,None,None
        
    return True, psicross, qcross_arg

def ffprime_from_jtor_pprime(jtor, pprime, R_avg, one_over_R_avg):
    r'''! Convert from J_toroidal to FF' using Grad-Shafranov equation
    @param jtor Toroidal current profile
    @param R_avg Flux averaged R, calculated by TokaMaker
    @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
    @param pprime dP/dPsi profile
    '''
    ffprime = (jtor -  R_avg * (-pprime)) * (mu0 / one_over_R_avg)
                #^the factor of 2.0 is wrong if you look at Freidberg Ideal MHD GS equation
                #However it's needed to match this function to the version of f*fp that Tokamaker outputs from get_profiles,
                #which is equal to twice the FFp from Ideal MHD (Freidberg)
    return ffprime

def jtor_from_ffprime_pprime(ffprime, pprime, R_avg, one_over_R_avg):
    r'''! Convert from FF' to J_toroidal using Grad-Shafranov equation
    @param ffprime F*(dF/dpsi), for flux function F in Grad-Shafranov equation
    @param R_avg Flux averaged R, calculated by TokaMaker
    @param one_over_R_avg Flux averaged 1/R, calculated by TokaMaker
    @param pprime dP/dPsi profile
    '''
    jtor = R_avg * (-pprime) + one_over_R_avg*ffprime/mu0
    return jtor

def cross_section_surf_int(profile,a_avgs):
    profile_inductive_integration_spline=CubicSpline(a_avgs,a_avgs*profile)
    cross_section_surf_int=2*numpy.pi*profile_inductive_integration_spline.integrate(min(a_avgs), max(a_avgs), extrapolate=False)
    return cross_section_surf_int

def rescale_Jtor(jtor_total,Ip_target,a_avgs,verbose=True):
    #eq_stats = self.get_stats() self,psi_norm,R_avgs,
    Jtorr_inductive_integration_spline=CubicSpline(a_avgs,a_avgs*jtor_total)
    cross_section_surf_int=2*numpy.pi*Jtorr_inductive_integration_spline.integrate(min(a_avgs), max(a_avgs), extrapolate=False)
    rescaling_factor=Ip_target/cross_section_surf_int
    if verbose:
        print(f'current rescaling factor = {rescaling_factor}')
    return jtor_total*rescaling_factor, rescaling_factor

def get_pprime(psi_norm,pressure,one_on_tot_pol_flux):
    pressure_spln=InterpolatedUnivariateSpline(psi_norm,pressure,k=3)
    pressure_gradient_points=numpy.zeros(len(psi_norm))
    for i in range(len(psi_norm)):
        pressure_gradient_points[i]=pressure_spln(psi_norm[i],1)
    return pressure_gradient_points * one_on_tot_pol_flux #units: Pa/Wb

# Function basically makes a loose spline of inputs, then re-samples at high res... gets rid of quirky discontinuities in Sauter bootstrap function, and 
# in your pressure profile input (if it has them). Also helps make sure forcing edge value to zero doesn't create a discontinuity at the edge.
def make_smooth(psi_norm,input_vec,ped_spot=(0.95**2),preped_grid=20,pedsepl=0.1,pedsepr=0.005,endsep=0.01,npts=200,endzero=True):
    input_spln=CubicSpline(psi_norm,input_vec)
    origsize=len(psi_norm)
    ped_grid=50
    preped_range=ped_spot-pedsepl
    if not endzero:
        endsep=0.0
    ped_range=(1.0-endsep)-(ped_spot+pedsepr)
    psigrid=[]
    valgrid=[]
    for i in range(preped_grid):
        psi_n=(i/(preped_grid-1))*preped_range
        psigrid.append(psi_n)
        samplept=input_spln(psi_n)
        valgrid.append(samplept)
    for i in range(10):
        psi_n=preped_range+((i+1)/(10))*0.9*pedsepl
        psigrid.append(psi_n)
        samplept=input_spln(psi_n)
        valgrid.append(samplept)
    for i in range(ped_grid):
        psi_n=ped_spot+pedsepr+(i/(ped_grid-1))*ped_range
        psigrid.append(psi_n)
        samplept=input_spln(psi_n)
        valgrid.append(samplept)
    if endzero:
        psigrid.append(1.0)
        valgrid.append(0.0)
    smoothed_spline=CubicSpline(numpy.array(psigrid),numpy.array(valgrid))
    new_psi=numpy.linspace(0.,1.,max(npts,origsize))
    output_vec=numpy.zeros(len(new_psi))
    for i in range(len(new_psi)):
        output_vec[i]=smoothed_spline(new_psi[i])
    return new_psi,output_vec

# Local interpretation of internal fortran error messages
def error_reason(error_flag):
    if error_flag==-1:
        return'Exceeded "maxits"'
    elif error_flag==-2:
        return'Total poloidal flux is zero'
    elif error_flag==-3:
        return'Closed flux volume lost'
    elif error_flag==-4:
        return'Axis dropped below "rmin"'
    elif error_flag==-5:
        return'Toroidal current droppped too low'
    elif error_flag==-6:
        return'Matrix solve failed for targets'
    elif error_flag==-7:
        return'Isoflux fitting failed'
    elif error_flag==-8:
        return'Wall eigenmode flux loop fitting failed'

    return'Unkown reason'