# --------------------------------------------------------------------------
# DM.py
# This class generates a line of sight integrated DM template.
# Author: Eric Carlson (erccarls@ucsc.edu) 11/26/2014
# --------------------------------------------------------------------------
from numpy import *
import numpy as np
import multiprocessing as mp
from functools import partial
import time
import pyfits
# from scipy import ndimage
# from scipy.interpolate import RegularGridInterpolator
import Tools
import healpy
import tempfile
import imp

def __LOS_DM(tnum, n_thread, l_max, b_max, res, tmp1, tmp2, r_min, r_max, z_step=0.02, ):
    """
    (INTERNAL) LOS Integration Kernel for two passed distributions
    """
    # ==============================================
    # Integration Parameters
    # ==============================================
    # import tmp1
    # import tmp2

    # time.sleep(.1)
    # reload(tmp)
    # reload(tmp2)

    tmp1 = imp.load_source('tmp1', tmp1)
    tmp2 = imp.load_source('tmp2', tmp2)

    func1 = tmp1.func
    func2 = tmp2.func

    R_solar = 8.5  # Sun at 8.5 kpc
    kpc2cm = 3.08568e21
    z_start, z_stop, z_step = max(R_solar - r_max, 0), R_solar + r_max, z_step  # los start,stop,step-size in kpc

    # distances along the LOS
    zz = np.linspace(start=z_start, stop=z_stop, num=int(np.ceil((z_stop - z_start) / z_step)))
    deg2rad = np.pi / 180.
    # List of lat/long to loop over.
    bb = np.linspace(-b_max + res / 2., b_max + res / 2., 1 + int(np.ceil(2 * b_max / res))) * deg2rad
    ll = np.linspace(-l_max + res / 2., l_max + res / 2., 1 + np.ceil(int(2 * l_max / res))) * deg2rad

    # divide according to thread
    stride = len(bb) / n_thread
    if tnum == n_thread - 1:
        bb = bb[tnum * stride:]
    else:
        bb = bb[tnum * stride:(tnum + 1) * stride]

    # Master projected skymap
    proj_skymap = np.zeros(shape=(len(bb), len(ll)))

    # Loop latitudes
    for bi in range(len(bb)):
        # loop longitude
        for lj in range(len(ll)):
            los_sum = 0.
            l, b = ll[lj], bb[bi]
            # z in cylindrical coords
            z = zz * sin(b)
            x, y = -zz * cos(l) * cos(b) + R_solar, +zz * sin(l) * cos(b)
            # r_2d = sqrt(x**2+y**2)

            los_sum += sum(func1(x, y, z) * func2(x, y, z))

            proj_skymap[bi, lj] = los_sum * z_step * kpc2cm
    return proj_skymap


def LOS_DM(l_max, b_max, res, z_step=0.02, func1='func = lambda x,y,z: 1.', func2='func = lambda x,y,z: 1.'):
    """
    (INTERNAL) Manages multithreaded integrator.
    :param l_max:
    :param b_max:
    :param res:
    :param z_step:
    :param func1:
    :param func2:
    :return:
    """
    """
    :param l_max:
    :param b_max:
    :param res:
    :param z_step:
    :param func1:
    :param func2:
    :return:
    """
    # See LOS_Gas_Ferierre for better documentation
    # Open a file tmp.py and write the passed function to it.  This is required because of
    # some issues surrounding multithreading in python.

    # Write func to a file so it is importable by child threads
    #f = open('tmp.py', 'wb')
    f = tempfile.NamedTemporaryFile(delete=False)
    tmp1 = f.name
    f.write(func1)
    f.flush()
    f.close()

    #f = open('tmp2.py', 'wb')
    f = tempfile.NamedTemporaryFile(delete=False)
    tmp2 = f.name
    f.write(func2)
    f.flush()
    f.close()
    time.sleep(0.05)
    n_threads = 1 # mp.cpu_count()

    kernel = partial(__LOS_DM,
                     n_thread=n_threads, l_max=l_max, b_max=b_max,
                     res=res, tmp1=tmp1, tmp2=tmp2, r_min=0, r_max=20, z_step=z_step)

    p = mp.Pool(n_threads)
    slices = p.map(kernel, range(n_threads))
    p.close()

    proj_skymap = slices[0]
    for slice_ in slices[1:]:
        proj_skymap = np.append(proj_skymap, slice_, axis=0)

    return proj_skymap


def GenNFW(nside=256, profile='NFW', decay=False, gamma=1, axesratio=1, rotation=0., offset=(0, 0), res=.125, size=60.,
           fitsout=None, r_s=20., mult_solid_ang=False, cartesian=False):
    """
    Generates a dark matter annihilation or decay skymap combined with instrumental and point source maps.

    :param nside: healpix nside.
    :param profile: 'NFW', 'Ein', 'Bur', 'Baryonic', or 'Iso'
    :param decay: If false, skymap is for annihilating dark matter
    :param gamma: Inner slope of DM profile for NFW.  Shape parameter for Einasto. Unused for Burk
    :param r_s: Scale factor
    :param axesratio: Stretch the *projected* dark matter profile along the +y axis
    :param rotation: *NOT SUPPORTED* In degrees, the CCW rotation of the DM profile.
    :param offset: offsets from (glon,glat)=(0,0) in degrees
    :param res: width in degrees of interpolation.  Should be ~0.5*size of healpix radius or less.
    :param size: max dist in degrees from the GC before all values zero.
    :param fitsout: write fits file to this path
    :param mult_solid_ang: Mutiply pixels by there solid angle.
    :param cartesian: output cartesian map instead of healpix
    :returns ndarray: A mask of DM convolved with XMM-Newton observations and point source maskes.\n
             Normalization is:\n
                Integral_l.o.s. [ pho ] dz\n
                Integral_l.o.s. [ pho^2 ] dz\n
             rho_0=0.4 GeV cm^-3 at R=8.5 kpc assumed.
             Units at this point are GeV cm^-2 sr^-1. or (ann) GeV^2 cm^5 sr^-1
             Finally, we multiply each pixel by it's area in sr.
            **This is not multiplied by Effective area**

             Thus return units are GeV cm^-2 for decay or or GeV^2 cm^-5 for annihilations
    """

    #TODO: Make rotation and offset for the actual profile.

    # -------------------------------------------------
    # Define Dark matter Profiles
    func = '''
import numpy as np
gamma=''' + str(gamma) + '''
r_s=''' + str(r_s) + '''
#offset=''' + str(offset) + '''
#rotation=''' + str(rotation) + '''
#axesratio=''' + str(axesratio) + '''

#sinAng = np.sin(np.deg2rad(rotation))
#cosAng = np.cos(np.deg2rad(rotation))
    '''

    NFW = '''
def func(x,y,z):
    #x = X
    #y = ((Y-offset[0])*cosAng+(Z-offset[1])*sinAng) / axesratio
    #z = -(Y-offset[0])*sinAng+(Z-offset[1])*cosAng

    r=np.sqrt(x*x+y*y+z*z)
    return r**-gamma*(1/(1+r/r_s)**(3-gamma))
    '''

    Ein = '''
def func(x,y,z):
    r=np.sqrt(x*x+y*y+z*z)
    return np.exp(-2/gamma*((r/r_s)**gamma-1))
    '''

    Bur = '''
def func(x,y,z):
    r=np.sqrt(x*x+y*y+z*z)
    return r_s**3/((r+r_s)*(r*r+r_s*r_s))
    '''

    # This is taken from arxiv: 1509.02166
    Baryonic = '''
import numpy as np

r_DM, rho_DM = np.array([[  1.47644469e-01,   8.47861095e+00],
       [  1.83846913e-01,   8.46988620e+00],
       [  2.23943289e-01,   8.74097384e+00],
       [  2.78897282e-01,   8.18358051e+00],
       [  3.39750047e-01,   8.17600109e+00],
       [  4.42092948e-01,   7.65305885e+00],
       [  5.75309021e-01,   6.93496863e+00],
       [  7.48724967e-01,   6.08372093e+00],
       [  9.74489091e-01,   5.16665429e+00],
       [  1.26823027e+00,   4.53246216e+00],
       [  1.54566213e+00,   3.72755260e+00],
       [  1.92614543e+00,   2.69234062e+00],
       [  2.29728963e+00,   1.94502708e+00],
       [  2.73994869e+00,   1.40514552e+00],
       [  3.26689343e+00,   1.15572793e+00],
       [  3.98246553e+00,   8.62365176e-01],
       [  4.64680862e+00,   6.23062442e-01],
       [  5.42113818e+00,   4.80331717e-01],
       [  6.32449958e+00,   3.70297650e-01],
       [  7.37782458e+00,   2.94879996e-01],
       [  8.60790670e+00,   2.20074812e-01],
       [  1.02665429e+01,   1.58988601e-01],
       [  1.27947638e+01,   1.11169985e-01],
       [  1.49291483e+01,   8.03207787e-02],
       [  1.78058090e+01,   5.80261237e-02],
       [  1.98845899e+01,   4.19327503e-02],
       [  2.37124401e+01,   3.23234960e-02],
       [  2.76680790e+01,   2.33538609e-02],
       [  3.15808865e+01,   1.74312459e-02],
       [  3.85101770e+01,   1.14241888e-02],
       [  4.59377016e+01,   7.73484638e-03],
       [  5.60127538e+01,   5.23640586e-03],
       [  6.68057127e+01,   3.78293561e-03],
       [  7.79500437e+01,   2.73318678e-03],
       [  9.29772239e+01,   1.91152639e-03],
       [  1.08495786e+02,   1.33701412e-03],
       [  1.23887034e+02,   8.48559721e-04],
       [  1.57891600e+02,   4.88372579e-04],
       [  1.96880096e+02,   2.72132354e-04],
       [  2.34925251e+02,   1.61833282e-04]]).T

rho_DM_interp  = lambda r: np.exp(np.interp(np.log(r), np.log(r_DM), np.log(rho_DM)))

def func(x,y,z):
    r=np.sqrt(x*x+y*y+z*z)
    return rho_DM_interp(r)

    '''



    # Normalize Dark matter Profiles to the Solar Position (rho=0.4 GeV/cm^3 for r=8.5kpc )
    if profile == 'NFW':
        func += NFW
        dm_norm = 0.4 / (8.5 ** -gamma * (1 / (1 + 8.5 / 20) ** (3 - gamma)))
    elif profile == 'Ein':
        func += Ein
        dm_norm = 0.4 / np.exp(-2 / gamma * ((8.5 / 20.) ** gamma - 1))
    elif profile == 'Bur':
        func += Bur
        dm_norm = 0.4 / (r_s ** 3 / ((8.5 + r_s) * (8.5 ** 2 + r_s ** 2)))
    elif profile == 'Iso':
        dm_norm = 1.
        func += '''
func = lambda x,y,z: 1.
'''
    elif profile == 'Baryonic':
        func += Baryonic
        dm_norm = 0.4 / 0.22540632663301324
    else:
        raise Exception("DM Halo type not supported")

    axesratio = 1./axesratio

    #-------------------------------------------------------
    # Integrate the DM profile along l.o.s. in 1 pc steps
    if decay:
        dm_prof = dm_norm * LOS_DM(0, 1.5 * axesratio * size, res=res, z_step=0.002, func1=func,
                                   func2='func=lambda x,y,z:1.')[:, 0]
    else:
        dm_prof = dm_norm ** 2 * LOS_DM(0, 1.5 * axesratio * size, res=res, z_step=0.002, func1=func,
                                        func2=func)[:, 0]
    
    
    # dm_prof is a list of LOS integrated DM profile as a function of angular displacement from GC.
    # Build the interpolator..
    dm_interp = lambda r: np.interp(r, np.linspace(-1.5 * axesratio * size, 1.5 * axesratio * size,
                                                   len(dm_prof)), dm_prof, left=0, right=0)
    # Currently unsupported for hpix maps
    rotation = np.deg2rad(-rotation)

    if cartesian is False:

        # Init healpix array and get the pixel sky locations
        l, b = Tools.hpix2ang(np.arange(12*nside**2))

        # Get angular distance from Vincenty's formula for great circle
        b, l = np.deg2rad((axesratio*(b-offset[1]), l-offset[0]))

        out_of_bounds = np.where((b*axesratio > np.pi) | (b*axesratio < -np.pi))[0]

        d = np.rad2deg(np.arctan2(np.sqrt(np.square(np.cos(b)*np.sin(l))+np.square(np.sin(b))), np.cos(b)*np.cos(l)))
        solidAngle = healpy.pixelfunc.nside2pixarea(nside)

        # Evaluate DM integral at each point.
        if mult_solid_ang:
            hpix = dm_interp(d)*solidAngle
            hpix[out_of_bounds]=0
        else:
            hpix = dm_interp(d)
            hpix[out_of_bounds]=0

        #########################################
        # Write skymap to FITS File
        #########################################
        if fitsout is not None:
            header = {'PROFILE': (profile, ''),
                      'DECAY': (decay, 'True=decay, False=annihilating'),
                      'GAMMA': (gamma, 'inner slope '),
                      'AXESRAT': (axesratio, 'Axes ratio of projected DM profile'),
                      'ROT': (rotation, 'Rotation angle of DM profile in degrees from +l toward +b'),
                      'OFFLON': (offset[0], 'LON offset of DM profile center'),
                      'OFFLAT': (offset[1], 'LAT offset of DM profile center'),
                      'COMMENT1': 'Skymap generated by GammaLike.DM (erccarls@ucsc.edu) 2014',
                      'COMMENT2': 'Normalization: (spherical) DM density=0.4 GeV/cm^3 at R_solar=8.5',
            }

            hdu = pyfits.PrimaryHDU(hpix)
            for key in header:
                if 'COMMENT' not in key:
                    hdu.header.update(key, header[key][0], header[key][1])
                else:
                    hdu.header.add_comment(header[key])

            hdu.writeto(fitsout, clobber=True)

        return hpix

    else:
        # ------------------------------------------------------
        # CARTESIAN MAP ROUTINES
        # ------------------------------------------------------

        # Generate the skymap!
        skymap_dim = np.linspace(-size, size, 2 * size / res + 1)
        skymap = np.zeros(shape=(skymap_dim.shape[0] - 1, skymap_dim.shape[0] - 1))
        center_bin = 0.5 * (skymap_dim[1] - skymap_dim[0])
        for i in range(len(skymap_dim) - 1):
            # Adjust for offset
            x, y = skymap_dim[i] - offset[0] + center_bin, skymap_dim[:-1] - offset[1] + center_bin
            # Rotate coordinates
            x, y = x * np.cos(rotation) + y * np.sin(rotation), -x * np.sin(rotation) + y * np.cos(rotation)
            # Evaluate the DM profile
            r = np.sqrt((x / float(axesratio)) ** 2. + y ** 2)
            dm_contrib = dm_interp(r)
        
            skymap[:, i] = dm_contrib
        
        if fitsout is not None:
            #########################################
            # Write skymap to FITS File
            #########################################
            header = {'NAXIS': (2, ''),
                      'NAXIS1': (int(1. / res) + 1, ''),
                      'CTYPE1': ('GLON---NCP', ''),
                      'CRVAL1': (0., ''),
                      'CRPIX1': (int(size / res) + 1 / 2., ''),
                      'CUNIT1': ('deg', ''),
                      'CDELT1': (res, ''),
                      'NAXIS2': (int(size / res) + 1, ''),
                      'CTYPE2': ('GLAT--NCP', ''),
                      'CRVAL2': (0., ''),
                      'CRPIX2': (int(size / res) + 1, ''),
                      'CUNIT2': ('deg', ''),
                      'CDELT2': (res, ''),
                      'PROFILE': (profile, ''),
                      'DECAY': (decay, 'True=decay, False=annihilating'),
                      'GAMMA': (gamma, 'inner slope '),
                      'AXESRAT': (axesRatio, 'Axis ratio of projected DM profile'),
                      'ROT': (rotation, 'Rotation angle of DM profile in degrees from +l toward +b'),
                      'OFFLON': (offset[0], 'LON offset of DM profile center'),
                      'OFFLAT': (offset[1], 'LAT offset of DM profile center'),
                      'COMMENT1': 'Skymap generated by GammaLike.DM (erccarls@ucsc.edu) 2014',
                      'COMMENT2': 'Normalization: (spherical) DM profile at .5 deg.=1 (before PSF)',
                      'COMMENT3': 'Then PSF is applied and map is multplied by ',
                      'COMMENT4': 'sum_i mask_i*GTI_i/Total_GTI'
            }
        
            hdu = pyfits.PrimaryHDU(skymap)
            for key in header:
                if 'COMMENT' not in key:
                    hdu.header.update(key, header[key][0], header[key][1])
                else:
                    hdu.header.add_comment(header[key])
        
            hdu.writeto(fitsout, clobber=True)
        
        return skymap
