#--------------------------------------------------------------------------
# Tools.py
# This module contains helper functions for likelihood analysis including 
# PSF convolutions and pixel transformations to and from healpix space. 
# Author: Eric Carlson (erccarls@ucsc.edu) 11/20/2014
#--------------------------------------------------------------------------
import numpy as np 
import pyfits
import healpy
from scipy.integrate import quad
from scipy.special import expn
from scipy.interpolate import RegularGridInterpolator


def hpix2ang(hpix, nside=256):
    """Transform the healpix index into lat/lon"""
    b, l = np.rad2deg(healpy.pix2ang(nside, hpix))
    b = 90-b
    return l, b


def ang2hpix(l, b, nside=256):
    """Transform lat/lon into healpix index"""
    l, b = np.deg2rad(l), np.deg2rad(-b+90)

    l = np.array(l)
    if l.ndim == 0: 
        if l > 180:
            l -= 360
    else:
        idx = np.where(l > 180)[0]
        l[idx] -= 360
    
    return healpy.ang2pix(nside, b, l, nest=False)


def GetSpec(specType):
    """
    Given a 2FGL Spectral type return lambdas for the spectrum and integrated spectrum
    params:
        specType: Can be 'PowerLaw','PLExpCutoff', or 'LogParabola'
    returns:
        Spec,IntegratedSpec: the spectrum and integrated spectrum.  See function def for param ordering.
    """
    if specType == 'PowerLaw':
        Spec = lambda e, gamma: e**-gamma
        IntegratedSpec = lambda e1, e2, gamma: (e1*e2)**-gamma * (e1*e2**gamma - e1**gamma*e2)/ (gamma-1)

    elif specType == 'PLExpCutoff':
        Spec = lambda e, gamma, cutoff: e**-gamma * np.exp(-e/cutoff)
        IntegratedSpec = lambda e1, e2, gamma, cutoff: (e1**(1-gamma)*expn(gamma, e1/cutoff)
                                                        -e2**(1-gamma)*expn(gamma, e2/cutoff))

    elif specType == 'LogParabola':
        Spec = lambda e, alpha, beta, pivot: e**-(alpha+beta*np.log(e/pivot))
        IntegratedSpec = lambda e1, e2, alpha, beta, pivot: quad(Spec, e1, e2, args=(alpha, beta, pivot))[0]
    else:
        raise Exception("Spectral type not supported.")

    return Spec, IntegratedSpec


#--------------------------------------------------------------------------
# Here we generate models and convolve them with 
# instrumental response functions. 
#--------------------------------------------------------------------------
def GetPSF(E_min, E_max, psfFile='/data/fermi_data_1-8-14/psf_P7REP_SOURCE_BOTH.fits'):
    """
    #---------------------------------------------------------------------------------------------
    # Spectrally weight the PSF(E) and return the average.
    #---------------------------------------------------------------------------------------------
    """
    hdu = pyfits.open(psfFile)
    
    thetas = np.array([theta[0] for theta in hdu[2].data])
    energies = np.array([energy[0] for energy in hdu[1].data])
    PSFs = np.array([psf[2] for psf in hdu[1].data])
    
    E_min_bin = np.argmin(np.abs(energies-E_min))
    E_max_bin = np.argmin(np.abs(energies-E_max))+1
    
    if E_max_bin > len(PSFs): E_min_bin, E_max_bin = len(PSFs)-2, len(PSFs)-1

    weights = energies[E_min_bin:E_max_bin]**-GetSpectralIndex(E_min, E_max)
    return thetas, np.average(PSFs[E_min_bin:E_max_bin], weights=weights, axis=0)


def ApplyPSF(hpix, E_min, E_max, PSFFile='P7Clean_Front+Back.fits', sigma=.1, smoothed=False):
    """
    WARNING INCOMPLETE: Normalization off and smoothing below healpix scale explodes. 
    This method takes a healpix input 'hpix', and a spectrally averaged energy 'E'. It then looks up the corresponding PSF from tables -> calculates the legendre transform coefficients -> transforms the input pixles into spherical harmonics -> re-weight the alm coefficients according to the PSF -> returns the inverse transform
    """

    # Spherical Harmonic transform 
    alm  = healpy.sphtfunc.map2alm(hpix)
    
    # get maximum value of l 
    l_max = healpy.sphtfunc.Alm.getlmax(len(alm))
    x = np.linspace(-30, 30, 10000)  # Cos(0) to Cos(pi)
    psf_y = (sigma*np.sqrt(2*np.pi))*np.exp(-x*x/(2*sigma**2))
    psf_x = np.cos(np.deg2rad(x))
    
    # Fit the PSF to l_max legendre polynomials
    cls = np.sqrt(4*np.pi/(2*np.arange(0, l_max+1)+1))*np.polynomial.legendre.legfit(psf_x, psf_y, l_max)
    conv_alm=healpy.sphtfunc.almxfl(alm, cls)

    # Find nside and return inverse transformed map. 
    nside = healpy.get_nside(hpix)
    if smoothed is False:
        return healpy.sphtfunc.alm2map(conv_alm, nside=nside, verbose=False)
    else:
        return healpy.sphtfunc.alm2map(alm, nside=nside, sigma=np.deg2rad(sigma), verbose=False)
      

def ApplyGaussianPSF(hpix, E_min, E_max, psfFile):
    """
    Finds the spectral weighted average PSF, determines the FWHM and blurs by Gaussian kernel of that size.
        hpix: input healpix map
        E_min: Minimum energy for spectral weighting
        E_max: Max energy for spectral weighting
        psfFile: Output of gtpsf 
    """
    theta, psf = GetPSF(E_min, E_max, psfFile)
    
    # Find FWHM
    halfmax = np.argmin(np.abs(0.5-psf/psf.max()))
    FWHM = np.deg2rad(2*theta[halfmax])
    
    # Spherical Harmonic transform 
    alm  = healpy.sphtfunc.map2alm(hpix)
    # inverse transform with gaussian beam
    return healpy.sphtfunc.alm2map(alm, nside=healpy.get_nside(hpix), fwhm=FWHM, verbose=False)


currentExpCube = None  # keeps track of the current gtexpcube2
def GetExpMap(E_min, E_max, l, b, expcube='/data/fermi_data_1-8-14/gtexpcube2_ALL_BOTH'):
    '''
    Returns the effective area given the energy range and angular coordinates.
    params:     
        E_min: Min energy in MeV
        E_max: Max Energy in MeV
        l: Galactic longitude.
        b: Galactic latitude.
        expcube: Exposure cube file over observation from Fermitools gtexpcube2.

    returns:
        value of the exposure in cm^2*s 
        "
    '''
    # check if the expCube has already been opened. 
    global currentExpCube
    if currentExpCube is None:
        currentExpCube = pyfits.open(expcube)
    hdu = currentExpCube

    # Find the average photon energy over the band
    alpha = GetSpectralIndex(E_min, E_max)
    if E_min == E_max:
        average_E = E_min
    else:
        average_E = (1-alpha)/(alpha-2)  *  (E_min**(2-alpha)-E_max**(2-alpha))  /  (E_min**(1-alpha)-E_max**(1-alpha))

    # Find the energy bin in the expcube file
    Ebin = int(np.round((np.log(average_E)-hdu[0].header['CRVAL3'])/hdu[0].header['CDELT3']))
    if Ebin >= hdu[0].header['NAXIS3']:
        Ebin=hdu[0].header['NAXIS3']-1

    # convert 0-360 to -180-180
    l,b = np.array(l), np.array(b)
    if l.ndim == 0: 
        if l>180:
            l -= 360
    else:
        idx = np.where(l > 180)[0]
        l[idx] -= 360.

    # Find lat/lon bin on expmap 
    l_bin = np.round((l-hdu[0].header['CRVAL1'])/hdu[0].header['CDELT1']+hdu[0].header['CRPIX1']).astype(np.int32)
    b_bin = np.round((b-hdu[0].header['CRVAL2'])/hdu[0].header['CDELT2']+hdu[0].header['CRPIX2']).astype(np.int32)
    # Return the exposure map in cm*s
    return hdu[0].data[Ebin,b_bin,l_bin]

def GetSpectralIndex(E_min, E_max):
    """
    Returns the spectal index between evaluated at the two endpoints E_min and E_max based on the averaged P7REPv15 diffuse model
    params:
        E_min: Min energy in MeV
        E_max: Max energy in MeV
    returns:
        index: Spectral index (positive value).
    """
    E = np.array([58.473133087158203, 79.970359802246108, 109.37088726489363, 149.58030713742139, 204.57243095354417,
                  279.7820134691566, 382.64185792772452, 523.31738421248383, 715.71125569520109, 978.83734991844688,
                  1338.6998597146596, 1830.8632323330951, 2503.9669281982829, 3424.532355440329, 4683.5370393234753,
                  6405.4057377695362, 8760.3070758200847, 11980.97094929486, 16385.688725918291, 22409.769304923335,
                  30648.559770668966, 41916.282280063475, 57326.501908367274, 78402.17791960774, 107226.17459483664,
                  146647.10628359963, 200560.85990769751, 274295.61718814232, 375138.42752394517, 513055.37160154793])
    dnde = np.array([1.3259335, 0.94195729, 0.66580701, 0.46162829, 0.30296713, 0.18484889, 0.10698333, 0.059697378,
                     0.032260861, 0.01673951, 0.0082548652, 0.0039907703, 0.0018546022, 0.00082937587, 0.0003599966,
                     0.00015557533, 6.7215013e-05, 2.8863404e-05, 1.2341489e-05, 5.3399754e-06, 2.2966778e-06,
                     9.9477847e-07, 4.53333e-07, 2.1135656e-07, 9.9832157e-08, 4.6697188e-08, 2.1986754e-08,
                     1.0368451e-08, 5.0197251e-09, 2.4097735e-09])

    # Find bin and check bounds 
    E_bin_min, E_bin_max = int(np.argmin(np.abs(E-E_min))),  int(np.argmin(np.abs(E-E_max)))
    if E_bin_min == E_bin_max:
        E_bin_max=E_bin_min+1
    if E_bin_max >= len(dnde):
        E_bin_min, E_bin_max= len(dnde)-3, len(dnde)-1

    return -np.log(dnde[E_bin_min]/dnde[E_bin_max])/np.log(E[E_bin_min]/E[E_bin_max])


def CartesianCountMap2Healpix(cartCube, nside):
    """
    This is a static function which takes an input cartesian datacube and returns a 2d healpix array.
    It simply maps each cartesian pixel to the healpix grid (individually for each spectral bin). 
    Primarily intended for converting gtsrcmaps output into healpix format.
    params: 
        input:
            cartCube: Fits filename containing the source cartesian countmap
            nside: healpix nside
        return: 
            hpixcube: First index corresponds to the energies in cartCube, second dimension is the healpix grid. 
    """
    # open the fits
    hdu = pyfits.open(cartCube)
    # initialize the target array
    hpix = np.zeros(shape=(hdu[0].data.shape[0],12*nside**2))
    
    def Getlatlon(i, j):
        l = (i-hdu[0].header['CRPIX1']+1)*hdu[0].header['CDELT1']+hdu[0].header['CRVAL1']
        b = (j-hdu[0].header['CRPIX2']+1)*hdu[0].header['CDELT2']+hdu[0].header['CRVAL2']
        return l, b

    # iterate over latitudes
    i_list = np.arange(hdu[0].data.shape[2]) # list of longitude bins 
    for j in range(hdu[0].data.shape[1]):
        l,b = Getlatlon(i_list,j)
        # convert l,b to healpix index
        hpixIndex = ang2hpix(l,b,nside)
        # iterate over energy bins, summing the contibution from each pixel.
        for i_E in range(hpix.shape[0]):
            hpix[i_E,hpixIndex]+=hdu[0].data[i_E,j]
    return hpix


def SampleCartesianMap(fits, E_min, E_max, nside, E_bins=5):
    """
    Given a cartesian fits mapcube and energy range, returns a spectrally weighted average diffuse model
    in units of (s cm^2)^-1.  Just need to multiply by effective area and PSF in order to
    params:
        fits: fits filename.  Assumed to run from -180, 180 and -90,-90 and have energy keywords like fermi diffuse model
        E_min: Min energy in MeV
        E_max: Max energy in MeV
        nside: healpix nside
        E_bins: Number of subbins for integration.
    returns:
        Healpix pixels with units in (s cm^2)^-1
    """
    hdu = pyfits.open(fits)

    # Define the grid spacings
    energies = np.log10([e[0] for e in hdu[1].data])
    lats = np.linspace(-90, 90, hdu[0].header['NAXIS2'])
    lons = np.linspace(-180, 180, hdu[0].header['NAXIS1'])
    print 10**energies
    # Build the interpolator
    rgi = RegularGridInterpolator((energies, lats, lons), hdu[0].data, method='linear',
                                  bounds_error=False, fill_value=np.float32(0.))
    # Init the healpix grid and compute the energy bins.
    master = np.zeros(12*nside**2)
    bin_edges = np.logspace(np.log10(E_min), np.log10(E_max), E_bins+1)
    # Get the latitude and longitude.
    l, b = hpix2ang(np.arange(12*nside**2))
    idx = np.where(l > 180)[0]
    l[idx] -= 360.

    # Iterate over the sub-bins to return the integrated spectrum.
    for i_E in range(len(bin_edges)-1):
        central_energy = 10.**(0.5*(np.log10(bin_edges[i_E]) + np.log10(bin_edges[i_E+1])))
        print central_energy
        bin_width = bin_edges[i_E+1]-bin_edges[i_E]
        # Units of diffuse model are (sr s cm^2 MeV)^-1
        master += rgi((np.log10(central_energy), b, l))*bin_width

    # Units of returned model are (s cm^2)^-1
    return master*healpy.pixelfunc.nside2pixarea(nside)

#----------------------------------------------------------------------------
# Testing for ApplyPSF
#----------------------------------------------------------------------------
# nside = 256
# hpix_im = np.zeros(12*nside**2)
# hpix_im[ang2hpix(0,0)]=1

# # set center pixel to 1
# hpix_im[ang2hpix(0,0)]=1
# hpix_im = ApplyPSF(hpix_im,100,1000,sigma=1)
# test = healpy.cartview(hpix_im,return_projected_map=True,latra=[-5,5],lonra=[-5,5])
# plt.clf()
# im = plt.imshow(test,extent=[-5,5,-5,5])
# plt.colorbar(im)
# print 'Sum', np.sum(test)

# hpix_im = np.zeros(12*nside**2)
# hpix_im[ang2hpix(0,0)]=1
# hpix_im = ApplyPSF(hpix_im,100,1000,sigma=1,smoothed=True)
# test2 = healpy.cartview(hpix_im,return_projected_map=True,latra=[-5,5],lonra=[-5,5])
# plt.clf()
# im = plt.imshow(test2,extent=[-5,5,-5,5])
# plt.colorbar(im)
# print 'Sum', np.sum(test2)
# plt.show()

# plt.plot(np.linspace(-5,5,800),test[399,:])#/np.max(test))
# plt.plot(np.linspace(-5,5,800),test2[399,:])#/np.max(test2))

#----------------------------------------------------------------------------
# End Testing for ApplyPSF
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Testing for ApplyGaussianPSF
#----------------------------------------------------------------------------
# withPSF = ApplyGaussianPSF(hpix_im,E_min=100,E_max=150)
# test2 = healpy.cartview(withPSF,return_projected_map=True,latra=[-5,5],lonra=[-5,5])
# im = plt.imshow(test2,extent=[-5,5,-5,5])
