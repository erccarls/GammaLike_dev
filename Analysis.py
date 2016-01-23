#--------------------------------------------------------------------------
# Analysis.py
# This class contains the settings for a binned likelihood analysis.
# Author: Eric Carlson (erccarls@ucsc.edu) 11/20/2014
#--------------------------------------------------------------------------
import numpy as np
import healpy
import pyfits
import DM
import Tools
import Template
import GenFermiData
import SourceMap
import GammaLikelihood
import copy
from scipy.integrate import quad
import multiprocessing as mp
import sys
import cPickle as pickle
import h5py
from scipy.sparse import csr_matrix

from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord


class Analysis():
    #--------------------------------------------------------------------
    # Most binning settings follows Calore et al 2014 (1409.0042)
    #--------------------------------------------------------------------

    def __init__(self, E_min=5e2, E_max=5e5, nside=256, gamma=1.45, n_bins=20, prefix_bins=[300, 350, 400, 450, 500],
                    tag='P7REP_CLEAN_V15_calore', basepath='/data/GCE_sys/',
                    phfile_raw='/data/fermi_data_1-8-14/phfile.txt',
                    scfile='/data/fermi_data_1-8-14/lat_spacecraft_merged.fits',
                    evclass=2, convtype=-1,  zmax=100, irf='P7REP_CLEAN_V15', fglpath='/data/gll_psc_v14.fit',
                    gtfilter="DATA_QUAL>0 && LAT_CONFIG==1 && ABS(ROCK_ANGLE)<52", templateDir='/data/Extended_archive_v15/Templates/',evtype='INDEF',):
        """
        :param    E_min:        Min energy for recursive spectral binning
        :param    E_max:        Max energy for recursive spectral binning
        :param    n_bins:       Number of recursive spectal bins. Specify zero if custom bins are supplied.
        :param    gamma:        Power-law index for recursive binning
        :param    nside:        Number of healpix spatial bins
        :param    prefix_bins:  manually specify bin edges. These are prepended to the recursive bins
        :param    tag:          an analysis tag that is included in generated files
        :param    basepath:     the base directory for relative paths
        :param    phfile_raw:   Photon file or list of files from fermitools (evfile you would input to gtselect)
        :param    scfile:       Merged spacecraft file
        :param    evclass:      Fermi event class (integer)
        :param    evtype:       Fermi event type subselection.  Default is INDEF which applies no subselection
        :param    convtype:     Fermi conversion type (integer)
        :param    zmax:         zenith cut
        :param    irf:          Fermi irf name
        :param    fglpath:      Path to the 2FGL fits file
        :param    gtfilter:     Filter string to pass to gtselect.
        :param    templateDir:  Path to the directory containing spatial FITS templates for FGL sources.
        """

        self.E_min = E_min
        self.E_max = E_max
        self.nside = nside
        self.gamma = gamma
        self.n_bins = n_bins
        self.phfile_raw = phfile_raw
        self.tag = tag
        self.basepath = basepath
        self.scfile = scfile
        self.evclass = evclass
        self.evtype = evtype
        self.convtype = convtype
        self.zmax = zmax
        self.irf = irf
        self.bin_edges = 0
        self.bin_edges = copy.copy(prefix_bins)
        self.gtfilter = gtfilter

        self.phfile = basepath + '/photons_merged_cut_'+str(tag)+'.fits'
        self.psfFile = basepath + '/gtpsf_' + str(tag)+'.fits'
        self.expCube = basepath + '/gtexpcube2_' + str(tag)+'.fits'
        self.fglpath = fglpath
        self.templateDir = templateDir + '/'

        # Currently Unassigned
        self.binned_data = None  # master list of bin counts. 1st index is spectral, 2nd is pixel_number
        self.mask = None         # Mask. 0 is not analyzed. between 0-1 corresponds to weights.
        self.templateList = {}   # Dict of analysis templates
        self.fitted = False      # True if fit has been run with the current templateList.
        self.residual = None     # After a fit, this is automatically updated.
        self.dm_renorm = 1e19    # renormalization constant for DM template
        self.m = None            # iMinuit results object
        self.res = None          # scipy minimizer results object.
        self.res_vals = None     # dict with best fit values for basinhopping minimizer.
        self.jfactor = 0.        # Dark matter j-factor
        self.psc_weights = None  # Pixel weighting for likelihood analysis
        self.expMap =None        # Stores the precomputed exposure map in memory.
        self.central_energies = None  # Log-central energies for each bin.
        self.loglike = None  # Stores the fit likelihood value.

        prefix_n_bins = len(prefix_bins)-1
        # --------------------------------------------------------------------
        # Recursively generate bin edges
        Ej = self.E_min
        for j in range(self.n_bins):
            # Eqn 2.3 (1409.0042)
            Ej = ((Ej**(1-self.gamma) - (self.E_min**(1-self.gamma)
                                         - self.E_max**(1-self.gamma))/self.n_bins)**(1/(1-self.gamma)))
            self.bin_edges += [Ej, ]
        # Add the prefix bins to the total bincount.
        self.n_bins += prefix_n_bins
        # Calculate the log-center of each bin.
        self.central_energies = 10**np.array([0.5*(np.log10(self.bin_edges[i])+np.log10(self.bin_edges[i+1]))
                                              for i in range(self.n_bins)])

        try:
            self.expMap = np.load('expmap_'+self.tag+'.npy')
        except:
            print "Warning: precomputed exposure map not found.  Reverting to slower methods, but you should run" \
                  " Analysis.GenExposureMap() for substantial speed increase."

    def BinPhotons(self, infile=None, outfile=None):
        """
        Spatially and spatially bin the Photons in self.phfile
        """
        if infile is None:
            # Load Fermi Data
            data = pyfits.open(self.phfile)[1].data

            # --------------------------------------------------------------------
            # Perform Spectral binning
            bin_idx = []  # indices of the photons in each spectral bin.
            for i in range(len(self.bin_edges)-1):
                bin_low, bin_high = self.bin_edges[i], self.bin_edges[i+1]
                idx = np.where((data['ENERGY'] > bin_low) & (data['ENERGY'] < bin_high))[0]
                bin_idx.append(idx)

            # Now for each spectral bin, form the list of healpix pixels.
            self.binned_data = np.zeros(shape=(len(bin_idx), 12*self.nside**2))
            for i in range(self.binned_data.shape[0]):
                # Convert sky coords to healpix pixel number
                idx = bin_idx[i]
                pix = Tools.ang2hpix(data['L'][idx], data['B'][idx], nside=self.nside)
                # count the number of events in each healpix pixel.
                np.add.at(self.binned_data[i], pix, 1.)
            if outfile is not None:
                np.save(open(outfile, 'wb'), self.binned_data.astype(np.float32))
        else:
            self.binned_data = np.load(infile)


    def GenSquareMask(self, l_range, b_range, plane_mask=0, merge=False):
        """
        Generate a square analysis mask (square in glat/glon)

        :param    l_range: range for min/max galactic longitude
        :param    b_range: range for min/max galactic latitude
        :param    plane_mask: Masks out |b|<plane_mask
        :param    merge: False will replace the current Analysis.mask.  In case one wants to combine multiple masks,
                    merge=True will apply the or operation between the exisiting and new mask
        :returns  mask: mask healpix array of dimension nside:
        """
        b_min, b_max = b_range
        l_min, l_max = l_range

        mask = np.zeros(shape=12*self.nside**2)
        # Find lat/lon of each healpix pixel
        l_pix, b_pix = Tools.hpix2ang(hpix=np.arange(12*self.nside**2), nside=self.nside)
        # Find elements that are masked
        l_pix[l_pix>180.] = l_pix[l_pix>180.]-360.
	
	    # Don't exclude the b=0 pixels
    	if plane_mask == 0:
    		plane_mask -= 1 

        idx = np.where(((l_pix < l_max) & (l_pix > l_min))
                       & (b_pix < b_max) & (b_pix > b_min)
                       & (np.abs(b_pix) > plane_mask))[0]
        mask[idx] = 1.  # Set unmasked elements to 1

        if merge is True:
            self.mask = (self.mask.astype(np.int32) | mask.astype(np.int32))
        else:
            self.mask = mask
        return mask



    def CaloreRegion(self, region):
        '''
        Returns a mask corresponding to Calore et al (1409.0042  figure 15 regions)
        
        :params region: Integer region index from 1 to 10
        :params l: Longitude vector of pixels between -180,180
        :params b: Latitude vector of pixels between -90,90
        '''
        
        pixels = np.arange(12*self.nside**2)
        l, b = Tools.hpix2ang(pixels, nside=self.nside) # Find l,b of each pixel
        mask = np.zeros(pixels.shape)
        l[l>180] -= 360
        
        if region==1:
            idx = (np.sqrt(l**2+b**2)<5) & (b>2)
        elif region==2:
            idx = (np.sqrt(l**2+b**2)<5) & (-b>2)
        
        elif region==3:
            idx = (5<np.sqrt(l**2+b**2)) & (b>np.abs(l)) & (np.sqrt(l**2+b**2)<10) 
        elif region==4:
            idx = (5<np.sqrt(l**2+b**2)) & (-b>np.abs(l)) & (np.sqrt(l**2+b**2)<10) 
        
        elif region==5:
            idx = (10<np.sqrt(l**2+b**2)) & (b>np.abs(l)) & (np.sqrt(l**2+b**2)<15) 
        elif region==6:
            idx = (10<np.sqrt(l**2+b**2)) & (-b>np.abs(l)) & (np.sqrt(l**2+b**2)<15) 
        
        elif region==7:
            idx = (5<np.sqrt(l**2+b**2)) & (l>np.abs(b)) & (np.sqrt(l**2+b**2)<15)
        elif region==8:
            idx = (5<np.sqrt(l**2+b**2)) & (-l>np.abs(b)) & (np.sqrt(l**2+b**2)<15) 
        
        elif region==9:
            idx = (15<np.sqrt(l**2+b**2)) & (np.sqrt(l**2+b**2)<20)
        
        elif region==10:
            idx =  np.sqrt(l**2+b**2)>20 
        else: 
            raise('Invalud Calore region. Must be int in 1...10')
        
        mask = (idx & (np.abs(l)<20) & (np.abs(b)<20) & (np.abs(b)>2))
        
        return mask


    def GenRadialMask(self, r1, r2, plane_mask=2, merge=False, start_angle=0., stop_angle=360.):
        '''
        Generate an annular mask with a plane cut out.

        :params r1: Inner radius in deg
        :params r2: Outer radius in deg
        :params plane_mask: Mask plus-minus "plane_mask" degrees in latitude
        :params merge: False will replace the current Analysis.mask.  In case one wants to combine multiple masks,
                    merge=True will apply the or operation between the exisiting and new mask
        :params start_angle: Stop and start angle must be specified in order to cut out an angular wedge beginning
                    from 0=+lon, 90=+lat, and 180=-lon. Start angle can be > stop angle to go through theta=0
        :params stop_angle: Stop and start angle must be specified in order to cut out an angular wedge beginning
                    from 0=+lon, 90=+lat, and 180=-lon.  Start angle can be > stop angle to go through theta=0
        :returns  mask: mask healpix array of dimension nside:
        '''

        pixels = np.arange(12*self.nside**2)
        l, b = Tools.hpix2ang(pixels, nside=self.nside) # Find l,b of each pixel
        r = Tools.Dist(0, l, 0, b)  # Find distance from origin
        mask = np.zeros(pixels.shape)
        l[l>180] -= 360

        angle_of_pixel = np.rad2deg(np.arctan2(-b,-l))

        if stop_angle < 0:
            stop_angle+=360

        if start_angle < 0:
            start_angle+=360

        if stop_angle<start_angle:
            mask[(r>r1) & (r<r2) & (np.abs(b)>=plane_mask) & (start_angle <= (angle_of_pixel+180)) & ( (angle_of_pixel+180) <= 360)] = 1 # Unmask the annulus
            mask[(r>r1) & (r<r2) & (np.abs(b)>=plane_mask) & (0 <= (angle_of_pixel+180)) & ( (angle_of_pixel+180) <= stop_angle)] = 1 # Unmask the annulus

        else:
            mask[(r>r1) & (r<r2) & (np.abs(b)>=plane_mask) & (start_angle <= (angle_of_pixel+180)) & ( (angle_of_pixel+180) <= stop_angle)] = 1 # Unmask the annulus

        if merge is True:
            self.mask = (self.mask.astype(np.int32) | mask.astype(np.int32))
        else:
            self.mask = mask

        return mask

    # def AddPointSourceTemplateFermi(self, pscmap='gtsrcmap_All_Sources.fits', name='PSC',
    #                                 fixSpectrum=False, fixNorm=False, limits=[None, None], value=1,):
    #     """
    #     Adds a point source map to the list of templates.  Cartesian input from gtsrcmaps is then converted
    #     to a healpix template.
    #
    #     :param    pscmap: The point source map should be the output from gtsrcmaps in cartesian coordinates.
    #     :param    name:   Name to use for this template.
    #     :param    fixSpectrum: If True, the relative normalizations of each energy bin will be held fixed for this
    #                 template, but the overall normalization is free
    #     :param    fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
    #     :param    limits:      Specify range of template normalizations.
    #     :param    value:       Initial value for the template normalization.
    #     """
    #     # Convert the input map into healpix.
    #     hpix = Tools.CartesianCountMap2Healpix(cartCube=pscmap, nside=self.nside)[:-1]/1e9
    #     for i in range(len(hpix)):
    #         hpix[i] /= (float(self.bin_edges[i])/self.bin_edges[0])
    #
    #     self.AddTemplate(name, hpix, fixSpectrum, fixNorm, limits, value, ApplyIRF=False, sourceClass='PSC')


    def AddFGLSource(self, idx, fixed=False):
        '''
        Add a single point source to the fit at index idx of the active PSC catalog (defined at Analysis.fglpath).
        :param idx: index of the point source to add.
        :param fix: Fix the normalization and spectrum of this source if True.
        :return:
        '''
        # Generate point source template and convert to sparse matrix for lower memory profile
        try:
            t_sparse = csr_matrix(self.GenPointSourceTemplate(onlyidx=[idx, ], save=False, verbosity=0), dtype=np.float32)
            name = pyfits.open(self.fglpath)[1].data['Source_Name'][idx]
            # Extended sources not yet supported
            if name[-1] == 'e':
                print 'Warning: Extended source found in ROI. These are not yet supported.'
                return
            name = name.replace('+','p').replace(' ', '_').replace('-', 'n').replace('.', 'd')[1:]
            # Already convolved with IRF, so just add template to stack.
            self.AddTemplate(name, t_sparse, fixSpectrum=fixed, fixNorm=fixed,
                             limits=[None, None], value=1., ApplyIRF=False, sourceClass='FGL')
        except:
            raise Exception("Index Error: Point Source index out of range for active catalog.")


    def PopulateROI(self, center, radius, fix_radius=5., include_point=True, include_extended=True):
        """
        Fills a rectangular region of interest with all FGL point sources.
        :param center: (lon,lat) of the ROI center
        :param width: ROI is a square box with this half-width in degrees.
        :param fix_radius: Sources farther than this from the center of the ROI have fixed normalization.
        :param:
        :return: None
        """

        fgl = pyfits.open(self.fglpath)[1].data
        l, b = center
        # Find soures impacting the ROI. We add 5 deg to this to ensure sources outside the ROI are included
        # since their PSF can leak in.  These source mush be fixed (radius must be > fix_radius).
        fgl_idx = np.nonzero((np.abs(b-fgl['GLAT']) < (radius+5.)) & (np.abs(l-fgl['GLON']) < (radius+5.)))[0]
        # Compute angular distance from the center to each source.
        dists = Tools.Dist(l, fgl['GLON'][fgl_idx], b, fgl['GLAT'][fgl_idx])

        # Add each source to the template stack
        for i, src_idx in enumerate(fgl_idx):

            fixed = False
            if dists[i] > fix_radius:
                fixed = True

            # If extended...
            if fgl['Source_Name'][src_idx][-1] == 'e':
                if include_extended:
                    print '\rPopulating ROI with point sources: %i of %i' % (i+1, len(fgl_idx))
                    self.AddExtendedSource(src_idx, fixed=fixed)

            elif include_point:
                print '\rPopulating ROI with point sources: %i of %i' % (i+1, len(fgl_idx)),
                self.AddFGLSource(src_idx, fixed=fixed)



    def AddExtendedSource(self, idx_fgl, fixed=True, add_template=True):
        """
        Add a 3FGL extended source to the analysis.
        :param source: the source name ('2FGL_Name' in the extended source table, but 'Source_Name' in the main FGL catalog column.)
        :param template_dir: Path to the directory containing FGL extended source fits templates.
        :param fixed: Fix the source, or let it float.
        :param add_template: Add the template to the stack?

        :return: scipy.sparse.csr_matrix with extended source. To convert to full healpix array call ".toarray()"
        """

        # Open FGL and look up the source name.
        hdu = pyfits.open(self.fglpath)
        ext_name = hdu[1].data['Extended_Source_Name'][idx_fgl]
        ext_idx = np.where(hdu[5].data['Source_Name']==ext_name)[0]
        fname = hdu[5].data['Spatial_Filename'][ext_idx][0]
        # Open the file and remove erroneous header info.
        hdu_spatial = pyfits.open(self.templateDir+fname, mode='readonly')
        try:
            hdu_spatial[0].header.pop('COMMENT')
        except:
            pass
        try:
            hdu_spatial[0].header.pop('HISTORY')
        except:
            pass

        # Read the WCS coordinates
        w = WCS(hdu_spatial[0].header, relax=False, fix=True)

        # Init a blank healpix template
        t = np.zeros(12*self.nside**2, dtype=np.float32)
        # Map the FGL extended template into healpix space, row by row.
        for i_row in range(hdu_spatial[0].data.shape[0]):
            # get lat/lon for each row
            lon, lat = w.all_pix2world(i_row,np.arange(hdu_spatial[0].data.shape[1]), 0)
            c_icrs = SkyCoord(ra=lon, dec=lat, frame='icrs', unit='degree')
            lon, lat = c_icrs.galactic.l.degree, c_icrs.galactic.b.degree
            # transform lat/lon to healpix
            hpix_idx = Tools.ang2hpix(lon, lat, nside=self.nside)
            # Add these counts to the healpix template
            np.add.at(t, hpix_idx, hdu_spatial[0].data[i_row, :])

        # Get the total number of counts from this source in each energy bin.  This will set the normalization
        total_counts = np.sum(self.GenPointSourceTemplate(pscmap=None, onlyidx=[idx_fgl,],
                                                          save=False, verbosity=0, ignore_ext=False), axis=1)

        # Generate a master sourcemap for the extended source (spatial template for each energy bin).
        master_t = np.array([t for i_E, count in enumerate(total_counts)])
        for i_E, count in enumerate(total_counts):
            master_t[i_E] = master_t[i_E]/np.sum(master_t[i_E])*count # normed to the expected PSC flux
            # Apply the PSF
            master_t[i_E] = Tools.ApplyGaussianPSF(master_t[i_E], E_min=self.bin_edges[i_E], E_max=self.bin_edges[i_E+1], psfFile=self.psfFile, multiplier=1.)
            #master_t[i_E] = Tools.ApplyPSF(master_t[i_E], E_min=self.bin_edges[i_E], E_max=self.bin_edges[i_E+1], PSFFile=self.psfFile, smoothed=False)
        # Don't allow the ALM of the PSF convolution to produce small negative values.
        # This is a count map so 1e-5 is totally negligible. This also cuts out artifacts from PSF convolution.
        master_t = master_t.clip(1e-5, 1e50)
        # Convert to sparse matrix for memory profile.
        t_sparse = csr_matrix(master_t, dtype=np.float32)
        self.AddTemplate(hdu[5].data['Source_Name'][ext_idx][0].replace(' ', '').replace('-', 'n').replace('+', 'p').replace('.', '_'), t_sparse, fixSpectrum=fixed, fixNorm=fixed,
                         limits=[None, None], value=1., ApplyIRF=False, sourceClass='FGL')
        return t_sparse

    def RemoveAllPointSources(self):
        """
        Removes all FGL sources from the template stack.  Useful if changing ROI, but still want to keep diffuse templates.
        :return: None
        """
        for t in self.templateList.keys():
            if t.sourceClass == 'FGL':
                self.DeleteTemplate(t)


    def AddPointSourceTemplate(self, pscmap=None, name='PSC', fixNorm=False,
                               limits=[None, None], value=1, multiplier=1.):
        """
        Adds a point source map to the list of templates.

        :param    pscmap: Filename of the pscmap.  If none, assumes default value.
        :param    name:   Name to use for this template.
        :param    fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
        :param    limits:      Specify range of template normalizations.
        :param    value:       Initial value for the template normalization.
        :param    multiplier: Sigma = multiplier*FWHM from fermi gtpsf.
        """
        if pscmap is None:
            pscmap = self.basepath + '/PSC_' + self.tag + '.npy'
        try:
            hpix = np.load(open(pscmap, 'r'))
        except:
            raise Exception('No point source map found at path '+str(pscmap))

        self.AddTemplate(name, hpix, fixSpectrum=True, fixNorm=fixNorm, limits=limits, value=value,
                         ApplyIRF=False, sourceClass='PSC', multiplier=multiplier)


    def GenPointSourceTemplate(self, pscmap=None, onlyidx=None, save=False, verbosity=1, ignore_ext=True,
                                l_range=(-180, 180), b_range=(-90, 90)):
        """
        Generates a point source count map valid for the current analysis based on 2fgl catalog.  This can take a long
        time so it is usually done once and then saved.

        :param pscmap: Specify the point source map filename.  If None, then the default path
            self.basemap+'PSC_'+self.tag+'.npy' is used.
        :param onlyidx: Generate template for only a single point source in the fglpath catalog at this index.
        :return PSCMap:  The healpix 'cube' for the point sources.
        """
        if (pscmap is None) and (save is True):
            pscmap = self.basepath + '/PSC_' + self.tag + '.npy'

        total_map = SourceMap.GenSourceMap(self.bin_edges, l_range=l_range, b_range=b_range,
                                           fglpath=self.fglpath,
                                           expcube=self.expCube,
                                           psffile=self.psfFile,
                                           maxpsf = 7.5,
                                           res=0.125,
                                           nside=self.nside,
                                           ignore_ext=ignore_ext,
                                           filename=pscmap, onlyidx=onlyidx, verbosity=verbosity)

        return total_map

    def PrintTemplates(self):
        """
        Prints the names and properties of each template in the template list.
        """
        print '%20s' % 'NAME', '%25s' % 'LIMITS', '%10s' % 'VALUE', '%10s' % 'FIXNORM', '%10s' % 'FIXSPEC',
        print '%10s' % 'SRCCLASS'
        for key in self.templateList:
            temp = self.templateList[key]
            if np.ndim(temp.value) == 0:
                print '%20s' % key, '%25s' % temp.limits, '%10s' % ('%3.3e' % temp.value), '%10s' % temp.fixNorm,
                print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
            else:
                print '%20s' % key, '%25s' % str(temp.limits), '%10s' % 'Vector', '%10s' % temp.fixNorm,
                print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
                print '         --------------------------------------------------------------------------------------'

                for i, val in enumerate(temp.value):
                    print '%20s' % ('[' + str(i) + ']'), '%25s' % str(temp.limits), '%10s' % ('%3.3e' % val), '%10s' % temp.fixNorm,
                    print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
                print '         --------------------------------------------------------------------------------------'


    def AddTemplate(self, name, healpixCube, fixSpectrum=False, fixNorm=False, limits=[None, None], value=1, ApplyIRF=True,
                    sourceClass='GEN', multiplier=1., valueUnc=None, noPSF=False):
        """
        Add Template to the template list.

        :param    name:   Name to use for this template.
        :param    healpixCube: Actually a 2-d array with first index selecting energy and second index selecting the
                    healpix index
        :param    fixSpectrum: If True, the relative normalizations of each energy bin will be held fixed for this
                    template, but the overall normalization is free
        :param    fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
        :param    limits:      Specify range of template normalizations.
        :param    value:       Initial value for the template normalization.
        :param    valueUnc:    Uncertainty on the fitting value. This is set automatically after fitting unless fixedNorm==True.
        :param    multiplier: Sigma = multiplier*FWHM from fermi gtpsf.
        :param    noPSF: Defaults to true, but can be disabled for speed.
        """

        self.fitted = False

        # Error Checking on shape of input cube.
        if (healpixCube.shape[0] != (len(self.bin_edges)-1)) or (healpixCube.shape[1] != (12*self.nside**2)):
            raise(Exception("Shape of template does not match binning"))

        healpixCube2 = copy.copy(healpixCube)
        if ApplyIRF:
            for i_E in range(len(self.bin_edges)-1):
                if sourceClass == 'ISO':
                    healpixCube2[i_E] = Tools.ApplyIRF(healpixCube2[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1],
                                                      self.psfFile, self.expCube, noPSF=True, expMap=self.expMap[i_E])
                else:
                    # This can be expensive if applying the PSF due to spherical harmonic transforms.
                    # This is already multithreaded in healpy.
                    healpixCube2[i_E] = Tools.ApplyIRF(healpixCube2[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1],
                                                      self.psfFile, self.expCube, multiplier=multiplier,
                                                      expMap=self.expMap[i_E], noPSF=noPSF)

            # Clip at zero. For delta functions, the PSF convolution from healpix ALM's can produce small negative numbers.
            healpixCube = healpixCube2.clip(0, 1e50)

        # Instantiate the template object.
        template = Template.Template(healpixCube2.astype(np.float32), fixSpectrum, fixNorm, limits, value, sourceClass,
                                     valueUnc)
        # Add the instance to the master template list.
        self.templateList[name] = template

    def DeleteTemplate(self, name):
        """
        Removes a template from templateList if it exists.

        :param name: Name of template to delete.
        """
        self.templateList.pop(name, None)

    def AddIsotropicTemplate(self, isofile='./IGRB_ackerman_2014_modA.dat', fixNorm=False, fixSpectrum=False):
        """
        Generates an isotropic template from a spectral file and add it to the current templateList

        :param isofile:  The file should be the same format as the isotropic files
        :param fixNorm: Fix the overall normalization of the isotropic template.
         from fermi (3 columns with E,flux,fluxUnc.)
        """
        # Load isotropic emission file
        E, flux, fluxUnc = np.genfromtxt(isofile).T
        # Build a power law interpolator
        fluxInterp = lambda x: np.exp(np.interp(np.log(x), np.log(E), np.log(flux)))
        fluxUncInterp = lambda x: np.exp(np.interp(np.log(x), np.log(E), np.log(fluxUnc)))

        # Units are in ph/cm^2/s/MeV/sr
        healpixCube = np.ones(shape=(len(self.bin_edges)-1, 12*self.nside**2))
        # Solid Angle of each pixel
        solidAngle = 4*np.pi/(12*self.nside**2)
        # Find
        valueUnc = []
        for i_E in range(len(self.bin_edges)-1):
            # multiply by integral(flux *dE )*solidangle
            counts = quad(fluxInterp, self.bin_edges[i_E], self.bin_edges[i_E+1])[0]*solidAngle
            healpixCube[i_E] *= counts
            counts_unc = quad(fluxUncInterp, self.bin_edges[i_E], self.bin_edges[i_E+1])[0]*solidAngle
            if counts != 0:
                val_unc = counts_unc/counts  # fractional change allowed in value
            else:
                val_unc = 1e20  # if the flux is zero, just make this bin unimportant by setting the error bars to inf.
            valueUnc.append(val_unc)

        # Add the template to the list.
        # IRFs are applied during add template.  This multiplies by cm^2 s
        self.AddTemplate(name='Isotropic', healpixCube=healpixCube, fixSpectrum=fixSpectrum,
                         fixNorm=fixNorm, limits=[None, None], value=1, ApplyIRF=True, sourceClass='ISO', valueUnc=valueUnc)

    def AddDMTemplate(self, profile='NFW', decay=False, gamma=1, axesratio=1, offset=(0, 0), r_s=20.,
                      spec_file=None, limits=[None, None],size=60.):
        """
        Generates a dark matter template and adds it to the current template stack.

        :param profile: 'NFW', 'Ein', or 'Bur'
        :param decay: If false, skymap is for annihilating dark matter
        :param gamma: Inner slope of DM profile for NFW.  Shape parameter for Einasto. Unused for Burk
        :param axesratio: Stretch the *projected* dark matter profile along the +y axis
        :param offset: offsets from (glon,glat)=(0,0) in degrees
        :param r_s: Scale factor
        :param spec_file: 2-column ascii file.  First column is E in MeV, second column is dN/dE per annihilation
                in units ph/MeV.  The differential spectrum dN/dE will then be integrated over each energy bin.
        :param limits: Limiting range for template normalization.  This usually does not need to be changed since '
                    the template will be automatically renormalized to have a realistic flux in the max pixel.
        :return: A healpix 'cube'. 2-dimensions: 1st is energy second is healpix pixel.  If spectrum is not supplied
                this is redundent.
        """


        # Generate the DM template.  This gives units in J-fact so we divide by something reasonable for the fit.
        # Say the max value.
        tmp = DM.GenNFW(nside=self.nside, profile=profile, decay=decay, gamma=gamma, axesratio=axesratio, rotation=0.,
                        offset=offset, r_s=r_s, mult_solid_ang=True,size=size)

        self.jfactor = np.sum(tmp*self.mask)

        # Just get an idea of the typical exposure aoround 1 GeV so we can prescale the fitting inputs
        exposure = Tools.GetExpMap(E_min=1e3, E_max=1e3, l=0., b=0., expcube=self.expCube, subsamples=1)


        # renormalize to b=5 degrees
        hpix_idx = Tools.ang2hpix(l=5.,b=5.,nside=self.nside)
        self.dm_renorm = 10./exposure/tmp[hpix_idx]*3.970529e-03

        tmp *= self.dm_renorm
        # Get as close as possible with start values.

        interp_energies = [324.03703492039307, 374.16573867739419, 424.2640687119287,
                           474.34164902525703, 527.94469228916375, 590.25685873722261,
                           663.83046421623033, 751.49059552671963, 856.99418582631142,
                           985.4208873128407, 1143.7679806383551, 1341.8800040696817,
                           1593.9455798525282, 1920.9916888346736, 2355.2088149126948,
                           2947.809536549059, 3784.1246893597918, 5014.6305456169475,
                           6924.3073867148269, 10105.305538385186, 15953.294664193767,
                           28418.14169014274, 62530.722509233441, 222864.51625256846]
        interp_vals = np.array([ 1.24738513e-01,   2.08671867e+01,   1.64419688e+01,
                                 3.42117678e+00,   2.05122617e+00,   2.19951630e+01,
                                 2.12563087e+01,   1.70642640e+01,   2.03808461e+01,
                                 1.77586504e+01,   1.79422137e+01,   1.80384238e+01,
                                 1.36724396e+01,   1.55134638e+01,   1.15991307e+01,
                                 1.01560749e+01,   8.78687207e+00,   6.67325932e+00,
                                 3.29639464e+00,   3.26640905e+00,   1.13648053e+00,
                                 9.12566623e-01,   3.62550033e-01,   1.69769179e-03])*np.array([120.50832071218372, 0.50544260402220376, 0.50269215354447727, 2.1982291034590968, 3.1800375465552153, 0.41759734422687789, 0.35581511608133187, 0.53834125119502851, 0.41882282691705752, 0.49581986277255824, 0.59661337163712325, 0.61080739355640457, 0.7443144976742635, 0.69448242983174446, 0.81671518607132365, 0.97517355786381898, 1.1233995847324556, 1.2384213058382529, 1.7765317070481652, 1.8696432782653931, 2.5742130977666853, 3.4623426965568389, 5.7954405589161926, 537.62256335695929])

        startvals = np.interp(self.central_energies, interp_energies, interp_vals)
        healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))

        if spec_file is None:
            # values = []
            for i in range(self.n_bins):
                 healpixcube[i] = tmp*startvals[i]*(self.bin_edges[i]/10e3)**-.5
            # Add the template.  Scale the values so that we get a roughly flat spectrum for faster convergence.
            self.AddTemplate(name='DM', healpixCube=healpixcube, fixSpectrum=False, fixNorm=False, limits=limits,
                             value=1., ApplyIRF=True, sourceClass='GEN')
        else:
            energies, dNdE = np.genfromtxt(spec_file).T[:2]
            spec = lambda e: np.interp(e, energies, dNdE)
            for i in range(self.n_bins):
                healpixcube[i] = tmp*quad(spec, self.bin_edges[i], self.bin_edges[i+1])[0]
            self.AddTemplate(name='DM', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False, limits=limits,
                             value=1., ApplyIRF=True, sourceClass='GEN')
        return tmp


    def GenFermiData(self, runscript=False):
        """
        Given the analysis properties, this will generate a script for the fermi data.

        :param scriptname: file location for the output script relative to the basepath.

        """

        scriptname = '/GenFermiData_' + self.tag + '_.sh'
        print "Run this script to generate the required fermitools files for this analysis."
        print "The script can be found at", self.basepath + scriptname

        GenFermiData.GenDataScipt(self.tag, self.basepath, self.bin_edges, scriptname, self.phfile_raw,
                                  self.scfile, self.evclass, self.convtype,  self.zmax, min(self.bin_edges),
                                  max(self.bin_edges), self.irf, self.gtfilter, self.evtype)
        if runscript:
            import sys
            import subprocess
            p = subprocess.Popen([self.basepath + scriptname, ], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
            # Grab stdout line by line as it becomes available.  This will loop until p terminates.
            while p.poll() is None:
                l = p.stdout.readline()  # This blocks until it receives a newline.
                sys.stdout.flush()
                print l.rstrip('\n')
            # When the subprocess terminates there might be unconsumed output
            # that still needs to be processed.
            print p.stdout.read()
            print p.stderr.read()



    def AddFermiDiffuseModel(self, diffuse_path, infile=None, outfile=None, multiplier=1., fixSpectrum=False):
        """
        Adds a fermi diffuse model to the template.  Input map is a fits file containing a cartesian mapcube.
        This gets resampled into a healpix cube, integrated over energy,
         applies PSF & effective exposure, and gets added to the templateList.
        :param diffuse_path: path to the diffuse model.
        :param infile: Save the template to this path (reduce initial load time)
        :param outfile: Save the template to this path (reduce initial load time) if infile is None

        :return: None
        """

        if infile is None:
            healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))

            # For each energy bin, sample and interpolate the map, then
            for i in range(self.n_bins):
                # SampleCartesianMap takes care of sub-binning.
                healpixcube[i] = Tools.SampleCartesianMap(fits=diffuse_path,
                                                          E_min=self.bin_edges[i], E_max=self.bin_edges[i+1],
                                                          nside=self.nside)
                print '\rRemapping Fermi Diffuse Model to Healpix Grid %.2f' % ((float(i+1)/self.n_bins)*100.), "%",

            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=fixSpectrum, fixNorm=False,
                             value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

            if outfile is not None:
                np.save(open(outfile, 'wb'), self.templateList['FermiDiffuse'].healpixCube)

        else:
            healpixcube = np.load(infile)
            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=fixSpectrum, fixNorm=False,
                             value=1., ApplyIRF=False, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

    def AddGalpropTemplate(self, basedir='/data/fermi_diffuse_models/galprop.stanford.edu/PaperIISuppMaterial/OUTPUT',
               tag='SNR_z4kpc_R20kpc_Ts150K_EBV2mag', verbosity=0, multiplier=1., bremsfrac=None, E_subsample=3,
               fixSpectrum=False, noPSF=False):
        """
        This method takes a base analysis prefix, along with an X_CO profile and generates the combined diffuse template,
        or components of the diffuse template.

        :param basedir: Base directory to read from
        :param tag: Tag for the galprop file.  This is the part between '_54_' and '.gz'.
        :param verbosity: 0 is quiet, >1 prints status.
        :param multiplier: Blur each map using Gaussian kernel with sigma=FWHM_PSF*multiplier/2
        :param bremsfrac: If None, brems is treated as independent.  Otherwise Brem normalization
            is linked to Pi0 normalization, scaled by a factor bremsfrac.
        :param E_subsample: Number of energy sub bins to use when integrating over each energy band.
        :param fixSpectrum: Allow the spectrum to float in each energy bin.
        :param noPSF: Do not apply PSF if True.  Can enhance speed.
        """

        #---------------------------------------------------------------------------------
        # Load templates

        if verbosity>0:
            print 'Loading FITS'

        energies = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[2].data.field(0)

        # For some reason, older versions of galprop files have slightly different data structures.  This try/except
        # will detect the right one to use.
        comps, comps_new = {}, {}
        try:
            comps['ics'] = pyfits.open(basedir+'/ics_isotropic_healpix_54_'+tag+'.gz')[1].data.field(0).T
            nside_in = np.sqrt(comps['ics'].shape[1]/12)
            comps['pi0'] = pyfits.open(basedir+'/pi0_decay_healpix_54_'+tag+'.gz')[1].data.field(0).T
            comps['brem'] = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[1].data.field(0).T

        except:
            def ReadFits(fname, length):
                d = pyfits.open(fname)[1].data
                return np.array([d.field(i) for i in range(length)])

            comps['ics'] = ReadFits(basedir+'/ics_isotropic_healpix_54_'+tag+'.gz', len(energies))
            nside_in = np.sqrt(comps['ics'].shape[1]/12)
            comps['brem'] = ReadFits(basedir+'/bremss_healpix_54_'+tag+'.gz', len(energies))
            comps['pi0'] = ReadFits(basedir+'/pi0_decay_healpix_54_'+tag+'.gz', len(energies))


        # Init new templates
        comps_new['ics'] = np.zeros((self.n_bins, 12*self.nside**2))
        comps_new['pi0'] = np.zeros((self.n_bins, 12*self.nside**2))
        comps_new['brem'] = np.zeros((self.n_bins, 12*self.nside**2))

        #---------------------------------------------------------------------------------
        # Now we integrate each model over the energy bins...
        #
        # Multiprocessing for speed. There is an async callback which applies each result to
        # the arrays.  Not sure why RunAsync needs new thread pool for each component, but this
        # works and decreases memory footprint.
        def callback(result):
            idx, comp, dat = result
            comps_new[comp][idx] = dat

        def RunAsync(component):
            p = mp.Pool(mp.cpu_count())
            for i_E in range(self.n_bins):
                p.apply_async(Tools.AsyncInterpolateHealpix,
                              [comps[component], energies, self.bin_edges[i_E], self.bin_edges[i_E+1],
                               i_E, component, E_subsample, self.nside],
                              callback=callback)
            p.close()
            p.join()

        # For each component, run the async sampling/sizing.
        for key in comps:
            if verbosity>0:
                print 'Integrating and Resampling', key, 'templates...'
                sys.stdout.flush()
            RunAsync(key)


        #---------------------------------------------------------------------------------
        # Now we just need to add the templates to the active template stack

        # Delete previous keys for diffuse model
        for key in ['Brems', 'Pi0', 'ICS', 'FermiDiffuse', 'Pi0_Brems']:
            self.templateList.pop(key, None)


        self.AddTemplate(name='ICS', healpixCube=comps_new['ics'], fixSpectrum=fixSpectrum, fixNorm=False,
                           value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)

        if bremsfrac is None:
            self.AddTemplate(name='Brems', healpixCube=comps_new['brem'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)
            self.AddTemplate(name='Pi0', healpixCube=comps_new['pi0'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)

        else:
            self.AddTemplate(name='Pi0_Brems', healpixCube=comps_new['pi0']+bremsfrac*comps_new['brem'],
                               fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)


    def AddHDF5Template(self, hdf5file, verbosity=0, multiplier=1., bremsfrac=None, E_subsample=3,
               fixSpectrum=False, noPSF=False, separate_ics=True, fix_ics=False, fix_brem=False):
        """
        This method takes a base analysis prefix, along with an X_CO profile and generates the combined diffuse template,
        or components of the diffuse template.

        :param basedir: Base directory to read from
        :param tag: Tag for the galprop file.  This is the part between '_54_' and '.gz'.
        :param verbosity: 0 is quiet, >1 prints status.
        :param multiplier: Blur each map using Gaussian kernel with sigma=FWHM_PSF*multiplier/2
        :param bremsfrac: If None, brems is treated as independent.  Otherwise Brem normalization
            is linked to Pi0 normalization, scaled by a factor bremsfrac.
        :param E_subsample: Number of energy sub bins to use when integrating over each energy band.
        :param fixSpectrum: Allow the spectrum to float in each energy bin.
        :param noPSF: Do not apply PSF if True.  Can enhance speed.
        :param separate_ics: If true, the CMB template and optical+far-infrared are treated as two templates.
            (normalization of OPT and FIR are linked)
        :param fix_ics: If true and separate_ics==False, the norm of the ICS template is fixed to the galprop prediction
        :param fix_brem: If true and bremsfrac==None, the norm of the brem template is fixed to the galprop prediction
        """

        #---------------------------------------------------------------------------------
        # Load templates

        if verbosity>0:
            print 'Loading HDF5 file'

        h5 = h5py.File(hdf5file, 'r')
        energies = h5['/templates/energies'][()]

        # For some reason, older versions of galprop files have slightly different data structures.  This try/except
        # will detect the right one to use.
        comps, comps_new = {}, {}

        if separate_ics:
            comps['ics_cmb'] = h5['/templates/ics_cmb'][()]
            comps['ics_optfir'] = h5['/templates/ics_opt'][()]+h5['/templates/ics_fir'][()]
            comps_new['ics_cmb'] = np.zeros((self.n_bins, 12*self.nside**2))
            comps_new['ics_optfir'] = np.zeros((self.n_bins, 12*self.nside**2))
        else:
            comps['ics'] = h5['/templates/ics_cmb'][()] + h5['/templates/ics_opt'][()]+h5['/templates/ics_fir'][()]
            comps_new['ics'] = np.zeros((self.n_bins, 12*self.nside**2))

        comps['pi0'] = h5['/templates/pi0'][()]
        comps['brem'] = h5['/templates/brem'][()]
        comps_new['pi0'] = np.zeros((self.n_bins, 12*self.nside**2))
        comps_new['brem'] = np.zeros((self.n_bins, 12*self.nside**2))

        #---------------------------------------------------------------------------------
        # Now we integrate each model over the energy bins...
        #
        # Multiprocessing for speed. There is an async callback which applies each result to
        # the arrays.  Not sure why RunAsync needs new thread pool for each component, but this
        # works and decreases memory footprint.
        def callback(result):
            idx, comp, dat = result
            comps_new[comp][idx] = dat

        def RunAsync(component):
            #p = mp.Pool(mp.cpu_count())
            for i_E in range(self.n_bins):
                # p.apply_async(Tools.AsyncInterpolateHealpix,
                #               [comps[component], energies, self.bin_edges[i_E], self.bin_edges[i_E+1],
                #                i_E, component, E_subsample, self.nside],
                #               callback=callback)

                comps_new[component][i_E] = Tools.InterpolateHealpix(comps[component], energies,  self.bin_edges[i_E],
                                                                     self.bin_edges[i_E+1], E_bins=E_subsample,
                                                                     nside_out=self.nside)
            # p.close()
            # p.join()

        # For each component, run the async sampling/sizing.
        for key in comps:
            if verbosity > 0:
                print 'Integrating and Resampling', key, 'templates...'
                sys.stdout.flush()
            RunAsync(key)


        #---------------------------------------------------------------------------------
        # Now we just need to add the templates to the active template stack

        # Delete previous keys for diffuse model
        for key in ['Brems', 'Pi0', 'ICS', 'ICS_CMB', 'ICS_OPTFIR', 'FermiDiffuse', 'Pi0_Brems']:
            self.templateList.pop(key, None)

        if separate_ics:
            self.AddTemplate(name='ICS_CMB', healpixCube=comps_new['ics_cmb'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)

            self.AddTemplate(name='ICS_OPT_FIR', healpixCube=comps_new['ics_optfir'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)
        else:
            self.AddTemplate(name='ICS', healpixCube=comps_new['ics'], fixSpectrum=fixSpectrum, fixNorm=fix_ics,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)

        if bremsfrac is None:
            self.AddTemplate(name='Brems', healpixCube=comps_new['brem'], fixSpectrum=fixSpectrum, fixNorm=fix_brem,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)
            self.AddTemplate(name='Pi0', healpixCube=comps_new['pi0'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)

        else:
            self.AddTemplate(name='Pi0_Brems', healpixCube=comps_new['pi0']+bremsfrac*comps_new['brem'],
                               fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1., ApplyIRF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier, noPSF=noPSF)


    def RunLikelihood(self, print_level=0, use_basinhopping=False, start_fresh=False, niter_success=50, tol=1e2,
                      precision=None, error=0.1, minos=True, force_cpu=False, statistic='Poisson', clip_model=False):
        """
        Runs the likelihood analysis on the current templateList.

        :param print_level: 0=Quiet, 1=Verbosse
        :param use_basinhopping: Refine migrad optimization with a basinhopping algortithm. Generally recommended.
        :param start_fresh: Skip migrad and just use basinhopping.
        :param niter_success: Number of successful iterations required before stopping basinhopping.
        :param tol: Tolerance for EDM in migrad convergence.
        :param precision: Migrad internal precision override.
        :param error: Migrad initial param error to use.
        :param minos: If true, runs minos to determine fitting errors. False uses Hesse method.
        :param statistic: 'Poisson' is only one supprted now. 'Gaussian' does not work correctly yet.
        :param clip_model: If true, negative values of the model are converted to a very small number to help convergence. 
                           This *is* unphysical so results should be double checked. 
        
        :returns m, res: m is an iMinuit object and res is a scipy minimization object.
        """

        # First, check whether all bins need to be fit simultaneously, or can be decoupled.
        binbybin = True
        for key, t in self.templateList.items():
            if ((not t.fixNorm) and t.fixSpectrum):
                binbybin = False

        if not binbybin:
                self.m, self.res, self.res_vals = GammaLikelihood.RunLikelihood(self, print_level=print_level,
                                                                                use_basinhopping=use_basinhopping,
                                                                                start_fresh=start_fresh,
                                                                                niter_success=niter_success,
                                                                                tol=tol,
                                                                                precision=precision,
                                                                                error=error,
                                                                                force_cpu=force_cpu)
                # -------------------------------------------------------------
                # UPDATE THE TEMPLATE VALUES AND ERRORS WITH THE FIT RESULTS
                for key, t in self.templateList.items():
                    if t.fixSpectrum:
                        if self.res_vals is not None:
                            t.value = self.res_vals[key]
                        else:
                            t.value = self.m.values[key]
                            t.valueError = self.m.errors[key]

                    else:
                        if self.res is not None:
                            t.value = np.array([self.res_vals[key+'_'+str(i)] for i in range(self.n_bins)])
                        else:
                            t.value = np.array([self.m.values[key+'_'+str(i)] for i in range(self.n_bins)])
                            t.valueError = np.array([self.m.errors[key+'_'+str(i)] for i in range(self.n_bins)])

        # Otherwise, Run the bin-by-bin fit.
        else:
            results = [GammaLikelihood.RunLikelihoodBinByBin(bin=i, analysis=self, print_level=print_level,
                                                             error=error, precision=precision, tol=tol,
                                                             statistic=statistic, clip_model=clip_model)
                       for i in range(self.n_bins)]


            for i_E in range(self.n_bins):
                if results[i_E].get_fmin().is_valid is False:
                    print "Warning: Bin", i_E, 'fit did not converge. Trying with lower limit at -10,10'
                    
                    for key, t in self.templateList.items():
                        t.limits = [0,10]
                    if key == 'DM':
                        t.limits = [-10,10]                        
                    
                    results[i_E] = GammaLikelihood.RunLikelihoodBinByBin(bin=i_E, analysis=self, print_level=print_level,
                                                                         error=error, precision=precision, tol=tol,
                                                                         statistic=statistic, clip_model=clip_model)
                
                # if results[i_E].get_fmin().is_valid is False:
                #     print "Warning: Bin", i_E, 'fit did not converge. Trying with lower limit at 0,10'
                #     for key, t in self.templateList.items():
                #         t.limits = [0,10]
                #     results[i_E] = GammaLikelihood.RunLikelihoodBinByBin(bin=i_E, analysis=self, print_level=print_level,
                #                                                          error=error, precision=precision, tol=tol,
                #                                                          statistic=statistic)



            # Calculate Fitting Errors.
            hesseList, minosList = [], []
            for m in results:
                try:
                    if minos:
                        minosList.append(m.minos())
                    else:
                        minosList.append(None)
                except:  # Sometimes we don't converge...
                    minosList.append(None)
                try:
                    hesseList.append(m.hesse())
                except:  # Sometimes we don't converge...
                    hesseList.append(None)

            # -------------------------------------------------------------
            # UPDATE THE TEMPLATE VALUES AND ERRORS WITH THE FIT RESULTS
            for key, t in self.templateList.items():
                if t.fixNorm:
                    continue
                else:
                    t.value = np.zeros(self.n_bins)
                    t.valueError = []
                    self.loglike = []

            for i_E in range(self.n_bins):
                if results[i_E].get_fmin().is_valid is False:
                    print 'WARNING: Fit for bin', i_E, 'is reported invalid'

                # Append the log-likelihood of each bin.
                self.loglike.append(results[i_E].fval)

                if minosList[i_E] is None:
                    if hesseList[i_E] is None:
                        if type(t.valueError) == type(0.):
                            t.valueError = []
                        t.valueError.append(np.array((1e10, 1e10)))
                        continue
                    for h in hesseList[i_E]:
                        name = "_".join(h['name'].split('_')[:-1])
                        if h['is_fixed']:
                            continue
                        t = self.templateList[name]

                        t.value[i_E] = h['value']
                        t.valueError.append(np.array((h['error'], h['error'])))
                else:
                    for key, err in minosList[i_E].items():
                        name = "_".join(key.split('_')[:-1])
                        t = self.templateList[name]
                        t.value[i_E] = err['min']
                        t.valueError.append(np.array((err['lower'], err['upper'])))

            #t.valueError = np.array(t.valueError)
            # END UPDATE TEMPLATE SECTION
            # -------------------------------------------------------------


        self.fitted = True
        self.residual = self.GetResidual()

        return self.m, self.res, self.res_vals

    def GetResidual(self):
        """
        Obtain the residual by subtracting off the best fit components.

        :return healpixcube: a healpix image of the residuals.
        """
        if not self.fitted:
            raise Exception('No fit run, or template added since last fit. Call "RunLikelihood()"')

        # Start with a copy of the binned photons and iterate through each template.
        residual = copy.copy(self.binned_data).astype(np.float32)
        for key, t in self.templateList.items():
            # Make sure this template has been fit already
            if t.fixSpectrum or t.fixNorm:

                if t.sourceClass == 'FGL':
                    #print t.healpixCube.toarray().shape
                    residual -= t.value*t.healpixCube.toarray()[0]*self.mask
                else:
                    residual -= t.value*t.healpixCube*self.mask
            else:
                for i_E in range(self.n_bins):
                    if t.sourceClass == 'FGL':
                        residual[i_E] -= t.value[i_E]*t.healpixCube.toarray()[i_E]*self.mask
                    else:
                        residual[i_E] -= t.value[i_E]*t.healpixCube[i_E]*self.mask

        return residual

    def GetSpectrum(self, name):
        """
        Given a template name, calculates the spectrum averaged over the unmasked area in each energy bin.

        :params name: name of template.  Use 'Data' to get the data spectrum
        :returns E, flux, stat_errors: E is the logarithmic center of each energy bin
                                       flux is the flux averaged over the region in [s^-1 cm^-2 sr^-1 MeV^-1]
                                       stat_errors is the statistical error on the flux.
        """
        if (name not in self.templateList) and (name is not 'Data'):
            raise KeyError("name '" + name + "' not in templateList.")

        if not self.fitted:
            print 'Warning! Template fitting has not been done. returned spectrum may equal input spectrum.'

        # Run through the template and obtain the spectrum in total counts over the masked area
        mask_idx = np.nonzero(self.mask)[0]

        # If looking for the data spectrum we need to temporarily create a template.
        if name is 'Data':
            t = Template.Template(healpixCube=self.binned_data)
        else:
            t = self.templateList[name]

        if t.valueError is None and t.fixNorm is False and name is not 'Data':
            print 'Warning! No fitting errors for component', name

        flux, errors = [], []
        # Iterate over each energy bin
        for i_E in range(self.n_bins):

            # Get energy bin boundaries
            E_min, E_max = self.bin_edges[i_E], self.bin_edges[i_E+1]

            if self.expMap[i_E] is not None:
                eff_area = self.expMap[i_E][mask_idx] * (E_max-E_min) * healpy.pixelfunc.nside2pixarea(self.nside)
            else:
                # Get the effective area for the masked region.
                l, b = Tools.hpix2ang(mask_idx, self.nside)
                # eff_area*bin width*solid_angle
                eff_area = (Tools.GetExpMap(E_min, E_max, l, b, self.expCube)
                            * (E_max-E_min)
                            * healpy.pixelfunc.nside2pixarea(self.nside))
            # if value is not a vector
            if np.ndim(t.value) == 0:
                #stat_error = (np.sqrt(np.sum(t.healpixCube[i_E][mask_idx])*t.value)
                #              / np.average(eff_area)/len(mask_idx))  # also divide by num pixels.
                stat_error = 0 # Now that we are using iMinuit minos etc.. instead of just poisson errors.

                if t.sourceClass == 'FGL':
                    count = np.average(t.healpixCube.toarray()[i_E][mask_idx]/eff_area)*t.value
                else:

                    count = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.value
                # No fit errors on data.
                if name is not 'Data' and t.fixNorm is False:
                    try:
                        if t.sourceClass == 'FGL':
                            fit_error = np.average(t.healpixCube.toarray()[i_E][mask_idx]/eff_area)*t.valueError
                        else:
                            fit_error = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.valueError
                    except:
                        fit_error = 0.
                if t.fixNorm:
                    fit_error = 0.

            # if value is a vector
            elif np.ndim(t.value) == 1 and len(t.value) == self.n_bins:

                #stat_error = (np.sqrt(np.sum(t.healpixCube[i_E][mask_idx])*t.value[i_E])
                #              / np.average(eff_area)/len(mask_idx))  # also divide by num pixels.
                stat_error = 0 # Now that we are using iMinuit minos etc.. instead of just poisson errors.

                if name is not 'Data':

                        try:
                            if t.sourceClass == 'FGL':
                                fit_error = np.average(t.healpixCube.toarray()[i_E][mask_idx]/eff_area)*t.valueError[i_E]
                            else:
                                fit_error = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.valueError[i_E]

                        except:
                            fit_error = 0
                if t.fixNorm:
                    fit_error = 0.
                if t.sourceClass == 'FGL':
                    count = np.average(t.healpixCube.toarray()[i_E][mask_idx]/eff_area)*t.value[i_E]
                else:
                    count = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.value[i_E]
            else:
                raise Exception("template.value attribute has invalid dimension or type.")
            flux.append(count)
            if name is 'Data':
                errors.append(np.array(stat_error))
            else:
                if np.ndim(fit_error) == 0:
                    err = np.sqrt(np.array(stat_error)**2+np.array(fit_error)**2)
                    errors.append((err,err))
                if np.ndim(fit_error) == 1:
                    errors.append((np.sqrt(np.array(stat_error)**2+np.array(fit_error[0])**2),
                                            np.sqrt(np.array(stat_error)**2+np.array(fit_error[1])**2)))

        return self.central_energies, np.array(flux), np.array(errors)


    def SaveSpectra(self, fname=None):
        """
        Save the spectrum to a pickled dictionary.  For each key there are 3 components: Energy, Flux, FluxUnc
        :param fname: Filename for the saved file.  Defaults to './spec_'+ tag +'.pickle'
        :return: None
        """
        """
        :param fname:
        :return:
        """
        tmp = {}
        for key in self.templateList:
            tmp[key] = self.GetSpectrum(key)

        if fname is not None:
            pickle.dump(tmp, open(fname,'wb'))
        else:
            pickle.dump(tmp, open('spec_'+self.tag+'.pickle','wb'))


    def AddFermiBubbleTemplate(self, template_file='./bubble_templates_diskcut30.0.fits',
                               spec_file='./reduced_bubble_spec_apj_793_64.dat', fixSpectrum=False, fixNorm=False):
        """
        Adds a fermi bubble template to the template stack.

        :param template_file: Requires file 'bubble_templates_diskcut30.0.fits'
            style file (from Su & Finkbeiner) with an extension table with a NAME column containing "Whole bubble"
            and a TEMPLATE column with an order 8 healpix array.
        :param spec_file: filename containing three columns (no header).  First col is energy in MeV, second is
            dN/dE in units (s cm^2 sr MeV)^-1 third is the uncertainty in dN/dE in (s cm^2 sr MeV)^-1.
        :param fixSpectrum: If True, the spectrum is not allowed to float.
        """

        try:
            # Load the template and spectrum
            hdu = pyfits.open(template_file)
            bub_idx = np.where(hdu[1].data['NAME'] == 'Whole bubble')
            bubble = hdu[1].data['TEMPLATE'][bub_idx][0]
        except: 
            # The new bubble template
            hdu = pyfits.open(template_file)
            bubble = hdu[0].data

        # Resize template if need be.
        nside_in = int(np.sqrt(bubble.shape[0]/12))
        if nside_in != self.nside:
            bubble = Tools.ResizeHealpix(bubble, self.nside, average=True)

        energy, dnde, dnde_unc = np.genfromtxt(spec_file).T
        spec = lambda e: np.interp(e, energy, dnde)
        spec_unc = lambda e: np.interp(e, energy, dnde_unc)

        # Get lat/lon for each pixel
        l,b = Tools.hpix2ang(np.arange(12*self.nside**2))

        healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))
        valueUnc = []
        for i_E in range(self.n_bins):
            # Determine the counts in each bin.
            e1, e2 = self.bin_edges[i_E], self.bin_edges[i_E+1]
            # integrate spectrum over energy band
            flux = quad(spec, e1, e2)[0]
            flux_unc = quad(spec_unc, e1, e2)[0]

            if flux != 0:
                val_unc = flux_unc/flux  # fractional change allowed in value
            else:
                val_unc = 1e20  # if the flux is zero, just make this bin unimportant by setting the error barto inf.
            valueUnc.append(val_unc)

            # Multiply mask by counts.
            healpixcube[i_E] = bubble*flux*(healpy.nside2pixarea(self.nside))

        # Now each bin is in ph cm^-2 s^-1.  Apply IRF takes care of the rest.
        self.AddTemplate(healpixCube=healpixcube, name='Bubbles', fixSpectrum=fixSpectrum,
                         fixNorm=fixNorm, limits=[None, None], value=1., ApplyIRF=True,
                         sourceClass='GEN', valueUnc=valueUnc)



    def AddParabolicTemplate(self, scale_factor,
                               spec_file='./reduced_bubble_spec_apj_793_64.dat', fixSpectrum=False, fixNorm=False):
        """
        Adds a parabolic template 

        :param template_file: Requires file 'bubble_templates_diskcut30.0.fits'
            style file (from Su & Finkbeiner) with an extension table with a NAME column containing "Whole bubble"
            and a TEMPLATE column with an order 8 healpix array.
        :param spec_file: filename containing three columns (no header).  First col is energy in MeV, second is
            dN/dE in units (s cm^2 sr MeV)^-1 third is the uncertainty in dN/dE in (s cm^2 sr MeV)^-1.
        :param fixSpectrum: If True, the spectrum is not allowed to float.
        """

        hpix = np.arange(12*self.nside**2)
        l,b = Tools.hpix2ang(hpix)
        l[l>180]-=360
        template = np.zeros(12*self.nside**2)
        template[np.abs(b)>scale_factor*l**2] = 1

        energy, dnde, dnde_unc = np.genfromtxt(spec_file).T
        spec = lambda e: np.interp(e, energy, dnde)
        spec_unc = lambda e: np.interp(e, energy, dnde_unc)

        healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))
        valueUnc = []
        for i_E in range(self.n_bins):
            # Determine the counts in each bin.
            e1, e2 = self.bin_edges[i_E], self.bin_edges[i_E+1]
            # integrate spectrum over energy band
            flux = quad(spec, e1, e2)[0]
            flux_unc = quad(spec_unc, e1, e2)[0]

            if flux != 0:
                val_unc = flux_unc/flux  # fractional change allowed in value
            else:
                val_unc = 1e20  # if the flux is zero, just make this bin unimportant by setting the error barto inf.
            valueUnc.append(val_unc)

            # Multiply mask by counts.
            healpixcube[i_E] = template*flux*(healpy.nside2pixarea(self.nside))

        # Now each bin is in ph cm^-2 s^-1.  Apply IRF takes care of the rest.
        self.AddTemplate(healpixCube=healpixcube, name='Parabola', fixSpectrum=fixSpectrum,
                         fixNorm=fixNorm, limits=[None, None], value=1., ApplyIRF=True,
                         sourceClass='GEN', valueUnc=valueUnc)

    def GenExposureMap(self):
        """
        This is intended for precomputation of the exposure map.  It performs a more precise integration over energy
        bins and saves the result to file for quick use.

        :return: None
        """

        l, b = Tools.hpix2ang(np.arange(12*self.nside**2), nside=self.nside)
        healpixcube = np.zeros((self.n_bins, len(l)), dtype=np.float32)
        for i_E in range(self.n_bins):
            healpixcube[i_E] = Tools.GetExpMap(E_min=self.bin_edges[i_E], E_max=self.bin_edges[i_E+1],
                                               l=l, b=b, expcube=self.expCube, subsamples=5, spectral_index=-2.25)
            print '\rGenerating exposure map %.2f' % ((float(i_E+1)/self.n_bins)*100.), "%",
        np.save('expmap_'+self.tag+'.npy', healpixcube)
        self.expMap = healpixcube


    def CalculatePixelWeights(self, diffuse_model, psc_model, alpha_psc=5, f_psc=0.1):
        """
        Calculates pixel weights for the likelihood analysis based on Eqn. 2.6 of Calore et al (1409.0042).

        :param diffuse_model: Path to healpixcube of diffuse model. Should be one of the fermi models.  Can obtain this
            by using AddFermiDiffuseModel() with outfile.
        :param outfile: If infile is None then PixelWeights are output to this path.
        :param infile: reads the pixel weights from file.
        """

        diff_model = np.load(diffuse_model)
        psc = np.load(psc_model)

        self.psc_weights = 1./((psc/(f_psc*diff_model))**alpha_psc+1)


    def SaveFit(self, filename=None):
        '''
        Returns a dictionary of fit results and saves a pickle object to the specified filepath if not None.
        This contains the following entries:\n
        'loglike': A single value, or list of the log likelihoods for each energy bin.\n
        'energies': The central energy of each bin (in log-space).\n

        For each template in the fit there is also a key corresponding to the template name
         which has a dictionary value containing:\n
            'flux': dNdE in [s^-1 cm^-2 sr^-1 MeV^-1]\n
            'fluxunc': Statistical (and fitting) error on dNdE in same units (possibly 2-col asymmetric errors)\n
        :param filename: filename to save the dictionary (saved as a pickle object).
        :return: The dict saved above
        '''

        saveDict = {'loglike': self.loglike, 'energies': self.central_energies}

        for key, t in self.templateList.items():
            e, flux, fluxunc = self.GetSpectrum(key)
            saveDict[key] = {'flux': flux, 'fluxunc': fluxunc}
        # Also append the data.
        e, flux, fluxunc = self.GetSpectrum('Data')
        saveDict['Data'] = {'flux': flux, 'fluxunc': fluxunc}

        if filename is not None:
            pickle.dump(saveDict, open(filename, 'wb'))

        return saveDict

    def ResetFit(self):
        '''
        Resets the fit values and initial fitting errors back to the defaults.  If fits are not converging after changing
        out some templates, use this.  Does not change values for fixed templates.
        '''

        for key, t in self.templateList.items():
            if not t.fixNorm:
                t.value = 1.
                t.valueError = .1

    def SetLimits(self, limits=[None, None], value=1., valueError=.1, exceptionList=None):
        '''
        Resets the fit values limits to the parameter values  If fits are not converging after changing
        out some templates, use this.  Does not change values for fixed templates.
        '''

        for key, t in self.templateList.items():
            if exceptionList is not None:
                if key in exceptionList:
                    continue
            if not t.fixNorm:
                t.limits = limits
                # t.value = 1.
                # t.valueError = .1
