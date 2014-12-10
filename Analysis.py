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

class Analysis():
    #--------------------------------------------------------------------
    # Most binning settings follows Calore et al 2014 (1409.0042)
    #--------------------------------------------------------------------

    def __init__(self, E_min=5e2, E_max=5e5, nside=256, gamma=1.45, n_bins=20, prefix_bins=[300, 350, 400, 450, 500],
                    tag='P7REP_CLEAN_V15_test', basepath='/data/GCE_sys/',
                    phfile_raw='/data/fermi_data_1-8-14/phfile.txt',
                    scfile='/data/fermi_data_1-8-14/lat_spacecraft_merged.fits',
                    evclass=2, convtype=-1,  zmax=100, irf='P7REP_CLEAN_V15', fglpath='/data/gll_psc_v08.fit'):
        """
        :param    E_min:        Min energy for recursive spectral binning
        :param    E_max:        Max energt for recursive spectral binning
        :param    n_bins:       Number of recursive spectal bins. Specify zero if custom bins are supplied.
        :param    gamma:        Power-law index for recursive binning
        :param    nside:        Number of healpix spatial bins
        :param    prefix_bins:  manually specify bin edges. These are prepended to the recursive bins
        :param    tag:          an analysis tag that is included in generated files
        :param    basepath:     the base directory for relative paths
        :param    phfile_raw:   Photon file or list of files from fermitools (evfile you would input to gtselect)
        :param    scfile:       Merged spacecraft file
        :param    evclass:      Fermi event class (integer)
        :param    convtype:     Fermi conversion type (integer)
        :param    zmax:         zenith cut
        :param    irf:          Fermi irf name
        :param    fglpath:      Path to the 2FGL fits file
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
        self.convtype = convtype
        self.zmax = zmax
        self.irf = irf
        self.bin_edges = 0
        self.bin_edges = copy.copy(prefix_bins)

        self.phfile = basepath + '/photons_merged_cut_'+str(tag)+'.fits'
        self.psfFile = basepath + '/gtpsf_' + str(tag)+'.fits'
        self.expCube = basepath + '/gtexpcube2_' + str(tag)+'.fits'
        self.fglpath = fglpath
        
        # Currently Unassigned 
        self.binned_data = None  # master list of bin counts. 1st index is spectral, 2nd is pixel_number
        self.mask = None         # Mask. 0 is not analyzed. between 0-1 corresponds to weights. 
        self.templateList = {}   # Dict of analysis templates 
        self.fitted = False      # True if fit has been run with the current templateList.
        self.residual = None     # After a fit, this is automatically updated.
        self.dm_renorm = 1e19    # renormalization constant for DM template

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
                np.save(open(outfile, 'wb'), self.binned_data)
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
        l_pix, b_pix = Tools.hpix2ang(hpix=np.arange(12*self.nside**2),nside=self.nside)
        # Find elements that are masked
        idx = np.where(((l_pix < l_max) | (l_pix > (l_min+360)))
                       & (b_pix < b_max) & (b_pix > b_min)
                       & (np.abs(b_pix) > plane_mask))[0]
        mask[idx] = 1.  # Set unmasked elements to 1
        
        if merge is True:
            masked_idx = np.where(mask == 0)[0]
            self.mask[masked_idx] = 0
        else: 
            self.mask = mask
        return mask

    def ApplyIRF(self, hpix, E_min, E_max, noPSF=False):
        """
        Apply the Instrument response functions to the input healpix map. This includes the effective area and PSF.
        These quantities are automatically computed based on the spectral weighted average with spectrum from P7REP_v15
        diffuse model.

        :param    hpix: A healpix array.
        :param    E_min: low energy boundary
        :param    E_max: high energy boundary
        :param    noPSF: Do not apply the PSF (Exposure only)
        """
        # Apply the PSF.  This is automatically spectrally weighted
        if noPSF is False:
            hpix = Tools.ApplyGaussianPSF(hpix, E_min, E_max, self.psfFile) 
        # Get l,b for each healpix pixel 
        l, b = Tools.hpix2ang(np.arange(len(hpix)), nside=self.nside)
        # For each healpix pixel, multiply by the exposure. 
        hpix *= Tools.GetExpMap(E_min, E_max, l, b, expcube=self.expCube)

        return hpix

    def AddPointSourceTemplateFermi(self, pscmap='gtsrcmap_All_Sources.fits', name='PSC',
                                    fixSpectrum=False, fixNorm=False, limits=[0., 10.], value=1,):
        """
        Adds a point source map to the list of templates.  Cartesian input from gtsrcmaps is then converted
        to a healpix template.

        :param    pscmap: The point source map should be the output from gtsrcmaps in cartesian coordinates.
        :param    name:   Name to use for this template.
        :param    fixSpectrum: If True, the relative normalizations of each energy bin will be held fixed for this
                    template, but the overall normalization is free
        :param    fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
        :param    limits:      Specify range of template normalizations.
        :param    value:       Initial value for the template normalization.
        """
        # Convert the input map into healpix.
        hpix = Tools.CartesianCountMap2Healpix(cartCube=pscmap, nside=self.nside)[:-1]/1e9
        for i in range(len(hpix)):
            hpix[i] /= (float(self.bin_edges[i])/self.bin_edges[0])

        self.AddTemplate(name, hpix, fixSpectrum, fixNorm, limits, value, ApplyIRF=False, sourceClass='PSC')


    def AddPointSourceTemplate(self, pscmap=None, name='PSC', fixNorm=False,
                               limits=[0, 1e2], value=1):
        """
        Adds a point source map to the list of templates.

        :param    pscmap: Filename of the pscmap.  If none, assumes default value.
        :param    name:   Name to use for this template.
        :param    fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
        :param    limits:      Specify range of template normalizations.
        :param    value:       Initial value for the template normalization.
        """
        if pscmap is None:
            pscmap = self.basepath + '/PSC_' + self.tag + '.npy'
        try:
            hpix = np.load(open(pscmap, 'r'))
        except:
            raise Exception('No point source map found at path '+str(pscmap))


        self.AddTemplate(name, hpix, fixSpectrum=True, fixNorm=fixNorm, limits=limits, value=value,
                         ApplyIRF=False, sourceClass='PSC')


    def GenPointSourceTemplate(self, pscmap=None, onlyidx=None):
        """
        Generates a point source count map valid for the current analysis based on 2fgl catalog.  This can take a long
        time so it is usually done once and then saved.

        :param pscmap: Specify the point source map filename.  If None, then the default path
            self.basemap+'PSC_'+self.tag+'.npy' is used.
        :return PSCMap:  The healpix 'cube' for the point sources.
        """
        if pscmap is None:
            pscmap = self.basepath + '/PSC_' + self.tag + '.npy'

        reload(SourceMap)
        total_map = SourceMap.GenSourceMap(self.bin_edges, l_range=(-180, 180), b_range=(-90, 90),
                                           fglpath=self.fglpath,
                                           expcube=self.expCube,
                                           psffile=self.psfFile,
                                           maxpsf = 10.,
                                           res=0.125,
                                           nside=self.nside,
                                           filename=pscmap, onlyidx=onlyidx)

        return total_map

    def PrintTemplates(self):
        """
        Prints the names and properties of each template in the template list.
        """
        print '%20s' % 'NAME', '%25s' % 'LIMITS', '%10s' % 'VALUE', '%10s' % 'FIXNORM', '%10s' % 'FIXSPEC',
        print '%10s' % 'SRCCLASS'
        for key in self.templateList:
            temp = self.templateList[key]
            print '%20s' % key, '%25s' % temp.limits, '%10s' % temp.value, '%10s' % temp.fixNorm,
            print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass

    def AddTemplate(self, name, healpixCube, fixSpectrum=False, fixNorm=False, limits=[0, 1e5], value=1, ApplyIRF=True,
                    sourceClass='GEN'):
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
        """

        self.fitted = False

        # Error Checking on shape of input cube. 
        if (healpixCube.shape[0] != (len(self.bin_edges)-1)) or (healpixCube.shape[1] != (12*self.nside**2)):
            raise(Exception("Shape of template does not match binning"))

        if ApplyIRF:
            for i_E in range(len(self.bin_edges)-1):
                if sourceClass == 'ISO':
                    healpixCube[i_E] = self.ApplyIRF(healpixCube[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1],
                                                     noPSF=True)
                else:
                    healpixCube[i_E] = self.ApplyIRF(healpixCube[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1])

        # Instantiate the template object. 
        template = Template.Template(healpixCube.astype(np.float32), fixSpectrum, fixNorm, limits, value, sourceClass)
        # Add the instance to the master template list.
        self.templateList[name] = template

    def RemoveTemplate(self, name):
        """
        Removes a template from the template list.

        :param    name: Name of template to remove.
        """
        self.templateList.pop(name)

    def AddIsotropicTemplate(self, isofile='iso_clean_v05.txt', fixNorm=True):
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
        from scipy.integrate import quad
        for i_E in range(len(self.bin_edges)-1):
            # multiply by integral(flux *dE )*solidangle
            healpixCube[i_E] *= quad(fluxInterp, self.bin_edges[i_E], self.bin_edges[i_E+1])[0]*solidAngle
        
        # Add the template to the list. 
        # IRFs are applied during add template.  This multiplies by cm^2 s
        self.AddTemplate(name='Isotropic', healpixCube=healpixCube, fixSpectrum=True,
                        fixNorm=fixNorm, limits=[0, 5], value=1, ApplyIRF=True, sourceClass='ISO')

        # TODO: NEED TO DEAL WITH UNCERTAINTY VECTOR in likelihood fit.

    def AddDMTemplate(self, profile='NFW', decay=False, gamma=1, axesratio=1, offset=(0, 0), r_s=20., spectrum=None):
        """
        Generates a dark matter template and adds it to the current template stack.

        :param profile: 'NFW', 'Ein', or 'Bur'
        :param decay: If false, skymap is for annihilating dark matter
        :param gamma: Inner slope of DM profile for NFW.  Shape parameter for Einasto. Unused for Burk
        :param axesratio: Stretch the *projected* dark matter profile along the +y axis
        :param offset: offsets from (glon,glat)=(0,0) in degrees
        :param r_s: Scale factor
        :param spectrum: Input vector for normalizations the DM spectrum should be fixed (needs to be
                pre-integrated over bins).
        :return: A healpix 'cube'. 2-dimensions: 1st is energy second is healpix pixel.  If spectrum is not supplied
                this is redundent.
        """
        # TODO: Allow spectrum input via file. e.g. (E, dNdE) ASCII file which will then be integrated in each bin.

        # Generate the DM template.  This gives units in J-fact so we divide by something reasonable for the fit.
        # Say the max value.
        tmp = DM.GenNFW(nside=self.nside, profile=profile, decay=decay, gamma=gamma, axesratio=axesratio, rotation=0.,
                        offset=offset, r_s=r_s, mult_solid_ang=True)/self.dm_renorm

        healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))

        if spectrum is None:
            for i in range(self.n_bins):
                healpixcube[i] = tmp
            self.AddTemplate(name='DM', healpixcube=healpixcube, fixSpectrum=False, fixNorm=False, limits=[0,1e5],
                             value=1, ApplyIRF=True, sourceClass='GEN')
        else:
            for i in range(self.n_bins):
                healpixcube[i] = tmp
            self.AddTemplate(name='DM', healpixcube=healpixcube, fixSpectrum=True, fixNorm=False, limits=[0,1e5],
                             value=1, ApplyIRF=True, sourceClass='GEN')
        return tmp


    def GenFermiData(self, scriptname):
        """
        Given the analysis properties, this will generate a script for the fermi data.

        :param scriptname: file location for the output script relative to the basepath.

        """

        scriptname = self.basepath + '/' +  scriptname

        print "Run this script to generate the required fermitools files for this analysis."
        print "The script can be found at", scriptname

        print GenFermiData.GenDataScipt(self.tag, self.basepath, self.bin_edges, scriptname, self.phfile_raw,
                                        self.scfile, self.evclass, self.convtype,  self.zmax, self.E_min, self.E_max,
                                        self.irf)

    def AddFermiDiffuseModel(self, diffuse_path, infile=None, outfile=None):
        """
        Adds a fermi diffuse model to the template.  Input map is a fits file containing a cartesian mapcube.
        This gets resampled into a healpix cube, integrated over energy,
         applies PSF & effective exposure, and gets added to the templateList.
        :param diffuse_path: path to the diffuse model.
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
            if outfile is not None:
                np.save(open(outfile, 'wb'), healpixcube)

            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False,
                             value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 5.])

        else:
            healpixcube = np.load(infile)
            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False,
                             value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 5.])

    def RunLikelihood(self, print_level=0, use_basinhopping=False, start_fresh=False, niter_success=30):
        """
        Runs the likelihood analysis on the current templateList.

        :param print_level: 0=Quiet, 1=Verbosse
        :param use_basinhopping: Refine migrad optimization with a basinhopping algortithm. Generally recommended.
        :param start_fresh: Skip migrad and just use basinhopping.
        :param niter_success: Number of successful iterations required before stopping basinhopping.
        :returns m, res: m is an iMinuit object and res is a scipy minimization object.
        """
        self.m, self.res = GammaLikelihood.RunLikelihood(self, print_level=print_level,
                                                         use_basinhopping=use_basinhopping,
                                                         start_fresh=start_fresh, niter_success=niter_success)
        
        # Need mapping to list output of res. m.values is a dict for looping over
        # this gives the correct order corresponding to res.x.  This is bad python practice, but it works....
        val_idx = {}
        count = 0
        for key in self.m.values:
            val_idx[key] = count
            count += 1

        # Run through the templates and update values to the best fit.
        for key, t in self.templateList.items():
            if t.fixSpectrum:
                if self.res is not None:
                    t.value = self.res.x[val_idx[key]]
                else:
                    t.value = self.m.values[key]
            else:
                if self.res is not None:
                    t.value = np.array([self.res.x[val_idx[key+'_'+str(i)]] for i in range(self.n_bins)])
                else:
                    t.value = np.array([self.m.value[key+'_'+str(i)] for i in range(self.n_bins)])

        self.fitted = True
        self.residual = self.GetResidual()

        return self.m, self.res

    def GetResidual(self):
        """
        btain the residual by subtracting off the best fit components.

        :return healpixcube: a healpix image of the residuals.
        """
        if not self.fitted:
            raise Exception('No fit run, or template added since last fit. Call "RunLikelihood()"')

        # Start with a copy of the binned photons and iterate through each template.
        residual = copy.copy(self.binned_data)
        for key, t in self.templateList.items():
            # Make sure this template has been fit already
            if t.fixSpectrum:
                residual -= t.value*t.healpixCube
            else:
                for i_E in range(self.n_bins):
                    residual[i_E] -= t.value[i_E]*t.healpixCube[i_E]

        return residual*self.mask

    # TODO: ADD INTERFACES TO GALPROP MAPS
    # TODO: Add calculate point source map weights for fitting..










