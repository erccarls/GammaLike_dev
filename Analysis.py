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

class Analysis():
    #--------------------------------------------------------------------
    # Most binning settings follows Calore et al 2014 (1409.0042)
    #--------------------------------------------------------------------

    def __init__(self, E_min=5e2, E_max=5e5, nside=256, gamma=1.45, n_bins=20, prefix_bins=[300, 350, 400, 450, 500],
                    tag='P7REP_CLEAN_V15_calore', basepath='/data/GCE_sys/',
                    phfile_raw='/data/fermi_data_1-8-14/phfile.txt',
                    scfile='/data/fermi_data_1-8-14/lat_spacecraft_merged.fits',
                    evclass=2, convtype=-1,  zmax=100, irf='P7REP_CLEAN_V15', fglpath='/data/gll_psc_v08.fit',
                    gtfilter="DATA_QUAL>0 && LAT_CONFIG==1 && ABS(ROCK_ANGLE)<52"):
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
        :param    gtfilter:       Filter string to pass to gtselect.
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
        self.gtfilter = gtfilter

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
        self.m = None            # iMinuit results object
        self.res = None          # scipy minimizer results object.
        self.res_vals = None     # dict with best fit values for basinhopping minimizer.
        self.jfactor = 0.        # Dark matter j-factor
        self.psc_weights = None  # Pixel weighting for likelihood analysis



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

    # def ApplyIRF(self, hpix, E_min, E_max, noPSF=False, noExp=False, multiplier=1.):
    #     """
    #     Apply the Instrument response functions to the input healpix map. This includes the effective area and PSF.
    #     These quantities are automatically computed based on the spectral weighted average with spectrum from P7REP_v15
    #     diffuse model.
    #
    #     :param    hpix: A healpix array.
    #     :param    E_min: low energy boundary
    #     :param    E_max: high energy boundary
    #     :param    noPSF: Do not apply the PSF
    #     :param    noExp: Do not apply the effective exposure
    #     :param    multiplier: Sigma = multiplier*FWHM from fermi gtpsf.
    #     """
    #     # Apply the PSF.  This is automatically spectrally weighted
    #     if noPSF is False:
    #         hpix = Tools.ApplyGaussianPSF(hpix, E_min, E_max, self.psfFile, multiplier=multiplier)
    #     # Get l,b for each healpix pixel
    #     l, b = Tools.hpix2ang(np.arange(len(hpix)), nside=self.nside)
    #     # For each healpix pixel, multiply by the exposure.
    #     if noExp is False:
    #         hpix *= Tools.GetExpMap(E_min, E_max, l, b, expcube=self.expCube,)
    #     return hpix

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
                               limits=[0, 1e2], value=1, multiplier=1.):
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

        # TODO: Adaptively set l_range and b_range based on current mask? At least make entire map.
        total_map = SourceMap.GenSourceMap(self.bin_edges, l_range=(-30, 30), b_range=(-30, 30),
                                           fglpath=self.fglpath,
                                           expcube=self.expCube,
                                           psffile=self.psfFile,
                                           maxpsf = 7.5,
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
            if np.ndim(temp.value) == 0:
                print '%20s' % key, '%25s' % temp.limits, '%10s' % ('%3.3e' % temp.value), '%10s' % temp.fixNorm,
                print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
            else:
                print '%20s' % key, '%25s' % temp.limits, '%10s' % 'Vector', '%10s' % temp.fixNorm,
                print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
                print '         --------------------------------------------------------------------------------------'

                for i, val in enumerate(temp.value):
                    print '%20s' % ('[' + str(i) + ']'), '%25s' % temp.limits, '%10s' % ('%3.3e' % val), '%10s' % temp.fixNorm,
                    print '%10s' % temp.fixSpectrum, '%10s' % temp.sourceClass
                print '         --------------------------------------------------------------------------------------'


    def AddTemplate(self, name, healpixCube, fixSpectrum=False, fixNorm=False, limits=[0, 1e5], value=1, ApplyIRF=True,
                    sourceClass='GEN', multiplier=1., valueUnc=None):
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
        :param    multiplier: Sigma = multiplier*FWHM from fermi gtpsf.
        """

        self.fitted = False

        # Error Checking on shape of input cube. 
        if (healpixCube.shape[0] != (len(self.bin_edges)-1)) or (healpixCube.shape[1] != (12*self.nside**2)):
            raise(Exception("Shape of template does not match binning"))

        if ApplyIRF:
            for i_E in range(len(self.bin_edges)-1):
                if sourceClass == 'ISO':
                    healpixCube[i_E] = Tools.ApplyIRF(healpixCube[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1],
                                                      self.psfFile, self.expCube, noPSF=True)
                else:
                    # This can be expensive if applying the PSF due to spherical harmonic transforms.
                    # This is already multithreaded in healpy.
                    healpixCube[i_E] = Tools.ApplyIRF(healpixCube[i_E], self.bin_edges[i_E], self.bin_edges[i_E+1],
                                                      self.psfFile, self.expCube, multiplier=multiplier)




        # Instantiate the template object. 
        template = Template.Template(healpixCube.astype(np.float32), fixSpectrum, fixNorm, limits, value, sourceClass,
                                     valueUnc)
        # Add the instance to the master template list.
        self.templateList[name] = template

    def DeleteTemplate(self, name):
        """
        Removes a template from templateList if it exists.

        :param name: Name of template to delete.
        """
        self.templateList.pop(name, None)

    def AddIsotropicTemplate(self, isofile='./IGRB_ackerman_2014_modA.dat', fixNorm=False):
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
        self.AddTemplate(name='Isotropic', healpixCube=healpixCube, fixSpectrum=True,
                        fixNorm=fixNorm, limits=[0, 5.0], value=1, ApplyIRF=True, sourceClass='ISO', valueUnc=valueUnc)

        # TODO: NEED TO DEAL WITH UNCERTAINTY VECTOR in likelihood fit.

    def AddDMTemplate(self, profile='NFW', decay=False, gamma=1, axesratio=1, offset=(0, 0), r_s=20.,
                      spec_file=None, limits=[0, 50.]):
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
                        offset=offset, r_s=r_s, mult_solid_ang=True)

        self.jfactor = np.sum(tmp*self.mask)

        exposure = Tools.GetExpMap(E_min=1e3, E_max=2e3, l=0., b=0., expcube=self.expCube)
        self.dm_renorm = 10./exposure/np.max(tmp)
        tmp *= self.dm_renorm

        healpixcube = np.zeros(shape=(self.n_bins, 12*self.nside**2))

        if spec_file is None:
            values = []
            for i in range(self.n_bins):
                healpixcube[i] = tmp
                values.append((self.bin_edges[i]/10e3)**-.5)
            # Add the template.  Scale the values so that we get a roughly flat spectrum for faster convergence.
            self.AddTemplate(name='DM', healpixCube=healpixcube, fixSpectrum=False, fixNorm=False, limits=limits,
                             value=values, ApplyIRF=True, sourceClass='GEN')
        else:
            energies, dNdE = np.genfromtxt(spec_file).T[:2]
            spec = lambda e: np.interp(e, energies, dNdE)
            for i in range(self.n_bins):
                healpixcube[i] = tmp*quad(spec, self.bin_edges[i], self.bin_edges[i+1])[0]
            self.AddTemplate(name='DM', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False, limits=limits,
                             value=1, ApplyIRF=True, sourceClass='GEN')
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
                                  max(self.bin_edges), self.irf, self.gtfilter)
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



    def AddFermiDiffuseModel(self, diffuse_path, infile=None, outfile=None, multiplier=1.):
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

            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False,
                             value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 5.], multiplier=multiplier)

            if outfile is not None:
                np.save(open(outfile, 'wb'), self.templateList['FermiDiffuse'].healpixCube)

        else:
            healpixcube = np.load(infile)
            self.AddTemplate(name='FermiDiffuse', healpixCube=healpixcube, fixSpectrum=True, fixNorm=False,
                             value=1, ApplyIRF=False, sourceClass='GEN', limits=[0, 5.], multiplier=multiplier)

    def AddGalpropTemplate(self, basedir='/data/fermi_diffuse_models/galprop.stanford.edu/PaperIISuppMaterial/OUTPUT',
               tag='SNR_z4kpc_R20kpc_Ts150K_EBV2mag', verbosity=0, multiplier=1., bremsfrac=None, E_subsample=3,
               fixSpectrum=False):
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
        :para fixSpectrum: Allow the spectrum to float in each energy bin.
        """

        #---------------------------------------------------------------------------------
        # Load templates

        if verbosity>0:
            print 'Loading FITS'

        comps, comps_new = {}, {}
        comps['ics'] = pyfits.open(basedir+'/ics_isotropic_healpix_54_'+tag+'.gz')[1].data.field(0).T
        comps['pi0'] = pyfits.open(basedir+'/pi0_decay_healpix_54_'+tag+'.gz')[1].data.field(0).T
        comps['brem'] = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[1].data.field(0).T

        energies = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[2].data.field(0)
        nside_in = np.sqrt(comps['ics'].shape[1]/12)

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
                           value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 10.], multiplier=multiplier)

        if bremsfrac is None:
            self.AddTemplate(name='Brems', healpixCube=comps_new['brem'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 10.], multiplier=multiplier)
            self.AddTemplate(name='Pi0', healpixCube=comps_new['pi0'], fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 10.], multiplier=multiplier)

        else:
            self.AddTemplate(name='Pi0_Brems', healpixCube=comps_new['pi0']+bremsfrac*comps_new['brem'],
                               fixSpectrum=fixSpectrum, fixNorm=False,
                               value=1, ApplyIRF=True, sourceClass='GEN', limits=[0, 10.], multiplier=multiplier)


    def RunLikelihood(self, print_level=0, use_basinhopping=False, start_fresh=False, niter_success=50):
        """
        Runs the likelihood analysis on the current templateList.

        :param print_level: 0=Quiet, 1=Verbosse
        :param use_basinhopping: Refine migrad optimization with a basinhopping algortithm. Generally recommended.
        :param start_fresh: Skip migrad and just use basinhopping.
        :param niter_success: Number of successful iterations required before stopping basinhopping.
        :returns m, res: m is an iMinuit object and res is a scipy minimization object.
        """
        self.m, self.res, self.res_vals = GammaLikelihood.RunLikelihood(self, print_level=print_level,
                                                                        use_basinhopping=use_basinhopping,
                                                                        start_fresh=start_fresh,
                                                                        niter_success=niter_success)

        # Run through the templates and update values to the best fit.
        for key, t in self.templateList.items():
            if t.fixSpectrum:
                if self.res_vals is not None:
                    t.value = self.res_vals[key]
                else:
                    t.value = self.m.values[key]
            else:
                if self.res is not None:
                    t.value = np.array([self.res_vals[key+'_'+str(i)] for i in range(self.n_bins)])
                else:
                    t.value = np.array([self.m.values[key+'_'+str(i)] for i in range(self.n_bins)])

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
        residual = copy.copy(self.binned_data)
        for key, t in self.templateList.items():
            # Make sure this template has been fit already
            if t.fixSpectrum:
                residual -= t.value*t.healpixCube
            else:
                for i_E in range(self.n_bins):
                    residual[i_E] -= t.value[i_E]*t.healpixCube[i_E]

        return residual*self.mask

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

        # Get the bin centers
        bin_centers = np.array([10**(0.5*(np.log10(self.bin_edges[i+1])+np.log10(self.bin_edges[i])))
                                for i in range(self.n_bins)])

        # Run through the template and obtain the spectrum in total counts over the masked area
        mask_idx = np.nonzero(self.mask)[0]

        # If looking for the data spectrum we need to temporarily create a template.
        if name is 'Data':
            t = Template.Template(healpixCube=self.binned_data)
        else:
            t = self.templateList[name]

        flux, stat_errors = [], []
        # Iterate over each energy bin
        for i_E in range(self.n_bins):

            # Get energy bin boundaries
            E_min, E_max = self.bin_edges[i_E], self.bin_edges[i_E+1]

            # Get the effective area for the masked region.
            l, b = Tools.hpix2ang(mask_idx, self.nside)
            # eff_area*bin width*solid_angle
            eff_area = (Tools.GetExpMap(E_min, E_max, l, b, self.expCube)
                        * (E_max-E_min)
                        * healpy.pixelfunc.nside2pixarea(self.nside))
            # if value is not a vector
            if np.ndim(t.value) == 0:

                stat_error = (np.sqrt(np.sum(t.healpixCube[i_E][mask_idx])*t.value)
                              / np.average(eff_area)/len(mask_idx))  # also divide by num pixels.
                count = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.value

            # if value is a vector
            elif np.ndim(t.value) == 1 and len(t.value) == self.n_bins:
                stat_error = (np.sqrt(np.sum(t.healpixCube[i_E][mask_idx])*t.value[i_E])
                              / np.average(eff_area)/len(mask_idx))  # also divide by num pixels.
                count = np.average(t.healpixCube[i_E][mask_idx]/eff_area)*t.value[i_E]
            else:
                raise Exception("template.value attribute has invalid dimension or type.")
            flux.append(count)
            stat_errors.append(stat_error)

        return bin_centers, np.array(flux), np.array(stat_errors)

    def AddFermiBubbleTemplate(self, template_file='./bubble_templates_diskcut30.0.fits',
                               spec_file='./reduced_bubble_spec_apj_793_64.dat', fixSpectrum=True):
        """
        Adds a fermi bubble template to the template stack.

        :param template_file: Requires file 'bubble_templates_diskcut30.0.fits'
            style file (from Su & Finkbeiner) with an extension table with a NAME column containing "Whole bubble"
            and a TEMPLATE column with an order 8 healpix array.
        :param spec_file: filename containing three columns (no header).  First col is energy in MeV, second is
            dN/dE in units (s cm^2 sr MeV)^-1 third is the uncertainty in dN/dE in (s cm^2 sr MeV)^-1.
        :param fixSpectrum: If True, the spectrum is not allowed to float.
        """

        # Load the template and spectrum
        hdu = pyfits.open(template_file)
        bub_idx = np.where(hdu[1].data['NAME'] == 'Whole bubble')
        bubble = hdu[1].data['TEMPLATE'][bub_idx][0]

        # Resize template if need be.
        nside_in = np.sqrt(bubble.shape[0]/12)
        if nside_in is not self.nside:
            Tools.ResizeHealpix(bubble, self.nside, average=True)

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
                         fixNorm=False, limits=[0., 5.], value=1., ApplyIRF=True,
                         sourceClass='GEN', valueUnc=valueUnc)

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










