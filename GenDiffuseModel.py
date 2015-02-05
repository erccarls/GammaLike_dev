import Analysis
import cPickle as pickle
import Tools
import multiprocessing as mp 
import pyfits
import numpy as np

A = Analysis.Analysis(tag='P7REP_CLEAN_V15_calore', )

# A.GenPointSourceTemplate(pscmap=(A.basepath + '/PSC_all_sky_3fgl.npy'))
# A.BinPhotons(outfile='binned_photons_all_sky.npy')
A.GenSquareMask(l_range=[180.,180], b_range=[-40.,40.], plane_mask=0.)
A.BinPhotons(infile='binned_photons_all_sky.npy')
# Load 2FGL 
A.AddPointSourceTemplate(fixNorm=True, pscmap=(A.basepath + '/PSC_all_sky_3fgl.npy'))
A.CalculatePixelWeights(diffuse_model='fermi_diffuse_'+A.tag+'.npy',psc_model='PSC_' + A.tag + '.npy',
                        alpha_psc=5., f_psc=0.1)
A.AddIsotropicTemplate(fixNorm=True, fixSpectrum=True) # External chi^2 used to fix normalization within uncertainties
#A.AddDMTemplate(profile='NFW', limits=[None,None], decay=False, gamma=1.26, 
#                r_s=20.0, axesratio=1, offset=(0, 0), spec_file=None,)
A.PrintTemplates()


def GenDiffuse(self, basedir='/data/galprop2/output/',
               tag='NSPEB_no_secondary_HI_H2', verbosity=0, multiplier=1., bremsfrac=None, E_subsample=3,
               fixSpectrum=True, nrings=9):
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

        # A.GenPointSourceTemplate(pscmap=(A.basepath + '/PSC_all_sky_3fgl.npy'))
        # A.BinPhotons(outfile='binned_photons_all_sky.npy')
        A.GenSquareMask(l_range=[180.,180], b_range=[-40.,40.], plane_mask=0.)
        A.BinPhotons(infile='binned_photons_all_sky.npy')
        # Load 2FGL 
        A.AddPointSourceTemplate(fixNorm=True, pscmap=(A.basepath + '/PSC_all_sky_3fgl.npy'))
        A.CalculatePixelWeights(diffuse_model='fermi_diffuse_'+A.tag+'.npy',psc_model='PSC_' + A.tag + '.npy',
                                alpha_psc=5., f_psc=0.1)
        A.AddIsotropicTemplate(fixNorm=True, fixSpectrum=True) # External chi^2 used to fix normalization within uncertainties
        #A.AddDMTemplate(profile='NFW', limits=[None,None], decay=False, gamma=1.26, 
        #                r_s=20.0, axesratio=1, offset=(0, 0), spec_file=None,)
        A.PrintTemplates()



        if verbosity>0:
            print 'Loading FITS'

        energies = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[2].data.field(0)
        comps, comps_new = {}, {}
#         # For some reason, older versions of galprop files have slightly different data structures.  This try/except
#         # will detect the right one to use. 
         
#         try:
#             comps['ics'] = pyfits.open(basedir+'/ics_isotropic_healpix_54_'+tag+'.gz')[1].data.field(0).T
#             nside_in = np.sqrt(comps['ics'].shape[1]/12)
#             comps['pi0'] = pyfits.open(basedir+'/pi0_decay_healpix_54_'+tag+'.gz')[1].data.field(0).T
#             comps['brem'] = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')[1].data.field(0).T

        #except:
        def ReadFits(fname, length):
            d = pyfits.open(fname)[1].data
            return np.array([d.field(i) for i in range(length)])
        
        # Add up the HI and HII contributions into a single template since nothing there is varying.
        pi0HIHII = np.zeros((len(energies), 12*self.nside**2))
        bremHIHII = np.zeros((len(energies), 12*self.nside**2))
        
        for i_ring in range(1,nrings+1):
            print "Adding HI/HII ring", i_ring
            bremHIHII += ReadFits(basedir+'/bremss_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            bremHIHII += ReadFits(basedir+'/bremss_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0HIHII += ReadFits(basedir+'/pi0_decay_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0HIHII += ReadFits(basedir+'/pi0_decay_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
        comps['pi0HIHII'] = pi0HIHII + 1.25*bremHIHII
        #comps['bremHIHII'] = bremHIHII
        comps_new['pi0HIHII'] =  np.zeros((self.n_bins, 12*self.nside**2))
        #comps_new['bremHIHII'] =  np.zeros((self.n_bins, 12*self.nside**2))
        
        
        for i_ring in range(1,nrings+1):
            print "Adding H2 ring", i_ring
            #comps['brem_H2_'+str(i_ring)]= ReadFits(basedir+'/bremss_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            brem = ReadFits(basedir+'/bremss_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi = ReadFits(basedir+'/pi0_decay_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            comps['pi0_H2_'+str(i_ring)] = pi + 1.25*brem
            #comps_new['brem_H2_'+str(i_ring)] = np.zeros((self.n_bins, 12*self.nside**2))
            comps_new['pi0_H2_'+str(i_ring)] =  np.zeros((self.n_bins, 12*self.nside**2))
        
        
        comps['ics'] = np.zeros((len(energies), 12*self.nside**2))
        comps_new['ics'] = np.zeros((self.n_bins, 12*self.nside**2))
        for i_ics in range(1,4):
            print "Adding ics", i_ics
            comps['ics'] += ReadFits(basedir+'/ics_isotropic_comp_'+str(i_ics)+'_healpix_54_'+tag+'.gz', len(energies))
#             comps['ics_'+str(i_ics)] = ReadFits(basedir+'/ics_isotropic_comp_'+str(i_ics)+'_healpix_54_'+tag+'.gz', len(energies))
#             comps_new['ics_'+str(i_ics)] = np.zeros((self.n_bins, 12*self.nside**2))
        
        nside_in = np.sqrt(comps['pi0HIHII'].shape[1]/12)
        
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
        print 'Adding Templates to stack'
        
        self.AddTemplate(name='pi0HIHII', healpixCube=comps_new['pi0HIHII'], fixSpectrum=fixSpectrum, fixNorm=False,
                           value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        #self.AddTemplate(name='bremHIHII', healpixCube=comps_new['bremHIHII'], fixSpectrum=fixSpectrum, fixNorm=False,
        #                   value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

        for i_ring in range(1,nrings+1):
            self.AddTemplate(name='pi0_H2_'+str(i_ring), healpixCube=comps_new['pi0_H2_'+str(i_ring)], fixSpectrum=fixSpectrum, fixNorm=False,
                           value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
            #self.AddTemplate(name='brem_H2_'+str(i_ring), healpixCube=comps_new['brem_H2_'+str(i_ring)], fixSpectrum=fixSpectrum, fixNorm=False,
            #               value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        
#         for i_ics in range(1,4):
#             self.AddTemplate(name='ics_'+str(i_ics), healpixCube=comps_new['ics_'+str(i_ics)], fixSpectrum=fixSpectrum, fixNorm=False,
#                            value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        self.AddTemplate(name='ics', healpixCube=comps_new['ics'], fixSpectrum=fixSpectrum, fixNorm=False,
                       value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

        #-----------------------------------------------------
        # Templates are now added so we fit X_CO
        
        import GammaLikelihood as like
        for key, t in A.templateList.items():
            if key not in [ 'PSC' ,'Isotropic'] :
                t.fixNorm = False
                t.fixSpectrum= True
                t.limits = [0.0,10.]
                t.value=1.
                
        
        m = like.RunLikelihood(A, print_level=1, precision=None, tol=1e2, force_cpu=False, use_basinhopping=False)

        nrings = 9
        vals = np.array([m[0].values['pi0_H2_'+str(i)] for i in range(1,nrings+1)])
        print "X_CO fit (not modulated by MS04 yet):", vals












AddGalpropRings(A, basedir='/data/galprop2/output/',
               tag='base_2D', verbosity=0, multiplier=1., bremsfrac=None, E_subsample=3,
               fixSpectrum=True, nrings=9)