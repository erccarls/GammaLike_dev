import Analysis
import cPickle as pickle
import Tools
import multiprocessing as mp 
import pyfits
import numpy as np
import h5py
import sys

def GenDiffuse(self, basedir='/data/galprop2/output/',
               tag='NSPEB_no_secondary_HI_H2', verbosity=0, multiplier=1., nrings=9.,
                E_subsample=1, fixSpectrum=True):
        """
        This method takes a base analysis prefix, along with an X_CO profile and generates the combined diffuse template,
        or components of the diffuse template.

        :param basedir: Base directory to read from
        :param tag: Tag for the galprop file.  This is the part between '_54_' and '.gz'.
        :param verbosity: 0 is quiet, >1 prints status.
        """

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
        # comps['pi0HIHII'] = pi0HIHII
        # comps['bremHIHII'] = bremHIHII
        comps_new['pi0HIHII'] =  np.zeros((self.n_bins, 12*self.nside**2))
        # comps_new['bremHIHII'] =  np.zeros((self.n_bins, 12*self.nside**2))
        

        for i_ring in range(1,nrings+1):
            print "Adding H2 ring", i_ring
            brem = ReadFits(basedir+'/bremss_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi = ReadFits(basedir+'/pi0_decay_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            comps['pi0_H2_'+str(i_ring)] = pi + 1.25*brem

            # comps['pi0_H2_'+str(i_ring)] = pi 
            # comps['brem_H2_'+str(i_ring)]= brem
            comps_new['pi0_H2_'+str(i_ring)] =  np.zeros((self.n_bins, 12*self.nside**2))
            # comps_new['brem_H2_'+str(i_ring)] = np.zeros((self.n_bins, 12*self.nside**2))
            
        
        
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
            for i_E in range(self.n_bins):
                comps_new[key][i_E] = Tools.InterpolateHealpix(comps[key], energies,  
                    self.bin_edges[i_E], self.bin_edges[i_E+1], E_bins=E_subsample, nside_out=self.nside)
            # Parallel version. (very memory hungry)
            #RunAsync(key)


        #---------------------------------------------------------------------------------
        # Now we just need to add the templates to the active template stack
        print 'Adding Templates to stack'
        
        self.AddTemplate(name='pi0HIHII', healpixCube=comps_new['pi0HIHII'], fixSpectrum=fixSpectrum, fixNorm=False,
                           value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        # self.AddTemplate(name='bremHIHII', healpixCube=comps_new['bremHIHII'], fixSpectrum=fixSpectrum, fixNorm=False,
                           # value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

        for i_ring in range(1,nrings+1):
            self.AddTemplate(name='pi0_H2_'+str(i_ring), healpixCube=comps_new['pi0_H2_'+str(i_ring)], fixSpectrum=fixSpectrum, fixNorm=False,
                           value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
            
            # self.AddTemplate(name='brem_H2_'+str(i_ring), healpixCube=comps_new['brem_H2_'+str(i_ring)], fixSpectrum=fixSpectrum, fixNorm=False,
            #                value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        
#         for i_ics in range(1,4):
#             self.AddTemplate(name='ics_'+str(i_ics), healpixCube=comps_new['ics_'+str(i_ics)], fixSpectrum=fixSpectrum, fixNorm=False,
#                            value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)
        self.AddTemplate(name='ics', healpixCube=comps_new['ics'], fixSpectrum=fixSpectrum, fixNorm=False,
                       value=1., ApplyIRF=True,noPSF=True, sourceClass='GEN', limits=[None, None], multiplier=multiplier)

        return self


def RunFit(A, nrings=9):
    #-----------------------------------------------------
    # Templates are now added so we fit X_CO
    
    import GammaLikelihood as like
    
    fval, res = [], []
    for key, t in A.templateList.items():
        # local 
        if key not in ['pi0_H2_7', 'Isotropic','pi0HIHII', 'ics', 'Bubbles']:
            t.fixNorm = True
            t.fixSpectrum= True
            t.limits = [0.0,200.]
            t.value=1.
        else: 
            t.fixNorm = False
            t.limits=[0., None]

            
    print 'Running Local Ring Fit...'
    A.GenSquareMask(l_range=[-180.,180], b_range=[-80.,80.], plane_mask=10.)
    m, R = A.RunLikelihood( print_level=1, precision=None, tol=1e3)[:2]
    fval.append(m.fval)
    res.append(R)

    print 'isotropic value:', A.templateList['Isotropic'].value

    vals = np.array([m.values['pi0_H2_'+str(i)] for i in range(1,nrings+1)])
    print "X_CO adjustment (this is not XCO value, it is multiplier for galdef values of xco):", vals

    #-------------------------------------------------------------------
    # Now we have fit the local X_CO (fixed).  Next we fit the outer galaxy
    A.templateList['pi0_H2_7'].fixNorm = True
    A.templateList['Isotropic'].fixNorm = True
    A.templateList['Bubbles'].fixNorm = True
    # Let the outer two rings float
    A.templateList['pi0_H2_8'].fixNorm = False
    A.templateList['pi0_H2_9'].fixNorm = False

    print 'Running Outer Rings Fit...'
    A.GenSquareMask(l_range=[-180,-80], b_range=[-10.,10.], plane_mask=0)
    A.GenSquareMask(l_range=[80,180], b_range=[-10.,10.], plane_mask=0, merge=True)

    m, R = A.RunLikelihood( print_level=1, precision=None, tol=1e3)[:2]
    fval.append(m.fval)
    res.append(R)
    vals = np.array([m.values['pi0_H2_'+str(i)] for i in range(1,nrings+1)])
    print "X_CO adjustment (this is not XCO value, it is multiplier for galdef values of xco):", vals


    #-------------------------------------------------------------------
    # Now we fit the inner galaxy X_CO.
    A.templateList['pi0_H2_8'].fixNorm = True
    A.templateList['pi0_H2_9'].fixNorm = True
    # Let the inner 6 rings float
    for i in range(1,7):
        A.templateList['pi0_H2_' + str(i)].fixNorm=False

    print 'Running Inner Rings Fit...'
    A.GenSquareMask(l_range=[-80,80], b_range=[-10.,10.], plane_mask=0)
    m, R = A.RunLikelihood( print_level=1, precision=None, tol=1e3)[:2]
    fval.append(m.fval)
    res.append(R)
    vals = np.array([m.values['pi0_H2_'+str(i)] for i in range(1,nrings+1)])
    print "X_CO adjustment (this is not XCO value, it is multiplier for galdef values of xco):", vals



    return m, fval, res





def WriteHDF5(fname, basedir, tag, m , nrings=9):
    """
    Build the diffuse model according to the best fit parameters and writes model+metadata to an HDF5 file.
    
    :param fname: Output filename for the HDF5 file
    :param basedir: Directory with galprop output files
    :param tag: galprop tag
    :param basedir: iminuit object. 
    :param nrings: number of galprop rings.
    """
    modf = h5py.File(fname, 'w')

    if m is not None:
        X_CO = np.array([m.values['pi0_H2_'+str(i)] for i in range(1,nrings+1)])
    
    #modf = h5py.File(fname, 'w')
    #try:
    # Generate Groups
    # template_group = modf.create_group("templates")
    # fit_group = modf.create_group("fit_results")


    # Get data dimensions
    tmp = pyfits.open(basedir+'/bremss_healpix_54_'+tag+'.gz')
    energies = tmp[2].data.field(0)
    tShape = (len(energies), tmp[1].data.shape[0])
    print tShape
    del tmp # free memory

    pi0     = modf.create_dataset("/templates/pi0", tShape, dtype='float32',compression="gzip")
    pi0_0   = modf.create_dataset("/templates/pi0_0", tShape, dtype='float32',compression="gzip")
    brem   = modf.create_dataset("/templates/brem", tShape, dtype='float32',compression="gzip")
    brem_0 = modf.create_dataset("/templates/brem_0", tShape, dtype='float32',compression="gzip")
    ics_opt = modf.create_dataset("/templates/ics_opt", tShape, dtype='float32',compression="gzip")
    ics_fir = modf.create_dataset("/templates/ics_fir", tShape, dtype='float32',compression="gzip")
    ics_cmb = modf.create_dataset("/templates/ics_cmb", tShape, dtype='float32',compression="gzip")
    modf.create_dataset("/templates/energies", data=energies, dtype='float32',compression="gzip")
    # Now fill in the templates one by one.
    # Add fit metadata.
    # Add galdef metadata.


    #---------------------------------------------------------------
    # Create Diffuse Template from fitting results.
    def ReadFits(fname, length):
        d = pyfits.open(fname)[1].data
        return np.array([d.field(i) for i in range(length)])

    if m is not None:
        for i_ring in range(1,nrings+1):
            print "Adding HI/HII ring", i_ring

            pi0[...] += m.values['pi0HIHII']*ReadFits(basedir+'/pi0_decay_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0[...] += m.values['pi0HIHII']*ReadFits(basedir+'/pi0_decay_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0[...] += X_CO[i_ring-1]*ReadFits(basedir+'/pi0_decay_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))

            brem[...] += m.values['pi0HIHII']*1.25*ReadFits(basedir+'/bremss_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            brem[...] += m.values['pi0HIHII']*1.25*ReadFits(basedir+'/bremss_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            brem[...] += 1.25*X_CO[i_ring-1]*ReadFits(basedir+'/bremss_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))

            if i_ring == 1:
                pi0_0[...] += pi0
                brem_0[...] += brem

        ics_opt[...] += m.values['ics']*ReadFits(basedir+'/ics_isotropic_comp_1_healpix_54_'+tag+'.gz', len(energies))
        ics_fir[...] += m.values['ics']*ReadFits(basedir+'/ics_isotropic_comp_2_healpix_54_'+tag+'.gz', len(energies))
        ics_cmb[...] += m.values['ics']*ReadFits(basedir+'/ics_isotropic_comp_3_healpix_54_'+tag+'.gz', len(energies))

    else:
        for i_ring in range(1,nrings+1):
            print "Adding HI/HII ring", i_ring

            pi0[...] += ReadFits(basedir+'/pi0_decay_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0[...] += ReadFits(basedir+'/pi0_decay_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            pi0[...] += ReadFits(basedir+'/pi0_decay_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))

            brem[...] += ReadFits(basedir+'/bremss_HIR_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            brem[...] += ReadFits(basedir+'/bremss_HII_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))
            brem[...] += ReadFits(basedir+'/bremss_H2R_ring_'+str(i_ring)+'_healpix_54_'+tag+'.gz', len(energies))

            if i_ring == 1:
                pi0_0[...] += pi0
                brem_0[...] += brem

        ics_opt[...] += ReadFits(basedir+'/ics_isotropic_comp_1_healpix_54_'+tag+'.gz', len(energies))
        ics_fir[...] += ReadFits(basedir+'/ics_isotropic_comp_2_healpix_54_'+tag+'.gz', len(energies))
        ics_cmb[...] += ReadFits(basedir+'/ics_isotropic_comp_3_healpix_54_'+tag+'.gz', len(energies))

#     except:
#         modf.close()
    
    try: 
        modf.close()
    except: pass
    return


def AddMetadata(fname, basedir, tag, A, m, fval=None):
    # Parse the galprop file into a dict.
    galdef_dict = {}
    with open(basedir + '/galdef_54_'+tag) as galdef:
        for line in galdef: 
            if line[0] != "#" and line.strip()!='':
                s = line.strip('\n').split('=')
                if len(s)<2: 
                    continue
                key = s[0].strip()
                if key in ["Title", 'X_CO_values', 'n_X_CO_values', 'X_CO_radius']: continue
                galdef_dict[key] = s[1].strip().split(" ")[0]
    try: h5.close() 
    except: pass
    h5 = h5py.File(fname)
    try:
        galdef_group = h5.create_group("/galdef")
    except: 
        galdef_group = h5['/galdef']
    for key, val in galdef_dict.items():
        #print key, val
        galdef_group.attrs.create(key,val)
    
    if m is not None and A is not None:
        try:
            fit_results = h5.create_group("/fit_results/global")
        except: 
            fit_results = h5['/fit_results/global']
        

        


        vals = np.array([m.values['pi0_H2_'+str(i)] for i in range(1,10)])

        fa = fit_results.attrs
        fa.create('globalvalues', m.values.items())
        fa.create('globalvaluesUnc', m.errors.items())
        fa.create('globalfval', m.fval)

        if fval is not None:
            fa.create('localfval', fval[0])            
            fa.create('outerfval', fval[1])            
            fa.create('innerfval', fval[2])            

        fa.create('global_XCO', vals)
        fa.create('globale_bins', A.bin_edges)
        fa.create('globalirf', A.irf)
        fa.create('globalevclass', A.evclass)
        fa.create('globalconvtype', A.convtype)
        fa.create('globalphfile', A.phfile)
        fa.create('globaltag', A.tag)
        h5.create_dataset('/fit_results/globalmask', data=A.mask, dtype='float32')
    h5.close()
    
try:
    modf.close()
except:pass



if __name__ == "__main__":
    if len(sys.argv) != 4: 
        raise("Incorrect number of args: <galprop output dir> <galprop tag> <galdef dir>")
    
    basedir, tag, galdefdir, = sys.argv[1:4]
    fname = basedir+'/'+tag+'_XCO.hdf5'

    # Load the analysis
    A = Analysis.Analysis(tag='P7REP_CLEAN_V15_calore', fglpath='/pfs/carlson/gll_psc_v14.fit',  
        templateDir='/home/carlson/pfs/Extended_archive_v15/Templates', basepath='/pfs/carlson/GCE_sys/')
    # A.GenPointSourceTemplate(pscmap=(A.basepath + '/PSC_all_sky_3fgl.npy'))
    # A.BinPhotons(outfile='binned_photons_all_sky.npy')
    #A.GenSquareMask(l_range=[-180.,180], b_range=[-40.,40.], plane_mask=1.)
    A.BinPhotons(infile='binned_photons_all_sky.npy')
    # Load 2FGL 
    A.AddPointSourceTemplate(fixNorm=True, pscmap=('PSC_all_sky_3fgl.npy'))
    A.CalculatePixelWeights(diffuse_model='fermi_diffuse_'+A.tag+'.npy',psc_model='PSC_' + A.tag + '.npy',
                            alpha_psc=5., f_psc=0.05)
    A.AddIsotropicTemplate(fixNorm=False, fixSpectrum=True) # External chi^2 used to fix normalization within uncertainties
    #A.PopulateROI([0,0],radius=360, fix_radius=360., include_point=False)


    # OPEN THE Extended PSC file and add it to the template list. 
    A.templateList['PSCExt'] = pickle.load(open('PSCExt.pickle', 'rb'))
    A.AddFermiBubbleTemplate(template_file='./bubble_templates_diskcut30.0.fits', 
                         spec_file='./reduced_bubble_spec_apj_793_64.dat', fixSpectrum=True, fixNorm=False)

    #A.AddDMTemplate(profile='NFW', limits=[None,None], decay=False, gamma=1.26, 
    #                r_s=20.0, axesratio=1, offset=(0, 0), spec_file=None,)
    
    # Run the analysis

    A = GenDiffuse(A, basedir=basedir, tag=tag, verbosity=1, nrings=9)
    m, fval, res = RunFit(A, nrings=9)
    WriteHDF5(fname=fname, basedir=basedir, tag=tag, m=m, nrings=9)
    AddMetadata(fname,basedir=galdefdir, tag=tag, A=A, m=m, fval=fval)

    # WriteHDF5(fname=fname, basedir=basedir, tag=tag, m=None, nrings=9)
    # AddMetadata(fname,basedir=galdefdir, tag=tag, A=None, m=None)    
