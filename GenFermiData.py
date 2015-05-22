import os
import numpy as np

def GenDataScipt(tag, basepath, bin_edges, scriptname, phfile, scfile, evclass=3, convtype=-1,  zmax=100, emin=200,
                 emax=6e5,
                 irf='P7REP_CLEAN_V15', filter="DATA_QUAL>0 && LAT_CONFIG==1 && ABS(ROCK_ANGLE)<52"):
    '''
    Given a set of input params, this calls the fermi tools to generate the necessary files for
    diffuse likelihood analysis.
    params:
        tag: Add this suffix to each file name.
        basepath: Base directory for relative paths and file storage.
        ebins: a list of energy bin edges to use. Energies in MeV
        scriptname: The output script will be written to the basepath with this name.
        ** The remaining parameters are all described in the various Fermi Tools documents.

    returns:
        runstring
    '''

    # Perhaps catch a common mistake....
    if evclass == 2 and 'SOURCE' not in irf:
        print 'WARNING: Possible mismatch between irf and evclass. Check Carefully.'
    elif evclass == 3 and 'CLEAN' not in irf:
        print 'WARNING: Possible mismatch between irf and evclass. Check Carefully.'


    # Generate the binning file definition for fermitools
    # This should be done for logspaced bins with high energy resolution.
    with open(basepath + '/bin_edges_'+str(tag)+'.dat', 'wb') as f:
        # This is for custom binning.
        # for i in range(len(bin_edges)-1):
        #     f.write(str(bin_edges[i]) + " " + str(bin_edges[i+1])+"\n")

        # We just care about finely log-spaced binning since exposure maps are later integrated over each bin.
        E = np.logspace(np.log10(np.min(bin_edges)), np.log10(np.max(bin_edges)), 76)
        for i in range(len(E)-1):
            f.write(str(E[i]) + " " + str(E[i+1])+"\n")


    runstring = '''
    cd '''+ basepath + '''

    echo "Get a beer. This will take a while..."

    echo "running gtselect"
    gtselect '''+str(phfile)+''' photons_merged_'''+str(tag)+'''.fits\
        ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF\
        zmax='''+str(zmax)+''' emin='''+str(emin)+''' emax='''+str(emax)+'''\
        convtype='''+str(convtype)+''' evclass='''+str(evclass)+''' clobber=True

    echo "running gtmktime"
    gtmktime scfile='''+str(scfile)+''' filter="'''+filter + '''"\
        roicut=no evfile=photons_merged_'''+str(tag)+'''.fits outfile=photons_merged_cut_'''+str(tag)+'''.fits\
        clobber=True

    rm photons_merged_'''+str(tag)+'''.fits

    echo "running gtltcube"

    gtltcube evfile=photons_merged_cut_'''+str(tag)+'''.fits scfile='''+str(scfile)+'''\
        outfile="cube_'''+str(tag)+'''.fits" dcostheta=0.1 binsz=1 zmin=0 zmax=180 clobber=True

    # make ebin file
    echo "running gtbindef"
    gtbindef bintype=E binfile=bin_edges_'''+str(tag)+'''.dat outfile=ebins_'''+str(tag)+'''.fits energyunits=MeV\
    clobber=True

    # make psf file
    echo "running gtpsf"
    gtpsf expcube="cube_'''+str(tag)+'''.fits" outfile=gtpsf_'''+str(tag)+'''.fits irfs='''+str(irf)+''' \
    emin=20 emax=1e6 nenergies=50 clobber=True ra=0 dec=0 thetamax=10 ntheta=200

    echo "running gtbin"
    gtbin evfile=photons_merged_cut_'''+str(tag)+'''.fits \
        scfile='''+str(scfile)+''' outfile="gtbin_'''+str(tag)+'''.fits" algorithm=CCUBE nxpix=721 nypix=361\
        ebinalg=FILE ebinfile=ebins_'''+str(tag)+'''.fits coordsys=GAL proj=CAR xref=0 yref=0 axisrot=0 binsz=0.5\
        clobber=True

    echo "running gtexpcube2"
    gtexpcube2 infile="cube_'''+str(tag)+'''.fits" cmap="gtbin_'''+str(tag)+'''.fits"\
        coordsys=GAL outfile="gtexpcube2_'''+str(tag)+'''.fits"\
        irf='''+str(irf)+''' ebinfile=ebins_'''+str(tag)+'''.fits ebinalg=FILE clobber=True'''

    f = open(basepath + scriptname, 'wb')
    f.write(runstring)
    f.close()
    os.chmod(basepath + scriptname, 0755)

    return runstring




# bins = [300, 350, 400, 450, 500, 557.4511962326, 624.9931144482831, 705.0811841511901, 800.9547380631948,
#         916.9544789959854, 1058.9994895010252, 1235.32183584898, 1457.6298200740125, 1743.0094229290717,
#         2117.148088832825, 2620.038055486477, 3316.5858204132596, 4317.5724796525965, 5824.226374320851,
#         8232.171915073328, 12404.648624640446, 20517.115667189668, 39361.808463774716, 99337.18520898951,
#         499999.9999999654]

#print GenDataScipt(tag='P7REP_CLEAN_V15_test', basepath='/data/GCE_sys/',
#                   scriptname='genFermiData.sh', bin_edges=bins,
#                   phfile='/data/fermi_data_1-8-14/phfile.txt',
#                   scfile='/data/fermi_data_1-8-14/lat_spacecraft_merged.fits')

