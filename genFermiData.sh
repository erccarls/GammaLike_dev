
    cd /data/GCE_sys/

    echo "Get a beer. This will take a while..."

    echo "running gtselect"
    gtselect /data/fermi_data_1-8-14/phfile.txt photons_merged_P7REP_CLEAN_V15_test.fits        ra=INDEF dec=INDEF rad=INDEF tmin=INDEF tmax=INDEF        zmax=100 emin=200 emax=600000.0        convtype=-1 evclass=2 clobber=True

    echo "running gtmktime"
    gtmktime scfile=/data/fermi_data_1-8-14/lat_spacecraft_merged.fits filter="DATA_QUAL>0 && LAT_CONFIG==1 && ABS(ROCK_ANGLE)<52"        roicut=no evfile=photons_merged_P7REP_CLEAN_V15_test.fits outfile=photons_merged_cut_P7REP_CLEAN_V15_test.fits        clobber=True

    rm photons_merged_P7REP_CLEAN_V15_test.fits

    echo "running gtltcube"
    gtltcube evfile=photons_merged_cut_P7REP_CLEAN_V15_test.fits scfile=/data/fermi_data_1-8-14/lat_spacecraft_merged.fits        outfile="cube_P7REP_CLEAN_V15_test.fits" dcostheta=0.5 binsz=2 zmax=100 clobber=True

    # make ebin file
    echo "running gtbindef"
    gtbindef bintype=E binfile=bin_edges_P7REP_CLEAN_V15_test.dat outfile=ebins_P7REP_CLEAN_V15_test.fits energyunits=MeV    clobber=True

    # make psf file
    echo "running gtpsf"
    gtpsf expcube="cube_P7REP_CLEAN_V15_test.fits" outfile=gtpsf_P7REP_CLEAN_V15_test.fits irfs=P7REP_CLEAN_V15     emin=20 emax=1e6 nenergies=50 clobber=True ra=0 dec=0 thetamax=10 ntheta=200

    echo "running gtbin"
    gtbin evfile=photons_merged_cut_P7REP_CLEAN_V15_test.fits         scfile=/data/fermi_data_1-8-14/lat_spacecraft_merged.fits outfile="gtbin_P7REP_CLEAN_V15_test.fits" algorithm=CCUBE nxpix=721 nypix=361        ebinalg=FILE ebinfile=ebins_P7REP_CLEAN_V15_test.fits coordsys=GAL proj=CAR xref=0 yref=0 axisrot=0 binsz=0.5        clobber=True

    echo "running gtexpcube2"
    gtexpcube2 infile="cube_P7REP_CLEAN_V15_test.fits" cmap="gtbin_P7REP_CLEAN_V15_test.fits"        coordsys=GAL outfile="gtexpcube2_P7REP_CLEAN_V15_test"        irf=P7REP_CLEAN_V15 ebinfile=ebins_P7REP_CLEAN_V15_test.fits ebinalg=FILE clobber=True