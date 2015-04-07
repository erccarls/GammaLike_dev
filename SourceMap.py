import Tools
import pyfits
import numpy as np
import sys


def GenSourceMap(bin_edges, l_range=(-180, 180), b_range=(-90, 90),
                 fglpath='/data/gll_psc_v08.fit',
                 expcube='/data/fermi_data_1-8-14/gtexpcube2_ALL_BOTH',
                 psffile='/data/fermi_data_1-8-14/psf_P7REP_SOURCE_BOTH.fits',
                 maxpsf = 10.,
                 res=0.125,
                 nside=256,
                 onlyidx=None,
                 filename=None,
                 verbosity=1,
                 ignore_ext=True):
    """
    This method generates a source map based on an input catalog implementing the following procedure:
    1. Integrate given spectrum to 2FGL over each energy bin to obtain counts/cm^2/s
    2. Multiply  by effective exposure to obtain counts
    3. Convolve with PSF to obtain the source map.
    **WARNING CURRENTLY ONLY VALID for |b|<85 degrees latitude due to treatment at celestial poles**

    :param bin_edges: Energy bin edges in MeV
    :param l_range: longitude range.  First number needs to be lower than second.  Should be at least 5 degrees larger
                    than your masked analysis area.
    :param b_range: latitude range.   First number needs to be lower than second.  Should be at least 5 degrees larger
                    than your masked analysis area.
    :param fglpath: absolute path to 2FGL file.
    :param expcube: absolute path to the exposure cube file (output of gtexpcube2)
    :param psffile: absolute path to output from gtpsf
    :param maxpsf: Maximum radius in degrgees, to consider for PSF smearing of point sources.
    :param res: subresolution for mapping PSF smearing onto healpix grid.
    :param nside: healpix nside parameter.
    :param filename: output the resultant map to a numpy array.
    :param ignore_ext: If true, does not include extended sources. If false, only the total counts
            for the extended maps are correct.
    :returns PSC Map: A healpix 'cube' for the point sources.
    """
    # Load FGL cat
    fgl_data = pyfits.open(fglpath)[1].data
    # Init the master point source template
    pscmap = np.zeros(shape=(len(bin_edges)-1, 12*nside**2)).astype(np.float32)

    # Determine which sources are inside the spatial window.
    idx_all = np.where((fgl_data['GLAT'] < b_range[1]) & (fgl_data['GLAT'] > b_range[0])
                       & ((fgl_data['GLON'] < l_range[1]) | (fgl_data['GLON'] > (l_range[0]+360))))[0]

    # Pre-load all the point spread functions as a function of fine energy binning so we can re-weight against spectra.
    hdu = pyfits.open(psffile)

    thetas = np.array([theta[0] for theta in hdu[2].data])

    energies = np.array([energy[0] for energy in hdu[1].data])
    psfs = np.array([psf[2] for psf in hdu[1].data])

    # Iterate over sources
    for i_idx, idx in enumerate(idx_all):
        if (fgl_data['Source_Name'][idx][-1] == 'e') and ignore_ext:
            continue
        # Debug
        if onlyidx is not None:
            if idx not in onlyidx:
                continue


        # -----------------------------------------------------------
        # First we determine the number of counts.
        # Retreive the spectrum and integrated spectrum functions
        spectype = fgl_data['SpectrumType'][idx]
        spec, integratedspec = Tools.GetSpec(spectype)

        # Get the spectral parameters
        specindex = fgl_data['Spectral_Index'][idx]
        beta = fgl_data['beta'][idx]
        fluxdens = fgl_data['Flux_Density'][idx]
        pivot = fgl_data['Pivot_Energy'][idx]
        cutoff = fgl_data['Cutoff'][idx]
        glat = fgl_data['GLAT'][idx]
        glon = fgl_data['GLON'][idx]
        # 2FGL doesnt have this keyword
        try:
            b = fgl_data['Exp_Index'][idx]
        except:
            pass
        #print spectype
        # Find the Normalization and integrate the spectrum over each energy bin
        if spectype == 'PowerLaw':
            norm = fluxdens/spec(pivot, specindex)
            counts = norm*np.array([integratedspec(bin_edges[i_E], bin_edges[i_E+1], specindex)
                                    for i_E in range(len(bin_edges)-1)])


            psfWeights = spec(energies, specindex)

        elif spectype == 'PLExpCutoff':
            norm = fluxdens/spec(pivot, specindex, cutoff)
            counts = norm*np.array([integratedspec(bin_edges[i_E], bin_edges[i_E+1], specindex, cutoff)
                                    for i_E in range(len(bin_edges)-1)])
            psfWeights = spec(energies, specindex, cutoff)
        elif spectype == 'PLSuperExpCutoff':
            norm = fluxdens/spec(pivot, specindex, b, pivot, cutoff)
            counts = norm*np.array([integratedspec(bin_edges[i_E], bin_edges[i_E+1], specindex, b, pivot, cutoff)
                                    for i_E in range(len(bin_edges)-1)])
            psfWeights = spec(energies, specindex, b, pivot, cutoff)
        elif spectype == 'LogParabola':
            norm = fluxdens/spec(pivot, specindex, beta, pivot)
            counts = norm*np.array([integratedspec(bin_edges[i_E], bin_edges[i_E+1], specindex, beta, pivot)
                                    for i_E in range(len(bin_edges)-1)])
            psfWeights = spec(energies, specindex, beta, pivot)


        # Now counts contains ph/cm^2/s^2 for each bin so we need to get the effective area in each bin.
        exposure = np.array([Tools.GetExpMap(bin_edges[i_E], bin_edges[i_E+1], glon, glat, expcube)
                             for i_E in range(len(bin_edges)-1)])

        # Now the counts are in actual counts
        counts = counts*exposure

        #-----------------------------------------------------------
        # Apply PSF
        avgPSF = []
        size = 2*maxpsf/res+1
        lats, lons = np.linspace(-maxpsf, maxpsf, size), np.linspace(-maxpsf, maxpsf, size)



        # Calculate the reweighted PSF in each energy bin for this source's spectrum.
        for i_E in range(len(bin_edges)-1):
            e1, e2 = bin_edges[i_E], bin_edges[i_E+1]
            eb1, eb2 = np.argmin(np.abs(energies-e1)), np.argmin(np.abs(energies-e2))

            avgPSF = np.average(psfs[eb1:eb2+1], weights=psfWeights[eb1:eb2+1], axis=0)
            avgPSFInterp = lambda r: np.interp(r, thetas, avgPSF)

            # Form a cartesian array for this PSF which will be mapped to the healpix grid
            cartMap = np.zeros(shape=(size, size)).astype(np.float32)

            # Scan over latitudes and fill in PSF value
            for i_lat, lat in enumerate(lats):
                # Now calculate distances for each point
                r = np.sqrt(lat**2+lons**2)
                cartMap[i_lat, :] = avgPSFInterp(r)  # *res**2*(np.pi/180.)**2

            # Mult by solid angle/px and sum gives ~1. Good!
            # We will just renormalize sum to 1 since there is a small error
            cartMap = cartMap/cartMap.sum()*counts[i_E]  # Now units are photons falling in that pixel.

            # Now we need to map this grid onto the healpix grid.
            for i_lat, lat in enumerate(lats):
                realLat = glat+lat
                # If we crossed the pole, also flip the meridian.
                # TODO: FIX BEHAVIOR NEAR POLES Currently not handled correctly
                # A GOOD WAY TO DO THIS WOULD BE TO USE THE GAMMACAP CODE WHICH
                # ROTATES A GIVEN INITIAL POSITION UP TO A SKYMAP POSITION.
                # TODO: Implement extended 2fgl sources.
                if realLat > 90.:
                    realLat -= 90.
                if realLat < -90.:
                    realLat += 90.
                if glon > 180:
                    glon -= 360.
                # Take care of rescaling the effective longitude based on the latitude.
                # longitude doesn't really matter if we are right a N/S pole.
                # if np.abs(90-np.abs(realLat))>.125: new_lons = ((glon+lons)/np.cos(np.deg2rad(realLat)))%360
                new_lons = (glon+lons/np.cos(np.deg2rad(realLat))) % 360
                # Find the which hpix bin each pixel falls and add the counts
                hpix_idx = Tools.ang2hpix(new_lons, realLat, nside=nside)
                # Add the values to the healpix grid.  Must use this 'at' method in order to
                # correctly add repeated indices.
                np.add.at(pscmap[i_E], hpix_idx, cartMap[i_lat])

        if (i_idx % 1) == 0 and verbosity > 0:
            print '\rGenerating Point Source Map:', '%2.2f' % (np.float(i_idx)/len(idx_all)*100.), '%',
            sys.stdout.flush()
    if filename is not None:
        np.save(open(filename, 'wb'), pscmap.astype(np.float32))
    return pscmap


# bin_edges=[300, 350, 400, 450, 500, 557.4511962326, 624.9931144482831, 705.0811841511901, 800.9547380631948, 916.9544789959854, 1058.9994895010252, 1235.32183584898, 1457.6298200740125, 1743.0094229290717, 2117.148088832825, 2620.038055486477, 3316.5858204132596, 4317.5724796525965, 5824.226374320851, 8232.171915073328, 12404.648624640446, 20517.115667189668, 39361.808463774716, 99337.18520898951, 499999.9999999654]
# cmap = GenSourceMap(bin_edges,onlyidx=None)