import h5py, healpy, pyfits
import numpy as np
import sys

def Healpix2Cartesian(fname):                        
    # Read from hdf5 file
    f = h5py.File(fname,'r')
    
    for t in f['templates/']:
        print "Writing", t
        if t=="energies": 
            continue
        
        
        # Get the cartesian maps for each energy
        hpixcube = f['templates/'+t][()]
        cartcube = np.zeros((hpixcube.shape[0], 721,1440), dtype=np.float32)
        for i in range(hpixcube.shape[0]):
            cartcube[i] = healpy.cartview(hpixcube[i], hold=True, return_projected_map=True,
                                                  xsize=1440, lonra=[-179.875, 179.875],flip='geo')
            plt.gcf()
        
        # Generate new hdu object
        hdu_new = pyfits.PrimaryHDU(cartcube.astype(np.float32))
        

        # Copy galdef into header
        galdef = dict(f['/galdef'].attrs.items())
        hdu_new.header.add_comment("Diffuse model generated by Eric Carlson (erccarls@ucsc.edu)")
        for key, val in galdef.items():
            hdu_new.header.add_comment(key + "=" +val)
        
        hdu_new.header['CRVAL1'] = 0.0
        hdu_new.header['CRPIX1'] = 720
        hdu_new.header['CDELT1'] = 0.25
        hdu_new.header['CUNIT1']= 'deg'
        hdu_new.header['CTYPE2']= 'GLON-CAR'
        hdu_new.header['CRVAL2'] = 0
        hdu_new.header['CRPIX2'] = 361
        hdu_new.header['CDELT2'] = 0.25
        hdu_new.header['CUNIT2']= 'deg'
        hdu_new.header['CTYPE2']= 'GLAT-CAR'
        hdu_new.header['CRVAL3'] = float(galdef['E_gamma_min'])
        hdu_new.header['CRPIX3'] = 0
        hdu_new.header['CDELT3'] = np.log10(float(galdef['E_gamma_factor']))
        hdu_new.header['CTYPE2']= 'Energy'
        hdu_new.header['CUNIT3']= 'MeV'
        hdu_new.header['EXTEND']=True
        hdu_new.header['CREATOR'] = ('Eric Carlson (erccarls@ucsc.edu)', '')

        # Write energy extension table
        energies = np.array([50*float(galdef['E_gamma_factor'])**i for i in range(cartcube.shape[0])])
        tbhdu = pyfits.BinTableHDU.from_columns([
                pyfits.Column(name='Energy', format='D', array=energies),])
        tbhdu.header['EXTNAME']="ENERGIES"
        
        hdulist = pyfits.HDUList([hdu_new,tbhdu])
        
        # Write to file
        fname_out = fname.split('.')[0]+"_"+t +'_mapcube.fits.gz'
        hdulist.writeto(fname_out,clobber=True)
        
if __name__ == "__main__": 
    fname = sys.argv[1]
    Healpix2Cartesian(fname)