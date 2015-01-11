#--------------------------------------------------------------------------
# Analysis.py
# This class describes templates used for diffuse analysis.  This is really
# just a datacube combined with various analysis settings such as the 
# spectrum, fixed limits, etc.. 
# 
# Author: Eric Carlson (erccarls@ucsc.edu) 11/20/2014
#--------------------------------------------------------------------------

class Template():
    def __init__(self, healpixCube, fixSpectrum=False, fixNorm=False, limits=[0., 10.], value=1, sourceClass='GEN',
                 valueUnc=None):
        '''

        :param healpixCube: Actually a 2-d array with first index selecting energy and second index selecting the healpix index
        :param fixSpectrum: If True, the relative normalizations of each energy bin will be held fixed for this template, but
                        the overall normalization is free
        :param fixNorm:     Fix the overall normalization of this template.  This implies fixSpectrum=True.
        :param limits:      Specify range of template normalizations.
        :param value:       Initial value for the template normalization, or a vector of values for each spectral bin.
        :param sourceClass: Type of template, used for later processing.  Can be 'ISO' (isotropic), 'PSC' (point source),
                        'GEN' (General).
        :param valueUnc:    similar format to value, used to apply external constraints on the
        '''

        # Currently only type that matters is PSC as it is used for spectral weighting. 
        self.healpixCube = healpixCube
        self.healpixCubeMasked = None
        self.fixSpectrum = fixSpectrum
        self.fixNorm = fixNorm
        self.limits = limits
        self.value = value
        self.sourceClass = sourceClass
        self.valueUnc = valueUnc
        self.valueError = None  # Stores the fitting errors on the values

        #if fixNorm:
        #    self.limits = [value, value]


