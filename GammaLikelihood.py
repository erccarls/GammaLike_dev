#--------------------------------------------------------------------------
# Likelihood.py
# This class contains the settings for a binned likelihood analysis.
# Author: Eric Carlson (erccarls@ucsc.edu) 11/20/2014
#--------------------------------------------------------------------------

from iminuit import Minuit
import tempfile
import numpy as np
from scipy.optimize import *
import time
import imp


def RunLikelihood(analysis, print_level=0, use_basinhopping=False, start_fresh=False, niter_success=30, tol=100000.):
    """
    Calculates the maximum likelihood set of parameters given an Analyis object (see analysis.py).

    :param analysis: An Analysis object with valid binned_data and at least one template in templateList.
    :param print_level: Verbosity for the fitting 0=quiet, 1=full output.
    :param use_basinhopping: Slower, but more accurate use of scipy basinhopping algorithm after migrad convergence
    :param start_fresh: If use_basinhopping==True, then start_fresh==True does not run migrad before basinhopping.
    :param niter_success: Setting for basinhopping algorithm.  See scipy docs.
    :param tol: EDM Tolerance for migrad convergence.
    :returns (m, res):
            m: iminuit result object\n
            res: scipy.minimize result.  None if use_basinhopping==False

    """

    if analysis.binned_data is None:
        raise Exception("No binned photon data.")

    # Check that the input data all has the correct dimensionality.
    shape = analysis.binned_data.shape
    for key, t in analysis.templateList.items():
        if t.healpixCube.shape != shape:
            raise Exception("Input template " + key + " does not match the shape of the binned data.")

    if analysis.psc_weights is None:
        print 'Warning! Pixel weights not initialized.  Default to equal weighting.'

    #---------------------------------------------------------------------------
    # Select only unmasked pixels.  Less expensive to do here rather than later.
    start = time.time()
    mask_idx = np.where(analysis.mask != 0)[0]
    # Iterate through templates and apply masking
    for key, t in analysis.templateList.items():
        tmpcube = np.zeros(shape=(t.healpixCube.shape[0], len(mask_idx)))
        for Ebin in range(t.healpixCube.shape[0]):
            tmpcube[Ebin] = t.healpixCube[Ebin, mask_idx]
        t.healpixCubeMasked = tmpcube

    # mask the data as well.
    masked_data = analysis.binned_data[:, mask_idx].astype(np.float32)
    #if mask_idx.shape[0] < 10:
    #    raise Exception('Masked number of pixels <10. Has mask been initialized?')

    if print_level > 0:
        print "Masking completed in", "{:10.4e}".format(time.time()-start), 's'

    #---------------------------------------------------------------------------
    # We first enumerate all of the fit parameters and add them to the args and 
    # model strings.  These strings will then be used to generate the likelihood 
    # code. 
    start = time.time()
    model, args = ['        model['+str(i)+']=' for i in range(analysis.binned_data.shape[0])], ''

    # External constraint
    extConstraint ='        chi2_ext='

    # pyminut initial step size and print level
    kwargs = {'errordef': 0.5, 'print_level': print_level}

    # This map keeps track of the arguments order since scipy minimizer returns values in tuple not dict.
    argMap, x0, bounds = [], [], []


    # Iterate over the input templates and add each template to the fitting functions args.
    for key, t in analysis.templateList.items():
        if t.fixSpectrum:
            # Just add an overall normalization parameter and the same for the model.
            args += key + ','
            argMap.append(key)
            for Ebin in range(t.healpixCube.shape[0]):
                model[Ebin] += key + "*self.templateList['"+key+"'].healpixCubeMasked["+str(Ebin)+"]+"
            # Set initial value and limits 
            kwargs[key] = t.value  # default initial value
            kwargs['limit_'+key] = t.limits
            kwargs['error_'+key] = .25
            kwargs['fix_'+key] = t.fixNorm

            if t.valueUnc is not None:
                for Ebin in range(t.healpixCube.shape[0]):
                    extConstraint += '(('+key + '-1)/' + str(t.valueUnc[Ebin]) + ')**2+'

            # For the second minimizer
            x0.append(t.value)
            if t.fixNorm:
                bounds.append([t.value, t.value])
            else:
                bounds.append(t.limits)

        else:
            # Add a normalization component for each spectral bin in each template.
            for Ebin in range(t.healpixCube.shape[0]):
                args += key + '_' + str(Ebin)+','
                argMap.append(key + '_' + str(Ebin))
                model[Ebin] += key + '_' + str(Ebin) + "*self.templateList['"+key+"'].healpixCubeMasked["+str(Ebin)+"]+"
                kwargs['error_' + key + '_' + str(Ebin)] = .25  # Initial step size
                # If we have an array or list of initial values....
                if type(t.value) == type([]) or type(t.value) == type(np.array([])):
                    kwargs[key + '_' + str(Ebin)] = t.value[Ebin]  # default initial value
                    x0.append(t.value[Ebin])
                    if t.fixNorm:
                        bounds.append([t.value[Ebin], t.value[Ebin]])
                    else:
                        bounds.append(t.limits)
                else:
                    kwargs[key + '_' + str(Ebin)] = t.value  # default initial value
                    x0.append(t.value)
                    if t.fixNorm:
                        bounds.append([t.value, t.value])
                    else:
                        bounds.append(t.limits)
                if t.valueUnc is not None:
                    extConstraint += '(('+key + '_' + str(Ebin) + '-1)/' + str(t.valueUnc[Ebin]) + ')**2+'

                # Currently no support for limits on each bin normalization
                kwargs['limit_'+key + '_' + str(Ebin)] = t.limits
                # If we are supposed to fix the values, then fix them.
                kwargs['fix_'+key + '_' + str(Ebin)] = t.fixNorm




    # remove the trailing '+' sign and add newline. 
    model = [model[i][:-1] + '\n' for i in range(analysis.binned_data.shape[0])]
    extConstraint = extConstraint[:-1]
    if extConstraint is '        chi2_ext':
        extConstraint = '        chi2_ext=0'

    # Combine the individual strings into bins.
    master_model = ''
    for m in model:
        master_model += m


    #---------------------------------------------------------------------------
    # Generate the function! There are probably better ways to do this, but 
    # this works to generate the required function signatures dynamically via 
    # code generation. 

    f = tempfile.NamedTemporaryFile(delete=False)

    f.write("""
import numpy as np 
import time 



class like():
    def __init__(self, templateList, data, psc_weights):
        self.templateList = templateList
        self.data = data
        self.use_cuda = True
        self.psc_weights = psc_weights
        try:
            import cudamat as cm
            self.cm = cm
            cm.cublas_init()
            self.cmData = cm.CUDAMatrix(data)
            self.cm_psc_weights = cm.CUDAMatrix(psc_weights)
        except:
            self.use_cuda = False
        self.ncall=0


    def f(self,"""+args+"""):
        # init model array 
        model = np.zeros(shape=self.templateList[self.templateList.keys()[0]].healpixCubeMasked.shape)
        # sum the models 
""" + master_model +"""
""" + extConstraint + """
        

        #------------------------------------------------
        # Uncomment this for CPU mode (~10 times slower than GPU depending on array size)
        if self.use_cuda == False:
            neg_loglikelihood_cpu = np.sum(self.psc_weights*(model-self.data*np.log(model)))
        
        #------------------------------------------------
        # Uncomment here for GPU mode using CUDA + cudamat libraries
        else:
            cmModel = self.cm.CUDAMatrix(model)
            cmModel_orig = cmModel.copy()
            neg_loglikelihood = cmModel_orig.subtract(self.cm.log(cmModel).mult(self.cmData)).mult(self.cm_psc_weights).sum(axis=0).sum(axis=1).asarray()[0,0]

        #if self.ncall%500==0: print self.ncall, neg_loglikelihood
        #self.ncall+=1

        if np.isnan(neg_loglikelihood):
            neg_loglikelihood=1e10
        return neg_loglikelihood + chi2_ext/2.
        
        
    def f2(self,x0):
        """+args+""" = x0
        model = np.zeros(shape=self.data.shape)
""" + master_model +"""
""" + extConstraint + """
    
        # gpu mode
        if self.use_cuda:
            cmModel = self.cm.CUDAMatrix(model)
            cmModel_orig = cmModel.copy()
            neg_loglikelihood = cmModel_orig.subtract(self.cm.log(cmModel).mult(self.cmData)).mult(self.cm_psc_weights).sum(axis=0).sum(axis=1).asarray()[0,0]

        # cpu mode
        else:
            neg_loglikelihood = np.sum(self.psc_weights*(-self.flat_data*np.log(model)+model))

        if np.isnan(neg_loglikelihood):
            neg_loglikelihood=1e10

        return neg_loglikelihood + chi2_ext/2.
        """)
    f.close()
    #---------------------------------------------------------------------------
    # Now load the source 
    foo = imp.load_source('tmplike', f.name)
    if print_level > 0:
        print "Code generation completed in", "{:10.4e}".format(time.time()-start), 's'
        try:
            import cudamat
            print "Using GPU mode. (Successful import of cudamat module.)"
        except:
            print "Fallback to CPU mode.  (Failed to import cudamat libraries.)"

    start = time.time()
    like = foo.like(analysis.templateList, masked_data, analysis.psc_weights[:, mask_idx].astype(np.float32))

    # Init migrad 
    m = Minuit(like.f, **kwargs)
    m.tol = tol  # TODO: why does m.tol need to be so big to converge when errors are very small????
    #m.migrad(ncall=200000, precision=1e-15)
    if not start_fresh:
        m.migrad(ncall=1e6)

    #m.minos(maxcall=10000,sigma=2.)

    if print_level > 0:
        print "Migrad completed fitting", "{:10.2e}".format(time.time()-start), 's'

    for i, key in enumerate(argMap):
        if not start_fresh:
            x0[i] = m.values[key]

    if use_basinhopping:
        if print_level > 0:
            disp = True
        else:
            disp = False
        res = basinhopping(like.f2, x0, niter=20000, disp=disp,
                           stepsize=.1, minimizer_kwargs={'bounds': bounds}, niter_success=niter_success)

        # Can also try pswarm method.
        #lb, ub = np.array(bounds)[:,0], np.array(bounds)[:,1]

        #from openopt import GLP
        #p = GLP(like.f2, startpoint=x0, lb=np.zeros(len(x0)), ub=np.ones(len(x0)), maxIter=niter_success,  maxFunEvals=1e6)
        #res = p.minimize('pswarm', iprint=3, iterObjFunTextFormat='%0.8e')

        # Generate a dict of the best fit values against the keys like iMinuit object has.
        res_vals = {}
        for i, key in enumerate(argMap):
            #res_vals[key] = res.xf[i]
            res_vals[key] = res.x[i]

        if print_level > 0:
            print "Basinhopping completed fit completed in", "{:10.4e}".format(time.time()-start), 's'

        return m, res, res_vals
    else:
        return m, None, None
    
          












# def compute_likelihood(data,sidebands,print_level=0,fixed={},f0={},
#     use_basinhopping=False,negative_cont=False,niter_success=30,
#     startFresh=False, negative_iso=False):
#     '''
#     returns the log likelihood given the input counts (data) and the model expectation.
#     '''
#     model, args = '',''
#     for key in sidebands: args  += key + ','
#     for key in sidebands: model += key + "*self.sidebands['"+key+"']['photons_binned']+"
#     model +="0"
#     kwargs={'errordef':0.5,'print_level':print_level}
#     for key in sidebands: 
#         kwargs[key]=.1
#         kwargs['limit_'+key]=(0,1e10)
#         if negative_cont==True and 'Cont' in key:
#                     kwargs['limit_'+key]=(-1e10,1e12)
#         if negative_iso==True and 'Iso' in key:
#                     kwargs['limit_'+key]=(-1e10,1e12)
#         kwargs['error_'+key]=.02
#         #if key=='DM': 
#         #    kwargs['error_'+key]=100
#         #    kwargs[key]=3.248832e+02
#     for key in fixed:
#         kwargs['fix_'+key]=True
#         kwargs[key]=fixed[key]

#     for key in f0:
#         kwargs[key]=f0[key]
    
#     #norms={}
#     #sidebands_copy=copy.deepcopy(sidebands)
#     # normalize each component so we don't encounter overflows in the fit
#     #for key in sidebands: 
#     #    norms[key]= np.mean(sidebands[key]['photons_binned'])
#     #    sidebands_copy[key]['photons_binned']/=norms[key]
        
        
#     f = tempfile.NamedTemporaryFile(delete=False)
    
    
#     f.write("""
# import numpy as np 
# class like():
#     def __init__(self,sidebands,flat_data):
#         self.sidebands=sidebands
#         self.flat_data = flat_data
#         #for key in sidebands:print key
    
#     def f(self,"""+args+"""):
#         model = ("""+ model +""").flatten()


#         # Exlude zero photon bins in the model (outside instrumental range)
#         use_idx = np.where( (model!=0) & (self.flat_data!=0))[0]
#         #use_idx = np.nonzero(model)        
#         #use_idx = np.nonzero(self.flat_data)        
#         neg_loglikelihood = -np.sum(self.flat_data[use_idx]*np.log(model[use_idx])-model[use_idx])

        
#         #mu, n = model[use_idx], self.flat_data[use_idx]
#         #neg_loglikelihood = -np.sum(n*np.log(n/mu))
        
#         return neg_loglikelihood
        
        
#     def f2(self,x0):
#         """+args+""" = x0
#         model = ("""+ model +""").flatten()

#         # Exlude zero photon bins in the model (outside instrumental range)
#         #use_idx = np.nonzero(model)
#         use_idx = np.where( (model!=0) & (self.flat_data!=0))[0]
#         # Original 
#         neg_loglikelihood = -np.sum(self.flat_data[use_idx]*np.log(model[use_idx])-model[use_idx])
                
#         return neg_loglikelihood
#         """)
#     f.close()
    
#     import imp
#     foo = imp.load_source('tmplike', f.name)
#     like = foo.like(sidebands, data.flatten())
    
#     m = Minuit(like.f,**kwargs)
    
#     m.migrad(ncall=20000)
#     #m.minos()
#     #for key in m.values:
#     #    m.values[key] *=norms[key]  
#     x0 = []
#     bounds=[]
#     for key in sidebands:
#         if startFresh==False: x0.append(m.values[key])
#         else: x0.append(.5)
        
#         if key not in fixed: bounds.append((0,10000))
#         else: bounds.append((fixed[key],fixed[key]))

#     res = None
#     if (use_basinhopping==True):
#         res = basinhopping(like.f2,x0,niter=2000,disp=bool(print_level),
#                             stepsize=.02,minimizer_kwargs={'bounds':bounds},niter_success=niter_success)
       
#         return m, res
#     else: return m
          



# def compute_likelihood_gaussian(data,sidebands,print_level=0,fixed={},f0={},use_basinhopping=False,negative_cont=False,niter_success=30,negative_iso=False):
#     '''
#     returns the log likelihood given the input counts (data) and the model expectation.
#     '''
#     model, args = '',''
#     for key in sidebands: args  += key + ','
#     for key in sidebands: model += key + "*self.sidebands['"+key+"']['photons_binned']+"
#     model +="0"
#     kwargs={'errordef':0.5,'print_level':print_level}
#     for key in sidebands: 
#         kwargs[key]=.1
#         kwargs['limit_'+key]=(0,1e10)
#         if negative_cont==True and 'Cont' in key:
#                     kwargs['limit_'+key]=(-1e10,1e12)
#         if negative_iso==True and 'Iso' in key:
#                     kwargs['limit_'+key]=(-1e10,1e12)
#         kwargs['error_'+key]=.02
#         #if key=='DM': 
#         #    kwargs['error_'+key]=100
#         #    kwargs[key]=3.248832e+02
#     for key in fixed:
#         kwargs['fix_'+key]=True
#         kwargs[key]=fixed[key]

#     for key in f0:
#         kwargs[key]=f0[key]
    
#     #norms={}
#     #sidebands_copy=copy.deepcopy(sidebands)
#     # normalize each component so we don't encounter overflows in the fit
#     #for key in sidebands: 
#     #    norms[key]= np.mean(sidebands[key]['photons_binned'])
#     #    sidebands_copy[key]['photons_binned']/=norms[key]
        
        
#     f = tempfile.NamedTemporaryFile(delete=False)
    
    
#     f.write("""
# import numpy as np 
# class like():
#     def __init__(self,sidebands,flat_data):
#         self.sidebands=sidebands
#         self.flat_data = flat_data
#         #for key in sidebands:print key
    
#     def f(self,"""+args+"""):
#         model = ("""+ model +""").flatten()


#         # Exlude zero photon bins in the model (outside instrumental range)
#         use_idx = np.nonzero(model)
#         neg_loglikelihood = -np.sum(self.flat_data[use_idx]*np.log(model[use_idx])-model[use_idx])

#         return neg_loglikelihood
        
        
#     def f2(self,x0):
#         """+args+""" = x0
#         model = ("""+ model +""").flatten()


#         # Exlude zero photon bins in the model (outside instrumental range)
#         use_idx = np.where((model!=0) & (self.flat_data!=0))[0]
#         neg_loglikelihood =       np.sum(-2*np.log(2*np.pi*np.abs(self.flat_data[use_idx]))
#                          -(model[use_idx] - self.flat_data[use_idx])**2/(2.*np.abs(self.flat_data[use_idx])))
#         #print neg_loglikelihood
#         return neg_loglikelihood
#         """)

#     f.close()
    
#     import imp
#     foo = imp.load_source('tmplike', f.name)
#     like = foo.like(sidebands, data.flatten())
    
#     m = Minuit(like.f,**kwargs)
    
#     m.migrad(ncall=20000)
#     #m.minos()
#     #for key in m.values:
#     #    m.values[key] *=norms[key]  
#     x0 = []
#     bounds=[]
#     for key in sidebands:
#         x0.append(m.values[key])
#         if key not in fixed: bounds.append((0,10000))
#         else: bounds.append((fixed[key],fixed[key]))

#     res = None
#     if (use_basinhopping==True):
#         res = basinhopping(like.f2,x0,niter=2000,disp=bool(print_level),
#                             stepsize=.02,minimizer_kwargs={'bounds':bounds},niter_success=niter_success)
       
#         return m, res
#     else: return m
          


# def compute_likelihood_twomaps(data,sideband,print_level=0,f0=[],fixed=[],):
#     '''
#     returns the log likelihood given the input counts (data) and the model expectation.
#     '''
    
#     flat_data = data.flatten()
#     def f(sideband_norm=0.5):
#         model = ( sideband_norm*sideband).flatten()
#         # Exlude zero photon bins in the model (outside instrumental range)
#         use_idx = np.nonzero(model)
#         neg_loglikelihood = -np.sum(flat_data[use_idx]*np.log(model[use_idx])-model[use_idx])
        
#         return neg_loglikelihood
    
        
#     m = Minuit(f,
#                 sideband_norm=7.156462e-02,limit_sideband_norm=(0,1e2),error_sideband_norm=.025,
#                 errordef=0.5)
    
#     m.migrad(ncall=2000)
#     return m
