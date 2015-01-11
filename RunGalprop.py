def GenGaldef(
    filename,   # filename for output files and for galdef suffix
    HIModel=1, # 1=galprop classic, 2=3D cube NS,  3=3D F07 <1.5 kpc
    H2Model=1, # 1=galprop classic, 2=3D cube PEB, 3=3D F07 <1.5kpc
    n_spatial_dimensions=3, 
    dx=1., # kpc for dx and dy propagation grid
    dz=.1, # kpc for dz propagation grid
    zmax=4, # halo half-height 
    healpix_order=7,
    IC_isotropic=0,
    computeBremss=0,
    secondary_leptons=1,
    secondary_hadrons=1,
    source_model=1,
    spiral_fraction=0.):
    
    galdef_string='''
Title                = Lorimer distribution, z_h = 4, R_h = 20, T_S = 150, and E(B-V) cut = 5
n_spatial_dimensions = '''+str(n_spatial_dimensions)+'''
r_min                =00.0    min r 
r_max                = 20
dr                   = '''+str(dx)+'''    delta r
z_min                = '''+str(-zmax)+'''      min z 
z_max                = '''+str(zmax)+'''      max z 
dz                   = '''+str(dz)+'''   delta z

x_min                =-15.0   min x 
x_max                =+15.0   max x 
dx                   =  '''+str(dx)+'''   delta x
y_min                =-15.0   min y 
y_max                =+15.0   max y 
dy                   =  '''+str(dx)+'''   delta y

p_min                =1000    min momentum (MV)
p_max                =4000    max momentum  
p_factor             =1.50        momentum factor

Ekin_min             =1.0e1  min kinetic energy per nucleon (MeV)
Ekin_max             =1.0e9  max kinetic energy per nucleon
Ekin_factor          =1.2        kinetic energy per nucleon factor

p_Ekin_grid          = Ekin         p||Ekin alignment 

E_gamma_min          = 50.     min gamma-ray energy (MeV)
E_gamma_max          = 1.e6    max gamma-ray energy (MeV)
E_gamma_factor       = 1.2996566     gamma-ray energy factor
integration_mode     = 0       integr.over part.spec.: =1-old E*logE; =0-PL analyt.

nu_synch_min         = 1.0e6   min synchrotron frequency (Hz)
nu_synch_max         = 1.0e10  max synchrotron frequency (Hz)
nu_synch_factor      = 2.0         synchrotron frequency factor

long_min             =  0.25  gamma-ray intensity skymap longitude minimum (deg);   0 -automatic binning  required to get correct results!
long_max             =359.75  gamma-ray intensity skymap longitude maximum (deg); 360 -automatic binning
lat_min              =-89.75  gamma-ray intensity skymap latitude  minimum (deg); -90 -automatic binning
lat_max              =+89.75  gamma-ray intensity skymap latitude  maximum (deg); +90 -automatic binning
d_long               = 0.5    gamma-ray intensity skymap longitude binsize (deg)
d_lat                = 0.5    gamma-ray intensity skymap latitude  binsize (deg)
healpix_order        = '''+str(healpix_order)+'''      order for healpix skymaps.  7 gives ~0.5 deg and it changes by an order of 2
lat_substep_number   = 1      latitude bin splitting (0,1=no split, 2=split in 2...)
LoS_step             = 0.01   kpc, Line of Sight (LoS) integration step
LoS_substep_number   = 1      number of substeps per LoS integration step (0,1=no substeps)

D0_xx                = 5.35907e+28     diffusion coefficient at reference rigidity
D_rigid_br           =4.0e3    reference rigidity for diffusion coefficient in MV
D_g_1                = 0.33    diffusion coefficient index below reference rigidity
D_g_2                = 0.33    diffusion coefficient index above reference rigidity
diff_reacc           =1        0=no reacc.; 1,2=incl.diff.reacc.; -1==beta^3 Dxx; 11=Kolmogorov+damping; 12=Kraichnan+damping
v_Alfven             = 33.4303         Alfven speed in km s-1

damping_p0           = 1.e6    MV -some rigidity (where CR density is low)
damping_const_G      = 0.02    a const derived from fitting B/C
damping_max_path_L   = 3.e21   Lmax~1 kpc, max free path

convection           =0        1=include convection
v0_conv              =0.       km s-1        v_conv=v0_conv+dvdz_conv*dz   
dvdz_conv            =10.      km s-1 kpc-1  v_conv=v0_conv+dvdz_conv*dz

nuc_rigid_br         = 11491.1      reference rigidity for nucleus injection index in MV
nuc_g_1              = 1.87944        nucleus injection index below reference rigidity
nuc_g_2              = 2.38735        nucleus injection index index above reference rigidity

inj_spectrum_type    = rigidity     rigidity||beta_rig||Etot nucleon injection spectrum type 

electron_g_0         = 1.6      electron injection index below electron_rigid_br0
electron_rigid_br0   = 2178.46         reference rigidity0 for electron injection index in MV
electron_g_1         = 2.41769        electron injection index below reference rigidity
electron_rigid_br    = 2.20561e+06      reference rigidity for electron injection index in MV
electron_g_2         = 4        electron injection index index above reference rigidity

He_H_ratio           = 0.11     He/H of ISM, by number
n_X_CO               = 3
X_CO                 = 2.0E20  conversion factor from CO integrated temperature to H2 column density
X_CO_parameters_0    = 0.597733e20
X_CO_parameters_1    = -0.100183
X_CO_parameters_2    = 0.001284e20
X_CO_parameters_3    = 0.360597
COR_filename         = rbands_co10mm_v2_2001_qdeg.fits.gz
HIR_filename         = rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag2_limit.fits.gz  rbands_hi10_garbage.fits  

B_field_model        = 050100020   bbbrrrzzz    bbb=10*B(0)  rrr=10*rscale zzz=10*zscale
ISRF_file            = ISRF/Standard/Standard.dat ISRF_RMax20_ZMax5_DR0.5_DZ0.1_MW_BB_24092007.fits  (new) input ISRF file
ISRF_filetype        = 3
ISRF_factors         = 1.0,1.0,1.0         ISRF factors for IC calculation: optical, FIR, CMB
ISRF_healpixOrder    = 1

fragmentation        =1        1=include fragmentation
momentum_losses      =1        1=include momentum losses
radioactive_decay    =1        1=include radioactive decay
K_capture            =1        1=include K-capture
ionization_rate      =0        1=compute ionization rate

start_timestep       =1.0e9 
  end_timestep       =1.0e2
timestep_factor      =0.25         
timestep_repeat      =20   number of repeats per timestep in  timetep_mode=1
timestep_repeat2     =0    number of timesteps in timetep_mode=2
timestep_print       =10000  number of timesteps between printings
timestep_diagnostics =10000  number of timesteps between diagnostics
control_diagnostics  =0      control detail of diagnostics

network_iterations   = 2      number of iterations of entire network

prop_r               = 1  1=propagate in r (2D)
prop_x               = 1  1=propagate in x (2D,3D)
prop_y               = 1  1=propagate in y (3D)
prop_z               = 1  1=propagate in z (3D)
prop_p               = 1  1=propagate in momentum

use_symmetry         = 0  0=no symmetry, 1=optimized symmetry, 2=xyz symmetry by copying(3D)

vectorized           = 0  0=unvectorized code, 1=vectorized code

source_specification = 0  2D::1:r,z=0 2:z=0  3D::1:x,y,z=0 2:z=0 3:x=0 4:y=0
source_model         = '''+ str(source_model) +'''  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_model_elec    = 1  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_parameters_0  = 0.2       model 1:alpha
source_parameters_1  = 1.9       model 1:alpha
source_pars_elec_0   = 0.2       model 1:alpha
source_pars_elec_1   = 1.9       model 1:alpha
source_parameters_2  = 5.0    model 1:beta
source_pars_elec_2   = 5.0    model 1:beta
source_parameters_3  = 30.0   model 1:rmax
source_pars_elec_3   = 30.0   model 1:rmax
spiral_fraction      = +''' +str(spiral_fraction)+ '''

n_cr_sources         = 0     number of pointlike cosmic-ray sources   3D only!
cr_source_x_01       = 10.0  x position of cosmic-ray source 1 (kpc)
cr_source_y_01       = 10.0  y position of cosmic-ray source 1
cr_source_z_01       = 0.1   z position of cosmic-ray source 1
cr_source_w_01       = 0.1 sigma width  of cosmic-ray source 1
cr_source_L_01       = 1.0   luminosity of cosmic-ray source 1
cr_source_x_02       = 3.0   x position of cosmic-ray source 2
cr_source_y_02       = 4.0   y position of cosmic-ray source 2
cr_source_z_02       = 0.2   z position of cosmic-ray source 2
cr_source_w_02       = 2.4 sigma width  of cosmic-ray source 2
cr_source_L_02       = 2.0   luminosity of cosmic-ray source 2

SNR_events           = 0    handle stochastic SNR events
SNR_interval         = 1.0e4 time interval in years between SNR in 1 kpc^-3 volume
SNR_livetime         = 1.0e4 CR-producing live-time in years of an SNR
SNR_electron_sdg     = 0.00      delta electron source index Gaussian sigma
SNR_nuc_sdg          = 0.00      delta nucleus  source index Gaussian sigma
SNR_electron_dgpivot = 5.0e3     delta electron source index pivot rigidity (MeV)
SNR_nuc_dgpivot      = 5.0e3     delta nucleus  source index pivot rigidity (MeV)

proton_norm_Ekin     = 1.08e+5 proton kinetic energy for normalization (MeV)
proton_norm_flux     = 3.95063e-09    to renorm nuclei/flux of protons at norm energy (cm^-2 sr^-1 s^-1 MeV^-1)

electron_norm_Ekin   = 2.48e4 3.45e4  electron kinetic energy for normalization (MeV)
electron_norm_flux   = 1.0743e-09    0.40e-9  flux of electrons at normalization energy (cm^-2 sr^-1 s^-1 MeV^-1)
 
max_Z                = 2     maximum number of nucleus Z listed
use_Z_1              = 1
use_Z_2              = 1
use_Z_3              = 1
use_Z_4              = 1
use_Z_5              = 1
use_Z_6              = 1
use_Z_7              = 1
use_Z_8              = 1
use_Z_9              = 1
use_Z_10             = 1 
use_Z_11             = 1
use_Z_12             = 1
use_Z_13             = 1
use_Z_14             = 1
use_Z_15             = 1
use_Z_16             = 1
use_Z_17             = 1
use_Z_18             = 1
use_Z_19             = 1
use_Z_20             = 1 
use_Z_21             = 1
use_Z_22             = 1
use_Z_23             = 1
use_Z_24             = 1
use_Z_25             = 1
use_Z_26             = 1
use_Z_27             = 1
use_Z_28             = 1
use_Z_29             = 0
use_Z_30             = 0 

iso_abundance_01_001 = 1.06e+06   H 
iso_abundance_01_002 =     0.     34.8    
iso_abundance_02_003 =    9.033   He
iso_abundance_02_004 = 7.199e+04    
iso_abundance_03_006 =        0   Li
iso_abundance_03_007 =        0    
iso_abundance_04_009 =        0   Be
iso_abundance_05_010 =        0   B 
iso_abundance_05_011 =        0    
iso_abundance_06_012 =     2819   C 
iso_abundance_06_013 = 5.268e-07    
iso_abundance_07_014 =    182.8   N 
iso_abundance_07_015 = 5.961e-05    
iso_abundance_08_016 =     3822   O 
iso_abundance_08_017 = 6.713e-07    
iso_abundance_08_018 =    1.286    
iso_abundance_09_019 = 2.664e-08   F 
iso_abundance_10_020 =    312.5   Ne
iso_abundance_10_021 = 0.003556    
iso_abundance_10_022 =    100.1    
iso_abundance_11_023 =    22.84   Na
iso_abundance_12_024 =    658.1   Mg
iso_abundance_12_025 =     82.5    
iso_abundance_12_026 =    104.7    
iso_abundance_13_027 =    76.42   Al
iso_abundance_14_028 =    725.7   Si
iso_abundance_14_029 =    35.02    
iso_abundance_14_030 =    24.68    
iso_abundance_15_031 =    4.242   P 
iso_abundance_16_032 =    89.12   S 
iso_abundance_16_033 =   0.3056    
iso_abundance_16_034 =    3.417    
iso_abundance_16_036 = 0.0004281    
iso_abundance_17_035 =   0.7044   Cl
iso_abundance_17_037 = 0.001167    
iso_abundance_18_036 =    9.829   Ar
iso_abundance_18_038 =   0.6357    
iso_abundance_18_040 = 0.001744    
iso_abundance_19_039 =    1.389   K 
iso_abundance_19_040 =    3.022    
iso_abundance_19_041 = 0.0003339    
iso_abundance_20_040 =    51.13   Ca
iso_abundance_20_041 =    1.974    
iso_abundance_20_042 = 1.134e-06    
iso_abundance_20_043 = 2.117e-06    
iso_abundance_20_044 = 9.928e-05    
iso_abundance_20_048 =   0.1099    
iso_abundance_21_045 =    1.635   Sc
iso_abundance_22_046 =    5.558   Ti
iso_abundance_22_047 = 8.947e-06    
iso_abundance_22_048 = 6.05e-07    
iso_abundance_22_049 = 5.854e-09    
iso_abundance_22_050 = 6.083e-07    
iso_abundance_23_050 = 1.818e-05   V 
iso_abundance_23_051 = 5.987e-09    
iso_abundance_24_050 =    2.873   Cr
iso_abundance_24_052 =    8.065    
iso_abundance_24_053 = 0.003014    
iso_abundance_24_054 =   0.4173    
iso_abundance_25_053 =    6.499   Mn
iso_abundance_25_055 =    1.273    
iso_abundance_26_054 =    49.08   Fe
iso_abundance_26_056 =    697.7    
iso_abundance_26_057 =    21.67    
iso_abundance_26_058 =    3.335    
iso_abundance_27_059 =    2.214   Co
iso_abundance_28_058 =    28.88   Ni
iso_abundance_28_060 =     11.9    
iso_abundance_28_061 =   0.5992    
iso_abundance_28_062 =    1.426    
iso_abundance_28_064 =   0.3039

total_cross_section  = 2   total cross section option: 0=L83 1=WA96 2=BP01
cross_section_option = 012    100*i+j  i=1: use Heinbach-Simon C,O->B j=kopt j=11=Webber, 21=ST

t_half_limit         = 1.0e4 year - lower limit on radioactive half-life for explicit inclusion

primary_electrons    = 1    1=compute primary electrons
secondary_positrons  = '''+str(secondary_leptons)+'''    1=compute secondary positrons
secondary_electrons  = '''+str(secondary_leptons)+'''    1=compute secondary electrons
knock_on_electrons   = 0    1,2 1=compute knock-on electrons (p,He) 2= use factor 1.75 to scale pp,pHe
secondary_antiproton = 0    1,2= calculate: 1=uses nuclear scaling; 2=uses nuclear factors (Simon et al 1998)
tertiary_antiproton  = '''+str(secondary_hadrons)+'''    1=compute tertiary antiprotons
secondary_protons    = '''+str(secondary_hadrons)+'''    1=compute secondary protons

gamma_rays           = 2 2    1=compute gamma rays, 2=compute HI,H2 skymaps separately
pi0_decay            = 3    1= old formalism 2=Blattnig et al. 3=Kamae et al.
IC_isotropic         = '''+str(IC_isotropic)+'''    1,2= compute isotropic IC: 1=compute full, 2=store skymap components
IC_anisotropic       = 0    1,2,3= compute anisotropic IC: 1=full, 2=approx., 3=isotropic
synchrotron          = 0    1=compute synchrotron
bremss               = '''+str(computeBremss)+'''    1=compute bremsstrahlung

comment              = the dark matter (DM) switches and user-defined parameters
DM_positrons         = 0   1=compute DM positrons
DM_electrons         = 0   1=compute DM electrons
DM_antiprotons       = 0   1=compute DM antiprotons
DM_gammas            = 0   1=compute DM gammas  

DM_double0           = 2.8    core radius, kpc
DM_double1           = 0.43   local DM mass density, GeV cm-3
DM_double2           = 80.    neutralino mass, GeV
DM_double3           = 40.    positron width distribution, GeV
DM_double4           = 40.    positron branching
DM_double5           = 40.    electron width distribution, GeV
DM_double6           = 30.    electron branching
DM_double7           = 50.    pbar width distribution, GeV
DM_double8           = 40.    pbar branching
DM_double9           =3.e-25  <cross_sec*V>-thermally overaged, cm3 s-1

DM_int0              = 1    isothermal profile
DM_int1              = 1
DM_int2              = 1
DM_int3              = 1
DM_int4              = 1
DM_int5              = 1
DM_int6              = 1
DM_int7              = 1
DM_int7              = 1
DM_int8              = 1
DM_int9              = 1

skymap_format        = 1 1 0 3 0 3 0 3 0 3 1 3 1 3 0 3 0     fitsfile format: 0=old format (the default), 1=mapcube for glast science tools, 
2=both, 3=healpix
output_gcr_full      = 1  output full galactic cosmic ray array
warm_start           = 0  read in nuclei file and continue run

verbose              = 0 -456 -455 -454 -453   verbosity: 0=min,10=max <0: selected debugs
test_suite           = 0  run test suite instead of normal run

n_X_CO_values        = 17
X_CO_values          = 2.82007e+19, 7.2812e+19, 7.47291e+19, 7.56622e+19, 7.76574e+19, 7.92986e+19, 8.2545e+19, 8.54416e+19, 8.26322e+19, 5.55903e+19, 5.57515e+19, 6.99169e+19, 6.6117e+19, 1.09155e+21, 6.72909e+21, 1.43886e+23, 9.351e+23
X_CO_radius          = 0.782569, 1.88942, 2.2729, 2.78797, 3.26916, 3.74009, 4.25494, 4.73877, 5.21537, 5.94654, 6.74566, 7.51648, 8.7184, 11.0052, 14.2598, 17.0855, 19.426

GCR_data_filename    = GCR_data_4.dat
network_iter_compl   =2
electron_source_norm =1
rigid_min            =0
B_field_name         =galprop_original
source_norm          =1
source_values        =0
source_parameters_0  =.2
source_parameters_4  =100
source_parameters_5  =0
source_parameters_6  =0
source_parameters_7  =0
source_parameters_8  =0
source_parameters_9  =0
source_pars_elec_9   =0
source_pars_elec_8   =0
source_pars_elec_5   =0
source_pars_elec_4   =100
source_pars_elec_7   =0
source_pars_elec_6   =0
source_pars_elec_0   =0.2
source_radius        =0
n_source_values      =0
B_field_parameters   =-1,-2,-3,-4,-5,-6,-7,-8,-9,-10
network_iter_sec     =1
n_B_field_parameters =10
propagation_X_CO     =1
rigid_max            =1e100

nHI_model            ='''+str(HIModel)+'''
nH2_model            ='''+str(H2Model)+'''
nHII_model           = 3

#COCube_filename      = CO_PEB_galprop_8500.fits.gz
#HICube_filename      = HI_NS_galprop_8500.fits.gz
COCube_filename      = CO_PEB_galprop.fits.gz
HICube_filename      = HI_NS_kipac_galprop.fits.gz
'''
    
    f = open('//data/galprop2/GALDEF/galdef_54_' + filename, 'wb')
    f.write(galdef_string)
    f.close()
    
    
    # Run Galprop
    import sys
    import subprocess 
    p = subprocess.Popen(['/data/galprop2/bin/galprop -r ' + filename + ' -o /data/galprop2/output/' ,], 
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
    # Grab stdout line by line as it becomes available.  This will loop until p terminates.
    while p.poll() is None:
        l = p.stderr.readline() # This blocks until it receives a newline.
        sys.stderr.flush()
        print l.rstrip('\n') 
        
        #l = p.stdout.readline() # This blocks until it receives a newline.
        #print l.rstrip('\n') 
        #sys.stdout.flush()
    # When the subprocess terminates there might be unconsumed output 
    # that still needs to be processed.
    print p.stdout.read()
        
   

# GenGaldef('base_no_secondary', dx=3,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#             secondary_leptons=0,secondary_hadrons=0)
GenGaldef('NSPEB_no_secondary', dx=3,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
            secondary_leptons=0,secondary_hadrons=0,HIModel=2,H2Model=1)

#GenGaldef('F07_no_secondary', dx=3,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#            secondary_leptons=0,secondary_hadrons=0,HIModel=3,H2Model=1)

# GenGaldef('base_no_secondary', dx=1.5,dz=.25,IC_isotropic=1,computeBremss=1,secondary_leptons=0,secondary_hadrons=0)
# GenGaldef('NSPEB', dx=1.5,dz=.25,IC_isotropic=1,computeBremss=1,secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=2)




















    
