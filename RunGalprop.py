import numpy as np


def GenGaldef(
    filename,   # filename for output files and for galdef suffix
    HIModel=1, # 1=galprop classic, 2=3D cube NS,  3=3D F07 <1.5 kpc
    H2Model=1, # 1=galprop classic, 2=3D cube PEB, 3=3D F07 <1.5kpc
    n_spatial_dimensions=3, 
    dx=.5, # kpc for dx and dy propagation grid
    dz=.125, # kpc for dz propagation grid
    zmax=3.5, # halo half-height 
    xmax=20,
    healpix_order=8,
    IC_isotropic=1,
    computeBremss=1,
    secondary_leptons=1,
    secondary_hadrons=0,
    spiral_fraction=.2,
    skymap_format=3,
    single_component=0,
    H2_filename = 'CO_Pohl_galprop_8500.fits',
    HI_filename = 'HI_Pohl_galprop_8500.fits',  
    H2_filename_rlb='CO_Pohl_8500_rlb.fits',
    HI_filename_rlb='HI_Pohl_8500_rlb.fits',
    D_0 = 7.2e+28, 
    D_zz = 1., 
    v_Alfven = 35., 
    dvdz = 0,
    B_0 = 7.2,
    r_b = 5.,
    z_b = 1.,
    isrf_opt_fir=.86,
    COR_filename = "rbands_co10mm_v2_2001_qdeg_9R_new.fits.gz",
    HIR_filename = "rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag5_limit_9R_new.fits",
    cr_source = 'SNR',  # can be OB, Yusifov, Lorimer, or SNR
    n_XCO = 9, 
    numProc=32,
    ):

    

    if cr_source == 'Lorimer':
        CRDist = '''
source_specification = 0  2D::1:r,z=0 2:z=0  3D::1:x,y,z=0 2:z=0 3:x=0 4:y=0
source_model         = 1  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_model_elec    = 1  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_parameters_1  = 1.9       model 1:alpha
source_pars_elec_1   = 1.9       model 1:alpha
source_parameters_2  = 5.0    model 1:beta
source_pars_elec_2   = 5.0    model 1:beta
source_parameters_3  = 30.0   model 1:rmax
source_pars_elec_3   = 30.0   model 1:rmax
'''
    elif cr_source == 'OB':
        CRDist = '''
source_specification = 0  2D::1:r,z=0 2:z=0  3D::1:x,y,z=0 2:z=0 3:x=0 4:y=0
source_model         = 8  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_model_elec    = 8  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_parameters_1  = 1.25 2.35       model 1:alpha
source_pars_elec_1   = 1.25 2.35       model 1:alpha
source_parameters_2  = 3.56 5.56283    model 1:beta
source_pars_elec_2   = 3.56 5.56283    model 1:beta
source_parameters_3  = 15.0   model 1:rmax
source_pars_elec_3   = 15.0   model 1:rmax
n_source_values      = 18
source_values        =  .17,  0.75,  1.44,  4.53,  4.26,  4.,    2.18,  2.,    2.31,  0.99,   0.53,   0.71,   0.41,   0.25,   0.1,    0.03,   0.04,   0.01
source_radius        = 2.125, 2.975, 3.825, 4.675, 5.525, 6.375, 7.225, 8.075, 8.925, 9.775, 10.625, 11.475, 12.325, 13.175, 14.025, 14.875, 15.725, 16.575
'''
    elif cr_source == 'Yusifov':
        CRDist = '''
source_specification = 0  2D::1:r,z=0 2:z=0  3D::1:x,y,z=0 2:z=0 3:x=0 4:y=0
source_model         = 1  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_model_elec    = 1  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_parameters_1  = 1.64       model 1:alpha
source_pars_elec_1   = 1.64       model 1:alpha
source_parameters_2  = 4.01       model 1:beta
source_pars_elec_2   = 4.01       model 1:beta
source_parameters_3  = 35.0   model 1:rmax
source_pars_elec_3   = 35.0   model 1:rmax
source_parameters_4  = 30.0   model 1:rconst
source_pars_elec_4   = 30.0   model 1:rconst
source_parameters_5  = 0.55   model 1:roff
source_pars_elec_5   = 0.55   model 1:roff
n_source_values      = 7
source_values        = 0, 4.38, 2.31, 2.25, 0.82, 0.82,  0
source_radius        = 0,    2,    5,    7,    9,   12, 15
''' 
    elif cr_source == 'SNR':
        CRDist = '''
source_specification = 0  2D::1:r,z=0 2:z=0  3D::1:x,y,z=0 2:z=0 3:x=0 4:y=0
source_model         = 2  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_model_elec    = 2  0=zero 1=parameterized  2=Case&B 3=pulsars 4= 5=S&Mattox 6=S&Mattox with cutoff
source_parameters_1  = 1.25 2.35       model 1:alpha
source_pars_elec_1   = 1.25 2.35       model 1:alpha
source_parameters_2  = 3.56 5.56283    model 1:beta
source_pars_elec_2   = 3.56 5.56283    model 1:beta
source_parameters_3  = 15.0   model 1:rmax
source_pars_elec_3   = 15.0   model 1:rmax
'''
    else: 
        raise('Source Distribution Not Supported')
    
    aniso = 1
    if D_zz == 1: 
        aniso = 0


    galdef_string='''
Title                = Lorimer distribution, z_h = 4, R_h = 20, T_S = 150, and E(B-V) cut = 5
n_spatial_dimensions = '''+str(n_spatial_dimensions)+'''
r_min                = 00.0    min r 
r_max                = 20
dr                   = '''+str(dx)+'''    delta r
z_min                = '''+str(-zmax)+'''      min z 
z_max                = '''+str(zmax)+'''      max z 
dz                   = '''+str(dz)+'''   delta z

x_min                =-'''+str(xmax)+'''   min x 
x_max                =+'''+str(xmax)+'''   max x 
dx                   =  '''+str(dx)+'''   delta x
y_min                =-'''+str(xmax)+'''   min y 
y_max                =+'''+str(xmax)+'''   max y 
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

long_min             =  0.125  gamma-ray intensity skymap longitude minimum (deg);   0 -automatic binning  required to get correct results!
long_max             =359.875  gamma-ray intensity skymap longitude maximum (deg); 360 -automatic binning
lat_min              =-89.875  gamma-ray intensity skymap latitude  minimum (deg); -90 -automatic binning
lat_max              =+89.875  gamma-ray intensity skymap latitude  maximum (deg); +90 -automatic binning
d_long               = 0.25    gamma-ray intensity skymap longitude binsize (deg)
d_lat                = 0.25    gamma-ray intensity skymap latitude  binsize (deg)
healpix_order        = '''+str(healpix_order)+'''      order for healpix skymaps.  7 gives ~0.5 deg and it changes by an order of 2
lat_substep_number   = 1      latitude bin splitting (0,1=no split, 2=split in 2...)
LoS_step             = 0.02   kpc, Line of Sight (LoS) integration step
LoS_substep_number   = 1      number of substeps per LoS integration step (0,1=no substeps)


Diffusion_aniso      = '''+str(aniso)+'''      0=isotropic 1=anisotropic diffusion
D0_xx                = '''+str(D_0)+'''     diffusion coefficient at reference rigidity
D0_zz                = '''+str(D_zz*D_0)+'''

D_rigid_br           = 4.0e3    reference rigidity for diffusion coefficient in MV
D_g_1                = 0.33    diffusion coefficient index below reference rigidity
D_g_2                = 0.33    diffusion coefficient index above reference rigidity
diff_reacc           = 1        0=no reacc.; 1,2=incl.diff.reacc.; -1==beta^3 Dxx; 11=Kolmogorov+damping; 12=Kraichnan+damping
v_Alfven             = '''+str(v_Alfven)+'''         Alfven speed in km s-1


damping_p0           = 1.e6    MV -some rigidity (where CR density is low)
damping_const_G      = 0.02    a const derived from fitting B/C
damping_max_path_L   = 3.e21   Lmax~1 kpc, max free path

convection           =1        1=include convection
v0_conv              =0.       km s-1        v_conv=v0_conv+dvdz_conv*dz   
dvdz_conv            ='''+str(dvdz)+'''      km s-1 kpc-1  v_conv=v0_conv+dvdz_conv*dz

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
n_X_CO               = '''+str(n_XCO)+'''  #9 is MS2004 
X_CO                 = 2.0E20  conversion factor from CO integrated temperature to H2 column density
X_CO_parameters_0    = 0.597733e20
X_CO_parameters_1    = -0.100183
X_CO_parameters_2    = 0.001284e20
X_CO_parameters_3    = 0.360597

COR_filename         = '''+COR_filename+'''
HIR_filename         = '''+HIR_filename+'''

B_field_model        = '''+"%03.0f"%(B_0*10)+"%03.0f"%(r_b*10)+"%03.0f"%(z_b*10)+'''  050100020   bbbrrrzzz    bbb=10*B(0)  rrr=10*rscale zzz=10*zscale
ISRF_file            = ISRF/Standard/Standard.dat ISRF_RMax20_ZMax5_DR0.5_DZ0.1_MW_BB_24092007.fits  (new) input ISRF file
ISRF_filetype        = 3
ISRF_factors         = '''+str(isrf_opt_fir)+','+str(isrf_opt_fir)+''',1.0         ISRF factors for IC calculation: optical, FIR, CMB
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

''' + CRDist+'''

spiral_fraction      = ''' +str(spiral_fraction)+ '''

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

proton_norm_Ekin     = 1.00e+5 proton kinetic energy for normalization (MeV)
proton_norm_flux     = 4.8e-09    to renorm nuclei/flux of protons at norm energy (cm^-2 sr^-1 s^-1 MeV^-1)

electron_norm_Ekin   = 3.45e4 3.45e4  electron kinetic energy for normalization (MeV)
electron_norm_flux   = 0.49e-09    0.40e-9  flux of electrons at normalization energy (cm^-2 sr^-1 s^-1 MeV^-1)


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

skymap_format        =  '''+str(skymap_format)+''' 1 0 3 0 3 0 3 0 3 1 3 1 3 0 3 0     fitsfile format: 0=old format (the default), 1=mapcube for glast science tools, 2=both, 3=healpix
output_gcr_full      = 1  output full galactic cosmic ray array
warm_start           = 0  read in nuclei file and continue run

verbose              = 0 -456 -455 -454 -453   verbosity: 0=min,10=max <0: selected debugs
test_suite           = 0  run test suite instead of normal run

n_X_CO_values        = 17
X_CO_values          = 3.61934e+19, 1.01355e+20, 1.0438e+20, 1.05894e+20, 1.11434e+20, 1.08569e+20, 1.15424e+20, 1.18716e+20, 1.2047e+20, 1.22475e+20, 1.32743e+20, 1.40205e+20, 7.21006e+19, 7.00169e+20, 2.45472e+21, 1.31792e+22, 5.3247e+22
X_CO_radius          = 0.778294, 1.88945, 2.27302, 2.78825, 3.2712, 3.74291, 4.25497, 4.73837, 5.22469, 5.98212, 6.74356, 7.48884, 8.67888, 10.9788, 13.6244, 17.4333, 19.8056

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


COCube_filename      = '''+H2_filename+'''
COCube_rlb_filename  = '''+H2_filename_rlb+'''
HICube_filename      = '''+HI_filename+'''
HICube_rlb_filename  = '''+HI_filename_rlb+'''

uniform_emiss        = 0
renorm_off           = 0
single_component     = '''+str(single_component) + ''' 
gamma_rays           = 2 2    1=compute gamma rays, 2=compute HI,H2 skymaps separately

'''
    
    with open('/pfs/carlson/galprop/GALDEF/galdef_54_' + filename, 'wb') as f :
        f.write(galdef_string)
    
    # Run Galprop
    import sys
    import subprocess 
    import os 
    from time import sleep

    f = open('/pfs/carlson/galprop/GALDEF/run_script_' + filename+'.sh', 'wb') 
    f.write("""#!/bin/bash
        
while true; 
do
   num=`qstat -a | grep galprop | wc -l`
   test $num -le """+str(numProc)+""" && break
   sleep 1
done


cat <<EOS | qsub -V -q hyper -S /bin/bash -N galprop_"""+filename+""" -l nodes=1:ppn=32,walltime=24:00:00 - 
/pfs/carlson/galprop/bin/galprop -r """ + filename + """ -o /pfs/carlson/galprop/output/
cd /pfs/carlson/GCE_sys/
python /pfs/carlson/GCE_sys/GenDiffuseModel_X_CO_3zone.py /pfs/carlson/galprop/output """ + filename + """ /pfs/carlson/galprop/GALDEF
#python /pfs/carlson/GCE_sys/Healpix2Mapcube.py /pfs/carlson/galprop/output """ + filename +""".hdf5 

EOS
        """)
    f.close()
    
    #sleep(.3)
    os.chmod('/pfs/carlson/galprop/GALDEF/run_script_' + filename+'.sh', 0755)

    p = subprocess.Popen(['/pfs/carlson/galprop/GALDEF/run_script_'+filename+'.sh' ,], 
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
    # print p.stdout.read()
        
for sf in [3,]:
    continue
    # GenGaldef('base_2D', 
    #     HIModel=1, # 1=galprop classic, 2=3D cube NS,  3=3D F07 <1.5 kpc
    #     H2Model=1, # 1=galprop classic, 2=3D cube PEB, 3=3D F07 <1.5kpc
    #     n_spatial_dimensions=3, 
    #     dx=1, # kpc for dx and dy propagation grid
    #     dz=.1, # kpc for dz propagation grid
    #     zmax=6, # halo half-height 
    #     xmax=30,
    #     healpix_order=8,
    #     IC_isotropic=1,
    #     computeBremss=1,
    #     secondary_leptons=1,
    #     secondary_hadrons=0,
    #     spiral_fraction=0,
    #     skymap_format=sf,
    #     single_component=0,
    #     H2_filename = 'CO_Pohl_galprop_8500.fits',
    #     HI_filename = 'HI_Pohl_galprop_8500.fits',  
    #     H2_filename_rlb='CO_Pohl_8500_rlb.fits',
    #     HI_filename_rlb='HI_Pohl_8500_rlb.fits',
    #     D_0 = 7.86e+28, 
    #     v_Alfven = 35., 
    #     dvdz = 10,
    #     B_0 = 5,
    #     r_b = 10.,
    #     z_b = 2.,
    #     isrf_opt_fir=1,
    #     COR_filename = "rbands_co10mm_v2_2001_qdeg_9R.fits",
    #     HIR_filename = "rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag2_limit_9R.fits",
    #     cr_source = 'Yusifov',  # can be OB, Yusifov, Lorimer, or SNR
    #     n_XCO = 3, )
    # GenGaldef('base_3D', 
    #     HIModel=1, # 1=galprop classic, 2=3D cube NS,  3=3D F07 <1.5 kpc
    #     H2Model=1, # 1=galprop classic, 2=3D cube PEB, 3=3D F07 <1.5kpc
    #     n_spatial_dimensions=3, 
    #     dx=1, # kpc for dx and dy propagation grid
    #     dz=.1, # kpc for dz propagation grid
    #     zmax=6, # halo half-height 
    #     xmax=30,
    #     healpix_order=8,
    #     IC_isotropic=1,
    #     computeBremss=1,
    #     secondary_leptons=1,
    #     secondary_hadrons=0,
    #     spiral_fraction=0,
    #     skymap_format=sf,
    #     single_component=0,
    #     H2_filename = 'CO_Pohl_galprop_8500.fits',
    #     HI_filename = 'HI_Pohl_galprop_8500.fits',  
    #     H2_filename_rlb='CO_Pohl_8500_rlb.fits',
    #     HI_filename_rlb='HI_Pohl_8500_rlb.fits',
    #     D_0 = 7.86e+28, 
    #     v_Alfven = 35., 
    #     dvdz = 10,
    #     B_0 = 5,
    #     r_b = 10.,
    #     z_b = 2.,
    #     isrf_opt_fir=1,
    #     COR_filename = "rbands_co10mm_v2_2001_qdeg_9R.fits",
    #     HIR_filename = "rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag2_limit_9R.fits",
    #     cr_source = 'Yusifov',  # can be OB, Yusifov, Lorimer, or SNR
    #     n_XCO = 3, )
    #             secondary_leptons=1,secondary_hadrons=1, 
    

    #             skymap_format=sf)

    # GenGaldef('base', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1, 
    #             skymap_format=sf)
    


    # GenGaldef('NSPEB_HI_8500_interp0', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #              skymap_format=sf, HI_filename='HI_NS_galprop_r8500_interp0.fits.gz')
    
    # GenGaldef('NSPEB_HI_8500_interp25', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #              skymap_format=sf, HI_filename='HI_NS_galprop_r8500_interp25.fits.gz')

    # GenGaldef('NSPEB_HI_8500_interp50', dx=2,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #              skymap_format=sf, HI_filename='HI_NS_galprop_r8500_interp50.fits.gz')
    # GenGaldef('Pohl_HI_8500', dx=2,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #               secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #               skymap_format=sf, HI_filename='HI_Pohl_galprop_8500.fits')

    # GenGaldef('Pohl_H2_8500', dx=2,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=1,H2Model=2,
    #              skymap_format=sf, HI_filename='CO_Pohl_galprop_8500.fits')

    # GenGaldef('NSPEB_HI_8500_interp75', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #              skymap_format=sf, HI_filename='HI_NS_galprop_r8500_interp75.fits.gz')

    # GenGaldef('NSPEB_HI_8500_interp100', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #              secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,
    #              skymap_format=sf, HI_filename='HI_NS_galprop_r8500_interp100.fits.gz')

    # GenGaldef('NSPEB_H2_8500', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=1,H2Model=2,
    #             skymap_format=sf, H2_filename = 'CO_PEB_galprop_8500.fits.gz')


    # GenGaldef('Pohl_HI_8500_L', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,skymap_format=sf,
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Pohl_8500_rlb.fits', 
    #             cr_source='Lorimer')
    # GenGaldef('Pohl_HI_8500_O', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,skymap_format=sf,
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Pohl_8500_rlb.fits' ,
    #             cr_source='OB')
    # GenGaldef('Pohl_HI_8500_S', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,skymap_format=sf,
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Pohl_8500_rlb.fits' ,
    #             cr_source='SNR')
    # GenGaldef('Pohl_HI_8500_Y', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,skymap_format=sf,
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Pohl_8500_rlb.fits' ,
    #             cr_source='Yusifov')

    # GenGaldef('Tavakoli_HI_L', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=1,skymap_format=sf,
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Tavakoli_rlb.fits.gz', 
    #             cr_source='Lorimer')


    # GenGaldef('Pohl_H2_8500', dx=1.5,dz=.5, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=1,H2Model=2,
    #             skymap_format=sf, 
    #             H2_filename = 'CO_Pohl_galprop_8500.fits',
    #             H2_filename_rlb='CO_Pohl_8500_rlb.fits')

    # GenGaldef('Pohl_HI_H2_8500', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=2,skymap_format=sf, 
    #             H2_filename = 'CO_Pohl_galprop_8500.fits',
    #             H2_filename_rlb='CO_Pohl_8500_rlb.fits',
    #             HI_filename = 'HI_Pohl_galprop_8500_no_corr.fits',
    #             HI_filename_rlb='HI_Pohl_8500_rlb.fits')


    # GenGaldef('NSPEB_HI_H2', dx=1,dz=.25, healpix_order=8, IC_isotropic=2,computeBremss=1,
    #             secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=2,
    #             skymap_format=sf)

# GenGaldef('base_no_secondary_testing', dx=3,dz=1, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3)

# GenGaldef('NSPEB_no_secondary_HI_testing', dx=3,dz=1, healpix_order=7, IC_isotropic=0,computeBremss=0,
#              secondary_leptons=0,secondary_hadrons=0,HIModel=2,H2Model=1,
#              skymap_format=3)

# GenGaldef('NSPEB_no_secondary_H2_testing', dx=3,dz=1, healpix_order=7, IC_isotropic=0,computeBremss=0,
#              secondary_leptons=0,secondary_hadrons=0,HIModel=1,H2Model=2,
#              skymap_format=3)


#GenGaldef('F07_no_secondary', dx=3,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#            secondary_leptons=0,secondary_hadrons=0,HIModel=3,H2Model=1)

# GenGaldef('base_no_secondary', dx=1.5,dz=.25,IC_isotropic=1,computeBremss=1,secondary_leptons=0,secondary_hadrons=0)
# GenGaldef('NSPEB', dx=1.5,dz=.25,IC_isotropic=1,computeBremss=1,secondary_leptons=1,secondary_hadrons=1,HIModel=2,H2Model=2)




# Generate single component gas maps

# GenGaldef('base_single_HI', dx=2,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=1)

# GenGaldef('base_single_H2', dx=2,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=2)

# GenGaldef('NSPEB_single_HI', dx=2,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=1, HIModel=2)

# GenGaldef('NSPEB_single_H2', dx=2,dz=.5, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=2, HIModel=1, H2Model=2)

# GenGaldef('F07_single_HI', dx=6,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=1, HIModel=3, H2Model=1)

# GenGaldef('F07_single_H2', dx=6,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#                  secondary_leptons=0,secondary_hadrons=0, 
#                  skymap_format=3, single_component=2, HIModel=1, H2Model=3)

# GenGaldef('Pohl_HI_8500_no_corr_test', dx=3,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#               secondary_leptons=0,secondary_hadrons=0,HIModel=2,H2Model=1,
#               skymap_format=3, HI_filename='HI_Pohl_galprop_8500_no_corr.fits', single_component=1)
# GenGaldef('Pohl_HI_8500_test', dx=3,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#               secondary_leptons=0,secondary_hadrons=0,HIModel=2,H2Model=1,
#               skymap_format=3, HI_filename='HI_Pohl_galprop_8500.fits', single_component=1)

# GenGaldef('Pohl_H2_8500_test', dx=5,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#               secondary_leptons=0,secondary_hadrons=0,HIModel=1,H2Model=2,
#               skymap_format=3, H2_filename='CO_Pohl_galprop_8500.fits', single_component=2)

# GenGaldef('test_2', dx=3,dz=2, healpix_order=7, IC_isotropic=0,computeBremss=0,
#               secondary_leptons=0,secondary_hadrons=0,HIModel=1,H2Model=1,
#               skymap_format=3)



# -----------------------------------------------------
# Gen Set of Models
# -----------------------------------------------------
count = 0
prefix = 'mod_h_'

for HIModel,H2Model in [(2,2), (2,1), (1,2),(1,1),]:
    print HIModel, H2Model

    # for D_zz in np.linspace(.25, 4, 16):
    #     GenGaldef(
    #         filename=prefix+str(count),   # filename for output files and for galdef suffix
    #         HIModel=HIModel,
    #         H2Model=H2Model,  
    #         D_zz = D_zz, 
    #         H2_filename = 'CO_PEB_galprop_8500.fits.gz',
    #         HI_filename = 'HI_Pohl_galprop_8500.fits',  
    #         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
    #         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
    #     )
    #     count+=1

    for spiral_fraction in np.linspace(0,.3,8):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model,  
            spiral_fraction=spiral_fraction,
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1


    for spiral_fraction in np.linspace(0,.75,8):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model,  
            spiral_fraction=spiral_fraction,
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
        )
        count+=1


    for zmax in np.linspace(2,7,11):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model, 
            zmax=zmax, 
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1

    for primary_source in ('Lorimer','SNR','OB','Yusifov'):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model,  
            cr_source = primary_source,
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1

    for gas_renorm in ['rbands_hi12_v2_qdeg_zmax1_Ts100000_EBV_mag5_limit_9R_new.fits.gz',
                       'rbands_hi12_v2_qdeg_zmax1_Ts100000_EBV_mag2_limit_9R_new.fits.gz',
                       'rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag5_limit_9R_new.fits.gz',
                       'rbands_hi12_v2_qdeg_zmax1_Ts150_EBV_mag2_limit_9R_new.fits.gz']:

        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model, 
            HIR_filename = gas_renorm,
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1

    for D_0 in np.logspace(28,29.5,8):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model,  
            D_0 = D_0,
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1

    for v_a in np.linspace(20,150,8):
        GenGaldef(
            filename=prefix+str(count),   # filename for output files and for galdef suffix
            HIModel=HIModel,
            H2Model=H2Model, 
            v_Alfven = v_a, 
            H2_filename = 'CO_PEB_galprop_8500.fits.gz',
            HI_filename = 'HI_Pohl_galprop_8500.fits',  
            H2_filename_rlb='CO_Pohl_8500_rlb.fits',
            HI_filename_rlb='HI_Pohl_8500_rlb.fits',
        )
        count+=1


    with open('rungalprop_jobarray.sh','w') as f:
        f.write('''
    #!/bin/bash
    #PBS -N galprop_'''+prefix+'''
    #PBS -l nodes=1:ppn=32
    #PBS -l pmem=8gb
    #PBS -l walltime=18:00:00
    #PBS -t 0-'''+str(count+1)+'''%64
    #PBS -q hyper

    /pfs/carlson/galprop/GALDEF/run_script_'''+prefix+'''"$PBS_ARRAYID".sh

    ''')












# for B_0 in np.linspace(0,20,11):
#     GenGaldef(
#         filename=prefix+str(count),   # filename for output files and for galdef suffix
#         HIModel=2,
#         H2Model=2, 
#         B_0 = B_0,
#         H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )
#     count+=1


# GenGaldef(
#         filename='mod_e_center_high_res',   # filename for output files and for galdef suffix
#         dx = .125, 
#         dz = .125, 
#         HIModel=2,
#         H2Model=2, 
#         B_0 = B_0,
#         H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )







# THESE ARE NOT VERY IMPORTANT


# for dvdz in np.linspace(0,500,6):
#     GenGaldef(
#         filename='mod_c_'+str(count),   # filename for output files and for galdef suffix
#         HIModel=2,
#         H2Model=2, 
#         dvdz=dvdz,
#         H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )
#     count+=1




# for r_b in np.linspace(5,10,6):
#     GenGaldef(
#         filename='mod_c_'+str(count),   # filename for output files and for galdef suffix
#         HIModel=2,
#         H2Model=2, 
#         r_b = r_b, 
#         H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )
#     count+=1


# for z_b in np.linspace(1,4,6):
#     GenGaldef(
#         filename='mod_c_'+str(count),   # filename for output files and for galdef suffix
#         HIModel=2,
#         H2Model=2, 
#         z_b = z_b, 
#     H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )
#     count+=1




# for isrf_opt_fir in np.linspace(.5,3,8):
#     GenGaldef(
#         filename='mod_c_'+str(count),   # filename for output files and for galdef suffix
#         HIModel=2,
#         H2Model=2, 
#         isrf_opt_fir=isrf_opt_fir,
#     H2_filename = 'CO_PEB_galprop_8500.fits.gz',
#         HI_filename = 'HI_Pohl_galprop_8500.fits',  
#         H2_filename_rlb='CO_Pohl_8500_rlb.fits',
#         HI_filename_rlb='HI_Pohl_8500_rlb.fits',
#     )
#     count+=1