# Torben Ferber (torben.ferber@kit.edu), last change June 2022

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import time
import vector, os, re, argparse
from glob import glob

cmap = mpl.cm.get_cmap('viridis_r')
# plt.style.use('niceplots')

import utility as utils

# ---------------------------
# USER INPUT
parser = argparse.ArgumentParser(description='LUXE NPOD')
parser.add_argument('--mass',           action="store", type=float, default=0.2)
parser.add_argument('--targetend',      action="store", type=float, default=0.0)
parser.add_argument('--targetlength',   action="store", type=float, default=1.0)
parser.add_argument('--distance',       action="store", type=float, default=3.0)
parser.add_argument('--spectrum',       action="store", type=str,   default='all')
parser.add_argument('--detectorradius', action="store", type=float, default=1.0)
parser.add_argument('--minenergy',      action="store", type=float, default=0.5)
parser.add_argument('--minseparation',  action="store", type=float, default=0.0)
parser.add_argument('--ncpu',           action="store", type=int,   default=16)
parser.add_argument('--debug',          action='store_true')
parser.add_argument('--timing',         action='store_true')
args = parser.parse_args()

mass            = args.mass           # closest available mass will be used (changing this requires to read new files)
target_end      = args.targetend      # target end in m
target_length   = args.targetlength   # in m
distance        = args.distance       # in m
detector_radius = args.detectorradius # in m
min_energy      = args.minenergy      # minimum energy for photons
min_separation  = args.minseparation  # minimum photons separation at detector surface [m]
ncpu            = args.ncpu           # number of CPUs to read rootfiles
debug           = args.debug          # switch to debug mode
timing          = args.timing         # switch on timing measurements
spectrum        = args.spectrum       # photon spectrum (primary, secondary, all)

print('User inputs:')
print('mass',            mass)
print('target_end',      target_end)
print('target_length',   target_length)
print('distance',        distance)
print('detector_radius', detector_radius)
print('min_energy',      min_energy)
print('min_separation',  min_separation)
print('spectrum',        spectrum)
print('ncpu',            ncpu)
print('debug',           debug)

# photonfile = '/home/ferber/git/beam-dumps/LUXE-NPOD/Photons_spectra_all.root'
photonfile = '/storage/9/rquishpe/luxe/npod/phase1/g4/lxsim_bsm_npod_test/doubledump/run_dumpR20wrapR50L100_bsmL100_BX0/merged/Photons_spectra.root'
makeplots = True

# ---------------------------
#XFEL BEAM (PROBABLY DONT TOUCH)
N_e = 1.5e9 # electrons per bunch
N_bx = 1e7 # bunch crossings
photon_scale = 1.0 # 0.25 # scale photon spectra by this number if the MC corresponds to multiple bunch crossings

# ---------------------------
# TARGET MATERIAL (CHANGE THESE NUMBERS BUT BE CAREFUL BECAUSE TORBEN DOES NOT KNOW IF THE MADGRAPH FILES DEPEND ON NUCLEAR FORM FACTORS!)
rho_W = 19.3 # tungsten 
X_0 = 0.35 # tungsten
A_W = 184 # tungsten 
m0 = 1.661*pow(10,-24) #nucleus mass

# ---------------------------
# SCAN VARIABLES
laserconfig = [f'phase_1_{spectrum}'] #, 'phase0_ppw_xi_7_0']
lam_inv = np.logspace(-2.5, -6.5, 60)
lam_inv = np.append(lam_inv, (1e-5, 1e-6)) #add a few fixed values for nice plots
lam_inv.sort() #sort to have the inserted values at the right position
if debug:
    lam_inv = [1e-5]

# ---------------------------
# MADGRAPH (DO NOT CHANGE THESE NUMBERS EVER!)
lambda_mg = 1e4 #used in MG to calculate xsec -> careful, there is a factor 4 in the MadGraph model already (run cards show 4e4!!!)
n_mg = 10000 # number of generated ALP per MG sample

# ---------------------------
#SMEARING OF INCIDENT POINTS
r_x = 0.01 # alp production smearing sigma
r_y = 0.01 # alp production smearing sigma

# ---------------------------
# SIGNAL FILES
input_file_dir = '/ceph/ferber/LUXE/BSM/sig/1m/TungstenDump/'

# ---------------------------
#MADGRAPH CROSS SECTION FILES
input_banner_dir = '/ceph/ferber/LUXE/BSM/sig/1m/TungstenDump/RunTagBannerFiles/'

#calculate effective luminosity
# see LUXE-NPOD paper eq. 13
L_eff = N_e * N_bx * 9.0 * rho_W * X_0 / (7.0 * A_W * m0)
print(r'L_eff = {} cm$^{{-2}}$'.format(L_eff))
L_eff_pb = L_eff / 1e36
print(r'L_eff = {} pb'.format(L_eff_pb))

# ---------------------------
# find requested mass
# get all subfolders
list_subfolders = glob(os.path.join(input_file_dir, 'mass*/'),recursive = True)
# get list of basenames (that contains the mass)
templist_normdirs = [os.path.basename(os.path.normpath(x)) for x in list_subfolders]
# get a list of all simulated masses
list_masses = [float(x.replace('mass', '').replace('GeV', '')) for x in templist_normdirs]
requested_mass = mass
mass = utils.closest_in_list(list_masses, mass)
print(f'mass requested: {requested_mass}, mass used: {mass}')

# ---------------------------
#create output directory
save_dir = utils.create_output_directory(suffix=f'_mass_{mass}_distance_{distance}_targetlength_{target_length}_minenergy_{min_energy}_detectorradius_{detector_radius}_spectrum_{spectrum}_separation_{min_separation}', nrand=10, parent_dir = '/storage/9/rquishpe/luxe/npod/phase1/signal/LUXE-NPOD/output_condor')

# ---------------------------
#GET LIST OF ALL ROOT FILES
#get list of all rootfiles in that subfolder
foldername = os.path.join(input_file_dir, 'mass' + str(mass) + 'GeV')
list_files = glob(os.path.join(foldername, '*'),recursive = True)
templist_normfiles = [os.path.basename(os.path.normpath(x)) for x in list_files]

# get a list of rootfiles
list_energies_rootfiles = [[float(re.search('_beam(.*)GeV_', x).group(1)), os.path.join(foldername, x)] for x in templist_normfiles]

# ---------------------------
# READ PHOTON INPUT SPECTRA
# list_bincenters and list_weights contain two lists each for the two xi parameters
tree='h_all'
if spectrum == 'primary':
    tree='h_primary'
if spectrum == 'secondary':
    tree='h_secondary'
h1_photons, list_bincenters, list_weights = utils.read_photon_spectrum(file = photonfile, tree=tree, save_dir=save_dir, N_e=N_e, plot=makeplots, scale=photon_scale)

# ---------------------------
# READ CROSS SECTIONS FROM MADGRAPH
list_xsec = utils.get_cross_sections(input_banner_dir, mass)

# ---------------------------
# COMBINE ALL META INFORMATION INTO A SINGLE FILE
# photon energy, MG xsec, photon weight, root file
list_weights = utils.get_weights_combined(list_energies_rootfiles, list_xsec, list_bincenters[0], list_weights[0])
# list_weights_xi_7_0 = utils.get_weights_combined(list_energies_rootfiles, list_xsec, list_bincenters[1], list_weights[1])
laser_dict = {f'phase_1_{spectrum}': list_weights}

# ---------------------------
# PLOT INPUT SPECTRA
plotdict = {'plot': makeplots,
           'xlabel': 'Energy (GeV)',
           'ylabel': 'Entries / 0.1 GeV',
           'save_dir': save_dir,
           'plotname': 'photon_spectrum_final.pdf',
           'label': 'photon_spectrum_final'}
utils.plot_spectra(list_weights, **plotdict)

# plotdict = {'plot': makeplots,
#            'xlabel': 'Energy (GeV)',
#            'ylabel': 'Entries / 0.1 GeV',
#            'save_dir': save_dir,
#            'plotname': 'photon_spectrum_final_xi_7_0.pdf',
#            'label': 'photon_spectrum_final_xi_7_0'}
# utils.plot_spectra(list_weights_xi_7_0, **plotdict)

# ---------------------------
# PLOT CROSS SECTIONS
plotdict = {'plot': makeplots,
           'xlabel': 'Energy (GeV)',
           'ylabel': r'$\sigma$ (pb)',
           'save_dir': save_dir,
           'plotname': 'madgraph_xsec.pdf',
           'label': 'xsec'}
utils.plot_xsec(list_xsec, **plotdict)

# ---------------------------
# READ ROOT FILES (SUPPORTS MULTIPROCESSING)
list_results = utils.read_rootfiles(list_energies_rootfiles, ncpu=ncpu, debug=debug, timing=timing, dtype=np.float32)
list_results.sort()
print('number of files read', len(list_results))

# ---------------------------
# COUPLING AND PHOTON SPECTRUM SCAN
scan_events = []
for lc in laserconfig: 
    for li in lam_inv:
        
        lam = 1./li
        gagg = 1./lam #the factor 4 is not needed(TF 11.08.2022)

        # get decay vertex positions in meters
        w, ctau = utils.lifetime(mass, lam)

        # scale cross section
        xsec_rescale = li**2/(1/lambda_mg)**2

        list_all_temp = []
        
        # energy, xsec, photonweight, rootfile
        for x in laser_dict[lc]: #this runs over a full list of files with photon weights corresponding to the laser configuration
            energy = x[0]
            xsec_mg = x[1]
            weight = x[2]
            xsec_scaled = xsec_mg*xsec_rescale
            
            #find entry in list_results
            list_e = [x[0] for x in list_results]
            index = utils.find_nearest_index(list_e, energy)
            
            if index >= 0: #this is needed if not all xsec files are available as rootfiles
                results = list_results[index][1]
                list_all_temp.append([energy, weight, xsec_scaled, ctau, results])
            else:
                print('WARNING: did not add to list')
                      
        print('list_all_temp', len(list_all_temp))
        print('lambda_inverse', li)
            
        # create new subdir for laser configs and coupling
        save_dir_temp = utils.create_output_directory(suffix=f'lc_{lc}_li_{li}', parent_dir = save_dir)
        
        metadict = { 'mass': mass, 
                     'gagg': gagg,
                     'ctau': ctau,
                     'lam': lam, 
                     'lam_inv': li, 
                     'n_mg': n_mg,
                     'r_x': r_x,
                     'r_y': r_y,             
                     'N_e': N_e, 
                     'N_bx': N_bx, 
                     'rho_W': rho_W, 
                     'X_0': X_0, 
                     'A_W': A_W, 
                     'm0': m0, 
                     'L_eff_pb': L_eff_pb, 
                     'laserconfig': lc, 
                     'target_end': target_end, 
                     'target_length': target_length, 
                     'distance': distance, 
                     'detector_radius': detector_radius, 
                     'min_energy': min_energy}
            
        np.save(os.path.join(save_dir_temp, 'metainfo.npy'), metadict)
        
        # displace vertex 
        list_alp_vtx, list_photon = utils.displace(list_all_temp, mass, distance, target_end, target_length, ctau, r_x, r_y)

        # run event selection
        list_mask = []
        for idx, _ in enumerate(list_alp_vtx):
    
            if len(list_photon[idx])>0:
                mask1 = (np.array(list_photon[idx])[:,1:3] <= detector_radius).all(axis=1) # two photons: first test radius, than check that both are true
                mask2 = (np.array(list_photon[idx])[:,3:5] >= min_energy).all(axis=1) # two photons: first test energy, than check that both are true
                mask3 = (np.array(list_photon[idx])[:,0:1] >= min_separation).all(axis=1) # two photons: check distance at detector2 surface
                # print("Photons separation: ", np.array(list_photon[idx])[:,0])
                # print("Testing logic: ", np.array(list_photon[idx])[:,0] >= min_separation)
                # print("Mask 3: ", mask3)
                # list_mask.append(mask1*mask2)
                list_mask.append(mask1*mask2*mask3)
            else:
                list_mask.append([])
                
        # event selection without radius cut for cumulative distributions
        list_mask_noradcut = []
        for idx, _ in enumerate(list_alp_vtx):
            if len(list_photon[idx])>0:
                mask2 = (np.array(list_photon[idx])[:,3:5] >= min_energy).all(axis=1) # two photons: first test energy, than check that both are true
                list_mask_noradcut.append(mask2)
            else:
                list_mask_noradcut.append([])
                
        # PLOTTING
        # DECAY VERTEX Z
        plotdict = {'plot': makeplots,
                    'xmin': -0.5, 
                    'xmax': 10.5,
                    'nbins': 220+1,
                    'xlabel': 'decay vertex z (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'decay_vertex.npy',
                    'plotname': 'decay_vertex.pdf',
                    'label': 'vertex'}
        vertex_a_full, vertex_b_full = utils.plot_histogram(list_alp_vtx, 2, list_all_temp, list_mask, plotdict, metadict)
        
        # STORE NUMBER OF EVENTS
        nevents = 0
        if vertex_a_full is not None:
            nevents = np.sum(vertex_a_full)
            
        with open(os.path.join(save_dir, f'events_lc{lc}.txt'), 'a') as file_object:
            file_object.write(f'{mass}, {li}, {gagg}, {ctau}, {L_eff_pb}, {nevents}' + '\n')

        # DECAY VERTEX LENGTH
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 10.5,
                    'nbins': 220+1,
                    'xlabel': 'decay distance (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'decay_distance.npy',
                    'plotname': 'decay_distance.pdf',
                    'label': 'decay distance'}
        vertex_a_full, vertex_b_full = utils.plot_histogram(list_alp_vtx, 3, list_all_temp, list_mask, plotdict, metadict)
        
        # DECAY TIME
        plotdict = {'plot': makeplots,
                    'xmin': 0, 
                    'xmax': 60,
                    'nbins': 1200+1,
                    'xlabel': 'decay time (ns)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'decay_vertex_time.npy',
                    'plotname': 'decay_vertex_time.pdf',
                    'label': 'time'}
        vertex_a_full, vertex_b_full = utils.plot_histogram(list_alp_vtx, 4, list_all_temp, list_mask, plotdict, metadict)
        
        # DISTANCE
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 2.0,
                    'nbins': 200+1,
                    'xlabel': 'distance at detector (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photons_distance.npy',
                    'plotname': 'photons_distance.pdf',
                    'label': 'photon distance'}
        utils.plot_histogram(list_photon, 0, list_all_temp, list_mask, plotdict, metadict)

        # DISTANCE - ZOOM ON 0 to 10 cm
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 0.1,
                    'nbins': 100+1,
                    'xlabel': 'distance at detector (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photons_distance_zoom.npy',
                    'plotname': 'photons_distance_zoom.pdf',
                    'label': 'photon distance'}
        utils.plot_histogram(list_photon, 0, list_all_temp, list_mask, plotdict, metadict)
        
        # ENERGY 1
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 10.0,
                    'nbins': 100+1,
                    'xlabel': r'$E_{\gamma_{1}}$ (GeV)',
                    'ylabel_unit': 'GeV', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_energy_1.npy',
                    'plotname': 'photon_energy_1.pdf',
                    'label': 'photon energy'}
        utils.plot_histogram(list_photon, 3, list_all_temp, list_mask, plotdict, metadict)   
        
        # ENERGY 2
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 10.0,
                    'nbins': 100+1,
                    'xlabel': r'$E_{\gamma_{2}}$ (GeV)',
                    'ylabel_unit': 'GeV', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_energy_2.npy',
                    'plotname': 'photon_energy_2.pdf',
                    'label': 'photon energy'}
        utils.plot_histogram(list_photon, 4, list_all_temp, list_mask, plotdict, metadict)        
        
        # RADIUS 1
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 1.1,
                    'nbins': 110+1,
                    'xlabel': r'$r_{\gamma_{1}}$ (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_radius_1.npy',
                    'plotname': 'photon_radius_1.pdf',
                    'label': 'photon radius'}
        utils.plot_histogram(list_photon, 1, list_all_temp, list_mask, plotdict, metadict)   
        
        # RADIUS 2
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 1.1,
                    'nbins': 100+1,
                    'xlabel': r'$r_{\gamma_{2}}$ (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_radius_2.npy',
                    'plotname': 'photon_radius_2.pdf',
                    'label': 'photon radius'}
        utils.plot_histogram(list_photon, 2, list_all_temp, list_mask, plotdict, metadict)   
        
        # TIME 1
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 60.0,
                    'nbins': 1200+1,
                    'xlabel': r'$t_{\gamma_{1}}$ (ns)',
                    'ylabel_unit': 'ns', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_time_1.npy',
                    'plotname': 'photon_time_1.pdf',
                    'label': 'photon time'}
        utils.plot_histogram(list_photon, 9, list_all_temp, list_mask, plotdict, metadict)   
        
        # TIME 2
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 60.0,
                    'nbins': 1200+1,
                    'xlabel': r'$t_{\gamma_{2}}$ (ns)',
                    'ylabel_unit': 'ns', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_time_2.npy',
                    'plotname': 'photon_time_2.pdf',
                    'label': 'photon time'}
        utils.plot_histogram(list_photon, 10, list_all_temp, list_mask, plotdict, metadict)   
        
        
        # CUMULATIVE RADIUS
        plotdict = {'plot': makeplots,
                    'xmin': 0.0, 
                    'xmax': 2.0,
                    'nbins': 200+1,
                    'xlabel': r'$r_{max}$ (m)',
                    'ylabel_unit': 'm', # Entries per binwidth
                    'save_dir': save_dir_temp,
                    'filename': 'photon_radius_max.npy',
                    'plotname': 'photon_radius_max.pdf',
                    'label': 'photon radius'}
        rmax_entries_full, rmax_bins_full = utils.plot_cumulative_pair(list_photon, 1, 2, list_all_temp, list_mask_noradcut, plotdict, metadict, True)
        
        if rmax_entries_full is not None:
            list_bincenters = []
            list_cumrad_frac = []
            radcumsum = 0.0
            radcumsum_total = np.sum(rmax_entries_full)

            for idx, val in enumerate(rmax_entries_full):
                radcumsum += val
                list_bincenters.append((rmax_bins_full[idx]+rmax_bins_full[idx+1])/2.0)
                list_cumrad_frac.append(radcumsum/radcumsum_total)



            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            _ = axs.scatter(list_bincenters, list_cumrad_frac, color='black', s=2)
            axs.set_yscale('log')
            axs.set_xlabel('$r = \sqrt{x^2+y^2}$ (m)')
            axs.set_ylabel('fraction contained')
            xrange = [0,1]
            axs.set_xlim(xrange)

            _ = axs.text(0.975, 0.925-0.1, r'$\Lambda^{{-1}}$={} GeV$^{{-1}}$, $m_{{a}}={}$ GeV'.format(metadict['lam_inv'], metadict['mass']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.875-0.1, r'{}, $n_{{BX}}$={:4.2e}'.format(metadict['laserconfig'], metadict['N_bx']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.825-0.1, r'$L_{{V}}$={}m, $L_{{D}}$={}m'.format(metadict['distance'], metadict['target_length']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.755-0.1, r'$E_{{\gamma}}$>{}GeV'.format(metadict['min_energy']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)

            np.save(os.path.join(save_dir_temp, 'photon_radius_cumulative.npy'), { 'frac': np.asarray(list_cumrad_frac), 'centers': np.asarray(list_bincenters), 'xlabel': axs.get_xlabel(), 'ylabel': axs.get_ylabel(), 'xrange': xrange})

            plt.savefig( os.path.join(save_dir_temp, 'photon_radius_cumulative_log.pdf'), bbox_inches='tight') #speichern der Funktion als pdf, bbox==bounding box

            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            _ = axs.scatter(list_bincenters, list_cumrad_frac, color='black', s=2)
            axs.set_yscale('linear')
            axs.set_xlabel('$r = \sqrt{x^2+y^2}$ (m)')
            axs.set_ylabel('fraction contained')
            xrange = [0,1]
            axs.set_xlim(xrange)

            _ = axs.text(0.975, 0.925-0.1, r'$\Lambda^{{-1}}$={} GeV$^{{-1}}$, $m_{{a}}={}$ GeV'.format(metadict['lam_inv'], metadict['mass']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.875-0.1, r'{}, $n_{{BX}}$={:4.2e}'.format(metadict['laserconfig'], metadict['N_bx']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.825-0.1, r'$L_{{V}}$={}m, $L_{{V}}$={}m'.format(metadict['distance'], metadict['target_length']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)
            _ = axs.text(0.975, 0.755-0.1, r'$E_{{\gamma}}$>{}GeV'.format(metadict['min_energy']), 
                 horizontalalignment='right', 
                 verticalalignment='center', 
                 transform=axs.transAxes,
                 size=14)

            plt.savefig( os.path.join(save_dir_temp, 'photon_radius_cumulative_linear.pdf'), bbox_inches='tight') #speichern der Funktion als pdf, bbox==bounding box