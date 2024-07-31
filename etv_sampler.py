# ><
import datetime
start = datetime.datetime.now() # timekeeping is important

import numpy as np
import pandas as pd
from rebound_model import rebound_model as rbmod
from lite_delay import lite_delay
from astropy import units as u
from tess_stars2px import tess_stars2px_function_entry
import scipy.stats as stats
import ticgen
from astropy.timeseries import LombScargle
from multiprocessing import Pool

######################################
## CHANGE THE FOLLOWING VALUES
total_number_of_samples = 100
num_cpu = 10
planet_mass_range = (10,80)
######################################
## THE RESULTS WILL BE IN etv_sample.csv
######################################

# This is the file that contains time windows for TESS orbits.
# It is being used mostly in function "is_in_tess_obs"
tess_obs_range = pd.read_csv("TESS_orbit_times.csv")


def random_star(number_of_star):
	# return RA, Dec, ticid, brightness
	# loc and scale values come from tess eb mag histogram fits, are ONLY for Tmag < 9 currently
	upper_mag, lower_mag = 2, 9
	mag = stats.truncnorm.rvs(a=(upper_mag-10.31)/2.034, b=(lower_mag-10.31)/2.034, loc=10.31, scale=2.034, size=number_of_star)
	ra, dec, ticid = np.random.uniform(0,360,number_of_star),np.random.uniform(-90,90,number_of_star),range(number_of_star)
	return ra, dec, ticid ,mag

def find_sectors(ra,dec,ticid):
	# It does what the name says
	outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, outRowPix, scinfo = tess_stars2px_function_entry(ticid, ra, dec)
	return_array = []
	for i in ticid:
		ii = np.where(i == outID)[0]
		return_array.append([ra[i],dec[i],outSec[ii].tolist()])
	return return_array

def is_in_tess_obs(obstime,sectors_to_look):
	# This function checks if any given time is in TESS observation times based on TESS_orbit_times.csv
	results = []
	for time_float in obstime:
		result = any(row['StartD'] <= time_float <= row['EndD'] and (row['Sector'] in sectors_to_look) for index, row in tess_obs_range.iterrows())
		results.append(result)
	return results

def ttv_model(parameters,setup,pri_mass):
	# Wraps up the parameters and settings to rebound model
	times_, xyz_com, vxvyvz_com, axayaz_com, xyz_hel, vxvyvz_hel, axayaz_hel = rbmod(start = setup[0],\
							times = setup[1],\
							timestep=setup[2], \
							primary_mass = pri_mass,\
							masses = parameters[0], \
							periods = parameters[1], \
							eccs = parameters[2], \
							omegas = parameters[3], \
							time_peris = parameters[4], \
							incs = parameters[5], \
							nodes = parameters[6])
	return times_, xyz_com, vxvyvz_com, axayaz_com, xyz_hel, vxvyvz_hel, axayaz_hel

def polynomial_photometric_precision_model(magnitude, a=0.2, c=0.2):
	# NOT BEING USED AT THE MOMENT
	# return in ppm
	return 10**(a*np.array(magnitude) + c)

def eclipse_depth():
	# approximation of primary eclipse depth distribution from Prsa et al. 2022
	x = np.random.uniform(1,1000)
	loginv = (np.log10(x)-3)/-2.8
	return loginv

def etv_noise(depth,phot_noise_duration,eclipse_duration):
	# Calculates timing noise based on two refs below
	# https://www.aanda.org/articles/aa/pdf/2017/03/aa29350-16.pdf
	# https://www.mdpi.com/2075-4434/9/1/1
	etv_err = (phot_noise_duration * eclipse_duration) / (2 * depth)
	return etv_err

def eclipse_duration(period):
	# approximation from eclipse duration vs orbital periods from TESS EBCatalog
	dur = 0.1129 * (period**-0.6938)
	ecl_dur = stats.truncnorm.rvs(a=(0.0001-dur)/(dur/1), b=(0.2-dur)/(dur/1), loc=dur, scale=(dur/1))
	return ecl_dur

# This is the output file
etv_sample_df = open("etv_sample.csv","w")
etv_sample_df.write("ra,dec,sectors,Tmag,sigmaTmag,per_binary,t_ref_binary,mass_binary,ecl_depth,ecl_duration,delta_mag,sigma_flux_at_depth,\
			etv_error,planet_mass, planet_period, planet_ecc, planet_omega, planet_tperi, planet_inc, planet_node,period_lombscargle,\
			amplitude_etv,amplitude_lombscargle\n")
etv_sample_df.close()

def etv_recovery(tot_num_samples = 10):
	# printing time is a problem that needs some more time to solve
	timeswitch = 0
	if type(tot_num_samples) == str:
		timeswitch = 1
		tot_num_samples = int(tot_num_samples)
		
	for rnd in range(tot_num_samples):# loop for each random sample unit
	
		if timeswitch:
			finish_ = datetime.datetime.now()
			print("Pseudo-run number: ",(rnd+1)*num_cpu,"of total",total_number_of_samples,"| Time passed: ",(finish_ - start))

		np.random.seed() # randomization in each iteration

		coords = random_star(1)
		sects = find_sectors(*coords[:-1])[0][-1]
		Tmag = coords[-1][0]
		noise_1sig = ticgen.calc_star({"Tmag":Tmag,"integration":2})[-1]
		normalized_noise = noise_1sig/1e6

		# the following come from histogram fit of tess eb catalog
		per_binary = stats.lognorm.rvs(0.957, 0, 4.7258, size=1)[0] + 0.7# binary period in days # limits 0.5 - 14 days # +0.7 for small correction
		t_ref_binary = np.random.uniform(-per_binary,per_binary) # first eclipse time # limits +/- per_binary days
		mass_binary = stats.powerlaw.rvs(a= np.log10(2.35),loc=0.1,scale=5)#0.2, 10, 4.7258, size=1)[0]
		if mass_binary < 1:
			mass_binary = np.random.uniform(0.2,1) #Miller–Scalo (1979)
			
		ecl_depth = eclipse_depth()
		if ecl_depth >= 1:
			continue

		if ecl_depth < 1:
			delta_mag = -2.5*np.log10(1-ecl_depth)
		else:
			delta_mag = 0.
		ecl_duration = eclipse_duration(per_binary)*per_binary # days
		
		#sigma_flux_at_depth = ticgen.calc_star({"Tmag":Tmag + delta_mag,"integration":ecl_duration})[-1] # this is only to check noise for total eclipse duration
		sigma_flux_at_depth = ticgen.calc_star({"Tmag":Tmag + delta_mag,"integration":2})[-1]
		etv_error = ((sigma_flux_at_depth/1e6)*ecl_duration/(2*ecl_depth))*24*60*60 # seconds
		
		# Planet parameters
		masses_in = [(np.random.uniform(planet_mass_range[0],planet_mass_range[1])*u.jupiterMass).to(u.solMass).value] # solar mass - but you type in jupiter masses
		periods_in = [np.random.uniform(per_binary*10,1000)] # days # limits 1 - 1/2 total observation timerange
		eccs_in = [0.] # circular orbit fixed
		omegas_in = [0.] # fixed
		tperis_in = [np.random.uniform(-periods_in[0],periods_in[0])] # days # limits -period - +period
		incs_in = [np.degrees(np.arccos(np.random.uniform(0.,1.)))] # limits 0 - 90 degree
		nodes_in = [0.] # limits 0 - 90 degree
	
		# This list of planet parameters will be sent to orbit simulation
		params = [masses_in, periods_in, eccs_in, omegas_in, tperis_in, incs_in, nodes_in]

		if 0: # set 1 if you want to print binary+planet parameters
			print("BINARY")
			print("RA",np.round(coords[0][0],2)," Dec",np.round(coords[1][0],2)," Tmag",np.round(Tmag,2),"\nPeriod",np.round(per_binary,2),
				" Mass",np.round(mass_binary,2),"\nEclipseDepth",np.round(ecl_depth,3),"\nETV Error: ",etv_error,"\nSectors",sects)		
			print("PL PARAMS  ",params)

		eclipse_times = np.arange(t_ref_binary,tess_obs_range["EndD"][len(tess_obs_range["EndD"])-1], per_binary)

		# sim_setup list is required for orbit simulation: [start, times_of_integration, stepsize]
		# stepsize is ignored, if ias algorith is selected in rebound_model
		sim_setup_ = [0, eclipse_times,1.]

		if 1:# Orbit Simulation Block
			try:
				times,xyz_c,vel_c,acc_c, xyz_h,vel_h,acc_h = ttv_model(params,sim_setup_,mass_binary)
				lite = lite_delay(xyz_c[0][2])
				amp_lite = max(lite) - min(lite)
				amp_lite /= 2
			except:
				print("orbit sim crashed")
				continue

		if 1:# Lomb-Scargle Block
			period_ls = 0
			etv_x = []
			try:
				mask = is_in_tess_obs(eclipse_times,sects)
				etv_x, etv_y = eclipse_times[mask],np.random.normal(lite[mask],etv_error)
				if len(etv_x) > 2:
					max_freq = 1/(np.min(np.diff(etv_x))*2)
					min_freq = 1/((np.max(etv_x) - np.min(etv_x))/2)
					ls = LombScargle(etv_x, etv_y, normalization='standard')
					freq, power = ls.autopower(samples_per_peak = 200,maximum_frequency=max_freq,minimum_frequency=min_freq)
					freq_max = freq[np.where(power == power.max())][0]
					period_ls = 1./freq_max
					
					x_theo = np.linspace(times[0],times[-1],5000)
					freq_curve_theo = ls.model(x_theo, freq_max)
					amp_ls = max(freq_curve_theo) - min(freq_curve_theo)
					amp_ls /= 2
			except:
				print("lomb-scargle crashed")
				continue

		if period_ls:# Output Block
			new_row = (np.round(coords[0][0],4),np.round(coords[1][0],4),np.array(sects),np.round(Tmag,4),np.round(noise_1sig,4),np.round(per_binary,4),
				np.round(t_ref_binary,4),np.round(mass_binary,4),np.round(ecl_depth,10),np.round(ecl_duration,10),np.round(delta_mag,10),
				np.round(sigma_flux_at_depth,4),np.round(etv_error,4),np.round((masses_in[0]*u.solMass).to(u.jupiterMass).value,10),np.round(periods_in[0],4),eccs_in[0],omegas_in[0],
				np.round(tperis_in[0],4),np.round(incs_in[0],4),nodes_in[0],np.round(period_ls,4),np.round(amp_lite,4),np.round(amp_ls,4))
			etv_sample_df = open("etv_sample.csv","a")
			for icolmn, colmn in enumerate(new_row):
				if icolmn == 2:
					i2 = str(colmn).replace("\n","")[1:-1].lstrip().rstrip().replace("  ",";").replace(" ",";")
					etv_sample_df.write(i2)
				else:
					etv_sample_df.write(str(colmn))
				if icolmn < len(new_row) - 1:
					etv_sample_df.write(",")
				else:
					etv_sample_df.write("\n")
			etv_sample_df.close()


if total_number_of_samples < num_cpu:
	num_cpu = int(total_number_of_samples)
pooling_list = [int(total_number_of_samples/num_cpu)]*num_cpu
pooling_list[-1] = str(pooling_list[-1])

with Pool(num_cpu) as p:
	p.map(etv_recovery,pooling_list)


finish = datetime.datetime.now()
print("FINISHED IN",(finish - start))
