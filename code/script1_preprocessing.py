#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tactile2026
Authors: @rodlaje @spiousas @Tomas-cm @pazlodife @galilei83

"""

#%% Import libraries

import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from scipy.signal import find_peaks
import math
import json


#%% Setup paths and filenames

# set 'Tactile2026/code/' as pwd

# Input
raw_data_path = '../raw data/'
registered_participants_path = raw_data_path + 'registered_subjects.dat'
path_po = raw_data_path + 'Presentation_orders.csv'

# Output
preprocessed_data_path = '../preprocessed data/'
times_data_path = preprocessed_data_path + 'df_times.csv'
voltages_data_path = preprocessed_data_path + 'df_voltages.csv'


# Preload metadata once
with open(registered_participants_path, "r") as fp:
	participant_fullstrings = [line.strip() for line in fp]

raw_files = os.listdir(raw_data_path)

with open(path_po, 'r') as file:
	po_lines = file.readlines()
po_conditions = [line.split(",") for line in po_lines]


#%% Trial data selection
# data(participant,block,trial) = single trial data

def data(participant, block, trial):
	participant_fullstring = participant_fullstrings[participant]
	suffix = f"block{block}-trial{trial}.dat"

	for filename in raw_files:
		if filename.startswith('S' + participant_fullstring) and filename.endswith(suffix):
			file_to_load = os.path.join(raw_data_path, filename)
			with open(file_to_load, "r") as f_to_load:
				content = json.loads(f_to_load.read())
			return content

	raise FileNotFoundError(
		f"No file for participant={participant}, block={block}, trial={trial}"
	)


#%% Peak detection
# tap_separator(voltage) = taps, peaks, p_min

def tap_separator(voltage, tap_length):
	voltage = np.asarray(voltage)

	taps = []

	# Vectorized rising/falling edge detection
	rising = np.where((voltage[:-1] == 0) & (voltage[1:] > 0))[0]
	falling = np.where((voltage[:-1] > 0) & (voltage[1:] == 0))[0]
	falling = falling[falling < len(voltage) - 10]

	for end_i in falling:
		idx = np.searchsorted(rising, end_i, side='left') - 1
		if idx < 0:
			continue

		start_i = rising[idx]

		if start_i < len(voltage) - tap_length and (end_i - start_i + 2) > 8:
			taps.append(voltage[start_i + 1:start_i + tap_length + 1])

	peaks = []
	p_min = 0
	taps_posta = []

	for b in taps:
		pk, _ = find_peaks(b, distance=5, height=10, width=3)

		if len(pk) > 1:
			if b[pk[0]] > b[pk[-1]]:
				taps_posta.append(b)
				peaks.append(pk[:2])

		if len(pk) == 1:
			pk = np.concatenate((pk, [np.nan]))
			taps_posta.append(b)
			peaks.append(pk[:2])

		if len(pk) == 0:
			pk = [np.nan, np.nan]
			taps_posta.append(b)
			peaks.append(pk[:2])

		min_peaks, _ = find_peaks(-b, distance=5, width=3)
		if len(min_peaks) != 0:
			p_min = min_peaks[0]

	return taps_posta, peaks, p_min


#%% Get condition from trial

def cond_of_trial(participant, block, trial, trials_per_block):
	return po_conditions[1 + trial + block * trials_per_block][participant + 2]


#%% Times and voltages

tap_length = 75#50#100 # don't make it too large (as it would get voltages from next tap) or too small (as it would prematurely end taps)
trials_per_block = 9
blocks_per_participant = 4
heigth_interval = 10
ppf = 6

times_records = []
voltages_records = []

# participants 0 and 3 discarded (empty data)
participants = [
	1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
	38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52
]

# every row is a tap
for participant in participants:
	print(participant_fullstrings[participant])
	for block in tqdm(range(blocks_per_participant)):
		for t in range(trials_per_block):
			trial_data = data(participant, block, t)

			asyns = trial_data["Asynchrony"]
			voltage = trial_data["voltage_value"]
			assign_stim = trial_data['Stim_assigned_to_asyn']

			taps, peaks, p_min = tap_separator(np.asarray(voltage), tap_length)
			taps = np.asarray(taps)
			peaks = np.asarray(peaks)
			cond = cond_of_trial(participant, block, t, trials_per_block)

			if len(np.asarray(assign_stim)) > 1:
				cosito = np.argwhere(np.array(assign_stim) >= ppf - 1)

				if len(cosito) > 0:
					asyns_taps = asyns[len(asyns) - len(taps):]

					for tap in range(len(asyns_taps)):
						if not math.isnan(asyns_taps[tap]):
							if not math.isnan(peaks[tap][1]):
								if (
									peaks[tap][0] < p_min
									and peaks[tap][1] > p_min
									and (taps[tap][int(peaks[tap][0])] - taps[tap][int(p_min)]) > heigth_interval
								):
									c_asyn = asyns_taps[tap]
									p1_tap = peaks[tap][0]
									p2_tap = peaks[tap][1]
									asyns_p1 = c_asyn + p1_tap
									asyns_p2 = c_asyn + p2_tap
									peak_interval = p2_tap - p1_tap

									times_records.append([
										participant, block, t, cond, tap,
										p1_tap, p2_tap, c_asyn,
										asyns_p1, asyns_p2, peak_interval
									])

									for ms, v in enumerate(taps[tap]):
										voltages_records.append([
											participant, block, t, cond, tap, ms, v
										])


#%% Save data

df_times = pd.DataFrame(
	times_records,
	columns=['Participant', 'Block', 'Trial', 'Cond', 'Tap',
		'P1', 'P2', 'Asyns_c', 'Asyns_p1', 'Asyns_p2', 'Peak_interval']
)

df_voltages = pd.DataFrame(
	voltages_records,
	columns=['Participant', 'Block', 'Trial', 'Cond', 'Tap', 'Time', 'Voltages']
)

df_times[['Effector', 'Period']] = df_times['Cond'].str.extract('(\w+)(\w+)', expand=True)
df_times['Period'] = df_times['Period'].replace('1', '444')
df_times['Period'] = df_times['Period'].replace('2', '666')

df_voltages[['Effector', 'Period']] = df_voltages['Cond'].str.extract('(\w+)(\w+)', expand=True)
df_voltages['Period'] = df_voltages['Period'].replace('1', '444')
df_voltages['Period'] = df_voltages['Period'].replace('2', '666')

df_times.to_csv(times_data_path)
df_voltages.to_csv(voltages_data_path)

#%%
