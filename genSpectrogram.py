#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:06:03 2017

@author: cclu
"""
mode = 'sxx'

from scipy import signal
import scipy.io.wavfile as wav
import numpy as np
import math
frame_len = 80

import os
from collections import defaultdict
import pickle
binsize = 800

base_dir = '/path/to/dir/'

import matplotlib.pyplot as plt


def filterbank(sxx, fs=16000, NFFT=binsize, nfilt=40):
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (4000) / 700))  # Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
	bins = np.floor((NFFT + 1) * hz_points / fs)

	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
	    f_m_minus = int(bins[m - 1])   # left
	    f_m = int(bins[m])             # center
	    f_m_plus = int(bins[m + 1])    # right

	    for k in range(f_m_minus, f_m):
	        fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
	    for k in range(f_m, f_m_plus):
	        fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
	filter_banks = np.dot(sxx.T, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20.*np.log10(filter_banks)  # dB
	filter_banks -= np.mean(filter_banks, axis=0)
	return filter_banks.T

""" find consecutive zeros """
def findZeros(a, win_size):
	# Create an array that is 1 where a is 0, and pad each end with an extra 0.

	iszero = np.concatenate(([0], (abs(a)<150).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

	find = np.where(np.diff(ranges)>win_size)[0]
	if find.size:
		return ranges[find]
	return None

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, mode=None):
	win_len = 0.020
	win_overlap = 0.010

#	audiopath=root+'/'+f
	fs, x = wav.read(audiopath)

	# skip if too short
	if len(x) > fs*10:
		return None
	if len(x) < fs*((win_len-win_overlap)*frame_len+win_overlap):
		return None
	if x.ndim > 1:
		x = np.mean(x, axis=1)
	# trim consecutive zeros
	con_zero = findZeros(x, int(fs*win_len))
	if con_zero is not None:
		con_zero[:,1] = con_zero[:,1]-1
		zero_ind = con_zero.reshape(1,-1)[0]
		dif = np.diff(x)
		low = 0
		high = len(x)-1
		for ind in zero_ind:
			if dif[:ind].size:
				if max(abs(dif[:ind])) < 10:
					low = ind
		con_zero = np.flip(con_zero,0)
		for ind in zero_ind:
			if dif[ind:].size:
				if max(abs(dif[ind:])) < 10:
					high = ind
		x = x[low:high]
		con_zero = findZeros(x, int(fs*win_len))
		if con_zero is not None:
			drop = []
			for ind in con_zero:
				drop.extend(range(ind[0],ind[1]))
			x = np.delete(x,drop)
	if len(x) < fs*win_len:
		return None

	f,t,sxx = signal.spectrogram(x, fs,
										window=('hamming'),
										nperseg=int(fs*win_len),
										noverlap=int(fs*win_overlap),
										nfft=binsize,
										scaling='spectrum',
										mode='magnitude')

	if mode.lower() == 'sxx':
		f = f[f<4000]
		sxx = sxx[:len(f),:]
		sxx = 20.*np.log10(sxx)
		new = sxx
	elif mode.lower() == 'mel':
		new = filterbank(sxx, fs, binsize)

	new = (new - new.mean()) / (new.std() + 1e-8)

	# slicing
	t_width = len(t)
	if t_width < frame_len:
#		print('too short', audiopath)
		return None
	frame_num = math.ceil(t_width/frame_len)
	t_gap = t_width/frame_num
	frames = list()
	for i in range(frame_num):
		if (int(i*t_gap)+frame_len) >= t_width:
			frames.append(new[:, 0-frame_len:])
		else:
			frames.append(new[:, int(i*t_gap):int(i*t_gap)+frame_len])

	return frames


#%%==============================================================================
#
#==============================================================================
from glob import glob

##### IEMOCAP #####
src_dir = base_dir + 'IEMOCAP/'
sxx_dir = src_dir + mode.lower() + '_16_log_'+str(binsize)+'_'+str(frame_len)+'/'
stack = defaultdict(str)
for f in glob(src_dir+'wav/*/wav_mod/*/*.wav'):
	name = os.path.splitext(os.path.basename(f))[0]
	data = plotstft(f, binsize=binsize, mode=mode)
	if data: stack[name] = data
with open(sxx_dir, "wb") as f:
	pickle.dump(stack, f)


##### NNIME #####
src_dir = base_dir + 'NNIME/'
sxx_dir = src_dir + mode.lower() + '_16_log_'+str(binsize)+'_'+str(frame_len)+'/'
stack = defaultdict()
for root, dirs, files in os.walk(src_dir + 'wav_16b/'):
	for f in files:
		if f == ".DS_Store" or f=="._.DS_Store":
			os.remove(root+'/'+f)
			continue
		if f[-3:] == 'wav':
			full_name = '_'.join(root.split('/')[-2:]) +'_'+ root.split('/')[-3] + f[:-4]
			data = plotstft(root+'/'+f, binsize=binsize, mode=mode)
			if data is not None:
				stack[full_name] = data

with open(sxx_dir, "wb") as f:
	pickle.dump(stack, f)


##### DaAi #####
src_dir = base_dir + 'DaAi/'
sxx_dir = src_dir + mode.lower() + '_16_log_'+str(binsize)+'_'+str(frame_len)+'/'
if not os.path.exists(sxx_dir):
	os.mkdir(sxx_dir)

with open(src_dir+'weak_score.pkl', 'rb') as f:
	raw_data = pickle.load(f)

for root, dirs, files in os.walk(base_dir+'data/Sentence Audio/2014_16hz/'):
	if files:
		prog = root.split('/')[6]
		print(prog)
		stack = defaultdict()
		for f in files:#[:1000]:
			full_name = prog+'第'+f.replace('_', '集_')[:-4]
			if full_name in raw_data.index:
				data = plotstft(root+'/'+f, binsize=binsize, mode=mode)
				if data is not None:
					stack[full_name] = data
		with open(sxx_dir+prog+'.pkl', 'wb') as ff:
			pickle.dump(stack, ff)
