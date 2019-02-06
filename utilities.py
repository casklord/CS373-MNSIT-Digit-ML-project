from __future__ import print_function
import numpy as np
import random

def shuffle(samples, labels, returnsize):
	returnsamples = np.empty((returnsize, 196))
	returnlabels = np.empty(returnsize)
	S = set()
	while(len(S) < returnsize):
		a = random.randint(0,len(labels) - 1)
		if a not in S:
			S.add(a)
	index = 0
	for s in S:
		returnsamples[index] = samples[s]
		returnlabels[index] = labels[s]	
		index += 1	
	return returnsamples, returnlabels

def subsample(n,z):
	returnsample = np.zeros(z*z)
	for i in range(0,z):
		for j in range(0,z):
			returnsample[i*z + j] = max(max(n[4*z*i + 2*j], n[4*z*i + 2*j + 1]), max(n[4*z*i + 2*z + 2*j], n[4*z*i + 2*z + 2*j + 1]))			
	return returnsample

def round(n):
	if n/255.0 > 0.5:
		return 1
	else:
		return 0

def print_number(x, rows, cols):
	for j in range(0, rows):
		for k in range(0, cols):
			print(int(x[j*cols + k]), end='')
		print ("")
	print ("")

def resize(x):
	Srow = Scol = 0
	Erow = Ecol = 27
	for j in range(0, 28):
		total = 0
		for k in range(0, 28):
			total += int(x[j*28 + k])
		if total != 0:
			Srow = j
			break
	for j in range(27, -1, -1):
		total = 0
		for k in range(0, 28):
			total += int(x[j*28 + k])
		if total != 0:
			Erow = j
			break
	for j in range(0, 28):
		total = 0
		for k in range(0, 28):
			total += int(x[j + k*28])
		if total != 0:
			Scol = j
			break
	for j in range(27, -1, -1):
		total = 0
		for k in range(0, 28):
			total += int(x[j + k*28])
		if total != 0:
			Ecol = j
			break
	rows = Erow - Srow + 1
	cols = Ecol - Scol + 1
	new_image = np.zeros(rows*cols)
	for j in range(Srow , Erow + 1):
		for k in range(Scol, Ecol + 1):
			new_image[(j - Srow)*cols + (k - Scol)] = int(x[j*28 + k])	
	return_image = np.zeros(784)
	for j in range(0, 28):
		for k in range(0, 28):
			return_image[j*28 + k] = new_image[int((int((j*rows)/28))*cols + (k*cols)/28)]
	return return_image

def sgn(z):
	if z > 0:
		return 1
	else:
		return 0

def match(a,b):
	if a == b:
		return 1
	else:
		return -1


