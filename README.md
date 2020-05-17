# Multi-souce-Separation-and-Alignment-Network-for-EEG-Emotion-Classification
Course project of EI328, SJTU advised by Prof. Baoliang Lu



Introduction
==============
We provide code for MSSAN on the task of emotion classification.

Structure
==============
-data

	-test		data for testing
	-train		data for training
	
-code

	-mmd.py		basic mmd computation
	-model.py		model used on cpu
	-model_cuda.py	model used on gpu
	-mssan.py		task running and evaluation on cpu
	-mssan_cuda.py	task running and evaluation on gpu
	
-report.pdf

Usage
==============
In the directory of code:

python ./mssan.py		run on cpu

python ./mssan_cuda.py	run on gpu
