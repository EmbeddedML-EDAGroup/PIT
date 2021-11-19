#!/bin/bash

# PPG-DaLiA 
python nas.py -c config/config_Dalia.json \
	--schedule_dil same --schedule_filt same --schedule_ch same \
	--dilation True --filters True --channels True \
	--mask_dil binary --mask_filt binary --mask_ch binary \
	--fix_strength True --gr_bin True \
	--strength 1e-8 --warmup max

# ECG5000
python nas.py -c config/config_ECG5000.json \
	--schedule_dil same --schedule_filt same --schedule_ch same \
	--dilation True --filters True --channels True \
	--mask_dil binary --mask_filt binary --mask_ch binary \
	--fix_strength True --gr_bin True \
	--strength 1e-8 --warmup max

# NinaProDB1
python nas.py -c config/config_NinaProDB1.json \
	--schedule_dil same --schedule_filt same --schedule_ch same \
	--dilation True --filters True --channels True \
	--mask_dil binary --mask_filt binary --mask_ch binary \
	--fix_strength True --gr_bin True \
	--strength 1e-8 --warmup max

# GoogleSpeechCommands
python nas.py -c config/config_GoogleSpeechCommands.json \
	--schedule_dil same --schedule_filt same --schedule_ch same \
	--dilation True --filters True --channels True \
	--mask_dil binary --mask_filt binary --mask_ch binary \
	--fix_strength True --gr_bin True \
	--strength 1e-8 --warmup max
