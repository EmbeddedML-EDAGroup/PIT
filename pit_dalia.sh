#!/bin/bash

# PPG-DaLiA 
python nas.py -c config/config_Dalia_tmp.json \
	--schedule_dil same --schedule_filt same --schedule_ch same \
	--dilation False --filters False --channels True \
	--mask_ch binary \
	--fix_strength True --gr_bin True \
	--strength $1 --warmup 0