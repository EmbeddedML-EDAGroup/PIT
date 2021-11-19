Copyright (C) 2021 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Francesco Conti, Lorenzo Lamberti, Yukai Chen, Luca Benini, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari

![logo](Assets/logo.png)
# Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge

## Reference
If you use PIT in your experiments, please make sure to cite our paper:
```
@inproceedings{risso2021pit,
	author = {Risso, Matteo and Burrello, Alessio and Jahier Pagliari, Daniele and Conti, Francesco and Lamberti, Lorenzo and Macii, Enrico and Benini, Luca and Poncino, Massimo},
	title = {Pruning In Time (PIT): A Lightweight Network Architecture Optimizer for Temporal Convolutional Networks},
	year = {2021},
	publisher = {IEEE Press},
	booktitle = {Proceedings of the 58th ACM/EDAC/IEEE Design Automation Conference},
	series = {DAC '21}
}
```

## Datasets
The current version support the following datasets:
- PPG-DaLiA.
- ECG5000.
- NinaProDB1.
- Google Speech Commands.

Further deitails about the pre-processing and data-loading phases of these datasets are provided under the **./Dataset** directory.

## How to run
The sheel script `nas_script_sample.sh` contains four examples of typical usage of PIT with the four aforementioned datasets.
The main script used to perform experiments is `nas.py` which takes as input a configuration file (see examples in .**./config** directory) and a bunch of command line arguments.
In order to fastly start experiments, we suggest to leave every command-line arguments as-is only tweaking the `--strength` value.

## License
PIT is released under Apache 2.0, see the LICENSE file in the root of this repository for details.