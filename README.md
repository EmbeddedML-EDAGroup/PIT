**NEW RELEASE: we released our new, engineered and user-friendly DNAS library named [PLiNIO](https://github.com/eml-eda/plinio) which includes PIT among the different implemented methods. We highly suggest to try this new release for your experiments!**

Copyright (C) 2021 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Francesco Conti, Lorenzo Lamberti, Yukai Chen, Luca Benini, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari

![logo](Assets/logo.png)
# Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge

## Reference
If you use PIT in your experiments, please make sure to cite our paper:
```
@ARTICLE{pit_tcomp,
  author={Risso, Matteo and Burrello, Alessio and Conti, Francesco and Lamberti, Lorenzo and Chen, Yukai and Benini, Luca and Macii, Enrico and Poncino, Massimo and Pagliari, Daniele Jahier},
  journal={IEEE Transactions on Computers}, 
  title={Lightweight Neural Architecture Search for Temporal Convolutional Networks at the Edge}, 
  year={2023},
  volume={72},
  number={3},
  pages={744-758},
  doi={10.1109/TC.2022.3177955}}
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
