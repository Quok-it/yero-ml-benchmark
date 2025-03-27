<h1 align="center">yero-ml-benchmark</h1>

<p align="center">
  A training and inference benchmarking tool for machine learning.
</p>

***

## Why Is This Needed?
Relatability and reproducibility should be at the forefront of any benchmark. 
<br>
<br>
Current online benchmarks comparing different GPUs in the machine learning space either show inference throughput or a vague point system describing their score.
That alone is not enough for the average person who is more familiar with machine learning as a whole. 
A few of those benchmarks are reproducible with the code they provide (even if it is more obtuse to use) while others are not with only hints of what was used. 
<br>
<br>
This benchmark aims to <b>relate</b> with others who understands the basics that goes into machine learning: the <b>training</b>, the <b>inference</b>, the type of <b>model</b>, and <b>hardware</b> used; 
while maintaining <b>reproducibility</b> of performance regardless of different systems.
<br>
<br>
This benchmark is not indicative of the performance of the model itself in terms of accuracy, precision, and other metrics.*

## YouTube Video
Here is an indepth overview of yero-ml-benchmark, including an example comparing a RTX 3090 to a Titan X Pascal: 

## Current Support
* Linux
* NVIDIA GPUs
* Image Classification models such as ResNet and ViT

## Scripts
Currently there are three types of benchmark scripts: base, mid, and beyond.
* Base
	* Deterministic algorithms are used (unless nondeterministic is unavoidable)
  * CUDNN is enabled but limited
  * TF32 is disbaled
  * CUDA async is disabled
* Mid
	* Deterministic algorithms are used (unless nondeterministic is unavoidable)
  * CUDNN is enabled but less limited
  * TF32 is enabled
  * CUDA async is disabled
  * Use of torch.autocast with FP16
* Beyond
	* Nondeterministic algorithms are used
  * CUDNN is fully enabled
  * TF32 is enabled
  * CUDA async is enabled
  * Use of torch.autocast with FP16
  * Use of torch.compile

## Benchmark Results
The results are based on 100 epochs recorded during training and inference of a model under a specific combination of scripts and paramaters.
Results may vary as the user can select different models to use for benchmarking.
<br>
They results are shown in a text table and graphs. Current results pertain to:
* Model training/inference forward pass, backward pass, optimization, and loss calculation
* Kernels performance
* Cuda runtimes
* VRAM, power, and temperature recordings

## Limitations
Only one GPU can be benchmarked at a time.
<br>
A system must have a CPU with two cores.

## Future Improvements
* Support for AMD, Intel, Apple, and CPU to benchmark
* More models types
* Use of FP8 in a script called peak
* Better GUI
* Better file system
* Better graphs
* Saving the results in csv files for others to use for their own visuals

## GUI Overview
![Image](https://github.com/user-attachments/assets/2eff3daa-0c16-45b3-8fd9-5aed03f2c30f)
***
![Image](https://github.com/user-attachments/assets/741e8857-3696-40b0-af1d-c86817e912b8)
***
![Image](https://github.com/user-attachments/assets/70e7dd7d-327f-4f33-bbba-e1df7f326016)
***
![Image](https://github.com/user-attachments/assets/de88b856-401b-44c1-9869-a6dbfee3fb65)

## Some Example Results
![Image](https://github.com/user-attachments/assets/b275f671-7616-45de-8860-d5ba1d5714b5)
***
***
![Image](https://github.com/user-attachments/assets/c9358db1-9975-40e4-a75c-aae4c18a03a9)
***
***
![Image](https://github.com/user-attachments/assets/c8b23220-2dc4-48ed-9875-d621db070807)
***
***
![Image](https://github.com/user-attachments/assets/243d058a-1d5b-4d79-9328-edfa01388844)

## Copyright Information
Copyright 2025 Yeshua A. Romero

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0
   
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
