# Instance-Wise Domain Generalization for Cross-Scene Wetland Classification With Hyperspectral and LiDAR Data
## Abstract
Wetland is one of the three ecosystems in the world, and collaborative monitoring using hyperspectral images (HSIs) and light detection and ranging (LiDAR) has been important for wetland ecological protection. However, because of the domain shift of different images, cross-scene wetland classification of HSIs and LiDAR is a practical challenge, necessitating the development of models trained solely on the source domain (SD) and directly transferred to the target domain (TD) without retraining. To address this issue, an instance-wise domain generalization network (IDGnet) is proposed for HSI and LiDAR cross-scene wetland classification. An instance-wise random domain expansion module (IWR-DEM) is developed to simulate the domain shift, establishing the extended domain (ED). Specifically, the original HSI and LiDAR data are separated as semantic and background information in the frequency domain, a random background shift is applied to the HSI, and a semantic random shift is deployed to LiDAR. The HSI and LiDAR fusion features are extracted from the SD and ED by a weight-shared network. Multiple condition constraints are proposed for domain and class alignment, learning the domain-invariant and class-specific information and improving model generalization. Experiments conducted on two wetland datasets demonstrate the superiority of the proposed IDGnet for cross-scene wetland classification with HSI and LiDAR data.

Paper web page: 
[Instance-Wise Domain Generalization for Cross-Scene Wetland Classification With Hyperspectral and LiDAR Data](https://ieeexplore.ieee.org/document/10806800)
## Paper
Please cite our paper if you find the code useful for your research.
```
@ARTICLE{10806800,
  author={Guo, Fangming and Li, Zhongwei and Ren, Guangbo and Wang, Leiquan and Zhang, Jie and Wang, Jianbu and Hu, Yabin and Yang, Min},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Instance-Wise Domain Generalization for Cross-Scene Wetland Classification With Hyperspectral and LiDAR Data}, 
  year={2025},
  volume={63},
  number={},
  pages={1-12},
  keywords={Wetlands;Laser radar;Semantics;Feature extraction;Training;Frequency-domain analysis;Discrete cosine transforms;Monitoring;Fuses;Oceanography;Cross-scene wetland classification;domain generalization;hyperspectral image (HSI);light detection and ranging (LiDAR)},
  doi={10.1109/TGRS.2024.3519900}}

```

## Requirements
CUDA Version: 11.3.1

torch: 1.10.0

Python: 3.7.11
