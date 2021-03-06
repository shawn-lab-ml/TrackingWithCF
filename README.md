# TrackingWithCF
Re-implementation of 3 of the core algorithms for object tracking, namely Minimum Output Sum of Squared Error (MOSSE), Kernelized Correlation Filters (KCF) and Dual Correlation Filters using raw features (grayscale or RGB) and HOG features on the task of tracking from the point of view of a UAV using the UAV123 dataset.

While the KCF and DCF have proven much more performant than the MOSSE algorithm with MIOU scores nearly 50% higher, the use of HOG features seemed to be detrimental in the case of the KCF which goes against what was demonstrated in the paper from Henriques et al. (2015. These 3 algorithms form the basis of their more advanced counterparts and suffer from drawbacks from variations in the frames of the videos used. The most prominent difficulty is that of tracking in the context of Full Occlusion. The task was all the more difficult since we used a 10fps version of the dataset making variations between frames all the more important.



# Results

To note that the implmentation was done in python and hence does not benefit from the same advantages as C (notably speed), some improvement scan be made using numba and cython, however, the purpose of this repository is one of demonstration.

![alt text](https://github.com/shawn-lab-ml/TrackingWithCF/blob/main/images/OPE.png)

![alt text](https://github.com/shawn-lab-ml/TrackingWithCF/blob/main/images/multOPE.png)

![alt text](https://github.com/shawn-lab-ml/TrackingWithCF/blob/main/images/trackspeed.png)



Implementation from the following papers:

`
@INPROCEEDINGS{5539960,
  author={Bolme, David S. and Beveridge, J. Ross and Draper, Bruce A. and Lui, Yui Man},
  journal={2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition}, 
  title={Visual object tracking using adaptive correlation filters}, 
  year={2010},
  volume={},
  number={},
  pages={2544-2550},
  doi={10.1109/CVPR.2010.5539960}}
`

`
  @ARTICLE{6870486,
  author={Henriques, João F. and Caseiro, Rui and Martins, Pedro and Batista, Jorge},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={High-Speed Tracking with Kernelized Correlation Filters}, 
  year={2015},
  volume={37},
  number={3},
  pages={583-596},
  doi={10.1109/TPAMI.2014.2345390}}
`

# Requirements

`cyvlfeat==0.7.0
opencv-contrib-python==4.5.5.62
numpy==1.21.0
matplotlib==3.5.0
pandas==1.4.0`


# Dataset
Link to download the UAV123 10fps dataset: https://cemse.kaust.edu.sa/ivul/uav123c <br/>

`@Inbook{Mueller2016,
author="Mueller, Matthias and Smith, Neil and Ghanem, Bernard",
editor="Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max",
title="A Benchmark and Simulator for UAV Tracking",
bookTitle="Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11--14, 2016, Proceedings, Part I",
year="2016",
publisher="Springer International Publishing",
pages="445--461",
isbn="978-3-319-46448-0",
doi="10.1007/978-3-319-46448-0_27",
url="http://dx.doi.org/10.1007/978-3-319-46448-0_27"
}`

# Useful Repositories
- https://github.com/menpo/cyvlfeat <br/>
- https://github.com/uoip/KCFpy <br/>
- https://github.com/fengyang95/pyCFTrackers
