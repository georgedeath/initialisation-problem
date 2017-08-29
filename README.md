# Visual Object Tracking: The Initialisation Problem
Model initialisation is an important problem in object tracking. Tracking 
algorithms are generally provided with the first frame of a sequence and a 
bounding box indicating the location of the object. This bounding box may 
contain of a large number of background pixels in addition to the object and 
can lead to parts-based tracking algorithms initialising their object models 
in background regions of the bounding box.

We tackled this as a missing labels problem, marking pixels sufficiently away 
from the bounding box as belonging to the background and learning the labels 
of the unknown pixels. We adapted three techniques to this problem; One-Class 
SVM (OC-SVM), Sampled-Based Background Model (SBBM), a novel background model 
based on pixels samples, and Learning Based Digital Matting (LBDM).

These were evaluated with leave-one-video-out cross-validation on images taken 
from the [VOT2016](http://www.votchallenge.net/vot2016/) tracking benchmark. 
Our evaluation showed both OC-SVMs and SBBMs are capable of providing a good 
levels of segmentation accuracy but are too parameter-dependent to be used in 
real-world scenarios. We showed that LBDM achieved significantly increased 
performance with cross validation selected parameters and investigated its 
robustness to parameter variation.

This repository contains the three Python 3.6 implementations of the techniques
(LBDM, SBBM, OC-SVM) used in the paper.

## Prerequisites
+ All techniques: matplotlib, numpy, scipy, skimage
+ OC-SVM: sklearn, vlfeat-ctypes: <https://github.com/dougalsutherland/vlfeat-ctypes>

## Usage
Information on each method can be found by typing ```help(function_name)``` 
at the python interpreter.

To see a working example of each of the three techniques, run their respective 
python files:
```
python alpha_matting_segmentation.py
```
![Alpha matting segmentation](https://i.imgur.com/Lbw4iWe.png)

```
python ocsvm_segmentation.py
```
![One-class SVM segmentation](https://i.imgur.com/B2FYebC.png)

```
python sbbm_segmentation.py
```
![SBBM segmentation](https://i.imgur.com/JQdNH7R.png)


## Author
**[George De Ath](https://www.linkedin.com/in/georgedeath/)**

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) 
file for details.
