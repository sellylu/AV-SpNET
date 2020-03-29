# Learn an Arousal-Valence Speech Front-End Network using Media Data In-the-wild
> ###### Time: Mon, Nov 12, 2018 1:07 PM
> ###### Author: Chih-Chuan Lu
> ###### tags: `speech emotion recognition` `media data in-the-wild` `convolutional neural network` `speech front-end network`
> [paper link](https://dl.acm.org/citation.cfm?id=3266306)

## Abstract

語音情緒辨識現已廣泛利用深度學習，然而在情緒資料庫中的標記卻很難獲得並變異性很高，*「初始化與微調網路參數」* 的的技巧可以用來解決此問題。我們提出一個供語音情緒辨識用的架構及初始神經網路參數——情緒激發與向性語音前端網路(AV-SpNET)，透過大量非領域內語音資料與音訊及文字等多模態產生的代理標記一起學習。結果顯示經我們的AV-SpNET初始過的網路再被目標資料庫微調可以達到更好的表現，並在目標資料庫IEMOCAP與NNIME上僅需75%、50%的總資料微調網路，即可達到整體最好準確率，以及相較使用全部資料進行未初始化方法的網路結果更好。

Recent progress in speech emotion recognition (SER) technology has benefited from the use of deep learning techniques. However, difficulty in emotion database collection and diversity across domains make it still challenging. An *initialization - fine-tuning* strategy help mitigate these technical challenges.
In this work, we propose an initialization network that gears toward SER applications by learning the speech front-end network on a large media data collected in-the-wild jointly with proxy arousal-valence labels that are multimodally derived from audio and text information, termed as the Arousal-Valence Speech Front-End Network (AV-SpNET). The result shows networks that've been initialized by our AV-SpNET and then fine-tuned by target database can achieve better performance than random initialization. Furthermore, it requires as little as 75% and 50% of the fine-tuned data on target databases (IEMOCAP and NNIME) to achieve the best overall result and surpass method based on randomly-initialized network with
fine-tuning on the complete training set.

## Introduction

Emotion plays an important role in  human daily communication, the ability to recognize the internal emotion state can benefit the decision making in an wide range of applications, such as *healthcare*, *education* or *commerce*.
For example, a tele-home health tool may let healthcare provider to monitor patient's emotional state[1]; an e-learning system can monitor student's emotion to provide aids and enhance user experience[2].

This technology involved is what we so call Speech Emotion Recognition(SER). Recently, there're researches about SER starting applying deep neural network architecture. However, if we want to widely adopt SER across applications, there’re still problems stop us from it:

1. difficulty in emotion database collection
2. variability in collected data across domains

An important success strategy in deep learning is the ***initialization – fine-tuning***.

### Initialization - fine-tuning

Initialization network is often **learnt from a large number of background data**. For example, VGG is a well-known initialization network trained on ImageNet[3]. With the well-pretrained weights in convolutional layers, we can use many other kinds of data as input to perform fine-tuning and obtain higher accuracy. This technique can not only improve the performance but also reduce the require of in-domain labeled data, since we’ve already got a **better initialization starting point** from the background database.

### Related Work

Some Previous works in SER have also tried this strategy.

1. In 2016 Fayek et al. have use automatic speech recognition to examine the transferability of speech for emotion recognition.[4]
2. In 2013, and 2017, Deng et al. and Huang et al. have then performed transfer using sparse autoencoder and a purposed PCANet between emotion corpora. [5][6]
3. In 2018, Neumann et al. have also done transferring network weights between emotion corpora, bus specifically a cross-lingual SER from English to French.[7]

Most of the above works perform transfer between emotion corpora, or they have similarity between background and target domain. While unlike in other domain, their initialization were actually not pretrained from a large-scale background.

Badshah et al. in 2017 did so, they transfer AlexNet for object recognition to SER.[8] However, it showed non-successful due to the huge difference between background and target domain.

### Our Purposed Model

Therefore, in our work, instead of learning initialization between emotion corpora, we aim to learn a meaningful speech front-end network from a large scale media data in-the-wild. This can easily apply on different emotion contexts; and, of course, it can result in better performance and hoping to reduce the amount of label.

![avspnet](https://i.imgur.com/Y848Oau.png)


## Citation
```
@inproceedings{Lu:2018:LAS:3266302.3266306,
 author = {Lu, Chih-Chuan and Li, Jeng-Lin and Lee, Chi-Chun},
 title = {Learning an Arousal-Valence Speech Front-End Network Using Media Data In-the-Wild for Emotion Recognition},
 booktitle = {Proceedings of the 2018 on Audio/Visual Emotion Challenge and Workshop},
 series = {AVEC'18},
 year = {2018},
 isbn = {978-1-4503-5983-2},
 location = {Seoul, Republic of Korea},
 pages = {99--105},
 numpages = {7},
 url = {http://doi.acm.org/10.1145/3266302.3266306},
 doi = {10.1145/3266302.3266306},
 acmid = {3266306},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {convolutional neural network, media data in-the-wild, speech emotion recognition, speech front-end network},
} 
```

## Reference
1. Christine Lisetti and Cynthia LeRouge. 2004. Affective computing in tele-home health. In 37th Annual Hawaii International Conference on System Sciences, 2004. Proceedings of the. 8 pp.–. https://doi.org/10.1109/HICSS.2004.1265373
2. Krithika L.B and Lakshmi Priya GG. 2016. Student Emotion Recognition System (SERS) for e-learning Improvement Based on Learner Concentration Metric. Procedia Computer Science 85 (2016), 767 – 776. https://doi.org/10.1016/j.procs. 2016.05.264 International Conference on Computational Modelling and Security (CMS 2016).
3. Karen Simonyan and AndrewZisserman. 2015. Very deep convolutional networks for large-scale image recognition. In Proceedings of the International Conference on Learning Representations (ICLR).
3. Haytham M Fayek, Margaret Lech, and Lawrence Cavedon. 2016. On the Correlation and Transferability of Features Between Automatic Speech Recognition and Speech Emotion Recognition.. In Proceedings of the International Speech Communication Association (Interspeech). 3618–3622. https://doi.org/10.21437/Interspeech.2016-868
5. Jun Deng, Zixing Zhang, Erik Marchi, and Bjorn Schuller. 2013. Sparse Autoencoder-Based Feature Transfer Learning for Speech Emotion Recognition. In 2013 International Conference on Affective Computing and Intelligent Interaction (ACII). 511–516. https://doi.org/10.1109/ACII.2013.90
6. Zhengwei Huang, Wentao Xue, Qirong Mao, and Yongzhao Zhan. 2017. Unsupervised domain adaptation for speech emotion recognition using PCANet. Multimedia Tools and Applications 76, 5 (01 Mar 2017), 6785–6799. https://doi.org/10.1007/s11042-016-3354-x
7. Michael Neumann and Ngoc Thang Vu. 2018. Cross-lingual and Multilingual Speech Emotion Recognition on English and French. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
8. Abdul Malik Badshah, Jamil Ahmad, Nasir Rahim, and Sung Wook Baik. 2017. Speech Emotion Recognition from Spectrograms with Deep Convolutional Neural Network. In 2017 International Conference on Platform Technology and Service (PlatCon). 1--5. https://doi.org/10.1109/PlatCon.2017.7883728
