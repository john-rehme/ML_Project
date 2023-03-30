# ML Project Proposal
## Introduction/Background:
Alzheimer’s disease (“AD”) is an increasingly prevalent condition in aging adults characterized by a debilitating cognitive decline, most notably in memory function (“dementia”). Although AD is mostly seen in adults ages 70+, early onset dementia is known to occur in younger adults (Scheltens, et al., 2021). Pathogenesis of the disease has been associated with an accumulation of β-amyloid (“Aβ”) peptides into masses known as “amyloid plaques” that form in the brain. These plaques, visible on MRI scans, are the main diagnostic criteria for AD (Gouras, Olsson, & Hansson, 2014). The degree to which these plaques have spread and are visible on the scans is directly correlated to the degree of dementia experienced by an AD patient (Cummings & Cotman, 1995). Therefore, an automated tool to assess not only the presence of AD and/or dementia but also its degree would help speed up and further calibrate the diagnostic process. This would facilitate more accurate diagnoses and provide more time for patients and their loved ones to determine a course of action and intervention plan.


## Problem:
In this project, we want to design a highly accurate model that can look at a .jpg of an MRI scan and classify whether the patient’s brain is non-demented, very mildly demented, mildly demented, or moderately demented. 

## Methods:
To analyze the images we will mostly be utilizing convolutional neural networks in pytorch. Though CNNs will be the base of our model, depending on the necessary complexity, more features will be added. ResNets, or residual networks, are highly applicable to image classification as it was used to win the 2015 ImageNet competition (He, et al., 2016). From ResNets, we may pull the idea of skip connections to use in our model (Oyedotun, et al., 2021). Attention mechanisms may also be useful in our model to be able to specify which parts of the image are referenced for the final classification decision (Vaswani, et al, 2017).


## References for us:

[Image Processing](https://ieeexplore.ieee.org/document/8320684)

[Attention Intro](https://blog.paperspace.com/image-classification-with-attention/)

[Melanoma with Visual Attention](https://www2.cs.sfu.ca/~hamarneh/ecopy/ipmi2019.pdf) 

## Contribution Table:

![](assets/Contribution_Table.png)

## Gantt Chart

![](assets/Gantt_Chart.png)

## Dataset

Our dataset is [Alzheimer's Dataset ( 4 class of Images)](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) from Kaggle. It is composed of 6400 .jpg images, where each image represents a layer of an MRI scan. The images are divided into one of four categories: non-demented, very mildly demented, mildly demented, or moderately demented. The break-down of these files are as follows:

![](assets/Data_Categories.png)

These scans are sourced from Open Access of Imaging Studies (OASIS). Dementia severity was assessed using the Clinical Dementia Rating (CDR) scale (Marcus, et al., 2010).

We did not need to clean the dataset; this had already been done by OASIS.

Results and Discussion
Our current method for the midterm report involves a five-step forward feature selection. The network consists of three convolutional layers followed by two fully connected layers.

![](assets/NetworkArch.JPG)


## Works Cited

Cummings, B. J., & Cotman, C. W. (1995). Image analysis of β-amyloid load in Alzheimer's disease and relation to dementia severity. The Lancet, 346(8989), 1524-1528.

Gouras, G. K., Olsson, T. T., & Hansson, O. (2015). β-Amyloid peptides and amyloid plaques in Alzheimer’s disease. Neurotherapeutics, 12, 3-11.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

Oyedotun, O. K., Al Ismaeil, K., & Aouada, D. (2022). Why is everyone training very deep neural network with skip connections?. IEEE Transactions on Neural Networks and Learning Systems.

Scheltens, P., De Strooper, B., Kivipelto, M., Holstege, H., Chételat, G., Teunissen, C. E., ... & van der Flier, W. M. (2021). Alzheimer's disease. The Lancet, 397(10284), 1577-1590.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.





