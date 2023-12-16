# Assisted learning for land use classification: The important role of semantic correlation between heterogeneous images

This is the core code of our work published in ISPRS J. :kissing_heart:

This work propose an innovative assisted learning framework that employs a "teacher-student" architecture equipped with local and global distillation schemes for land use classification on heterogeneous data.

![Image of work](https://github.com/WHUlwb/Assisted_learning/blob/main/network.png)

It has several advantages as outlined below:

:star: **Ability to maintain performance during testing with missing modalities**

:star: **High interpretability demonstrating knowledge transferability between different modalities**
  
:star: **Simplicity and flexibility**

Below is the process of the proposed framework:

Step 1: Training teacher models using cross-entropy and dice loss. (Using train_t.py to train the teachr.)

Step 2: Training student models using our framework. (Using train_s.py to train the teachr.)

Step 3: Testing.

--------------------------------------------------------------
#### Thanks to [bubbliiiing](https://github.com/bubbliiiing) for providing the open-source project [HRnet](https://github.com/bubbliiiing/hrnet-pytorch/).
