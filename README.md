# This work is still under review :lock:

This work propose an innovative assisted learning framework that employs a "teacher-student" architecture equipped with local and global distillation schemes. 

It has several advantages as outlined below:

:star: **Ability to maintain performance during testing with missing modalities**

:star: **High interpretability demonstrating knowledge transferability between different modalities**
  
:star: **Simplicity and flexibility**

Below is the process of the proposed framework:

Step 1: Training teacher models using cross-entropy and dice loss.
        Using train_t.py to train the teachr.

Step 2: Training student models using our framework.
        Using train_s.py to train the teachr.

Step 3: Testing.
