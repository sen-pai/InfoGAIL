# InfoGAIL

A Pytorch implementation of InfoGAIL built on top of stable-baselines3 and imiation. 

Core changes to the imitation repository v0.2.0 are done to implement InfoGAIL 
We have kept only necessary files from the imitation repository. 


### Additional Modules for CNN-GAIL
To avoid any more core changes to the imitation library, all classes needed to execute a CNN version of GAIL and WGAIL are saved in the ``cnn_modules`` folder.

Two new discriminator classes in ``cnn_modules/cnn_discriminator.py``
* ``ActObsCNN``: uses a NaturCNN backbone from stable-baselines 3 to extract features from an image observation. Obs features are concatenated with the action and rest is as ``ActObsMLP`` would work. 
* ``ObsOnlyCNN``: same as ``ActObsCNN``, no action is used.