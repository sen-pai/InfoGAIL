# InfoGAIL

A Pytorch implementation of InfoGAIL built on top of stable-baselines3 and imiation. 

Core changes to the imitation repository v0.2.0 are done to implement InfoGAIL 
We have kept only necessary files from the imitation repository. 


#### Changes for WGAIL 
Two new classes in ``src\imitation\rewards\discrim_nets.py``
* ``WassersteinDiscrimNet``: Inherits ``DiscrimNet`` and overwrites ``disc_loss`` that implements the Wasserstein loss to train the discriminator
* ``DiscrimNetWGAIL``: Inherits ``WassersteinDiscrimNet`` and overwrites ``reward_train`` with -logits as the reward for the generator. 

Two new classes in ``src\imitation\algorithms\adversarial.py``
* ``WGAIL``: Core changes from ``GAIL`` class are ``DiscrimNetWGAIL`` as the discriminator and ``disc_opt_cls`` as RMSprop instead of Adam
* ``WassersteinAdversarialTrainer``: inherits ``AdversarialTrainer`` class to include gradient clipping in the ``train_disc`` function

Sample test script for WGAIL: ``python .\minigrid_wgail_training_script.py -r testing_wgail -t minigrid_empty_right_down -f --vis-trained ``

Policy was consistent even if env was changed from "MiniGrid-Empty-6x6-v0" to "MiniGrid-Empty-8x8-v0" and "MiniGrid-Empty-5x5-v0" while testing

#### Additional Modules for CNN-GAIL
To avoid any more core changes to the imitation library, all classes needed to execute a CNN version of GAIL and WGAIL are saved in the ``cnn_modules`` folder.

Two new discriminator classes in ``cnn_modules/cnn_discriminator.py``
* ``ActObsCNN``: uses a NaturCNN backbone from stable-baselines 3 to extract features from an image observation. Obs features are concatenated with the action and rest is as ``ActObsMLP`` would work. 
* ``ObsOnlyCNN``: same as ``ActObsCNN``, no action is used.

To use the CNN version of GAIL or WGAIL, exclude the ``-f`` arg. 
A sample test script for CNN-GAIL: ``python .\minigrid_gail_training_script.py -r testing_cnngail -t img_no_stack_minigrid_empty_down_right --vis-trained ``