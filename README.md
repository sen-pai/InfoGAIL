# InfoGAIL

A Pytorch implementation of InfoGAIL built on top of stable-baselines3 and imiation. 

Core changes to the imitation repository v0.2.0 are done to implement InfoGAIL 
We have kept only necessary files from the imitation repository. 





#### Changes for WGAIL 
Two new classes in ``src\imitation\rewards\discrim_nets.py``
* ``WassersteinDiscrimNet``: Inherits ``DiscrimNet`` and overwrites ``disc_loss`` that implements the Wasserstein loss to train the discriminator
* ``DiscrimNetWGAIL``: Inherits ``WassersteinDiscrimNet`` and overwrites ``reward_train`` with -logits as the reward for the generator. 

One new class in ``src\imitation\algorithms\adversarial.py``
* ``WGAIL``: Core changes from ``GAIL`` class are ``DiscrimNetWGAIL`` as the discriminator and ``disc_opt_cls`` as RMSprop instead of Adam