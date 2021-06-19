import pickle5 as pickle

from modules.rnn_attention_discriminator import ActObsCRNNAttn
from BaC.bac_triplet_rnn import BaC_RNN_Triplet

from imitation.data import rollout

import torch


def get_classifier(venv):
    bac_class = ActObsCRNNAttn(
        action_space=venv.action_space, observation_space=venv.observation_space
    )

    traj_dataset_path = "./traj_datasets/middle_empty_random_traj.pkl"
    with open(traj_dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    transitions = rollout.flatten_trajectories(trajectories)

    # bac_trainer = BaCRNN(
    #     venv,
    #     eval_env=None,
    #     bc_trainer=None,
    #     bac_classifier=bac_class,
    #     expert_data=transitions
    # )

    bac_trainer = BaC_RNN_Triplet(
        venv,
        bc_trainer=None,
        bac_classifier=bac_class,
        expert_data=trajectories
    )


    bac_trainer.bac_classifier.load_state_dict(torch.load("bac_weights/avoid_middle_6x6.pt", map_location=torch.device('cpu')))

    return bac_trainer