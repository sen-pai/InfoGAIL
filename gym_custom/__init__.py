from gym.envs.registration import register

register(
    id='FreeMovingContinuous-v0',
    entry_point='gym_custom.free_moving_env:FreeMovingContinuous'
)