from gym.envs.registration import register

register(
    id='FreeMovingContinuous-v0',
    entry_point='gym_custom.free_moving_continuous:FreeMovingContinuous'
)

register(
    id='CoverAllTargets-v0',
    entry_point='gym_custom.free_moving_continuous:CoverAllTargets'
)


register(
    id='FreeMovingDiscrete-v0',
    entry_point='gym_custom.free_moving_discrete:FreeMovingDiscrete'
)

register(
    id='CoverAllTargetsDiscrete-v0',
    entry_point='gym_custom.free_moving_discrete:CoverAllTargetsDiscrete'
)
