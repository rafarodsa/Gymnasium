import pytest
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.ant_variable_legs import AntVariableLegsEnv

@pytest.mark.parametrize("num_legs", [4, 6, 8])
def test_ant_variable_legs(num_legs):
    """Test the AntVariableLegs environment with different numbers of legs."""
    env = AntVariableLegsEnv(num_legs=num_legs)
    assert isinstance(env, AntVariableLegsEnv)
    
    # Test initialization
    obs, _ = env.reset()
    assert obs.shape == (27 + (num_legs - 4) * 6,)
    
    # Test action space
    action = env.action_space.sample()
    assert action.shape == (num_legs * 2,)  # 2 joints per leg
    
    # Test step
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (27 + (num_legs - 4) * 6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    # Test that the environment can be rendered
    env.render()
    env.close()

def test_ant_variable_legs_xml():
    """Test that the XML file is properly modified for different numbers of legs."""
    for num_legs in [4, 6, 8]:
        env = AntVariableLegsEnv(num_legs=num_legs)
        assert env.num_legs == num_legs
        
        # Check that the XML has the correct number of legs
        tree = env._get_xml_tree()
        legs = tree.findall(".//body[starts-with(@name, 'leg_')]")
        assert len(legs) == num_legs
        
        # Check that the actuators match the number of legs
        actuators = tree.findall(".//motor")
        assert len(actuators) == num_legs * 2  # 2 motors per leg
        
        env.close()

def test_ant_variable_legs_observation():
    """Test that the observation space is correctly sized for different numbers of legs."""
    for num_legs in [4, 6, 8]:
        env = AntVariableLegsEnv(num_legs=num_legs)
        obs, _ = env.reset()
        
        # Base dimensions (27) + additional dimensions for each leg beyond 4 (6 per leg)
        expected_dim = 27 + (num_legs - 4) * 6
        assert obs.shape == (expected_dim,)
        
        env.close() 