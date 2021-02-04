#!/usr/bin/env python3

import torch
import numpy as np

from mlagents_envs.environment import UnityEnvironment

class Unity():
    def __init__(self):
        self.connect()

    def __del__(self):
        try:
            self.env.close()
        except:
            pass

    def connect(self):
        # Connect with Unity
        self.env = UnityEnvironment(file_name=None, seed=1)
        self.env.reset()

        # Get behavior specification
        for behavior_name in self.env.behavior_specs:
            self.behavior = behavior_name
            break
        spec = self.env.behavior_specs[self.behavior]
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        self.agents = list(set(list(decision_steps.agent_id) + list(terminal_steps.agent_id)))

        print(f"Behavior name: {self.behavior}")
        print(f"Observation shapes: {spec.observation_shapes}")
        print(f"Action specifications: {spec.action_spec}")
        print(f"Agents: {self.agents}")

    def get_experience(self):
        """Get observation, done, reward from unity environment"""
        # observation[ 0:36] = scan results
        # observation[36:38] = linear/angular velocity of robot
        # observation[38:40] = distance/angle to target
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)
        exp = {}
            
        if len(set(decision_steps) | set(terminal_steps)) == len(self.agents):
            for agent in self.agents: exp[agent] = {}
            for agent in terminal_steps:
                exp[agent]['obs'] = torch.from_numpy(terminal_steps[agent].obs[0]).unsqueeze(0)
                exp[agent]['reward'] = torch.tensor([terminal_steps[agent].reward]).unsqueeze(0)
                exp[agent]['done'] = True
            for agent in list(set(decision_steps) - set(terminal_steps)):
                exp[agent]['obs'] = torch.from_numpy(decision_steps[agent].obs[0]).unsqueeze(0)
                exp[agent]['reward'] = torch.tensor([decision_steps[agent].reward]).unsqueeze(0)
                exp[agent]['done'] = False
        return exp
    
    def set_command(self, command):
        cmd = command.tolist()
        self.env.set_actions(
            self.behavior,
            np.array([[max(min(l, 1.), -1.), max(min(a, 1.), -1.)] for l, a in cmd]))

class Batch:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.temp = {}
        self.data = []

    def size(self):
        return sum([d.shape[0] for d in self.data])

    def store(self, exp, act):
        """Save experiences and wrap finished episodes"""
        # sample[ 0:40] = observation
        # sample[40:41] = reward
        # sample[41:  ] = action
        for i, agent in enumerate(exp):
            sample = torch.cat([exp[agent]['obs'], exp[agent]['reward'], act[i].unsqueeze(0)], dim=1)
            if agent not in self.temp.keys():
                self.temp[agent] = sample
            else:
                self.temp[agent] = torch.cat([self.temp[agent], sample], dim=0)
            if exp[agent]['done']:
                self.data.append(self.temp[agent])
                del self.temp[agent]
    
    def pop(self):
        episodes = self.data
        self.data = []
        return episodes