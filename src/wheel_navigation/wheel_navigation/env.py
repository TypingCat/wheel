#!/usr/bin/env python3

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
        exp = {}
        decision_steps, terminal_steps = self.env.get_steps(self.behavior)

        if len(set(decision_steps) - set(terminal_steps)) == len(self.agents):
            for agent in self.agents:
                exp[agent] = {}
            for agent in terminal_steps:
                exp[agent]['obs'] = list(map(float, terminal_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(terminal_steps[agent].reward)
                exp[agent]['done'] = True
            for agent in list(set(decision_steps) - set(terminal_steps)):
                exp[agent]['obs'] = list(map(float, decision_steps[agent].obs[0].tolist()))
                exp[agent]['reward'] = float(decision_steps[agent].reward)
                exp[agent]['done'] = False
        return exp
    
    def set_command(self, cmd):
        self.env.set_actions(self.behavior, cmd.detach().numpy())
