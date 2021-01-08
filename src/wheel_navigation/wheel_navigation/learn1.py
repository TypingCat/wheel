#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json
import torch
import copy
import threading
import time

from wheel_navigation.sim1 import Brain
from wheel_navigation.env import Batch

class SPG(Node):
    """Simple Policy Gradient"""

    def __init__(self):
        # Initialize brain
        self.brain = Brain(num_input=40, num_output=9)
        self.batch = Batch(size=1000)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.01)

        # Initialize ROS
        super().__init__('wheel_navigation_learning')
        self.time = time.time()
        self.sample_subscription = self.create_subscription(String, '/sample', self.sample_callback, 1)
        self.brain_update_publisher = self.create_publisher(String, '/brain/update', 10)

    def sample_callback(self, msg):
        # Accumulate samples into batch
        samples = json.loads(msg.data)
        self.batch.extend(
            [samples[agent] for agent in samples])
        print(self.batch.check_free_space())

        # Remove dead thread
        try:
            self.threads = [t for t in self.threads if t.isAlive()]
        except:
            self.threads = []

        # Insert batch into learning thread
        if self.batch.check_free_space() <= 0:
            t = threading.Thread(
                target=self.learning,
                args=[copy.deepcopy(self.batch)],
                daemon=True)
            self.threads.append(t)
            self.batch.reset()
            t.start()

    def learning(self, batch):
        obs = [samples['obs'] for samples in batch.data]
        obs = torch.tensor(obs).float()
        act = [samples['act'] for samples in batch.data]
        act = torch.tensor(act).float()
        reward = self.get_reward(
            [int(samples['agent']) for samples in batch.data],
            [samples['reward'] for samples in batch.data],
            [samples['done'] for samples in batch.data])
        
        # Train the brain
        self.optimizer.zero_grad()

        policy = self.brain.get_policy(obs)
        
        # logp = 


        # loss = -(logp * weights).mean()


        # loss.backward()
        self.optimizer.step()

        # Log
        # self.publish_brain_state()
        # self.get_logger().warning(f"Time {time.time()-self.time:.1f}s, Loss {float(loss):.4f}")

    def get_reward(self, agents, rewards, dones):
        # Initialize values
        idx = {}
        try:
            self.prev_reward
        except:
            self.prev_reward = {}
        for i, agent in enumerate(agents):
            try:
                idx[agent]
            except:
                idx[agent] = []
            idx[agent].append(i)
            try:
                self.prev_reward[agent]
            except:
                self.prev_reward[agent] = 0

        # For each agent,
        reward_spread = [0.]*len(agents)
        for a in idx:
            reward = [rewards[i] for i in idx[a]]
            done = [dones[i] for i in idx[a]]
            
            # Find episodes index
            episode_start = 0
            episode_idx = []
            episode_ends =[i for i, d in enumerate(done) if d == 1]
            for end in episode_ends:
                e = end + 1
                episode_idx.append([episode_start, e])
                episode_start = e
            if episode_start != len(done):
                episode_idx.append([episode_start, len(done)])

            # Calculate episode reward
            episode_rewards = [[i[1] - i[0], sum(reward[i[0]:i[1]])] for i in episode_idx]
            episode_rewards[0][1] += self.prev_reward[a]
            self.prev_reward[a] = episode_rewards[0][-1]
            episode_reward = [[r[1]]*r[0] for r in episode_rewards]
            episode_reward = sum(episode_reward, [])
            for i, r in enumerate(episode_reward):
                reward_spread[idx[a][i]] = r

        return reward_spread

    def publish_brain_state(self):
        # Extract brain state as list dictionary
        state_tensor = self.brain.state_dict()
        state_list = {}
        for key, value in state_tensor.items():
            state_list[key] = value.tolist()
        
        # Publish brain state
        state = String()
        state.data = json.dumps(state_list)
        self.brain_update_publisher.publish(state)

# class Accumulator:
#     def __init__(self):
#         self.data = {}

#     def update(self):
#         pass

def main(args=None):
    rclpy.init(args=args)
    node = SPG()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()