#!/usr/bin/env python3

import torch
import time
import rclpy

from rclpy.node import Node

from wheel_navigation.environment import Unity
from wheel_navigation.agent import MLP, Batch, discount_cumsum

class PPO(Node):
    """Proximal Policy Optimization"""

    def __init__(self):
        super().__init__('wheel_navigation_ppo')

        # Set parameters
        self.policy_std = torch.exp(torch.tensor([-0.5, -0.5]))
        self.discount_gamma = 0.99
        self.discount_lamda = 0.95
        self.batch_size_max = 500
        self.actor_optimizer_iter = 80
        self.critic_optimizer_iter = 80
        self.clip_ratio = 0.2
        self.max_kld_step = 0.015

        # Initialize networks
        self.actor = MLP(num_input=40, num_output=2)
        self.critic = MLP(num_input=40, num_output=1)
        self.batch = Batch()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        print(self.actor)
        
        # Connect to Unity
        self.get_logger().warning("Wait for Unity scene play")
        self.unity = Unity()
        self.get_logger().info("Unity environment connected")

        # Initialize ROS
        self.get_logger().info("Start simulation")
        self.time_start = time.time()
        self.timer = self.create_timer(0.2, self.timer_callback)

    def timer_callback(self):
        # Simulate environment one-step forward
        self.unity.env.step()
        exp = self.unity.get_experience()
        if not exp: return

        # Set commands
        obs = torch.cat([exp[agent]['obs'] for agent in exp], dim=0)
        with torch.no_grad():
            mu = self.actor(obs)
            policy = torch.distributions.normal.Normal(mu, self.policy_std)
            act = policy.sample()
            val = self.critic(obs)
        self.unity.set_command(act)
        self.batch.store(exp, torch.cat([act, val], dim=1))
        
        # Start learning
        print(f'\rBatch: {self.batch.size()}/{self.batch_size_max}', end='')
        if self.batch.size() >= self.batch_size_max:
            loss = self.learning(self.batch.pop())
            print()
            self.get_logger().warning(
                f"Time {time.time()-self.time_start:.1f}s, Loss {float(loss):.4f}")

    def learning(self, batch):
        """Training neural network using batch data"""

        # Generalized Advantage Estimation
        for episode in batch:
            if episode.shape[0] < 2: continue
            rew = episode[:, 40:41].squeeze().tolist()
            val = episode[:, 43:44].squeeze().tolist()

            delta = [rew[i] + self.discount_gamma*val[i+1] - val[i] for i in range(len(rew)-1)] + [self.discount_gamma*val[-1]]
            adv = discount_cumsum(delta, self.discount_gamma*self.discount_lamda)
            ret = discount_cumsum(rew, self.discount_gamma)

            episode = torch.cat([episode, torch.tensor(adv).unsqueeze(1), torch.tensor(ret).unsqueeze(1)], dim=1)

        # Optimize actor network
        data = torch.cat(batch, dim=0)
        obs, act, adv = data[:, 0:40], data[:, 41:43], data[:, -2]
        
        with torch.no_grad():
            mu = self.actor(obs)
            policy = torch.distributions.normal.Normal(mu, self.policy_std)
            logp_old = policy.log_prob(act).sum(axis=1)

        for _ in range(self.actor_optimizer_iter):
            mu = self.actor(obs)
            policy = torch.distributions.normal.Normal(mu, self.policy_std)
            logp = policy.log_prob(act).sum(axis=1)

            # Early stopping by KL-divergence limit
            kld = (logp_old - logp).mean().item()
            if kld > self.max_kld_step: break

            # PPO Clip
            logp_ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(logp_ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            actor_loss = -(torch.min(logp_ratio * adv, clip_adv)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Optimize critic
        ret = data[:, -1]
        critic_loss = torch.tensor([])
        for _ in range(self.critic_optimizer_iter):
            critic_loss = ((self.critic(obs).squeeze() - ret)**2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        return actor_loss.item()

def main(args=None):
    rclpy.init(args=args)
    node = PPO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()