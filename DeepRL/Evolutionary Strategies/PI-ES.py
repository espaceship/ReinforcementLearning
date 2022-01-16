import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class BasePolicy(tf.keras.Model):

    def __init__(self, state_size=4, action_size=2, layer_size=32, stddev=0.05, initial_weights=False):
        super(BasePolicy, self).__init__()
        self.policy_function=tf.keras.models.Sequential([
            tf.keras.layers.Dense(layer_size, input_shape=(state_size,), activation="elu"),
            tf.keras.layers.Dense(action_size, activation='softmax'),
        ])

        # perturb the base weights, if given
        if initial_weights:
            for index, weights in enumerate(initial_weights):
                noise = tf.random.normal(shape=weights.shape, mean=0.0, stddev=1.0)
                initial_weights[index] = weights + noise * stddev
            # set the weights
            self.policy_function.set_weights(initial_weights)

    def call(self, state):
        # no e-greedy required; exploration is induced by perturbation in the base parameters
        action_probs=self.policy_function(state)
        action = np.argmax(action_probs)

        return action


class Session():

    ''' Policy Iteration Evolutionary Strategy (PI-ES)

    - Very simple, robust, and highly parallelizable
    - It can converge even if the search space is completely random(but take longer time)
    - Each iteration always helps to find better policy

    '''

    def __init__(self, env,
                 max_n_steps=200,
                 n_samples=10,
                 n_iterations=5,
                 stddev=0.5,
                 stddev_step_size=0.005,
                 n_episodes=10):

        self.env=env
        self.max_n_steps=max_n_steps
        self.n_samples=n_samples
        self.n_iterations = n_iterations
        self.stddev = stddev
        self.stddev_step_size=stddev_step_size
        self.n_episodes=n_episodes

    def train(self, base_model, show_info=True):

        #use 42 for to reproduce result
        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)


        mean_reward_log, base_reward_log=[],[]
        last_best=0

        for iteration in range(self.n_iterations):

            # generate a list of policy
            policies_list=self._generate_policies(base_model)

            # policy evaluation can be parallelized
            all_rewards=[]
            for policy in policies_list:
                returns=self._evaluate(policy)
                all_rewards.append(returns)

            max_index=all_rewards.index(max(all_rewards))
            current_best = np.max(all_rewards)
            if current_best > last_best:
                last_best = current_best.copy()
                base_model.policy_function.set_weights(policies_list[max_index].get_weights())
                # decrease search region if it succeeds
                # should never down to 0
                self.stddev = max(self.stddev - self.stddev_step_size, 0.005)
            else:
                # increase search region if it fails
                # should eventually cover all search space
                self.stddev = min(self.stddev + self.stddev_step_size, 1)

            base_rewards=self._evaluate(base_model)
            base_reward_log.append(base_rewards)

            # debugging
            if show_info:
                current_mean=np.mean(all_rewards)
                print("\rIteration: {}, mean rewards: {:.1f}, base rewards: {}".format(iteration, current_mean, base_rewards), end="")
                mean_reward_log.append(current_mean)

        return mean_reward_log, base_reward_log

    def _evaluate(self, policy):

        episode_rewards=[]
        for episode in range(self.n_episodes):
            state = self.env.reset()
            for step in range(self.max_n_steps):
                state_ = tf.cast(state[np.newaxis], dtype = 'float32')
                action=policy(state_)
                state, reward, done, info = self.env.step(action)
                if done:
                    break
            episode_rewards.append(step+1)
            
        return np.mean(episode_rewards)

    def _generate_policies(self, base_policy):

        base_weights = base_policy.get_weights()
        policies_list=[BasePolicy(stddev=self.stddev, initial_weights=base_weights) for _ in range(self.n_samples)]

        return policies_list


if __name__ == '__main__':

    env = gym.make("CartPole-v1")

    # use 42 to reproduce result
    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)


    model_args={'state_size':4,
        'action_size':2,
        'layer_size':32,
        'initial_weights':False,
        }

    session_args={
        'env':env,
        'max_n_steps':200,
        'n_samples': 50,
        'n_iterations':10,
        'stddev': 0.5,
        'stddev_step_size':0.002,
        'n_episodes':10,
        }

    base_model=BasePolicy(**model_args)
    sess=Session(**session_args)

    mean_rewards, base_rewards=sess.train(base_model)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards, label='Mean rewards')
    plt.plot(base_rewards, label='Base rewards')
    plt.xlabel("Iteration(50 policies, 10 episodes)", fontsize=14)
    plt.ylabel("Total Rewards", fontsize=14)
    plt.legend(fontsize='x-large')
    plt.show()
