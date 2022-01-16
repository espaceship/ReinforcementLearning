import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym


class ActorCritic(tf.keras.Model):

    '''Vanilla Advantage Actor-critic

    This Advantage Actor-critic gets an update on every step;
    it uses entropy bonus to decorrelate training instances;
    it also applies delayed update to the actor network.
    The policy function and value function are trained by 2 seperate optimizers.
    Learning is very oscillated but otherwise fast.

    '''

    def __init__(self, state_size=4, action_size=2, layer_size=32):
        super(ActorCritic, self).__init__()
        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(layer_size, activation="elu", input_shape=[state_size,]),
            tf.keras.layers.Dense(layer_size, activation="elu"),
            tf.keras.layers.Dense(action_size) # output probability for each action
        ])
        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(layer_size, activation="elu", input_shape=[state_size,]),
            tf.keras.layers.Dense(layer_size, activation="elu"),
            tf.keras.layers.Dense(1) # output probability for each action
        ])

    def call(self, state):
        # wrap actions logits with e-greedy policy to select action
        action_probs = tf.keras.layers.Softmax()(self.actor(state))
        action = np.random.choice(2, p=action_probs.numpy()[0])

        return action


class Session():

    def __init__(self,
                 env=None,
                 discount_rate=0.95,
                 n_episdoes=200, # the number of iteration, each i runs n-episodes
                 n_max_steps = 200, # maximum steps per episode
                 gamma=[0.0001, 0.5],
                 loss_functions=[tf.keras.losses.categorical_crossentropy,
                                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)], # actor loss function
                 optimizers=[
                     tf.keras.optimizers.Adam(), # actor optimizer
                     tf.keras.optimizers.Adam(), # critic optimizer
                 ]):

        self.env=env
        self.discount_rate=discount_rate
        self.n_episdoes=n_episdoes
        self.n_max_steps=n_max_steps
        self.gamma=gamma
        self.loss_functions=loss_functions
        self.optimizers=optimizers

    def train(self, model, show_info=True):

        # use 42 to reproduce result
        env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        rewards=[]
        best_score=0
        for episode in range(self.n_episdoes):
            state = self.env.reset()
            episode_rewards = []
            for step in range(self.n_max_steps):
                state_ = tf.cast(state[np.newaxis], tf.float32)
                # player one step
                reward, done, next_state, experience = self._play_one_step(state_, model)
                # update every step
                self._train_step(experience, model, step)
                state = next_state
                if done:
                    break
            total_rewards = step+1
            if total_rewards > best_score:
                best_weights = model.actor.get_weights()
                best_score = step
            # debugging
            if show_info:
                print("\repisode: {}, episode rewards: {:.01f}".format(
                    episode, total_rewards), end="")
            rewards.append(total_rewards)
        # return best weights
        model.actor.set_weights(best_weights)

        return rewards

    def _play_one_step(self, state, model):
        # record actor's policy parameters
        with tf.GradientTape() as actor_tape:
            action_logits = model.actor(state)
            action = model(state) # e-greedy action
            next_state, reward, done, info = env.step(int(action)) # output stuff
            # get entropy loss
            entropy_loss = self.loss_functions[0](action_logits, action_logits)
            # get policy loss
            policy_loss = self.loss_functions[1](action, action_logits)
            # combined loss
            actor_loss = policy_loss  - self.gamma[0] * entropy_loss
            # return the gradients per variables as a list
            gradients = actor_tape.gradient(actor_loss, model.actor.trainable_variables)
            experience = (state, action, reward, next_state, done, gradients)

        return reward, done, next_state, experience

    def _train_step(self, experience, model, step):
        # sample "batch_size" of experiences from replay memory
        state, action, reward, next_state, done, gradients = experience
        # compute values for next state
        next_state_value = model.critic.predict(next_state[np.newaxis])

        # improve critic's Advantage gradients
        with tf.GradientTape() as critic_tape:
            state_value = model.critic(state)
            # A(s, a) = r + discount_rate * V(s') - V(s)
            advantage = reward + \
                     (1 - done) * \
                     self.discount_rate * \
                     tf.squeeze(next_state_value) - \
                     tf.squeeze(state_value)
            # advantage is also TD error
            critic_loss = tf.square(advantage) * self.gamma[1]
            critic_grads = critic_tape.gradient(critic_loss, model.critic.trainable_variables)
        self.optimizers[1].apply_gradients(zip(critic_grads, model.critic.trainable_variables))
         # delay update: 2 value updates per 1 policy update
        if step % 2 == 0:
            # Improve actor's Policy
            weighted_gradients = [advantage * grad for grad in gradients]
            self.optimizers[0].apply_gradients(zip(weighted_gradients, model.actor.trainable_variables))

    def plot(self, rewards, window=0):
        # smoothing window
        if window !=0:
            weights = np.repeat(1.0, window)/window
            rewards = np.convolve(rewards, weights, 'valid')
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel("Episode", fontsize=16)
        plt.ylabel("Total Rewards", fontsize=16)
        plt.show()


if __name__ == '__main__':


    # use 42 to reproduce result
    np.random.seed(42)
    tf.random.set_seed(42)


    env = gym.make("CartPole-v1")

    model_args={
        "state_size": 4,
        "action_size": 2,
        "layer_size": 32,
        }

    session_args={
        'env':env,
        'discount_rate': 0.95,
        'n_episdoes': 200,
        'n_max_steps': 200,
        'gamma':[0.0001, 0.5],
        'loss_functions':[tf.keras.losses.categorical_crossentropy,  # for entropy loss
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)], # for policy loss

        'optimizers':[tf.keras.optimizers.Adam(lr=1e-3),
            tf.keras.optimizers.Adam(lr=1e-3)],
        }


    ac=ActorCritic(**model_args)
    sess=Session(**session_args)

    rewards=sess.train(ac)
    sess.plot_rewards(rewards)
