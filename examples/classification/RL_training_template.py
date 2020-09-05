import torch
NUM_EPISODES = 400


class AgentOptimizer():
    def __init__(self, **kwargs):
        pass

    def reset(self):
        """
        Reset cureent episode info
        """
        pass

    def _train_agent_step(self):
        """

        :return:
        """
        pass

    def _predict_action(self):
        """
        Predict action for the last state
        :return:
        """
        pass

    def ask(self):
        """

        :return:
        """
        action = self._predict_action()
        return action

    def tell(self, state, reward, info):
        """
        Getting info about episode step
        :return:
        """
        # save state, reward and info
        self._train_agent_step()
        pass

class Environment():
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        """

        :param action:
        :return: new_state, reward, done, info
        """
        pass


def RL_agent_train(Environment, Optimizer, env_params, agent_params):
    env = Environment(env_params)

    agent = Optimizer(agent_params)

    rewards = []

    for episode in range(NUM_EPISODES):
        state, reward, done, info = env.reset()
        episode_reward = 0

        while not done:
            action = agent.ask(state)
            new_state, reward, done, info = env.step(action)
            agent.tell(state, reward, info)

            state = new_state
            episode_reward += reward

        rewards.append(episode_reward)
