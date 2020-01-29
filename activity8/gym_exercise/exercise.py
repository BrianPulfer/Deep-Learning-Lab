import gym


def main():
    environment = gym.make('CartPole-v0')
    environment.reset()

    for i in range(1000):
        environment.render()
        environment.step(environment.action_space.sample())     # Random action

    environment.close()


if __name__ == '__main__':
    main()
