import gym

def main():
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()

if __name__=="__main__":
    main()
