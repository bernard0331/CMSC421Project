from GeometryDashEnvironment import GDashEnv

game = GDashEnv(60,100,240,160)
run = 0
iterations = 10 # Number of iterations you want to run

for i in range(iterations):
    game.reset()
    terminated = False
    while not terminated:
        action = game.action_space.sample()
        observation, reward, terminated = game.step(action)
        game.render()                         
    run += 1   
    print("Run #", run, " Reward: ", reward)