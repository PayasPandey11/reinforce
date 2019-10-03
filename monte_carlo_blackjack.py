import gym
import numpy as np

def run_game(env,policy,display):

    state = env.reset()
    game_end = False
    episode = list()
    while not game_end:
        action = policy[state]
        if display:
            print("state-", state, "action-", action)
        else:
            pass
        next_state,reward,game_end,_ = env.step(action)
        #print("next state- ",next_state,"reward -> ",reward,"is end - >" ,game_end, _)
        episode.append([state,action,reward])
        state = next_state
    return episode

def train(env,num_episodes):

    #print(env.action_space.n, env.observation_space)

    Q = {}
    agentSumSpace = [i for i in range(4, 22)]
    dealerShowCardSpace = [i + 1 for i in range(10)]
    agentAceSpace = [False, True]
    actionSpace = [0, 1]  # stick or hit

    stateSpace = []
    returns = {}
    pairsVisited = {}
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    returns[((total, card, ace), action)] = 0
                    pairsVisited[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))

    policy = {}
    for state in stateSpace:
        policy[state] = np.random.choice(actionSpace)
    print(policy)
    for i in range(num_episodes):
        episode = run_game(env,policy,False)

        print("New episode ->", episode, "\n")
        for e_idx,event in enumerate(episode):
            s_t, a_t, r_t = episode[e_idx]

            pairsVisited[(s_t, a_t)] += 1
            returns[(s_t, a_t)] += r_t
            try:
                Q[(s_t, a_t)] = returns[(s_t, a_t)] / pairsVisited[(s_t, a_t)]
            except ZeroDivisionError:
                pass

            for j in range(1):
                best_reward=0
                if Q[s_t,j]> best_reward:
                    best_action = j
            policy[s_t] = j
    #print("\n\n---ppppppp---", Q, "\n\n\n policy", policy)
    return Q,policy



def test_policy(policy, env):
    wins = 0
    r = 1000
    for i in range(r):
        episode = run_game(env, policy,False)
        if i%100==0:

            print(i,episode)
        win = episode[-1][-1]
        if win == 1:
            wins += 1
    return wins / r



env = gym.make("Blackjack-v0")
Q,policy = train(env,num_episodes=1)
win_percentage = test_policy(policy,env)
print(win_percentage)

win_percentage = test_policy(policy,env)
print(win_percentage)

# # obtain the corresponding state-value function
# V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
#          for k, v in Q.items())
#
# # plot the state-value function
# plot_blackjack_values(V_to_plot)