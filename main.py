import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

input_size= 4
hidden_nodes= 150
output_size = 2

model= torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_nodes),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden_nodes,output_size),
    torch.nn.Softmax()
    )

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

def discounted_reward_function(rewards,gamma=0.99):
    length=len(rewards)
    discounted = torch.pow(gamma,torch.arange(length).float())
    discounted = discounted * rewards
    return discounted / discounted.max()

def loss_fun(rewards , prediction ):
    return -1 * torch.sum(rewards * torch.log(prediction))


env= gym.make("CartPole-v0")

MAX_EPISODES = 1000
MAX_DURATION = 500 
gamma = 0.99
scores = []
games = []
for i in range(MAX_EPISODES):

    transition=[]
    current_state = env.reset()
    done = False
    j=0
    env.render()

    while j<MAX_DURATION:
        probability = model(torch.from_numpy(current_state).float())
        action = np.random.choice(np.array([0,1]), p= probability.data.numpy())
        previous_state= current_state
        current_state , _ , done , info = env.step(action)
        transition.append((previous_state,j+1,action))
        j+=1
        if done:
            break
    episode_length = len(transition)
    scores.append(episode_length)
    games.append(i)
    state = torch.Tensor([s for s,r,a in transition])
    action = torch.Tensor([a for s,r,a in transition])
    rewards = torch.Tensor([r for s,r,a in transition]).flip(dims=(0,))
    discounted_reward = discounted_reward_function(rewards)
    prediction = model(state)
    prediction = prediction.gather(dim = 1 , index = action.long().view(-1,1)).squeeze()
    loss = loss_fun(discounted_reward,prediction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(games,scores)
plt.show()