import torch
import numpy as np
import wandb
from model import DQN, ReplayMemory


def get_epsilon(it, p=0.05, when_to_end=1000):
    if it >= when_to_end:
        return p
    else:
        return 1 - it / (when_to_end * (1 + p))


def select_action(model, state, epsilon, device="cpu"):
    state = torch.tensor(state).to(dtype=torch.float).to(device)
    scores = model(state)
    action = int(np.random.rand() * 2) if np.random.rand() < epsilon else torch.argmax(scores).item()
    return action


def compute_q_val(model, state, action):
    scores = model(state)
    q_val = torch.gather(scores, 1, action.unsqueeze(1))
    return q_val


def compute_target(model, reward, next_state, done, discount_factor, device= 'cpu'):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    scores = model(next_state)
    next_q = torch.max(scores, 1)[0].to(device)
    #discount_factor = discount_factor.to(device)
    target = reward + (torch.ones(done.size(), device=device) - done) * discount_factor * next_q
    target = target.unsqueeze(1)
    return target


def train(model, target_net, memory, optimizer, batch_size, discount_factor, device="cpu", reward_clip=0):

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float, device=device)
    action = torch.tensor(action, dtype=torch.int64, device=device)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float, device=device)
    reward = torch.tensor(reward, dtype=torch.float, device=device)
    done = torch.tensor(done, dtype=torch.float, device=device)

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(target_net, reward, next_state, done, discount_factor, device=device)

    # loss is measured from error between current and newly expected Q values
    loss = torch.nn.functional.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    if reward_clip != 0:
        if reward_clip == 1:
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
        else:
            for param in model.parameters():
                param.grad.data = param.grad.data * reward_clip
    optimizer.step()


    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(model,target_net, memory, env, num_episodes, batch_size, discount_factor, optimizer, target_update=10,
                 replay_bool=True, reward_clip=None, device="cpu"):
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    losses = []
    # reward for each time step for each episode
    rewards_per_episode = []
    for i in range(num_episodes):
        rewards = [] # For each time step

        # initialise inside the episode counter
        episode_steps = 0
        # initialise epsilon
        epsilon = get_epsilon(global_steps)
        # start the environment
        s = env.reset()
        action = select_action(model, s, epsilon, device=device)
        while True:
            s_new, r, is_terminal, prob = env.step(action)

            # store reward for each time step in order to calculate cumulative reward
            rewards.append(r)

            # store the state
            memory.push((s, action, r, s_new, is_terminal))
            # train the Qnet
            loss = train(model, target_net, memory, optimizer, batch_size, discount_factor, device=device)
            losses.append(loss)
            # update counters
            episode_steps += 1
            global_steps += 1
            #update target model
            if target_update > 0:
                if global_steps % target_update == 0:
                    target_net.load_state_dict(model.state_dict())
            else:
                target_net.load_state_dict(model.state_dict())
            # take new action
            epsilon = get_epsilon(global_steps)
            action = select_action(model, s_new, epsilon, device=device)
            s = s_new
            #parameter norm
            total_norm = 0
            for p in model.parameters():
                try:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                except NameError:
                    total_norm = np.NaN

            total_norm = total_norm ** (1. / 2)

            wandb.log({
                "Rewards_per step": r,
                "Loss": loss,
                "Grad_norm": total_norm})
            if is_terminal:
                episode_durations.append(episode_steps)
                break


        rewards_per_episode.append(rewards)
        print(np.sum(rewards))
        wandb.log({"Rewards_per episode": np.sum(rewards)})
    return episode_durations, losses, rewards_per_episode
