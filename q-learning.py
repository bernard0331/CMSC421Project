def update_q_values(network, optimizer, states, actions, rewards, next_states, done_flags, gamma=0.99):
    """
    Update Q-values for a batch of transitions.

    :param network: The neural network model that predicts Q-values.
    :param optimizer: The optimizer for updating network weights.
    :param states: Batch of states.
    :param actions: Batch of actions taken.
    :param rewards: Batch of rewards received.
    :param next_states: Batch of next states.
    :param done_flags: Batch of done flags (True if next state is terminal).
    :param gamma: Discount factor.
    """
    # Predict Q-values for starting states
    current_q_values = network.predict(states)

    # Predict Q-values for next states
    next_q_values = network.predict(next_states)

    # Calculate the maximum Q-value for each next state
    max_next_q_values = np.max(next_q_values, axis=1)

    # If done flag is True, just use the reward; else use reward + gamma * max_next_q_value
    target_q_values = rewards + (gamma * max_next_q_values * (1 - done_flags))

    # Update the Q-values for the actions taken
    targets = current_q_values.copy()
    for i, action in enumerate(actions):
        targets[i, action] = target_q_values[i]

    # Perform a gradient descent step to update the network's weights
    optimizer.zero_grad()
    loss = network.loss(current_q_values, targets)
    loss.backward()
    optimizer.step()

    return loss.item()
