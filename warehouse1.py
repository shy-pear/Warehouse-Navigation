import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Value iteration and optimal policy for robot that starts at 
random location in warehouse, picks up two items, and drops them off.
"""

"""
Warehouse environment - defines warehouse floor grid, start and end
 positions, pickup and dropoff positions, obstacles, actions the robot 
can take, and rewards/costs of taking actions.
"""
class WarehouseMultipleItems:
    def __init__(self, rows=10, cols=10, start=(0, 3), pickups=[(2, 3), (4, 6)], dropoff=(5, 9), obstacles=None):
        if obstacles is None:
            obstacles = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 6), (2, 7), (3, 7), (4, 4), (4, 5)]
        self.rows = rows
        self.cols = cols
        self.start = start
        self.pickups = pickups
        self.dropoff = dropoff
        self.obstacles = obstacles
        self.actions = ['U', 'D', 'L', 'R']
        self.state_rewards = self._create_rewards()

    """
    Defines reward grid. Rewards for reaching pickup points are
    based on whether either pickup point has already been reached. For
    all reached pickup points, there is a penalty. There is a also a
    big penalty to moving towards the dropoff point before all items
    have been picked up.

    state is now defined as a 4D array, with 2 binary elements to represent
    whether each item has been picked up for not.
    """
    def _create_rewards(self):
        
        rewards = np.full((self.rows, self.cols, 2, 2), -1)  # Default reward is -1 for each step
        for (i, j) in self.obstacles:
            rewards[i, j, :, :] = -10  # Penalty for hitting obstacles

        for idx, (pi, pj) in enumerate(self.pickups):
          rewards[pi, pj, 0, 0] = 10
          if idx == 0:
              rewards[pi, pj, 1, 0] = -10

          if idx == 1:
              rewards[pi, pj, 0, 1] = -10
        
        rewards[self.dropoff[0], self.dropoff[1], 0, 0] = -20
        rewards[self.dropoff[0], self.dropoff[1], 1, 0] = -20
        rewards[self.dropoff[0], self.dropoff[1], 0, 1] = -20
        rewards[self.dropoff[0], self.dropoff[1], 1, 1] = 20

        return rewards
    
    # returns position
    def get_position(self, state):
        return (state[0], state[1])

    """
    Calculates next state based on action, and if action
    cannot be taken, robot stays in same spot
    """
    def get_next_state(self, state, action):
        i, j, item_1, item_2 = state
        if action == 'U' and i > 0:
            i -= 1
        elif action == 'D' and i < self.rows - 1:
            i+= 1
        elif action == 'L' and j > 0:
            j -= 1
        elif action == 'R' and j < self.cols - 1:
            j += 1
        
        #Checks if robot is current on pickup location, and
        #if true, carrying to true since robot is now carrying item

        if (i, j) == self.pickups[0] and item_1 == 0:
            item_1 = 1
        if (i, j) == self.pickups[1] and item_2 == 0:
            item_2 = 1
        
        return (i, j, item_1, item_2)
    
"""
Value iteration and policy optimization for the warehouse robot 
navigation.
Arguments:
    env: Warehouse environment object
    gamma: Discount factor for future rewards
    theta: Convergence threshold
    max_iterations: Maximum number of iterations to prevent infinite loops
Returns:
    V: Value function for each state
    policy: Optimal policy for each state
"""
def value_iteration(env, gamma=0.9, theta=0.0001, max_iterations=1000):
    # Initialize the value function to zeros
    V = np.zeros((env.rows, env.cols, 2, 2))
    policy = {}

    for iteration in range(max_iterations):
        delta = 0.0  # Track the change in value function for convergence
        for i in range(env.rows):
            for j in range(env.cols):
                for item_1 in range(2):
                    for item_2 in range(2):
                      state = (i, j, item_1, item_2)

                      if (i, j) in env.obstacles or (i, j) == env.dropoff:
                        policy[state] = None
                        continue  # Skip obstacle and terminal states

                      # get the best action
                      max_value = float('-inf')
                      max_action = None
                      for action in env.actions:
                          next_state = env.get_next_state(state, action)
                          reward = env.state_rewards[next_state] #reward for next action
                          value = reward + gamma * V[next_state] #get value of the action
                          # value greater than max value, replace action
                          if value > max_value:
                              max_value = value # max value
                              max_action = action # max action


                      # Update the value function and policy for the current state
                      delta = max(delta, np.abs(V[state] - max_value))
                      V[state] = max_value
                      policy[state] = max_action

        # Check for convergence
        if delta < theta:
          print("Value function converged!")
          break

        # Print the value function for each iteration to monitor convergence
        print(f"Iteration: {iteration+1}, Max Change (Delta): {delta}")
        print(V)

    return V, policy

"""
Plots the optimal policy on the warehouse grid using matplotlib pyplot
based on whether one item, no items, or both items have been picked up.
Arguments:
    env: warehouse environment object
    policy: optimal policy
    item_state: tuple that shows whether item 1 and item 2 have been picked up
    or not
    title: title of diagram
"""
def plot_policy(env, policy, item_state=(0,0), title="Optimal Policy and Value Function"):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(title)

    # Create a color grid for visualizing the layout
    color_grid = np.zeros((env.rows, env.cols))
    for (i, j) in env.obstacles:
        color_grid[i][j] = -1  # Mark obstacles as -1

    # Plotting the grid world with state colors
    plt.imshow(color_grid, cmap='gray_r')

    # Mark the start, pickup, and drop-off locations
    plt.text(env.start[1], env.start[0], 'S', ha='center', va='center', color='red', fontsize=20, fontweight='bold')
    for idx, (pi, pj) in enumerate(env.pickups):
        plt.text(pj, pi, f'P{idx+1}', ha='center', va='center', color='green', fontsize=20, fontweight='bold')
    plt.text(env.dropoff[1], env.dropoff[0], 'D', ha='center', va='center', color='blue', fontsize=20, fontweight='bold')

    for i in range(env.rows):
        for j in range(env.cols):
            # Only plot arrows for the specified state configuration
            state = (i, j, item_state[0], item_state[1])
            action = policy.get(state)

            if action == 'U':
                plt.arrow(j, i, 0, -0.4, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
            elif action == 'D':
                plt.arrow(j, i, 0, 0.4, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
            elif action == 'L':
                plt.arrow(j, i, -0.4, 0, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
            elif action == 'R':
                plt.arrow(j, i, 0.4, 0, head_width=0.2, head_length=0.2, fc='orange', ec='orange')

    plt.grid()
    plt.xticks(np.arange(-0.5, env.cols, 1), [])
    plt.yticks(np.arange(-0.5, env.rows, 1), [])
    plt.show()

"""
Animate the robot's path from start to goal using the given policy using
matplotlib animation.
Args:
    env: Warehouse environment object.
    policy: The optimal policy for each state.
    start: Starting position (tuple).
    goal: Goal position (tuple).
"""
def animate_robot_path(env, policy, start, goal):
    # Initialize the starting position
    state = start

    path = [state]
    visited = set()  # Track visited states to detect loops

    # Follow the policy until reaching the goal or until a loop is detected
    while state != goal:
        # Prevent infinite loop by detecting repeated states
        if state in visited:
            print(f"Detected a loop at state: {state}. Exiting to prevent infinite loop.")
            break
        visited.add(state)

        action = policy.get(state)
        if action is None:
            print(f"No valid action for state {state}. Breaking the loop.")
            break

        # Move to the next state based on the action
        next_state = env.get_next_state(state, action)
        if next_state == state:
            print(f"Invalid move from state {state} using action {action}.")
            break

        # Debugging: Print each step
        print(f"Current state: {state}, Action: {action}, Next state: {next_state}")

        state = next_state
        path.append(state)

    # Visualize the path
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Warehouse Robot Navigation Path")

    # Create a color grid for visualizing the layout
    color_grid = np.zeros((env.rows, env.cols))
    for (i, j) in env.obstacles:
        color_grid[i][j] = -1  # Mark obstacles as -1

    # Plotting the grid world with state colors
    plt.imshow(color_grid, cmap='gray_r')

    # Plot start, pickup, and drop-off points
    plt.text(env.start[1], env.start[0], 'S', ha='center', va='center', color='red', fontsize=20, fontweight='bold')
    for idx, (pi, pj) in enumerate(env.pickups):
        plt.text(pj, pi, f'P{idx+1}', ha='center', va='center', color='green', fontsize=20, fontweight='bold')
    plt.text(env.dropoff[1], env.dropoff[0], 'D', ha='center', va='center', color='blue', fontsize=20, fontweight='bold')

    # Plot the path
    for step, (i, j, item_1, item_2) in enumerate(path):
        plt.text(j, i, f"{step}\n{item_1},{item_2}", ha='center', va='center', color='yellow', fontsize=12, fontweight='bold')

    # Animate the path
    robot_path = [plt.Circle((j, i), 0.3, color='orange') for i, j, _, _ in path]

    def animate(frame):
        """Update the robot's position on each frame."""
        if frame < len(robot_path):
            for patch in robot_path[:frame + 1]:
                ax.add_patch(patch)
        return robot_path

    ani = animation.FuncAnimation(fig, animate, frames=len(robot_path), interval=500, repeat=False)
    plt.grid()
    plt.xticks(np.arange(-0.5, env.cols, 1), [])
    plt.yticks(np.arange(-0.5, env.rows, 1), [])
    plt.show()

"""
Code to run algorithm and display visuals
To make policy clear, item states are defined as
(0, 0) - no items have been picked up
(1, 0) - first item has been picked up
(1, 1) - both items have been picked up
"""
# Create the warehouse environment
env = WarehouseMultipleItems()

# Run Value Iteration to get the optimal policy
optimal_values, optimal_policy = value_iteration(env)

# Plot the optimal policy using matplotlib
plot_policy(env, optimal_policy, item_state=(0, 0), title="Policy with No Items Collected")

# Optionally, plot the policy for other configurations (e.g., after picking up item 1 but not item 2)
plot_policy(env, optimal_policy, item_state=(1, 0), title="Policy with Item 1 Collected")

# Plot the policy for the state where both items have been picked up
plot_policy(env, optimal_policy, item_state=(1, 1), title="Policy with Both Items Collected")

#simulate_robot_movement(env, optimal_policy, env.start, env.dropoff)

animate_robot_path(env, optimal_policy, (env.start[0], env.start[1], 0, 0), (env.dropoff[0], env.dropoff[1], 1, 1))