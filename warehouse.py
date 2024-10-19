import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Value iteration and optimal policy for robot that starts at 
random location in warehouse, picks up one item, and drops it off.
"""

"""
Warehouse environment - defines warehouse floor grid, start and end
 positions, pickup and dropoff positions, obstacles, actions the robot 
can take, and rewards/costs of taking actions.
"""
class Warehouse:
    def __init__(self, rows=10, cols=10, start=(0, 3), pickup=(2, 3), dropoff=(5, 9), obstacles=None):
        if obstacles is None:
            obstacles = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 6), (2, 7), (3, 7), (4, 4), (4, 5)]
        self.rows = rows
        self.cols = cols
        self.start = start
        self.pickup = pickup
        self.dropoff = dropoff
        self.obstacles = obstacles
        self.actions = ['U', 'D', 'L', 'R']
        self.state_rewards = self._create_rewards()
    """
    Defines reward grid. There is small reward for reaching pickup point,
    however if dropoff point is reached before item is picked up, there 
    is a penalty. There is also a penalty to returning to pickup point
    after the item is picked up, so the robot can direct its attention
    to reaching the dropoff point.

    State is defined as 3D array, with a binary element to represent
    whether item has been picked up or not.
    """
    def _create_rewards(self):
        # Create a reward grid for the warehouse environment
        rewards = np.full((self.rows, self.cols, 2), -1)  # Default reward is -1 for each step
        for (i, j) in self.obstacles:
            rewards[i, j, :] = -10  # Penalty for hitting obstacles

        rewards[self.pickup[0], self.pickup[1], 0] = 5   # Small reward for reaching pickup point
        rewards[self.pickup[0], self.pickup[1], 1] = -10

        rewards[self.dropoff[0], self.dropoff[1], 0] = -10 # I wonder I can remove this
        rewards[self.dropoff[0], self.dropoff[1], 1] = 20

        return rewards
    
    # returns position
    def get_position(self, state):
        return (state[0], state[1])

    """
    Calculates next state based on action, and if action
    cannot be taken, robot stays in same spot
    """
    def get_next_state(self, state, action):
        i, j, carrying = state
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
        
        if (i, j) == self.pickup and carrying == 0:
            carrying = 1
        
        return (i, j, carrying)
    
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
    V = np.zeros((env.rows, env.cols, 2))
    policy = {}

    for iteration in range(max_iterations):
        delta = 0.0  # Track the change in value function for convergence
        for i in range(env.rows):
            for j in range(env.cols):
                for carrying_item in range(2):
                    state = (i, j, carrying_item)


                    if (i, j) in env.obstacles or (i, j) == env.dropoff:
                      policy[state] = None #policy of state is non existent
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
Plot the optimal policy on the warehouse grid using matplotlib pyplot.
Arguments:
    env: warehouse environment object
    policy: optimal policy
    title: title of diagram
"""
def plot_policy(env, policy, title="Optimal Policy and Value Function"):
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
    plt.text(env.pickup[1], env.pickup[0], 'P', ha='center', va='center', color='green', fontsize=20, fontweight='bold')
    plt.text(env.dropoff[1], env.dropoff[0], 'D', ha='center', va='center', color='blue', fontsize=20, fontweight='bold')

    # Create arrow annotations for the policy
    for i in range(env.rows):
        for j in range(env.cols):
            for carrying_item in range(2):
                
              state = (i, j, carrying_item)
              action = policy.get(state)

              if action == 'U':
                  plt.arrow(j, i, 0, -0.2, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
              elif action == 'D':
                  plt.arrow(j, i, 0, 0.2, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
              elif action == 'L':
                  plt.arrow(j, i, -0.2, 0, head_width=0.2, head_length=0.2, fc='orange', ec='orange')
              elif action == 'R':
                  plt.arrow(j, i, 0.2, 0, head_width=0.2, head_length=0.2, fc='orange', ec='orange')

    plt.grid()
    plt.xticks(np.arange(-0.5, env.cols, 1), [])
    plt.yticks(np.arange(-0.5, env.rows, 1), [])
    plt.show()

"""
Animates the robot's path from start to goal for given policy using 
matplotlib animation.
Args:
    env: Warehouse environment object.
    policy: optimal policy
    start: starting position (tuple)
    goal: goal position (tuple)
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
    plt.text(env.pickup[1], env.pickup[0], 'P', ha='center', va='center', color='green', fontsize=20, fontweight='bold')
    plt.text(env.dropoff[1], env.dropoff[0], 'D', ha='center', va='center', color='blue', fontsize=20, fontweight='bold')

    # Plot the path
    for step, (i, j, _) in enumerate(path):
        plt.text(j, i, f"{step}", ha='center', va='center', color='yellow', fontsize=12, fontweight='bold')

    # Animate the path
    robot_path = [plt.Circle((j, i), 0.3, color='orange') for i, j, _ in path]

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
Code to run algorithms and display visuals
"""
# Create the warehouse environment
env = Warehouse()

# Run Value Iteration to get the optimal policy
optimal_values, optimal_policy = value_iteration(env)

# Plot the optimal policy using matplotlib
plot_policy(env, optimal_policy)

# Animate optimal policy
animate_robot_path(env, optimal_policy, (env.start[0], env.start[1], 0), (env.dropoff[0], env.dropoff[1], 1))