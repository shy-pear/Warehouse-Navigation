# Warehouse Navigation
## Project Summary
Value iteration and policy iteration are tested as potential solutions to warehouse navigation issues. An agent at a start location has to navigate to pickup locations and finally to the dropoff location. Two scenarios are tested:
- One pickup location and one dropoff location
- Two pickup locations and one dropoff locations
### Results:
#### One pickup location:
Optimal policy plot:

<img width="545" alt="img1" src="https://github.com/user-attachments/assets/1acf15d3-2245-4208-aa00-e2a553372246">

Animation of robot navigating warehouse:

<img width="582" alt="img5" src="https://github.com/user-attachments/assets/51e218be-c740-4174-89d9-9aff5e39e9b6">


#### Two pickup locations:
Optimal policy plots:
- No items picked up

<img width="543" alt="img2" src="https://github.com/user-attachments/assets/443493d2-231e-410c-bfe9-158baaa5496b">

- First item picked up

<img width="545" alt="img3" src="https://github.com/user-attachments/assets/e4e809ee-5bd6-4b57-b5eb-d379c7a431be">

- Both items picked up

<img width="545" alt="img4" src="https://github.com/user-attachments/assets/3d69ab5c-196f-4405-ac46-627fe76f47e2">

Animation of robot navigating warehouse:

<img width="545" alt="img6" src="https://github.com/user-attachments/assets/9084e0f8-9b05-436e-bd3a-f71e929c8bde">

## Motivation and Goals
Warehouse navigation is a significant issue that impacts efficiency and costs for companies. The longer that robots take during picking up and storing processes, the longer it takes to fulfill orders and restock. Also, since more time is spent per task, labor costs and operations expenses increase. The challenge increases as inventory volumes increase or warehouse layouts become more complex. Therefore, my goal was to experiment with value iteration to help an agent find the optimal path as it picks up and drops off items.

## Tools
- Python
- Matplotlib

## Limitations and Next Steps
The layout is limited. Warehouses are much more complex than the simple layout that was used in this project, so this project can be expanded by adding more obstacles and dynamic obstacles, such as other moving machines or humans. There might also be more tasks than picking up or dropping off items, which can also be incorporated into a more complex model.
