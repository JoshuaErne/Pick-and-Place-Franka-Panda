# Pick and Place Challenge

## Our Project
<div style="text-align: justify">
Developed a stable and robust system for both static and dynamic block acquisition. Our code was extensivelytested in simulation and sufficiently tested on the physical robot and performs well in both environments. Different strategies and scenarios were tested in both simulation and on the physical robot, and we implemented a robust solution to precisely stack stationery and dynamic blocks on the target platform in an efficient and safe manner by implementing Forward and Inverse kinematics, obstacle avoidance and path planning concepts. Our code can also stack tag 5 blocks with the white side, but we chose to skip this step in the competition because the flipping process took more time than it would to stack two dynamic blocks.
</div>

## Usage:
1) Install dependencies (required only for running this in ROS)-
Packages- First make sure you have [panda_simulator](https://github.com/justagist/panda_simulator/tree/noetic-devel); then clone this repository. Rename the file rename_to_meam520_labs to meam520_labs
2) Run the required ROS nodes with the simulation open using the final project
3) View our competition at [Link](https://www.youtube.com/watch?v=U018lOmAtOI)

<!-- ![rrt_algo](imgs/rrt_algo.png) -->

<img src=Images/pickplace.gif height="489" width="567" > <p></p>
Live Competition Blue(Our Team) vs Red
