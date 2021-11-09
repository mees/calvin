# Dataset
The CALVIN dataset comes with 6 hours of teleoperated play data in each of the 4 environments.

## Download

We provide a download script to download the three different splits:

**1.** [Split A->A](http://calvin.cs.uni-freiburg.de/dataset/task_A_A.zip) (166GB):
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh A
```
**2.** [Split BCD->A](http://calvin.cs.uni-freiburg.de/dataset/task_BCD_A.zip) (?GB)
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh B
```
**3.** [Split ABCD->A](http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_A.zip) (?GB)
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh full
```
## Data Structure
Each interaction timestep is stored in a dictionary inside a numpy file and contains all corresponding sensory observations, different action spaces, state information and language annoations.
The keys to access the different camera observations are:
```
['rgb_static'], ['rgb_gripper'], ['rgb_tactile'], ['depth_static'], ['depth_gripper'], ['depth_tactile']
```
The keys to access the 7-DOF absolute and relative actions are:
```
['actions'], ['rel_actions']
```
The keys to access the scene state information containing the position and orientation of all objects in the scenes
(we do not use them to better capture challenges present in real-world settings):
```
['scene_obs']
```
The robot proprioceptive information, which also includes joint positions can be accessed with:
```
['robot_obs']
```
The key to access the MiniLM precomputed language embeddings:
```
['language']
```
