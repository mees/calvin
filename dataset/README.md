# Dataset
The CALVIN dataset comes with 6 hours of teleoperated play data in each of the 4 environments.

## Download

We provide a download script to download the three different splits:

**1.** [Split A->A](http://calvin.cs.uni-freiburg.de/dataset/task_A_A.zip) (166 GB):
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh A
```
**2.** [Split BCD->A](http://calvin.cs.uni-freiburg.de/dataset/task_BCD_A.zip) (517 GB)
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh BCD
```
**3.** [Split ABCD->A](http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_A.zip) (656 GB)
```bash
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh ABCD
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
The language annotations are in a subdirectory of the train and validation folders called `lang_annotations`.
The file `auto_lang_ann.npy` contains the language annotations and its embeddings besides of additional metadata such as the task id, the sequence indexes.
```
['language']['ann']: list of raw language
['language']['task']: list of task_id
['language']['emb']: precomputed miniLM language embedding
['info']['indx']: list of start and end indices corresponding to the precomputed language embeddings
```
The `embeddings.npy` file is only present on the validation folder, this file contains the embeddings used only during the Rollouts (test inference) to condition the policy.

## Visualize Language Annotations
We provide a script to generate a video that visualizes the language annotations of the recorded play data.
By default we visualize the first 100 sequences, but feel free to more sequences (just change this [line](https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/utils/visualize_annotations.py#L57)).
A example video is.
```
cd $CALVIN_ROOT/calvin_models/calvin_agent
python utils/visualize_annotations.py datamodule.root_data_dir=$CALVIN_ROOT/dataset/task_A_A/
```
