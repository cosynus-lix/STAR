# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
import numpy as np

def create_maze_env(env_name=None, seed=0):
  maze_id = None
  if env_name.endswith('Maze'):
    maze_id = 'Maze'
    if env_name.startswith('Point'):
      return PointMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed, manual_collision=True)
    
    elif env_name.startswith('Ant'):
      return AntMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed)
  
  elif env_name.endswith('MazeSparse'):
    maze_id = 'Maze2'
    if env_name.startswith('Point'):
      return PointMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed)
    elif env_name.startswith('Ant'):
      return AntMazeEnv(maze_id=maze_id, maze_size_scaling=2, seed=seed)
  
  elif env_name.endswith('Push'):
    maze_id = 'Push'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed)
  
  elif env_name.endswith('Fall'):
    maze_id = 'Fall'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed)
  
  elif env_name.endswith('MazeCam'):
    maze_id = 'MazeCam'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed)
  
  elif env_name.endswith('MazeStochastic'):
    maze_id = 'MazeStochastic'
    new_pos = [np.random.normal(), 2]
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=8, seed=seed, new_pos=new_pos)
  
  elif env_name.endswith('2Rooms'):
    maze_id = '2Rooms'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=4, seed=seed)
  
  elif env_name.endswith('3Rooms'):
    maze_id = '3Rooms'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=4, seed=seed)

  elif env_name.endswith('4Rooms'):
    maze_id = '4Rooms'
    return AntMazeEnv(maze_id=maze_id, maze_size_scaling=4, seed=seed)

  else:
    raise ValueError('Unknown maze environment %s' % env_name)
