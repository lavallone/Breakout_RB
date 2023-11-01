import gym
import numpy as np
from flloat.semantics import PLInterpretation
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace


class BreakoutWrapper(gym.ObservationWrapper):

    def __init__(self, config: BreakoutConfiguration, action_type: str):
        super().__init__(BreakoutDictSpace(config))
        self._previous_brick_matrix = None
        self._next_brick_matrix = None
        self.action_type = action_type

    def observation(self, observation):
        new_observation = observation
        new_observation["previous_bricks_matrix"] = self._previous_brick_matrix
        self._previous_brick_matrix = np.copy(self._next_brick_matrix)
        self._next_brick_matrix = new_observation["bricks_matrix"]
        return new_observation

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._previous_brick_matrix = np.copy(obs["bricks_matrix"])
        self._next_brick_matrix = self._previous_brick_matrix
        return obs


def make_goal(nb_columns: int = 3, direction: str = "sx2dx") -> str:
    """
    Define the goal expressed in LDLf logic.

    E.g. for nb_columns = 3:

        <(!c0 & !c1 & !c2)*;c0;(!c0 & !c1 & !c2)*;c1;(!c0 & !c1 & !c2)*;c2>tt

    :param nb_columns: the number of column
    :return: the string associated with the goal.
    """
    labels = ["c" + str(column_id) for column_id in range(nb_columns)]
    if direction == "dx2sx":
        labels.reverse()
    empty = "(!" + " & !".join(labels) + ")"
    f = "<" + empty + "*;{}>tt"
    regexp = (";" + empty + "*;").join(labels)
    f = f.format(regexp)
    return f

# function to extract the fluents from the actual world environment
def extract_breakout_fluents(obs, action) -> PLInterpretation:
    brick_matrix = obs["bricks_matrix"]  # type: np.ndarray
    previous_brick_matrix = obs["previous_bricks_matrix"]  # type: np.ndarray
    # here we analyze which columns are broken in the current and in the previous bricks matrix
    previous_broken_columns = np.all(previous_brick_matrix == 0.0, axis=1)
    current_broken_columns = np.all(brick_matrix == 0.0, axis=1)
    compare = (previous_broken_columns == current_broken_columns)  # type: np.ndarray
    # if nothing changed, we return {}
    if compare.all():
        result = PLInterpretation(set())
        return result
    # if a column 'i' has been broken, we return the fluent 'c_i'
    else:
        index = np.argmin(compare)
        fluent = "c" + str(index)
        result = PLInterpretation({fluent})
        return result
