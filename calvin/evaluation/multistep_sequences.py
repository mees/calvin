from copy import deepcopy

import numpy as np


def check_condition(state, condition):
    for k, v in condition.items():
        if not state[k] == v:
            return False
    return True


def update_state(state, effect):
    next_state = deepcopy(state)
    for k, v in effect.items():
        next_state[k] = v
    return next_state


def valid_task(curr_state, task):
    next_states = []
    for _task in task:
        if check_condition(curr_state, _task["condition"]):
            next_state = update_state(curr_state, _task["effect"])
            next_states.append(next_state)
    return next_states


def get_sequences():
    state = {"led": 0,
             "lightbulb": 0,
             "slider": "right",
             "drawer": "closed",
             "red_block": "table",
             "blue_block": "table",
             "pink_block": "slider_right",
             "grasped": 0}

    task_categories = {"rotate_red_block_right": 0,
                        "rotate_red_block_left": 0,
                        "rotate_blue_block_right": 0,
                        "rotate_blue_block_left": 0,
                        "rotate_pink_block_right": 0,
                        "rotate_pink_block_left": 0,
                        "push_red_block_right": 1,
                        "push_red_block_left": 1,
                        "push_blue_block_right": 1,
                        "push_blue_block_left": 1,
                        "push_pink_block_right": 1,
                        "push_pink_block_left": 1,
                        "move_slider_left": 2,
                        "move_slider_right": 2,
                        "open_drawer": 3,
                        "close_drawer": 3,
                        "lift_red_block_table": 4,
                        "lift_red_block_slider": 4,
                        "lift_red_block_drawer": 4,
                        "lift_blue_block_table": 4,
                        "lift_blue_block_slider": 4,
                        "lift_blue_block_drawer": 4,
                        "lift_pink_block_table": 4,
                        "lift_pink_block_slider": 4,
                        "lift_pink_block_drawer": 4,
                        "place_in_slider": 5,
                        "place_in_drawer": 5,
                        "turn_on_lightbulb": 6,
                        "turn_off_lightbulb": 6,
                        "turn_on_led": 6,
                        "turn_off_led": 6,
                        "push_in_drawer": 1,
                        "stack_block": 9,
                        "collapse_stacked_blocks": 10}

    tasks = {
        "rotate_red_block_right": [
        {"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
             "rotate_red_block_left": [
                 {"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
             "rotate_blue_block_right": [
                 {"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
             "rotate_blue_block_left": [
                 {"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
             "rotate_pink_block_right": [
                 {"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
             "rotate_pink_block_left": [
                 {"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],

             "push_red_block_right": [
                 {"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
             "push_red_block_left": [
                 {"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
             "push_blue_block_right": [
                 {"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
             "push_blue_block_left": [
                 {"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
             "push_pink_block_right": [
                 {"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
             "push_pink_block_left": [
                 {"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],

             "move_slider_left": [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
             "move_slider_right": [{"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}],

             "open_drawer": [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
             "close_drawer": [{"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}],

             "lift_red_block_table": [
                 {"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
             "lift_red_block_slider": [{"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0},
                                        "effect": {"red_block": "grasped", "grasped": 1}},
                                       {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0},
                                        "effect": {"red_block": "grasped", "grasped": 1}}],
             "lift_red_block_drawer": [{"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0},
                                        "effect": {"red_block": "grasped", "grasped": 1}}],

             "lift_blue_block_table": [
                 {"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
             "lift_blue_block_slider": [{"condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0},
                                         "effect": {"blue_block": "grasped", "grasped": 1}},
                                        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0},
                                         "effect": {"blue_block": "grasped", "grasped": 1}}],
             "lift_blue_block_drawer": [{"condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0},
                                         "effect": {"blue_block": "grasped", "grasped": 1}}],

             "lift_pink_block_table": [
                 {"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
             "lift_pink_block_slider": [{"condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0},
                                         "effect": {"pink_block": "grasped", "grasped": 1}},
                                        {"condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0},
                                         "effect": {"pink_block": "grasped", "grasped": 1}}],
             "lift_pink_block_drawer": [{"condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0},
                                         "effect": {"pink_block": "grasped", "grasped": 1}}],

             "place_in_slider": [{"condition": {"red_block": "grasped", "slider": "right", "grasped": 1},
                                  "effect": {"red_block": "slider_right", "grasped": 0}},
                                 {"condition": {"red_block": "grasped", "slider": "left", "grasped": 1},
                                  "effect": {"red_block": "slider_left", "grasped": 0}},
                                 {"condition": {"blue_block": "grasped", "slider": "right", "grasped": 1},
                                  "effect": {"blue_block": "slider_right", "grasped": 0}},
                                 {"condition": {"blue_block": "grasped", "slider": "left", "grasped": 1},
                                  "effect": {"blue_block": "slider_left", "grasped": 0}},
                                 {"condition": {"pink_block": "grasped", "slider": "right", "grasped": 1},
                                  "effect": {"pink_block": "slider_right", "grasped": 0}},
                                 {"condition": {"pink_block": "grasped", "slider": "left", "grasped": 1},
                                  "effect": {"pink_block": "slider_left", "grasped": 0}}],
             "place_in_drawer": [{"condition": {"red_block": "grasped", "drawer": "open", "grasped": 1},
                                  "effect": {"red_block": "drawer", "grasped": 0}},
                                 {"condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1},
                                  "effect": {"blue_block": "drawer", "grasped": 0}},
                                 {"condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1},
                                  "effect": {"pink_block": "drawer", "grasped": 0}}],
             "stack_block": [{"condition": {"red_block": "grasped", "blue_block": "table", "grasped": 1},
                              "effect": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
                             {"condition": {"red_block": "grasped", "pink_block": "table", "grasped": 1},
                              "effect": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
                             {"condition": {"blue_block": "grasped", "red_block": "table", "grasped": 1},
                              "effect": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
                             {"condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1},
                              "effect": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
                             {"condition": {"pink_block": "grasped", "red_block": "table", "grasped": 1},
                              "effect": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
                             {"condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1},
                              "effect": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}}],

             "collapse_stacked_blocks": [{"condition": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"red_block": "table", "blue_block": "table"}},
                                        {"condition": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"red_block": "table", "pink_block": "table"}},
                                        {"condition": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"blue_block": "table", "red_block": "table"}},
                                        {"condition": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"blue_block": "table", "pink_block": "table"}},
                                        {"condition": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"pink_block": "table", "red_block": "table"}},
                                        {"condition": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0},
                                         "effect": {"pink_block": "table", "blue_block": "table"}}],
             "turn_on_lightbulb": [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
             "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
             "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
             "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
             "push_in_drawer": [{"condition": {"red_block": "table", "drawer": "open", "grasped": 0},
                                 "effect": {"red_block": "drawer", "grasped": 0}},
                                {"condition": {"blue_block": "table", "drawer": "open", "grasped": 0},
                                 "effect": {"blue_block": "drawer", "grasped": 0}},
                                {"condition": {"pink_block": "table", "drawer": "open", "grasped": 0},
                                 "effect": {"pink_block": "drawer", "grasped": 0}}]}
    seq_len = 5
    valid_seqs = [[] for x in range(seq_len)]

    for step in range(seq_len):
        for task_name, task in tasks.items():
            if step == 0:
                for next_state in valid_task(state, task):
                    valid_seqs[0].append([(task_name, next_state)])
            else:
                for seq in valid_seqs[step - 1]:
                    curr_state = seq[-1][1]
                    for next_state in valid_task(curr_state, task):
                        valid_seqs[step].append([*seq, (task_name, next_state)])

    results = []
    result_set = []
    np.random.seed(0)
    for seq in np.random.permutation(valid_seqs[-1]):
        _seq = list(zip(*seq))[0]
        categories = [task_categories[name] for name in _seq]
        if len(categories) == len(set(categories)) and set(_seq) not in result_set:
            results.append(_seq)
            result_set.append(set(_seq))

    return results


if __name__ == "__main__":
    results = get_sequences()
    for seq in results:
        print(seq)
    print(len(results))













