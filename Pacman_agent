import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point, Queue


# import time

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.start_game_state = game_state
        self.closed_in_list = []
        self.middle_hover_counter = 0
        self.position_food_eaten = None
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)


        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 0}

    def min_distance_to_home(self, game_state, pos):
        """
        Computes the minimum distance to home
        """
        # Home first column
        if self.red:
            x = int(self.midWidth - 1)
        else:
            x = int(self.midWidth + 1)
        positions = []
        # Compute valid positions in first column
        for y in range(self.height):
            if not is_wall(game_state, (x, y)):
                positions.append((x, y))
        distance_to_home = 9999
        # Compute min distance to the valid positions
        for position in positions:
            distance = self.get_maze_distance(position, pos)
            if distance < distance_to_home:
                distance_to_home = distance
        return distance_to_home


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that attacks with different cases.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        twoWays = two_ways(self.start_game_state, self)

        current_capsule_list = self.get_capsules(game_state)
        current_food_list = self.get_food(game_state).as_list()
        next_food_list = self.get_food(successor).as_list()
        next_capsule_list = self.get_capsules(successor)

        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        next_state = successor.get_agent_state(self.index)
        next_pos = next_state.get_position()

        # All opponents for current game state
        current_enemies = [game_state.get_agent_state(agent) for agent in self.get_opponents(game_state)]

        # All ghosts for current game state that could eat the offensive agent
        current_ghosts = [enemy for enemy in current_enemies if
                          (not enemy.is_pacman) and (enemy.scared_timer == 0) and (enemy.get_position() is not None)]

        # Computes the distance to the closest ghost for current game state
        current_min_distance_to_ghost = 0
        if len(current_ghosts) > 0:
            current_min_distance_to_ghost = min(
                [self.get_maze_distance(current_pos, ghost.get_position()) for ghost in current_ghosts])

        # Computes all current pacman invaders in our half of the field for current game state
        current_invaders = [enemy for enemy in current_enemies if enemy.is_pacman and enemy.get_position() is not None]

        # Computes the closest distance to an invader in our half of the field for current game state
        current_min_distance_to_pacman = 0
        if len(current_invaders) > 0:
            current_min_distance_to_pacman = min(
                [self.get_maze_distance(current_pos, invader.get_position()) for invader in current_invaders])

        # Computes the amount of food pacman is currently carrying
        current_carrying_food = current_state.num_carrying

        # Computes whether the current position is closed in
        if current_pos in self.closed_in_list:
            current_not_closed_in = False
        else:
            current_not_closed_in = twoWays.breadthFirstSearch(current_pos)
            if not current_not_closed_in:
                self.closed_in_list.append(current_pos)

        # Computes the distance to home from current game state
        current_distance_to_home = self.min_distance_to_home(game_state, current_pos)

        # Computes whether the agent is currently on defense
        if not current_state.is_pacman:
            current_is_defense = True
        else:
            current_is_defense = False

        # Computes the minimal distance to the nearest capsule for the current state
        if len(current_capsule_list) > 0:
            current_min_distance_to_capsule = min(
                [self.get_maze_distance(current_pos, capsule) for capsule in current_capsule_list])

        # Computes the amount of food that is left for the successor
        features['successor_score'] = -len(next_food_list)

        # Computes whether the successor is on defense or offense
        if next_state.is_pacman:
            features['on_offense'] = 1
        else:
            features['on_defense'] = 1

        # Computes the distance to the nearest food for the successor state
        if len(next_food_list) > 0:
            next_min_distance_to_food = min([self.get_maze_distance(next_pos, next_food) for next_food in next_food_list])
            features['distance_to_food'] = next_min_distance_to_food

        # Computes the distance to the nearest capsule for the successor state
        if len(next_capsule_list) > 0:
            next_min_distance_to_capsule = min(
                [self.get_maze_distance(next_pos, capsule) for capsule in next_capsule_list])
            features['distance_to_capsule'] = next_min_distance_to_capsule

        # Compute the closest distance to ghost enemies for the successor
        next_enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]

        # Computes all ghosts for successor game state
        next_ghosts = [enemy for enemy in next_enemies if
                       (not enemy.is_pacman) and (enemy.scared_timer == 0) and (enemy.get_position() is not None)]

        # Computes the distance to the closest ghost for successor game state
        next_min_distance_to_ghost = 0
        if len(next_ghosts) > 0:
            next_min_distance_to_ghost = min([self.get_maze_distance(next_pos, next_ghost.get_position()) for next_ghost in next_ghosts])
            features['ghost_distance'] = next_min_distance_to_ghost

        # All invaders
        next_invaders = [next_enemy for next_enemy in next_enemies if next_enemy.is_pacman and next_enemy.get_position() is not None]

        # Compute the distance to the closest pacman enemy we can see for the successor
        next_min_distance_to_pacman = 0
        if len(next_invaders) > 0:
            next_min_distance_to_pacman = min(
                [self.get_maze_distance(next_pos, next_invader.get_position()) for next_invader in next_invaders])
            features['invader_distance'] = next_min_distance_to_pacman

        # Compute the amount of invaders
        features['num_invaders'] = len(next_invaders)

        # Computes whether the action is STOP
        if action == Directions.STOP:
            features['stop'] = 1

        # Computes whether the action is REVERSE
        rev = Directions.REVERSE[next_state.configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Computes all scared ghosts
        next_scared_ghosts = [next_enemy for next_enemy in next_enemies if
                              not next_enemy.is_pacman and not next_enemy.scared_timer == 0 and next_enemy.get_position() is not None]

        # Computes the distance to the closest scared ghost we can see for the successor
        if len(next_scared_ghosts) > 0:
            next_min_distance_to_scared_ghost = min(
                [self.get_maze_distance(next_pos, next_scared_ghost.get_position()) for next_scared_ghost in next_scared_ghosts])
            features['scared_ghost_distance'] = next_min_distance_to_scared_ghost

        # Computes the amount of food pacman is carrying for the successor state
        next_carrying_food = next_state.num_carrying
        features['carrying_food'] = next_carrying_food

        # Compute the amount of food returned for the successor state
        next_returned_food = next_state.num_returned
        features['returned_food'] = next_returned_food

        # Computes whether the successor eats a capsule
        if current_capsule_list > next_capsule_list:
            features['capsule_eaten'] = 1

        # Computes whether the successor is going to be closed in
        if next_pos in self.closed_in_list:
            next_not_closed_in = False
        else:
            next_not_closed_in = twoWays.breadthFirstSearch(next_pos)
            if not next_not_closed_in:
                self.closed_in_list.append(next_pos)
        if not next_not_closed_in:
            features['closed_in'] = 1

        # Computes the distance to home for the successor state
        next_distance_to_home = self.min_distance_to_home(successor, next_pos)
        features['distance_to_home'] = next_distance_to_home

        # Calculates if the agent is hovering in the middle because of a camping opponent agent
        if self.red:
            border_x = int(self.midWidth - 1)
        else:
            border_x = int(self.midWidth + 1)

        # The agent is hovering if it's in defense mode and 3 blocks away from the border
        hover_here = (not current_state.is_pacman) and (abs(current_pos[0] - border_x) <= 3)

        if hover_here:
            self.middle_hover_counter += 1
        else:
            self.middle_hover_counter = 0

        # Computes the distance to the closest border: top or bottom, and the x middle and y middle
        top_border = self.height
        bottom_border = 0
        distance_to_closest_border = 0
        distance_to_top_border = abs(next_pos[1] - top_border)
        distance_to_bottom_border = abs(next_pos[1] - bottom_border)
        # Check if current position is closer to the top or bottom border
        if distance_to_top_border < distance_to_bottom_border:
            distance_to_closest_border = distance_to_top_border
        else:
            distance_to_closest_border = distance_to_bottom_border
        features['distance_to_closest_border'] = distance_to_closest_border

        # Computes if the successor state eats a ghost
        features['eat_scared_ghost'] = 0
        for next_scared_ghost in next_scared_ghosts:
            if next_scared_ghost.get_position() is not None and self.get_maze_distance(next_pos, next_scared_ghost.get_position()) == 0:
                features['eat_scared_ghost'] = 1
                break

        """Different Modes"""

        # Offensive agent can go in defensive mode to eat an enemy on our side of the field
        if current_min_distance_to_pacman > 0 and current_min_distance_to_pacman < 6 and current_is_defense and current_state.scared_timer == 0:
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = features['on_defense'] * 10
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = 0
            features['stop']
            features['reverse']
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['eat_scared_ghost'] = 0
            features['returned_food'] = 0
            features['capsule_eaten'] = 0
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance']
            features['num_invaders']
            features['distance_to_closest_border'] = 0

        # If agent is being blocked from entering the enemy field from the middle, go to a border to enter the grid
        elif current_state.is_pacman and hover_here and self.middle_hover_counter > 40:
            if distance_to_closest_border < 6:
                features['distance_to_border'] = 0
            features['successor_score'] = 0
            features['distance_to_closest_border']
            features['on_offense'] = 0
            features['eat_scared_ghost'] = 0
            features['on_defense'] = features['on_defense'] * 10
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance']
            features['stop']
            features['reverse']
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food'] = 0
            features['capsule_eaten'] = 0
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0


        # If pacman agent has eaten a capsule, eat the scared ghosts if there is enough time
        elif next_state.is_pacman and any(scared_ghost.scared_timer > 5 for scared_ghost in next_scared_ghosts):
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food'] = 0
            features['eat_scared_ghost']
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = 0
            features['stop']
            features['reverse']
            features['scared_ghost_distance']
            features['carrying_food'] = 0
            features['returned_food'] = 0
            features['capsule_eaten'] = 0
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0
            features['distance_to_closest_border'] = 0


        # If being followed and there are capsules, go to capsule
        elif current_min_distance_to_ghost > 0 and current_min_distance_to_ghost < 5 and len(current_capsule_list) > 0:
            if current_min_distance_to_capsule < current_min_distance_to_ghost:
                features['ghost_distance'] = 0
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food'] = 0
            features['distance_to_capsule']
            features['ghost_distance']
            features['eat_scared_ghost'] = 0
            features['stop']
            features['reverse']
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0
            features['distance_to_closest_border'] = 0


        # If closed in, being followed and carrying food, go to a non closed-in location
        elif not current_not_closed_in and current_min_distance_to_ghost < 9 and current_min_distance_to_ghost > 0 and current_carrying_food > 0:
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance']
            features['stop'] = 0
            features['reverse'] = 0
            features['eat_scared_ghost'] = 0
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['distance_to_home']
            features['invader_distance'] = 0
            features['num_invaders'] = 0
            features['distance_to_closest_border'] = 0


        # If being followed and carrying food or time is almost up or carrying more than 10, go home
        elif (current_min_distance_to_ghost > 0 and current_min_distance_to_ghost < 6 and current_carrying_food > 0) or (
                len(current_food_list) <= 2) or game_state.data.timeleft < self.min_distance_to_home(game_state,
                                                                                        current_pos) + 60 or (
                current_distance_to_home < 5 and current_carrying_food > 0) or current_carrying_food > 10:
            features['successor_score'] = 0
            features['on_offense'] = 0
            features['on_defense']
            features['distance_to_food'] = 0
            features['distance_to_capsule'] = 0
            features['ghost_distance'] = next_min_distance_to_ghost * 3
            features['stop']
            features['reverse'] = 0
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in']
            features['eat_scared_ghost'] = 0
            features['distance_to_home'] = next_distance_to_home / 2
            features['invader_distance'] = 0
            features['num_invaders'] = 0
            features['distance_to_closest_border'] = 0


        # Else eat food
        else:
            features['successor_score']
            features['on_offense'] = 0
            features['on_defense'] = 0
            features['distance_to_food']
            features['distance_to_capsule']
            features['ghost_distance']
            features['eat_scared_ghost'] = 0
            features['stop']
            features['reverse'] = 0
            features['scared_ghost_distance'] = 0
            features['carrying_food'] = 0
            features['returned_food']
            features['capsule_eaten']
            features['closed_in'] = 0
            features['distance_to_home'] = 0
            features['invader_distance'] = 0
            features['num_invaders'] = 0
            features['distance_to_closest_border'] = 0

        return features

    def get_weights(self, game_state, action):

        return {'successor_score': 100, 'distance_to_food': -2, 'distance_to_capsule': -1, 'reverse': -2,
                'ghost_distance': 8, 'carrying_food': -2, 'scared_ghost_distance': -1, 'stop': - 100, 'on_offense': 10,
                'returned_food': 10, 'distance_to_home': - 10, 'closed_in': -100, 'capsule_eaten': 9999,
                'on_defense': 10, 'num_invaders': -1000, 'eat_scared_ghost': 100 ,'invader_distance': -10, 'distance_to_closest_border': -3,}


class DefensiveReflexAgent(ReflexCaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # initialize key variables for the agent's state
        self.start = None
        self.midWidth = None
        self.height = None
        self.width = None
        # list of important points to defend (food, capsules, choke points)
        self.defensive_targets = []
        # list of points for the agent to patrol when no invader is visible
        self.patrol_points = []
        # index to track the current patrol point
        self.patrol_index = 0
        # dictionary to store historical invader positions for prediction
        self.last_invader_positions = {}
        # location of the last piece of food eaten by an invader
        self.position_food_eaten = None
        # time when the last piece of food was eaten
        self.last_eaten_food_time = 0
        # timer for how long to retreat after being scared
        self.scared_retreat_timer = 0

    def register_initial_state(self, game_state):
        # call base class registration
        super().register_initial_state(game_state)
        # set up map dimensions
        self.midWidth = game_state.data.layout.width / 2
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.start = game_state.get_agent_position(self.index)

        # set up defensive points and patrol route
        self.initialize_defensive_setup(game_state)

    def initialize_defensive_setup(self, game_state):
        # get initial food and capsules to defend
        our_food = self.get_food_you_are_defending(game_state).as_list()
        our_capsules = self.get_capsules_you_are_defending(game_state)

        # identify the most critical points (e.g., near capsules or entrances)
        self.defensive_targets = self.identify_critical_points(game_state, our_food, our_capsules)

        # create a route for the agent to follow when idle
        self.patrol_points = self.create_patrol_route(game_state, our_food, our_capsules)

        # if only 2 or less food left, just defend those spots
        if len(our_food) <= 2:
            self.defensive_targets = our_food

    def identify_critical_points(self, game_state, food_list, capsules):
        # defines a list of high-priority locations to defend based on proximity to capsules,
        # spawn points, and choke points (border locations with few exits)
        critical_points = []

        # capsules are critical
        critical_points.extend(capsules)

        # if little food is left, defend the remaining food
        if len(food_list) <= 4:
            critical_points.extend(food_list[:2])

        # food near a capsule is critical
        for food in food_list:
            for capsule in capsules:
                if self.get_maze_distance(food, capsule) < 4:
                    critical_points.append(food)
                    break

        # food near enemy spawn points (potential entry points) is critical
        spawn_points = self.get_spawn_points(game_state)
        for food in food_list:
            for spawn in spawn_points:
                if self.get_maze_distance(food, spawn) < 5:
                    critical_points.append(food)
                    break

        # identify choke points on the border
        border_x = int(self.midWidth - 1) if self.red else int(self.midWidth + 1)
        for y in range(1, self.height - 1):
            pos = (border_x, y)
            if not is_wall(game_state, pos):
                exits = 0
                # count exits from this border tile
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = border_x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height and not is_wall(game_state, (nx, ny)):
                        exits += 1
                # if 2 or fewer non-wall exits, it's a choke point
                if exits <= 2:
                    critical_points.append(pos)

        # return unique critical points
        return list(set(critical_points))

    def get_spawn_points(self, game_state):
        # collects the starting positions of all agents in the game
        spawns = []
        for i in range(game_state.get_num_agents()):
            spawn = game_state.get_initial_agent_position(i)
            if spawn:
                spawns.append(spawn)
        return spawns

    def create_patrol_route(self, game_state, food_list, capsules):
        # creates a sequential route of points to visit when the agent is not chasing an invader
        patrol = []

        # patrol the capsules
        patrol.extend(capsules)

        # patrol the identified critical points
        patrol.extend(self.defensive_targets)

        # include key border points (middle, top, bottom quarters)
        border_x = int(self.midWidth - 1) if self.red else int(self.midWidth + 1)
        patrol.append((border_x, self.height // 2))
        patrol.append((border_x, self.height // 4))
        patrol.append((border_x, 3 * self.height // 4))

        # ensure patrol points are unique and limit the list size
        seen = set()
        unique_patrol = []
        for point in patrol:
            if point not in seen:
                seen.add(point)
                unique_patrol.append(point)

        return unique_patrol[:10] # use up to 10 unique points

    def get_features(self, game_state, action):
        # calculates features for the successor state that will be used to determine the best action

        features = util.Counter()
        # successor state after performing the action
        successor = self.get_successor(game_state, action)

        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        next_state = successor.get_agent_state(self.index)
        next_pos = next_state.get_position()

        # identify enemies, invaders (pacmen in our territory), and scared enemies
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        scared_enemies = [e for e in enemies if e.scared_timer > 0 and e.get_position() is not None]

        # food and capsules the agent is defending
        our_food = self.get_food_you_are_defending(successor).as_list()
        our_capsules = self.get_capsules_you_are_defending(successor)

        # --- invader features ---
        # count of invaders
        features['num_invaders'] = len(invaders)
        if invaders:
            # distance to the closest invader
            invader_positions = [invader.get_position() for invader in invaders]
            closest_invader = min(invader_positions,
                                  key=lambda pos: self.get_maze_distance(next_pos, pos))

            features['invader_distance'] = self.get_maze_distance(next_pos, closest_invader)

            # try to predict the invader's next position based on last few moves
            if closest_invader in self.last_invader_positions:
                last_pos = self.last_invader_positions[closest_invader]
                if len(last_pos) >= 2:
                    dx = last_pos[-1][0] - last_pos[-2][0]
                    dy = last_pos[-1][1] - last_pos[-2][1]
                    predicted = (closest_invader[0] + dx, closest_invader[1] + dy)
                    if not is_wall(game_state, predicted):
                        # distance to the predicted interception point
                        features['predicted_intercept'] = self.get_maze_distance(next_pos, predicted)

            # update historical invader positions
            if closest_invader not in self.last_invader_positions:
                self.last_invader_positions[closest_invader] = []
            self.last_invader_positions[closest_invader].append(closest_invader)
            if len(self.last_invader_positions[closest_invader]) > 5:
                self.last_invader_positions[closest_invader].pop(0)

            # check if any invader is carrying food
            carrying_invaders = [invader for invader in invaders if invader.num_carrying > 0]
            if carrying_invaders:
                features['invader_with_food'] = 1

        # --- capsule defense features ---
        if our_capsules:
            # distance to the closest capsule we are defending
            capsule_distances = [self.get_maze_distance(next_pos, capsule) for capsule in our_capsules]
            features['capsule_distance'] = min(capsule_distances)

            # check if a capsule is under threat from any enemy
            capsule_at_risk = False
            for capsule in our_capsules:
                for enemy in enemies:
                    if enemy.get_position():
                        dist = self.get_maze_distance(capsule, enemy.get_position())
                        if dist < 8:
                            capsule_at_risk = True
                            break
                if capsule_at_risk:
                    break
            if capsule_at_risk:
                features['capsule_at_risk'] = 1

        # --- food defense features ---
        if our_food:
            # distance to the closest food we are defending
            closest_food = min(our_food, key=lambda f: self.get_maze_distance(next_pos, f))
            features['food_distance'] = self.get_maze_distance(next_pos, closest_food)

            # special features if only 2 or less food left
            if len(our_food) <= 2:
                features['last_food_defense'] = 1
                features['last_food_distance'] = features['food_distance']

            # check if food is under threat from an invader
            food_at_risk = False
            for food in our_food:
                for invader in invaders:
                    dist = self.get_maze_distance(food, invader.get_position())
                    if dist < 6:
                        food_at_risk = True
                        break
                if food_at_risk:
                    break
            if food_at_risk:
                features['food_at_risk'] = 1

        # --- patrol features (used when no invaders are visible) ---
        if self.patrol_points and not invaders:
            # distance to the current patrol target
            target_patrol = self.patrol_points[self.patrol_index % len(self.patrol_points)]
            features['patrol_distance'] = self.get_maze_distance(next_pos, target_patrol)

            # advance to the next patrol point once the current one is reached
            if self.get_maze_distance(current_pos, target_patrol) < 2:
                self.patrol_index += 1

        # --- scared/retreat features ---
        if current_state.scared_timer > 0:
            # agent is scared
            features['scared'] = 1
            self.scared_retreat_timer = 5 # start retreat timer

            # distance to the border (home) for retreat
            features['distance_to_border'] = self.min_distance_to_home(successor, next_pos)

            # if a scared enemy is nearby, moving away is preferred (negative feature value)
            if scared_enemies:
                closest_scared = min(scared_enemies,
                                     key=lambda e: self.get_maze_distance(next_pos, e.get_position()))
                if self.get_maze_distance(next_pos, closest_scared.get_position()) < 3:
                    features['scared_enemy_nearby'] = -5

        # agent is recovering after being scared
        elif self.scared_retreat_timer > 0:
            features['scared_recovery'] = 1
            self.scared_retreat_timer -= 1
            # continue moving towards border
            features['distance_to_border'] = self.min_distance_to_home(successor, next_pos)

        # --- last eaten food tracking ---
        previous_game_state = self.get_previous_observation()
        if previous_game_state and not self.position_food_eaten:
            # check if food was just eaten in the previous step
            previous_food = self.get_food_you_are_defending(previous_game_state).as_list()
            if len(previous_food) > len(our_food):
                for food in previous_food:
                    if food not in our_food:
                        self.position_food_eaten = food
                        self.last_eaten_food_time = game_state.data.timeleft
                        break

        # if food was recently eaten, move towards that position
        if self.position_food_eaten:
            time_since_eaten = self.last_eaten_food_time - game_state.data.timeleft
            if time_since_eaten < 30: # only for 30 game steps
                features['last_eaten_food_distance'] = self.get_maze_distance(next_pos, self.position_food_eaten)

        # --- general action features ---
        # is the agent on defense (not a pacman)
        features['on_defense'] = 1 if not next_state.is_pacman else 0
        # discourage stopping
        if action == Directions.STOP:
            features['stop'] = 1
        # discourage reversing direction
        rev = Directions.REVERSE[current_state.configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # --- teamwork/border features ---
        teammate_index = 1 if self.index == 0 else 0
        teammate_state = successor.get_agent_state(teammate_index)
        if teammate_state.get_position():
            teammate_pos = teammate_state.get_position()

            # check if the teammate is closer to the single invader
            if invaders and len(invaders) == 1:
                invader_pos = invaders[0].get_position()
                my_dist_to_invader = self.get_maze_distance(next_pos, invader_pos)
                teammate_dist_to_invader = self.get_maze_distance(teammate_pos, invader_pos)

                # if teammate is significantly closer, this agent might back off (positive feature)
                if teammate_dist_to_invader < my_dist_to_invader - 2:
                    features['teammate_better_position'] = 1

        # optimal distance from the border
        border_dist = self.min_distance_to_home(successor, next_pos)
        if 1 < border_dist < 10:
            features['good_border_distance'] = 1
        # penalize being too far from the border
        elif border_dist >= 10:
            features['too_far_from_border'] = border_dist

        return features

    def get_weights(self, game_state, action):
        # defines the importance (weights) for each feature.
        # these weights are fixed and determine the agent's behavior strategy.

        return {'num_invaders': 1000, 'invader_distance': -10, 'invader_with_food': 100,
                'capsule_distance': -5, 'capsule_at_risk': 50,
                'food_distance': -1, 'last_food_defense': 500, 'last_food_distance': -10,
                'food_at_risk': 100,
                'patrol_distance': -2,
                'scared': -100, 'scared_enemy_nearby': 100, 'scared_recovery': -10,
                'distance_to_border': -5,
                'last_eaten_food_distance': -20,
                'on_defense': 100, 'stop': -10, 'reverse': -2,
                'predicted_intercept': -15,
                'teammate_better_position': -50,
                'good_border_distance': 1, 'too_far_from_border': -5,
                }

    def choose_action(self, game_state):
        if game_state.data.timeleft % 50 == 0:
            self.last_invader_positions.clear()

        return super().choose_action(game_state)


class two_ways:
    """ A class to determine whether a position has at least two different ways to get back into our half of the field """

    def __init__(self, game_state, agent):
        self.agent = agent
        self.game_state = game_state
        self.height = game_state.data.layout.height
        self.width = game_state.data.layout.width
        self.mid_width = game_state.data.layout.width / 2
        self.red = agent.red

    def is_goal_state(self, state):
        """ Checks whether the given state is the goal state (= a position in the first column in the home side) """
        if self.red:  # Left side of the field
            i = int(self.mid_width - 1)
        else:  # If the agent is from the blue team = right side of the field
            i = int(self.mid_width + 1)
        return state[0] == i

    def get_successors(self, state):
        """ Calculates the valid successors of the given position. """
        my_pos = state
        successors = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = my_pos
            dx, dy = Actions.direction_to_vector(action)  # Compute next positions
            nextx, nexty = int(x + dx), int(y + dy)
            next_pos = (nextx, nexty)

            # Only valid position if it's not a wall
            if not is_wall(self.game_state, next_pos):
                successors.append(next_pos)

        return successors

    def breadthFirstSearch(self, pos):
        """ Returns a boolean, which is True when the given position has two different ways to home, if not it returns False. """
        fringe = Queue()
        fringe.push(pos)
        visited = []
        states = Queue()
        states.push([])

        # How many paths lead to goal state
        count = 0

        # First path that reaches goal state
        first_way = []

        while not fringe.is_empty():
            node = fringe.pop()
            path = states.pop()

            # First time reaching goal state
            if self.is_goal_state(node):
                if count == 0:
                    # Save path
                    first_way = path
                    # Increase count
                    count += 1
                else:
                    # Check if new path shares any nodes with old path
                    one_way = False
                    for elem in first_way:
                        if elem in path:
                            one_way = True

                            # if they do not share nodes = another independent way to escape
                    # There are 2 independent ways to reach goal state
                    if not one_way:
                        return True

            if node in visited:
                continue
            else:
                visited.append(node)

            child_nodes = self.get_successors(node)
            for child_node in child_nodes:
                pathToNode = path + [child_node]
                if child_node not in visited:
                    fringe.push(child_node)
                    states.push(pathToNode)

        else:
            # There were no 2 routes to the goal state: the agent is closed in
            return False


def is_wall(game_state, pos):
    """ Determines whether the position is a wall """
    grid = game_state.data.layout.walls
    return grid[pos[0]][pos[1]]
