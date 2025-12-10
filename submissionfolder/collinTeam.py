# collinTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CollinAgent', second = 'CollinAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class CollinAgent(CaptureAgent):
    """
    A flexible agent that switches roles based on game state:
    - Defends when enemies invade
    - Attacks when safe or after scaring enemies
    - Returns home when carrying food or unsafe
    """

    threat = False
    initialFoodCount = 0
    lastEnemyScore = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Identify the border
        layoutWidth = gameState.data.layout.width
        if self.red:
            self.borderX = layoutWidth // 2 - 1
        else:
            self.borderX = layoutWidth // 2

        self.borderPositions = []
        walls = gameState.getWalls()
        height = walls.height

        for y in range(height):
            pos = (self.borderX, y)
            if not walls[pos[0]][pos[1]]:
                self.borderPositions.append(pos)

        # Precompute valid positions
        self.validPositions = [p for p in gameState.getWalls().asList(False)]

        # Track initial friendly food count for defense decisions
        self.initialFoodCount = len(self.getFoodYouAreDefending(gameState).asList())

        # Track enemy score to detect successful scoring
        self.lastEnemyScore = self.getScore(gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions = [a for a in actions if a != "Stop"]

        # Check if enemy successfully scored (food returned to their side)
        currentEnemyScore = self.getScore(gameState)
        if currentEnemyScore < self.lastEnemyScore:
            # Enemy scored (our score decreased), reset to normal play
            self.initialFoodCount = len(self.getFoodYouAreDefending(gameState).asList())
        self.lastEnemyScore = currentEnemyScore

        # Check how much friendly food has been eaten
        currentFriendlyFood = len(self.getFoodYouAreDefending(gameState).asList())
        foodEaten = self.initialFoodCount - currentFriendlyFood
        shouldDefend = foodEaten > 5

        # First: if an enemy invader is visible, prioritize defending them.
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos is not None:
            start = util.nearestPoint(myPos)
            start = (int(start[0]), int(start[1]))

            # Visible opponents who are pacmen (invaders)
            opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            enemyPacmen = [e for e in opponents if e.isPacman and e.getPosition() is not None]

            if enemyPacmen:
                # Choose nearest invader
                try:
                    invader = min(enemyPacmen, key=lambda e: self.getMazeDistance(start, e.getPosition()))
                except Exception:
                    invader = min(enemyPacmen, key=lambda e: abs(start[0]-e.getPosition()[0]) + abs(start[1]-e.getPosition()[1]))

                invaderPos = invader.getPosition()

                # Always chase visible invaders aggressively (low threshold)
                doDefend = False
                if not myState.isPacman:
                    # If on defense, always chase
                    doDefend = True
                else:
                    # If on offense but invader is close to our border (within 3 steps), chase
                    if self.borderPositions:
                        try:
                            borderDist = min(self.getMazeDistance(invaderPos, bp) for bp in self.borderPositions)
                        except Exception:
                            borderDist = min(abs(invaderPos[0]-bp[0]) + abs(invaderPos[1]-bp[1]) for bp in self.borderPositions)
                        if borderDist <= 6:
                            doDefend = True

                if doDefend:
                    path = self.aStarSearch(gameState, start, invaderPos)
                    if path:
                        return path[0]

            # If too much friendly food has been eaten, stop offensive operations
            if shouldDefend:
                if self.borderPositions:
                    path = self.aStarSearch(gameState, start, self.borderPositions)
                    if path:
                        return path[0]

            # Next: follow A* for clear objectives:
            carried = myState.numCarrying

            # Return home when carrying too much
            if carried > 3:
                if self.borderPositions:
                    path = self.aStarSearch(gameState, start, self.borderPositions)
                    if path:
                        return path[0]

        # Fallback to original feature-weighted policy
        values = []
        for a in actions:
            features = self.getFeatures(gameState, a)
            weights = self.getWeights(gameState, a)
            values.append(features * weights)

        if not values:
            return Directions.STOP

        bestValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == bestValue]
        return random.choice(bestActions)

    #  FEATURES

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Check if in defense mode (too much friendly food eaten)
        currentFriendlyFood = len(self.getFoodYouAreDefending(gameState).asList())
        foodEaten = self.initialFoodCount - currentFriendlyFood
        inDefenseMode = foodEaten > 5

        # Offensive/defensive recognition
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        enemyGhosts = [e for e in enemies if not e.isPacman and e.getPosition() is not None]

        # Distance to invading enemy pacman
        if enemyPacmen:
            d = min(self.getMazeDistance(myPos, e.getPosition()) for e in enemyPacmen)
            features["invaderDistance"] = -d

        # If on offense, encourage eating food
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minFoodDist = min(self.getMazeDistance(myPos, f) for f in foodList)
            features["foodDistance"] = -minFoodDist

        # Avoid enemy ghosts when we are pacman
        if myState.isPacman and enemyGhosts:
            nearestGhostDist = min(self.getMazeDistance(myPos, g.getPosition()) for g in enemyGhosts)
            if nearestGhostDist <= 3:
                features["ghostThreat"] = 1.0 / nearestGhostDist
                self.threat = True
            else:
                self.threat = False

        #redundant
        # Go home if carrying too much food
        # carried = myState.numCarrying
        # if carried > 3:
        #     if self.borderPositions:
        #         homePos = min(
        #             self.borderPositions,
        #             key=lambda p: self.getMazeDistance(myPos, p)
        #         )
        #         features["returnHome"] = -self.getMazeDistance(myPos, homePos)

        currentFoodCount = len(self.getFood(gameState).asList())
        newFoodCount = len(self.getFood(successor).asList())
        features["foodEaten"] = 1 if newFoodCount < currentFoodCount else 0

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        features["reverse"] = 1 if action == reverse else 0

        walls = gameState.getWalls()
        x, y = int(myPos[0]), int(myPos[1])

        # A* completely dominates and this never triggers
        # wallcount = 0
        # if walls[x+1][y]: wallcount += 1
        # if walls[x-1][y]: wallcount += 1
        # if walls[x][y+1]: wallcount += 1
        # if walls[x][y-1]: wallcount += 1
        # features["deadEnd"] = 1 if wallcount == 3 else 0

        # Incentivize teammate separation when in defense mode
        if inDefenseMode:
            teammates = [successor.getAgentState(i) for i in self.getTeam(successor) if i != self.index]
            if teammates:
                for teammate in teammates:
                    if teammate.getPosition() is not None:
                        try:
                            teammateDist = self.getMazeDistance(myPos, teammate.getPosition())
                        except Exception:
                            teammateDist = abs(myPos[0] - teammate.getPosition()[0]) + abs(myPos[1] - teammate.getPosition()[1])
                        # Negative distance penalty: encourages separation
                        features["teammateSeparation"] = -teammateDist if features.get("teammateSeparation", 0) == 0 else min(features["teammateSeparation"], -teammateDist)

            # Encourage patrolling deeper into friendly territory, not just border
            defendingFood = self.getFoodYouAreDefending(successor).asList()
            if defendingFood:
                minFoodDist = min(self.getMazeDistance(myPos, f) for f in defendingFood)
                features["defendFood"] = -minFoodDist

        return features

    
    #  WEIGHTS
    def getWeights(self, gameState, action):

        foodGrid = self.getFoodYouAreDefending(gameState)

        mapWidth = foodGrid.width

        eatBonus = mapWidth

        successor = self.getSuccessor(gameState, action)
        carried = successor.getAgentState(self.index).numCarrying

        returnHomeWeight = - (20.0 * max(1, carried))

        if self.threat:
            returnHomeWeight -= 20

        # Check if in defense mode
        currentFriendlyFood = len(self.getFoodYouAreDefending(gameState).asList())
        foodEaten = self.initialFoodCount - currentFriendlyFood
        inDefenseMode = foodEaten > 5

        # Higher weight for teammate separation in defense mode
        teammateSeparationWeight = -20.0 if inDefenseMode else 0.0

        return {
            "invaderDistance": 30.0,
            "foodDistance": 1.0,
            "ghostThreat": -100.0,
            "returnHome": returnHomeWeight,
            "foodEaten": eatBonus,
            "reverse": -50.0,
            "deadEnd": -10.0,
            "teammateSeparation": teammateSeparationWeight,
            "defendFood": -3.0 if inDefenseMode else 0.0
        }

    # HELPERS

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Allow half-steps
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def aStarSearch(self, gameState, start, goals):
        """
        A* search on the maze grid from start to a goal position.
        - `start` is a (x,y) tuple of integers.
        - `goals` may be a single (x,y) tuple or an iterable of goal tuples.
        - Treats enemy ghosts as obstacles (avoids paths through ghost positions).

        Returns a list of actions (e.g. ['North','East',...]) leading from
        start to the closest goal, or an empty list if no path found.
        """
        # Normalize goals to a set for fast membership tests
        if not hasattr(goals, '__iter__') or isinstance(goals, tuple) and len(goals) == 2:
            goals = [goals]
        goalSet = set(goals)

        walls = gameState.getWalls()

        # Identify ghost positions to treat as obstacles
        ghostPositions = set()
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        for opponent in opponents:
            if not opponent.isPacman and opponent.getPosition() is not None:
                ghostPositions.add(opponent.getPosition())

        # Heuristic: Manhattan distance to nearest goal
        def heuristic(pos):
            return min(abs(pos[0]-g[0]) + abs(pos[1]-g[1]) for g in goalSet)

        # Directions mapping
        dirs = {'North': (0, 1), 'South': (0, -1), 'East': (1, 0), 'West': (-1, 0)}

        frontier = util.PriorityQueue()
        frontier.push((start, []), heuristic(start))

        explored = set()
        cost_so_far = {start: 0}

        while not frontier.isEmpty():
            current, actions = frontier.pop()

            if current in goalSet:
                return actions

            if current in explored:
                continue
            explored.add(current)

            for action, delta in dirs.items():
                nx = current[0] + delta[0]
                ny = current[1] + delta[1]
                # skip walls
                if walls[nx][ny]:
                    continue
                # skip ghost positions
                if (nx, ny) in ghostPositions:
                    continue
                neighbor = (nx, ny)
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor)
                    frontier.push((neighbor, actions + [action]), priority)

        # No path found
        return []
