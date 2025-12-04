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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class CollinAgent(CaptureAgent):
    """
    A flexible agent that switches roles based on game state:
    - Defends when enemies invade
    - Attacks when safe or after scaring enemies
    - Returns home when carrying food or unsafe
    """

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

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions = [a for a in actions if a != "Stop"]

        values = []
        for a in actions:
            features = self.getFeatures(gameState, a)
            weights = self.getWeights(gameState, a)
            values.append(features * weights)

        bestValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == bestValue]
        return random.choice(bestActions)

    #  FEATURES

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

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
            if nearestGhostDist <= 4:
                features["ghostThreat"] = -nearestGhostDist

        # Go home if carrying too much food
        carried = myState.numCarrying
        if carried > 3:
            if self.borderPositions:
                homePos = min(
                    self.borderPositions,
                    key=lambda p: self.getMazeDistance(myPos, p)
                )
                features["returnHome"] = -self.getMazeDistance(myPos, homePos)

        currentFoodCount = len(self.getFood(gameState).asList())
        newFoodCount = len(self.getFood(successor).asList())
        features["foodEaten"] = 1 if newFoodCount < currentFoodCount else 0

        return features

    
    #  WEIGHTS
    def getWeights(self, gameState, action):

        foodGrid = self.getFoodYouAreDefending(gameState)
        try:
            mapWidth = foodGrid.width
        except Exception:
            # Fallback: use layout width
            mapWidth = gameState.data.layout.width

        eatBonus = mapWidth * 2

        successor = self.getSuccessor(gameState, action)
        carried = successor.getAgentState(self.index).numCarrying

        returnHomeWeight = - (4.0 * max(1, carried))

        return {
            "invaderDistance": 10.0,
            "foodDistance": 1.0,
            "ghostThreat": -20.0,
            "returnHome": returnHomeWeight,
            "foodEaten": eatBonus,
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
