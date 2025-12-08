# collinTeam_planned.py
# A CollinAgent with beam-search fallback to escape dead-ends / oscillation
# Drop-in replacement for your collinTeam.py

from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CollinAgent', second = 'CollinAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class CollinAgent(CaptureAgent):
    """
    CollinAgent: reflex agent with beam-search fallback to escape dead-ends / oscillation.
    Preserves your features and weights; adds limited-depth beam search used when necessary.
    """

    def __init__(self, index):
        super().__init__(index)
        self.threat = False
        self.prevPositions = []   # history of positions for simple oscillation detection

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Identify the border
        layoutWidth = gameState.data.layout.width
        if self.red:
            self.borderX = layoutWidth // 2 - 1
        else:
            self.borderX = layoutWidth // 2

        # collect border positions that are legal (not walls)
        self.borderPositions = []
        walls = gameState.getWalls()
        height = walls.height
        for y in range(height):
            pos = (self.borderX, y)
            if not walls[pos[0]][pos[1]]:
                self.borderPositions.append(pos)

        # Precompute valid positions
        self.validPositions = [p for p in gameState.getWalls().asList(False)]

        # init prev position record
        self.prevPositions = []

    ###############
    #  CHOICE     #
    ###############
    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        # avoid using string literal; use Directions.STOP constant
        legalActions = [a for a in legalActions if a != Directions.STOP]

        # Greedy evaluation over one-step features
        actionValues = []
        for a in legalActions:
            f = self.getFeatures(gameState, a)
            w = self.getWeights(gameState, a)
            actionValues.append((f * w, a))

        # pick best greedy action
        bestValue, bestAction = max(actionValues, key=lambda x: x[0])

        # Basic successor info
        successor = self.getSuccessor(gameState, bestAction)
        succPos = successor.getAgentState(self.index).getPosition()
        myStateNow = gameState.getAgentState(self.index)
        myPosNow = myStateNow.getPosition()

        # dead-end feature as evaluated for the greedy action
        deadEndFeature = self.getFeatures(gameState, bestAction).get("deadEnd", 0)
        # carrying after taking the greedy action
        carrying = successor.getAgentState(self.index).numCarrying

        # --- compute nearest visible ghost distance on CURRENT observation ---
        opponents_now = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible_ghosts_now = [g for g in opponents_now if (not g.isPacman) and g.getPosition() is not None]
        nearestGhostDist = None
        if visible_ghosts_now:
            try:
                dists = [self.getMazeDistance(myPosNow, g.getPosition()) for g in visible_ghosts_now]
                nearestGhostDist = min(dists)
            except Exception:
                nearestGhostDist = None

        # simple oscillation detection using previous positions history
        # maintain prevPositions (list) on the agent instance
        if not hasattr(self, "prevPositions"):
            self.prevPositions = []
        self.prevPositions.append(myPosNow)
        if len(self.prevPositions) > 6:
            self.prevPositions.pop(0)

        oscillating = False
        if len(self.prevPositions) >= 4:
            # detect A,B,A,B pattern
            if self.prevPositions[-1] == self.prevPositions[-3] and self.prevPositions[-2] == self.prevPositions[-4]:
                oscillating = True

        # --- DECIDE whether to run planner ---
        # Run planner if any of:
        # 1) in dead end while carrying food
        # 2) oscillating
        # 3) we are Pacman, carrying food, and nearest visible ghost is dangerously close (<=5)
        is_pacman_now = myStateNow.isPacman
        dangerous_ghost_close = (nearestGhostDist is not None and nearestGhostDist <= 5)

        shouldUsePlanner = False
        if carrying > 1:
            shouldUsePlanner = True
        # if (deadEndFeature == 1 and carrying > 1) or oscillating:
        #     shouldUsePlanner = True
        # elif is_pacman_now and carrying > 1 and dangerous_ghost_close:
        #     shouldUsePlanner = True

        if shouldUsePlanner:
            chosen = self.beamSearch(gameState, depth=6, beamSize=6)
            if chosen is None or chosen == Directions.STOP:
                return bestAction
            return chosen

        # otherwise go greedy
        return bestAction


    ###############
    #  FEATURES   #
    ###############
    def getFeatures(self, gameState, action):
        """
        Original feature logic preserved. This method expects gameState and action,
        computes successor internally via getSuccessor.
        """
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Offensive/defensive recognition
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [e for e in opponents if e.isPacman and e.getPosition() is not None]
        enemyGhosts = [e for e in opponents if not e.isPacman and e.getPosition() is not None]

        # Distance to invading enemy pacman (preserve original sign)
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
                # use inverse distance to get larger value when closer (as you wanted earlier)
                features["ghostThreat"] = 1.0 / nearestGhostDist
                self.threat = True
            else:
                self.threat = False

        # Go home if carrying too much food
        carried = myState.numCarrying
        if carried > 3:
            if self.borderPositions:
                homePos = min(self.borderPositions, key=lambda p: self.getMazeDistance(myPos, p))
                features["returnHome"] = -self.getMazeDistance(myPos, homePos)

        # Food eaten detection
        currentFoodCount = len(self.getFood(gameState).asList())
        newFoodCount = len(self.getFood(successor).asList())
        features["foodEaten"] = 1 if newFoodCount < currentFoodCount else 0

        # Reverse action feature (penalize reversing)
        # be careful: gameState.getAgentState(self.index).configuration may be None very early
        conf = gameState.getAgentState(self.index).configuration
        if conf and conf.direction:
            rev = Directions.REVERSE[conf.direction]
            features["reverse"] = 1 if action == rev else 0
        else:
            features["reverse"] = 0

        # Dead-end detection
        walls = gameState.getWalls()
        x, y = int(myPos[0]), int(myPos[1])
        wallcount = 0
        # bounds-safe checks
        try:
            if walls[x+1][y]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x-1][y]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x][y+1]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x][y-1]: wallcount += 1
        except Exception:
            pass
        features["deadEnd"] = 1 if wallcount == 3 else 0

        return features

    ###############
    #  WEIGHTS    #
    ###############
    def getWeights(self, gameState, action):
        """
        Kept your original weight structure; returnHome weight is dynamic and
        augmented by self.threat boolean as you requested previously.
        """
        foodGrid = self.getFoodYouAreDefending(gameState)
        mapWidth = foodGrid.width if hasattr(foodGrid, "width") else gameState.data.layout.width
        eatBonus = mapWidth

        successor = self.getSuccessor(gameState, action)
        carried = successor.getAgentState(self.index).numCarrying

        # Stronger home incentive: scales with carried amount
        returnHomeWeight = - (4.0 * max(1, carried))

        # if self.threat:
        #     returnHomeWeight -= 20

        return {
            "invaderDistance": 20.0,
            "foodDistance": 5.0,
            "ghostThreat": -100.0,
            "returnHome": returnHomeWeight,
            "foodEaten": eatBonus,
            "reverse": -2.0,
            "deadEnd": -10.0
        }

    ###############
    #  PLANNER    #
    ###############
    def beamSearch(self, rootState, depth=6, beamSize=6):
        """
        Limited breadth beam search that evaluates state-action sequences using
        the same feature/weight evaluator. Returns the first action of the best
        discovered plan (or None if none found).
        """
        # Each node: (score, action_sequence, resultingState)
        frontier = [(0.0, [], rootState)]

        for _ in range(depth):
            candidates = []
            for score, actions, state in frontier:
                legal = state.getLegalActions(self.index)
                # exclude STOP from search to prefer moving out
                legal = [a for a in legal if a != Directions.STOP]
                for a in legal:
                    # evaluate next step from 'state' with action a
                    # compute immediate score using getFeatures/getWeights with this state as base
                    f = self._features_for_state_action(state, a)
                    w = self._weights_for_state_action(state, a)
                    stepScore = f * w
                    # accumulate score (use additive here)
                    newScore = score + stepScore
                    try:
                        succ = state.generateSuccessor(self.index, a)
                    except Exception:
                        continue
                    candidates.append((newScore, actions + [a], succ))
            if not candidates:
                break
            # keep best beamSize candidates
            candidates.sort(key=lambda x: x[0], reverse=True)
            frontier = candidates[:beamSize]

        # choose highest scoring branch
        if not frontier:
            return None
        best = max(frontier, key=lambda x: x[0])
        _, bestActions, _ = best
        return bestActions[0] if bestActions else None

    def _features_for_state_action(self, state, action):
        """
        Evaluate features for (state, action) pair without calling getSuccessor again.
        This mirrors getFeatures but computes successor = state.generateSuccessor(...)
        so beam search depth is consistent.
        """
        features = util.Counter()
        try:
            successor = state.generateSuccessor(self.index, action)
        except Exception:
            return features  # return empty features if successor invalid

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Opponents from successor
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmen = [e for e in opponents if e.isPacman and e.getPosition() is not None]
        enemyGhosts = [e for e in opponents if not e.isPacman and e.getPosition() is not None]

        # invaderDistance (same sign as your getFeatures)
        if enemyPacmen:
            d = min(self.getMazeDistance(myPos, e.getPosition()) for e in enemyPacmen)
            features["invaderDistance"] = -d

        # foodDistance
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minFoodDist = min(self.getMazeDistance(myPos, f) for f in foodList)
            features["foodDistance"] = -minFoodDist

        # ghost threat
        if myState.isPacman and enemyGhosts:
            nearestGhostDist = min(self.getMazeDistance(myPos, g.getPosition()) for g in enemyGhosts)
            if nearestGhostDist <= 3:
                features["ghostThreat"] = 1.0 / nearestGhostDist

        # returnHome
        carried = myState.numCarrying
        if carried > 3 and self.borderPositions:
            homePos = min(self.borderPositions, key=lambda p: self.getMazeDistance(myPos, p))
            features["returnHome"] = -self.getMazeDistance(myPos, homePos)

        # foodEaten detection against 'state' (not rootState) -> compare food counts
        curFoodCount = len(self.getFood(state).asList())
        newFoodCount = len(self.getFood(successor).asList())
        features["foodEaten"] = 1 if newFoodCount < curFoodCount else 0

        # reverse detection: use state's agent configuration
        conf = state.getAgentState(self.index).configuration
        if conf and conf.direction:
            rev = Directions.REVERSE[conf.direction]
            features["reverse"] = 1 if action == rev else 0
        else:
            features["reverse"] = 0

        # dead end detection using successor's position
        walls = state.getWalls()
        x, y = int(myPos[0]), int(myPos[1])
        wallcount = 0
        try:
            if walls[x+1][y]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x-1][y]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x][y+1]: wallcount += 1
        except Exception:
            pass
        try:
            if walls[x][y-1]: wallcount += 1
        except Exception:
            pass
        features["deadEnd"] = 1 if wallcount == 3 else 0

        return features

    def _weights_for_state_action(self, state, action):
        """
        Mirror of getWeights that is safe to call on arbitrary state/action pairs
        during beam search. It uses the same dynamic returnHome scaling.
        """
        # compute map width safely
        foodGrid = self.getFoodYouAreDefending(state)
        mapWidth = foodGrid.width if hasattr(foodGrid, "width") else state.data.layout.width
        eatBonus = mapWidth

        # inspect successor to see carried amount
        try:
            succ = state.generateSuccessor(self.index, action)
            carried = succ.getAgentState(self.index).numCarrying
        except Exception:
            carried = 0

        returnHomeWeight = - (20.0 * carried)
        # determine threat using quick heuristic: is any ghost very close in succ?
        # Note: we keep self.threat unchanged here; beam search uses local successor info
        threat_local = False
        try:
            opponents = [succ.getAgentState(i) for i in self.getOpponents(succ)]
            enemyGhosts = [e for e in opponents if not e.isPacman and e.getPosition() is not None]
            if enemyGhosts:
                nearestGhostDist = min(self.getMazeDistance(succ.getAgentState(self.index).getPosition(), g.getPosition()) for g in enemyGhosts)
                if nearestGhostDist <= 3:
                    threat_local = True
        except Exception:
            threat_local = False

        # if threat_local:
        #     returnHomeWeight -= 20

        return {
            "invaderDistance": 20.0,
            "foodDistance": 1.0,
            "ghostThreat": -100.0,
            "returnHome": returnHomeWeight,
            "foodEaten": eatBonus,
            "reverse": -50.0,
            "deadEnd": -10.0
        }

    ###############
    # HELPERS     #
    ###############
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Allow half-steps
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
