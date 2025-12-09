# fsm.py
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
import random, util
from game import Directions, Actions

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffenseFSMAgent', second='DefenseFSMAgent'):
    """
    Return a list of two agents that form the team.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    Minimal random agent (for reference / sanity checks).
    """
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)


class FSMBaseAgent(CaptureAgent):
    """
    Base class for finite-state agents with a BFS navigation core.

    Modes:
      - OFFENSE: seek and eat food on opponent's side
      - DEFENSE: chase invaders on our side or patrol boundary
      - RETREAT: run back to home boundary to cash in food


    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.mode = "OFFENSE"
        self.start = gameState.getAgentPosition(self.index)

        # Precompute home boundary tiles (walkable squares on our side of midline)
        layout = gameState.data.layout
        walls = layout.walls
        width, height = layout.width, layout.height

        if self.red:
            boundary_x = (width // 2) - 1
        else:
            boundary_x = (width // 2)

        self.homeBoundary = []
        for y in range(height):
            if not walls[boundary_x][y]:
                self.homeBoundary.append((boundary_x, y))

    # ---------- Core helpers ----------

    def getGridPosition(self, gameState):
        """
        Return the agent's current position snapped to the nearest grid point.
        """
        pos = gameState.getAgentState(self.index).getPosition()
        if pos is None:
            return self.start
        nearest = util.nearestPoint(pos)
        return (int(nearest[0]), int(nearest[1]))

    def bfsToTargets(self, gameState, targets):
        """
        BFS from our current grid position to the *nearest* of the given targets.

        targets: iterable of (x, y) positions.
        Returns: list of actions (Directions.*); empty if no path exists.
        """
        start = self.getGridPosition(gameState)
        if not targets:
            return []

        # normalize targets to int coordinates
        targets_set = set((int(x), int(y)) for (x, y) in targets)
        if start in targets_set:
            return []

        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        from util import Queue
        frontier = Queue()
        frontier.push(start)
        # parents[pos] = (prevPos, actionToHere)
        parents = {start: (None, None)}

        directions = [Directions.NORTH, Directions.SOUTH,
                      Directions.EAST, Directions.WEST]

        while not frontier.isEmpty():
            pos = frontier.pop()

            if pos in targets_set:
                # Reconstruct path
                actions = []
                cur = pos
                while parents[cur][0] is not None:
                    prev, act = parents[cur]
                    actions.append(act)
                    cur = prev
                actions.reverse()
                return actions

            for action in directions:
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(pos[0] + dx), int(pos[1] + dy)

                if 0 <= nx < width and 0 <= ny < height and not walls[nx][ny]:
                    nxt = (nx, ny)
                    if nxt not in parents:
                        parents[nxt] = (pos, action)
                        frontier.push(nxt)

        # No reachable targets
        return []

    # ---------- Invader assignment helper ----------

    def getAssignedInvaders(self, gameState):
        """
        Return positions of invaders (enemy Pacmen) that THIS agent should chase:
        for each invader, whichever teammate is closer (in maze distance)
        is assigned that invader.

        This prevents both agents from chasing the same enemy.
        """
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        if not invaders:
            return []

        myPos = self.getGridPosition(gameState)

        team = self.getTeam(gameState)
        teammateIndex = None
        for t in team:
            if t != self.index:
                teammateIndex = t
                break

        teammatePos = None
        if teammateIndex is not None:
            tState = gameState.getAgentState(teammateIndex)
            tPos = tState.getPosition()
            if tPos is not None:
                tn = util.nearestPoint(tPos)
                teammatePos = (int(tn[0]), int(tn[1]))

        myTargets = []
        for inv in invaders:
            pos = inv.getPosition()
            if pos is None:
                continue
            pos = (int(pos[0]), int(pos[1]))
            myDist = self.getMazeDistance(myPos, pos)

            teammateDist = None
            if teammatePos is not None:
                teammateDist = self.getMazeDistance(teammatePos, pos)

            # If no teammate pos or I'm closer/equal, I'm responsible
            if teammateDist is None or myDist <= teammateDist:
                myTargets.append(pos)

        return myTargets

    # ---------- FSM driver ----------

    def chooseAction(self, gameState):
        """
        Generic FSM + BFS decision:
          1. updateMode()  (subclass)
          2. pick targets based on mode
          3. BFS path to nearest target
          4. take first step
          5. fallback to local scoring if BFS fails
        """
        if not hasattr(self, "numMoves"):
            self.numMoves = 0
        self.numMoves += 1

        # 1) mode transition
        self.updateMode(gameState)

        # 2) targets based on mode
        if self.mode == "OFFENSE":
            targets = self.getOffenseTargets(gameState)
        elif self.mode == "DEFENSE":
            targets = self.getDefenseTargets(gameState)
        else:  # "RETREAT"
            targets = self.getRetreatTargets(gameState)

        # 3) BFS path
        path = self.bfsToTargets(gameState, targets)
        if path:
            desired = path[0]
            legal = gameState.getLegalActions(self.index)
            if Directions.STOP in legal and len(legal) > 1:
                legal.remove(Directions.STOP)
            if desired in legal:
                return desired

        # 4) Fallback: local scoring (should be rare)
        legal = gameState.getLegalActions(self.index)
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)

        if not legal:
            return Directions.STOP

        if self.mode == "OFFENSE":
            scores = [self.offenseEval(gameState, a) for a in legal]
        elif self.mode == "DEFENSE":
            scores = [self.defenseEval(gameState, a) for a in legal]
        else:
            scores = [self.retreatEval(gameState, a) for a in legal]

        best_score = max(scores)
        best_actions = [a for a, s in zip(legal, scores) if s == best_score]
        return random.choice(best_actions)

    # ---------- To be customized by subclasses ----------

    def updateMode(self, gameState):
        """
        Subclasses must set self.mode to "OFFENSE", "DEFENSE", or "RETREAT".
        """
        util.raiseNotDefined()

    # Target selection hooks (can be overridden)

    def getOffenseTargets(self, gameState):
        """
        Default offensive targets: enemy food.
        """
        return self.getFood(gameState).asList()

    def getDefenseTargets(self, gameState):
        """
        Default defensive targets:
          - invaders assigned to THIS agent (via getAssignedInvaders),
          - otherwise home boundary tiles.
        """
        assigned = self.getAssignedInvaders(gameState)
        if assigned:
            return assigned
        return list(self.homeBoundary)

    def getRetreatTargets(self, gameState):
        """
        Default retreat targets: home boundary tiles (to cross home and score).
        """
        return list(self.homeBoundary)

    # ---------- Local evaluators (no/few global effects) ----------

    def offenseEval(self, gameState, action):
        """
        Simple offensive eval:
          - prefer higher team score
          - prefer being Pacman
          - prefer being close to enemy food
        """
        succ = gameState.generateSuccessor(self.index, action)
        my_state = succ.getAgentState(self.index)
        my_pos = my_state.getPosition()
        feats = util.Counter()

        feats["score"] = self.getScore(succ)
        feats["onOffense"] = 1 if my_state.isPacman else 0

        food = self.getFood(succ).asList()
        if food and my_pos is not None:
            d = min(self.getMazeDistance(my_pos, f) for f in food)
            feats["closestFood"] = float(d)
        else:
            feats["closestFood"] = 0.0

        weights = {
            "score":       80.0,
            "onOffense":   10.0,
            "closestFood": -3.0,
        }
        return feats * weights

    def defenseEval(self, gameState, action):
        """
        Simple defensive eval:
          - prefer being a ghost on our side (onDefense)
          - chase invaders
          - avoid stopping
        """
        succ = gameState.generateSuccessor(self.index, action)
        my_state = succ.getAgentState(self.index)
        my_pos = my_state.getPosition()
        feats = util.Counter()

        feats["onDefense"] = 0 if my_state.isPacman else 1

        enemies = [succ.getAgentState(i) for i in self.getOpponents(succ)]
        invaders = [e for e in enemies if e.isPacman and e.getPosition() is not None]
        feats["numInvaders"] = len(invaders)

        if invaders and my_pos is not None:
            d = min(self.getMazeDistance(my_pos, e.getPosition()) for e in invaders)
            feats["closestInvader"] = float(d)
        else:
            feats["closestInvader"] = 0.0

        feats["stop"] = 1 if action == Directions.STOP else 0

        weights = {
            "onDefense":       60.0,
            "numInvaders":    -80.0,
            "closestInvader":  -8.0,
            "stop":           -10.0,
        }
        return feats * weights

    def retreatEval(self, gameState, action):
        """
        Simple retreat eval:
          - get closer to home boundary
          - value carrying food
          - avoid stopping
        """
        succ = gameState.generateSuccessor(self.index, action)
        my_state = succ.getAgentState(self.index)
        my_pos = my_state.getPosition()
        carried = my_state.numCarrying
        feats = util.Counter()

        if self.homeBoundary and my_pos is not None:
            d = min(self.getMazeDistance(my_pos, b) for b in self.homeBoundary)
            feats["homeDist"] = float(d)
        else:
            feats["homeDist"] = 0.0

        feats["carrying"] = float(carried)
        feats["stop"] = 1 if action == Directions.STOP else 0

        weights = {
            "homeDist": -5.0,
            "carrying": 3.0,
            "stop":    -15.0,
        }
        return feats * weights


class OffenseFSMAgent(FSMBaseAgent):
    """
    Offensive FSM agent.

    Behavior:
      - OFFENSE:
          Use BFS to nearest enemy food, BUT do not cross into enemy
          territory if a non-scared ghost is very close right after crossing.
      - RETREAT:
          Triggered when carrying a lot of food or when already in danger.
      - DEFENSE:
          If we're a ghost on our side and see invaders, briefly help defend.

    """

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.mode = "OFFENSE"
        self.numMoves = 0

    def updateMode(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = my_state.getPosition()
        carrying = my_state.numCarrying

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible = [e for e in enemies if e.getPosition() is not None]
        invaders = [e for e in visible if e.isPacman]
        ghosts = [e for e in visible if not e.isPacman and e.scaredTimer == 0]

        ghostDanger = None
        if ghosts and my_pos is not None:
            ghostDanger = min(self.getMazeDistance(my_pos, g.getPosition()) for g in ghosts)

        # Early game: force offense so we leave spawn
        if self.numMoves < 8:
            self.mode = "OFFENSE"
            return

        # If we are a ghost on our side and there are invaders, help defend
        if invaders and not my_state.isPacman:
            self.mode = "DEFENSE"
            return

        # If we are Pacman and:
        #   - carrying a lot, OR
        #   - near a non-scared ghost
        # => RETREAT to cash in / avoid death
        carryingThreshold = 4
        dangerRadius = 4

        if my_state.isPacman:
            if carrying >= carryingThreshold:
                self.mode = "RETREAT"
                return
            if ghostDanger is not None and ghostDanger <= dangerRadius:
                self.mode = "RETREAT"
                return

        # Default: be on offense (seek more food)
        self.mode = "OFFENSE"

    def chooseAction(self, gameState):
        """
        Override FSMBaseAgent.chooseAction for OFFENSE mode to avoid
        suicidal border-crosses, while still using BFS+FSM logic.

        """
        # Let the FSM decide the mode as usual
        if not hasattr(self, "numMoves"):
            self.numMoves = 0
        self.numMoves += 1

        self.updateMode(gameState)

        # If we're not in OFFENSE, fall back to generic FSM behavior
        if self.mode != "OFFENSE":
            return super().chooseAction(gameState)

        # OFFENSE mode: plan toward food
        targets = self.getOffenseTargets(gameState)
        path = self.bfsToTargets(gameState, targets)

        legal = gameState.getLegalActions(self.index)
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)

        if not path:
            # No BFS path (weird, but possible) -> fallback to base behavior
            return super().chooseAction(gameState)

        desired = path[0]
        if desired not in legal:
            # BFS suggested something illegal, fallback
            return super().chooseAction(gameState)

        # --- Danger check on crossing the border ---
        my_state = gameState.getAgentState(self.index)
        succ = gameState.generateSuccessor(self.index, desired)
        succ_state = succ.getAgentState(self.index)
        succ_pos = succ_state.getPosition()

        # Did this action actually cross us into enemy territory?
        crossingBorder = (not my_state.isPacman) and succ_state.isPacman

        if crossingBorder and succ_pos is not None:
            enemies = [succ.getAgentState(i) for i in self.getOpponents(succ)]
            ghosts = [e for e in enemies
                      if not e.isPacman and e.getPosition() is not None and e.scaredTimer == 0]

            ghostDanger = None
            if ghosts:
                ghostDanger = min(self.getMazeDistance(succ_pos, g.getPosition()) for g in ghosts)

            # If a ghost will be very close right after we cross, don't cross
            safeCrossRadius = 3  # adjust if you want more/less caution
            if ghostDanger is not None and ghostDanger <= safeCrossRadius:
                # Choose a "safe" action that stays on our side instead
                safeActions = []
                for a in legal:
                    succ2 = gameState.generateSuccessor(self.index, a)
                    st2 = succ2.getAgentState(self.index)
                    # only keep actions that do NOT make us Pacman (stay home side)
                    if not st2.isPacman:
                        safeActions.append(a)

                if safeActions:
                    # Among safe actions, still bias towards good offenseEval
                    scores = [self.offenseEval(gameState, a) for a in safeActions]
                    bestScore = max(scores)
                    bestActs = [a for a, s in zip(safeActions, scores) if s == bestScore]
                    return random.choice(bestActs)
                # If no safe actions exist, we begrudgingly take the risky move

        # Either not crossing, or crossing is judged safe enough
        return desired


class DefenseFSMAgent(FSMBaseAgent):
    """
    Defensive FSM agent.

    Behavior:
      - DEFENSE:
          If any invaders are visible, BFS to those assigned to this agent
          (closer agent gets that invader). If none visible, patrol a set
          of boundary points spread along the whole side, cycling over time.
      - OFFENSE:
          If map is quiet (no invaders) and we cross into enemy side,
          we can steal some food.
      - RETREAT:
          If we're Pacman and carrying food, go home to score.


    """

    def registerInitialState(self, gameState):
        super().registerInitialState(gameState)
        self.mode = "DEFENSE"

        # Build wider patrol set: sample several boundary points spanning the height.
        boundary = sorted(self.homeBoundary, key=lambda p: p[1])  # sort by y
        if not boundary:
            self.patrolPoints = []
        else:
            n = len(boundary)
            if n <= 4:
                pts = boundary
            else:
                idxs = [0, n // 3, 2 * n // 3, n - 1]
                pts = [boundary[i] for i in idxs]
            # unique
            self.patrolPoints = list({p for p in pts})

        self.currentPatrolIndex = 0
        self.lastPatrolSwitchMove = 0

    def getDefenseTargets(self, gameState):
        """
        Override base:
          - If invaders assigned to us: chase them.
          - Else: patrol one boundary point at a time (cycling), to sweep area.
        """
        assigned = self.getAssignedInvaders(gameState)
        if assigned:
            return assigned

        if not self.patrolPoints:
            return list(self.homeBoundary)

        # Patrol only ONE point at a time so we sweep across the map.
        return [self.patrolPoints[self.currentPatrolIndex]]

    def updateMode(self, gameState):
        my_state = gameState.getAgentState(self.index)
        my_pos = my_state.getPosition()
        carrying = my_state.numCarrying

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        visible = [e for e in enemies if e.getPosition() is not None]
        invaders = [e for e in visible if e.isPacman]
        ghosts = [e for e in visible if not e.isPacman and e.scaredTimer == 0]

        ghostDanger = None
        if ghosts and my_pos is not None:
            ghostDanger = min(self.getMazeDistance(my_pos, g.getPosition()) for g in ghosts)

        # If there are ANY invaders, we want to be in DEFENSE mode.
        if invaders:
            self.mode = "DEFENSE"
            return

        # If we're Pacman and carrying food, RETREAT to cash it in.
        if my_state.isPacman and carrying >= 2:
            self.mode = "RETREAT"
            return

        # If we're Pacman with little/no food and not in immediate danger,
        # we can opportunistically steal a bit.
        dangerRadius = 3
        if my_state.isPacman:
            if ghostDanger is None or ghostDanger > dangerRadius:
                self.mode = "OFFENSE"
                return
            else:
                # ghost is close -> go home instead
                self.mode = "RETREAT"
                return

        # No invaders, we're a ghost on our side -> DEFENSE patrol
        self.mode = "DEFENSE"

        # --- Patrol cycling: sweep a wider area over time ---
        if self.patrolPoints and my_pos is not None:
            myGridPos = self.getGridPosition(gameState)
            currentTarget = self.patrolPoints[self.currentPatrolIndex]
            # If we've reached current target or been here a while, move to next
            reached = self.getMazeDistance(myGridPos, currentTarget) <= 1
            timeLimit = 20  # moves before forcing patrol shift
            if reached or (self.numMoves - self.lastPatrolSwitchMove > timeLimit):
                self.currentPatrolIndex = (self.currentPatrolIndex + 1) % len(self.patrolPoints)
                self.lastPatrolSwitchMove = self.numMoves
