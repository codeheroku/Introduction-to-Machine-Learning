"""
This project was developed by Peter Chen, Rocky Duan, Pieter Abbeel for the Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from CS188 project materials: http://ai.berkeley.edu/project_overview.html.

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""


import numpy as np
import sys

from gym import utils
from gym import Env, spaces
from gym.utils import seeding

import math
from math import pi as PI
import random

all_envs = []

class CrawlingRobotEnv(Env):

    def close_gui(self):
        if self.root is not None:
            self.root.destroy()
            self.root = None

    def __init__(self, horizon=np.inf, render=False):
        if render:
            import tkinter
            for env in all_envs:
                env.close_gui()
            #all_envs.clear()
            all_envs.append(self)
            root = tkinter.Tk()
            root.title('Crawler GUI')
            root.resizable(0, 0)
            self.root = root
            canvas = tkinter.Canvas(root, height=200, width=1000)
            canvas.grid(row=2, columnspan=10)

            def close():
                if self.root is not None:
                    self.root.destroy()
                    self.root = None
            root.protocol('WM_DELETE_WINDOW', lambda: close)
            root.lift()
            root.attributes('-topmost', True)
            import platform, subprocess, os
            if platform.system() == 'Darwin':
                tmpl = 'tell application "System Events" to set frontmost of every process whose unix id is {} to true'
                script = tmpl.format(os.getpid())
                output = subprocess.check_call(['/usr/bin/osascript', '-e', script])
            root.attributes('-topmost', False)
        else:
            canvas = None
            self.root = None
            
        robot = CrawlingRobot(canvas)
        self.crawlingRobot = robot

        self._stepCount = 0
        self.horizon = horizon

        # def update_gui():
        #     robot.draw(self.stepCount, )
        # update_gui()

        # The state is of the form (armAngle, handAngle)
        # where the angles are bucket numbers, not actual
        # degree measurements
        self.state = None

        self.nArmStates = 9
        self.nHandStates = 13

        # create a list of arm buckets and hand buckets to
        # discretize the state space
        minArmAngle, maxArmAngle = self.crawlingRobot.getMinAndMaxArmAngles()
        minHandAngle, maxHandAngle = self.crawlingRobot.getMinAndMaxHandAngles()
        armIncrement = (maxArmAngle - minArmAngle) / (self.nArmStates-1)
        handIncrement = (maxHandAngle - minHandAngle) / (self.nHandStates-1)
        self.armBuckets = [minArmAngle+(armIncrement*i) \
                           for i in range(self.nArmStates)]
        self.handBuckets = [minHandAngle+(handIncrement*i) \
                            for i in range(self.nHandStates)]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            [spaces.Discrete(self.nArmStates), spaces.Discrete(self.nHandStates)]
        )

        # Reset
        self.reset()

    @property
    def stepCount(self):
        return self._stepCount

    @stepCount.setter
    def stepCount(self, val):
        self._stepCount = val
        self.crawlingRobot.draw(val, self.root)

    def _legal_actions(self, state):
        """
          Returns possible actions
          for the states in the
          current state
        """

        actions = list()

        currArmBucket,currHandBucket = state
        if currArmBucket > 0: actions.append(0)
        if currArmBucket < self.nArmStates-1: actions.append(1)
        if currHandBucket > 0: actions.append(2)
        if currHandBucket < self.nHandStates-1: actions.append(3)

        return actions

    def step(self, a):
        """
          Returns:
            s, r, d, info
        """
        if self.stepCount >= self.horizon:
            raise Exception("Horizon reached")
        nextState, reward = None, None

        oldX, oldY = self.crawlingRobot.getRobotPosition()
        armBucket, handBucket = self.state

        if a in self._legal_actions(self.state):
            if a == 0:
                newArmAngle = self.armBuckets[armBucket-1]
                self.crawlingRobot.moveArm(newArmAngle)
                nextState = (armBucket-1,handBucket)
            elif a == 1:
                newArmAngle = self.armBuckets[armBucket+1]
                self.crawlingRobot.moveArm(newArmAngle)
                nextState = (armBucket+1,handBucket)
            elif a == 2:
                newHandAngle = self.handBuckets[handBucket-1]
                self.crawlingRobot.moveHand(newHandAngle)
                nextState = (armBucket,handBucket-1)
            elif a == 3:
                newHandAngle = self.handBuckets[handBucket+1]
                self.crawlingRobot.moveHand(newHandAngle)
                nextState = (armBucket,handBucket+1)
            else:
                raise Exception("action out of range")
        else:
            nextState = self.state

        newX, newY = self.crawlingRobot.getRobotPosition()

        # a simple reward function
        reward = newX - oldX

        self.state = nextState
        self.stepCount += 1

        return tuple(nextState), reward, self.stepCount >= self.horizon, {}


    def reset(self):
        """
         Resets the Environment to the initial state
        """
        ## Initialize the state to be the middle
        ## value for each parameter e.g. if there are 13 and 19
        ## buckets for the arm and hand parameters, then the intial
        ## state should be (6,9)
        ##
        ## Also call self.crawlingRobot.setAngles()
        ## to the initial arm and hand angle

        armState = self.nArmStates // 2
        handState = self.nHandStates // 2
        self.state = armState, handState
        self.crawlingRobot.setAngles(self.armBuckets[armState], self.handBuckets[handState])
        self.crawlingRobot.positions = [20, self.crawlingRobot.getRobotPosition()[0]]

        self.stepCount = 0

class CrawlingRobot:

    def __init__(self, canvas):

        ## Canvas ##
        self.canvas = canvas
        self.velAvg = 0
        self.lastStep = 0

        ## Arm and Hand Degrees ##
        self.armAngle = self.oldArmDegree = 0.0
        self.handAngle = self.oldHandDegree = -PI/6

        self.maxArmAngle = PI/6
        self.minArmAngle = -PI/6

        self.maxHandAngle = 0
        self.minHandAngle = -(5.0/6.0) * PI

        self.robotWidth = 80
        self.robotHeight = 40
        self.armLength = 60
        self.handLength = 40
        self.positions = [0,0]

        ## Draw Ground ##
        if canvas is not None:
            self.totWidth = canvas.winfo_reqwidth()
            self.totHeight = canvas.winfo_reqheight()
            self.groundHeight = 40
            self.groundY = self.totHeight - self.groundHeight

            self.ground = canvas.create_rectangle(
                0,
                self.groundY,self.totWidth,self.totHeight, fill='blue'
            )

            ## Robot Body ##
            self.robotPos = (self.totWidth / 5 * 2, self.groundY)
            self.robotBody = canvas.create_polygon(0,0,0,0,0,0,0,0, fill='green')

            ## Robot Arm ##
            self.robotArm = canvas.create_line(0,0,0,0,fill='orange',width=5)

            ## Robot Hand ##
            self.robotHand = canvas.create_line(0,0,0,0,fill='red',width=3)

            # canvas.focus_force()

        else:
            self.robotPos = (20, 0)

    def setAngles(self, armAngle, handAngle):
        """
            set the robot's arm and hand angles
            to the passed in values
        """
        self.armAngle = armAngle
        self.handAngle = handAngle

    def getAngles(self):
        """
            returns the pair of (armAngle, handAngle)
        """
        return self.armAngle, self.handAngle

    def getRobotPosition(self):
        """
            returns the (x,y) coordinates
            of the lower-left point of the
            robot
        """
        return self.robotPos

    def moveArm(self, newArmAngle):
        """
            move the robot arm to 'newArmAngle'
        """
        oldArmAngle = self.armAngle
        if newArmAngle > self.maxArmAngle:
            raise 'Crawling Robot: Arm Raised too high. Careful!'
        if newArmAngle < self.minArmAngle:
            raise 'Crawling Robot: Arm Raised too low. Careful!'
        disp = self.displacement(self.armAngle, self.handAngle,
                                 newArmAngle, self.handAngle)
        curXPos = self.robotPos[0]
        self.robotPos = (curXPos+disp, self.robotPos[1])
        self.armAngle = newArmAngle

        # Position and Velocity Sign Post
        self.positions.append(self.getRobotPosition()[0])
        #        self.angleSums.append(abs(math.degrees(oldArmAngle)-math.degrees(newArmAngle)))
        if len(self.positions) > 100:
            self.positions.pop(0)
            #           self.angleSums.pop(0)

    def moveHand(self, newHandAngle):
        """
            move the robot hand to 'newArmAngle'
        """
        oldHandAngle = self.handAngle

        if newHandAngle > self.maxHandAngle:
            raise 'Crawling Robot: Hand Raised too high. Careful!'
        if newHandAngle < self.minHandAngle:
            raise 'Crawling Robot: Hand Raised too low. Careful!'
        disp = self.displacement(self.armAngle, self.handAngle, self.armAngle, newHandAngle)
        curXPos = self.robotPos[0]
        self.robotPos = (curXPos+disp, self.robotPos[1])
        self.handAngle = newHandAngle

        # Position and Velocity Sign Post
        self.positions.append(self.getRobotPosition()[0])
        #       self.angleSums.append(abs(math.degrees(oldHandAngle)-math.degrees(newHandAngle)))
        if len(self.positions) > 100:
            self.positions.pop(0)
            #           self.angleSums.pop(0)

    def getMinAndMaxArmAngles(self):
        """
            get the lower- and upper- bound
            for the arm angles returns (min,max) pair
        """
        return self.minArmAngle, self.maxArmAngle

    def getMinAndMaxHandAngles(self):
        """
            get the lower- and upper- bound
            for the hand angles returns (min,max) pair
        """
        return self.minHandAngle, self.maxHandAngle

    def getRotationAngle(self):
        """
            get the current angle the
            robot body is rotated off the ground
        """
        armCos, armSin = self.__getCosAndSin(self.armAngle)
        handCos, handSin = self.__getCosAndSin(self.handAngle)
        x = self.armLength * armCos + self.handLength * handCos + self.robotWidth
        y = self.armLength * armSin + self.handLength * handSin + self.robotHeight
        if y < 0:
            return math.atan(-y/x)
        return 0.0


    ## You shouldn't need methods below here


    def __getCosAndSin(self, angle):
        return math.cos(angle), math.sin(angle)

    def displacement(self, oldArmDegree, oldHandDegree, armDegree, handDegree):

        oldArmCos, oldArmSin = self.__getCosAndSin(oldArmDegree)
        armCos, armSin = self.__getCosAndSin(armDegree)
        oldHandCos, oldHandSin = self.__getCosAndSin(oldHandDegree)
        handCos, handSin = self.__getCosAndSin(handDegree)

        xOld = self.armLength * oldArmCos + self.handLength * oldHandCos + self.robotWidth
        yOld = self.armLength * oldArmSin + self.handLength * oldHandSin + self.robotHeight

        x = self.armLength * armCos + self.handLength * handCos + self.robotWidth
        y = self.armLength * armSin + self.handLength * handSin + self.robotHeight

        if y < 0:
            if yOld <= 0:
                return math.sqrt(xOld*xOld + yOld*yOld) - math.sqrt(x*x + y*y)
            return (xOld - yOld*(x-xOld) / (y - yOld)) - math.sqrt(x*x + y*y)
        else:
            if yOld  >= 0:
                return 0.0
            return -(x - y * (xOld-x)/(yOld-y)) + math.sqrt(xOld*xOld + yOld*yOld)

        raise 'Never Should See This!'

    def draw(self, stepCount, root):
        if self.canvas is None or root is None:
            return
        x1, y1 = self.getRobotPosition()
        x1 = x1 % self.totWidth

        ## Check Lower Still on the ground
        if y1 != self.groundY:
            raise 'Flying Robot!!'

        rotationAngle = self.getRotationAngle()
        cosRot, sinRot = self.__getCosAndSin(rotationAngle)

        x2 = x1 + self.robotWidth * cosRot
        y2 = y1 - self.robotWidth * sinRot

        x3 = x1 - self.robotHeight * sinRot
        y3 = y1 - self.robotHeight * cosRot

        x4 = x3 + cosRot*self.robotWidth
        y4 = y3 - sinRot*self.robotWidth

        self.canvas.coords(self.robotBody,x1,y1,x2,y2,x4,y4,x3,y3)

        armCos, armSin = self.__getCosAndSin(rotationAngle+self.armAngle)
        xArm = x4 + self.armLength * armCos
        yArm = y4 - self.armLength * armSin

        self.canvas.coords(self.robotArm,x4,y4,xArm,yArm)

        handCos, handSin = self.__getCosAndSin(self.handAngle+rotationAngle)
        xHand = xArm + self.handLength * handCos
        yHand = yArm - self.handLength * handSin

        self.canvas.coords(self.robotHand,xArm,yArm,xHand,yHand)


        # Position and Velocity Sign Post
        #        time = len(self.positions) + 0.5 * sum(self.angleSums)
        #        velocity = (self.positions[-1]-self.positions[0]) / time
        #        if len(self.positions) == 1: return
        steps = (stepCount - self.lastStep)
        # if steps==0:return
        #       pos = self.positions[-1]
        #        velocity = (pos - self.lastPos) / steps
        #      g = .9 ** (10 * stepDelay)
        #        g = .99 ** steps
        #        self.velAvg = g * self.velAvg + (1 - g) * velocity
        #       g = .999 ** steps
        #       self.velAvg2 = g * self.velAvg2 + (1 - g) * velocity
        pos = self.positions[-1]
        velocity = pos - self.positions[-2]
        vel2 = (pos - self.positions[0]) / len(self.positions)
        self.velAvg = .9 * self.velAvg + .1 * vel2
        velMsg = '100-step Avg Velocity: %.2f' % self.velAvg
        #        velMsg2 = '1000-step Avg Velocity: %.2f' % self.velAvg2
        velocityMsg = 'Velocity: %.2f' % velocity
        positionMsg = 'Position: %2.f' % pos
        stepMsg = 'Step: %d' % stepCount
        if 'vel_msg' in dir(self):
            self.canvas.delete(self.vel_msg)
            self.canvas.delete(self.pos_msg)
            self.canvas.delete(self.step_msg)
            self.canvas.delete(self.velavg_msg)
            #           self.canvas.delete(self.velavg2_msg)
            #       self.velavg2_msg = self.canvas.create_text(850,190,text=velMsg2)
        self.velavg_msg = self.canvas.create_text(650,190,text=velMsg)
        self.vel_msg = self.canvas.create_text(450,190,text=velocityMsg)
        self.pos_msg = self.canvas.create_text(250,190,text=positionMsg)
        self.step_msg = self.canvas.create_text(50,190,text=stepMsg)
        #        self.lastPos = pos
        self.lastStep = stepCount
        root.update()
    #        self.lastVel = velocity
