import numpy as np
import cvxpy as cp
import json
import os

HEALTH_RANGE = 5
ARROWS_RANGE = 4
MATERIALS_RANGE=3
POSITION_RANGE=5
MONSTER_STATES_RANGE=2 
ACTION_RANGE=10

NUM_STATES = HEALTH_RANGE * ARROWS_RANGE * MATERIALS_RANGE * POSITION_RANGE * MONSTER_STATES_RANGE

HEALTH_VALUES = tuple(range(HEALTH_RANGE))
ARROWS_VALUES = tuple(range(ARROWS_RANGE))
MATERIALS_VALUES = tuple(range(MATERIALS_RANGE))
POSITION_VALUES = tuple(range(POSITION_RANGE))
MONSTER_STATE_VALUES=tuple(range(MONSTER_STATES_RANGE))
ACTION_VALUES=tuple(range(ACTION_RANGE))

HEALTH_FACTOR = 25 # 0, 25, 50, 75, 100
ARROWS_FACTOR = 1 # 0, 1, 2, 3
MATERIALS_FACTOR = 1 # 0, 1, 2
POSITION_FACTOR=1 # 0, 1, 2, 3, 4
MONSTER_STATES_FACTOR=1 #0: Ready, 1: Dormant
ACTION_FACTOR=1 

ACTION_SHOOT = 0
ACTION_HIT = 1
ACTION_UP=2
ACTION_DOWN=3
ACTION_RIGHT=4
ACTION_LEFT=5
ACTION_STAY=6
ACTION_GATHER=7
ACTION_CRAFT=8
ACTION_NONE=9

ACTION_NAMES=["SHOOT","HIT","UP","DOWN","RIGHT","LEFT","STAY","GATHER","CRAFT","NONE"]

TEAM = 34
Y = [1/2, 1,2]
COST = -10/Y[TEAM%3]
HashArr=["C","N","E","S","W"]
HashArr1=["R","D"]

class State:
    def __init__(self, num_position , num_materials, num_arrows ,monster_state, enemy_health):
        if (enemy_health not in HEALTH_VALUES) or (num_arrows not in ARROWS_VALUES) or (num_materials not in MATERIALS_VALUES) or (num_position not in POSITION_VALUES) or (monster_state not in MONSTER_STATE_VALUES) :
            #print(num_position, num_materials,num_arrows,monster_state,enemy_health)
            print(num_position)
            raise ValueError
        
        self.position = num_position
        self.materials = num_materials 
        self.arrows = num_arrows 
        self.monsterState = monster_state
        self.health = enemy_health 

    def as_tuple(self):
        return (self.position, self.materials, self.arrows, self.monsterState, self.health)

    def as_list(self):
        return [HashArr[self.position], self.materials, self.arrows,HashArr1[self.monsterState], self.health]
 
    def get_hash(self):
        return (self.position *(MATERIALS_RANGE * ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE) + 
                self.materials *(ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE)+
                self.arrows*(MONSTER_STATES_RANGE* HEALTH_RANGE)+
                self.monsterState * (HEALTH_RANGE)+
                self.health)

    def is_action_valid(self, action):
        if action == ACTION_NONE: # NONE is valid only for terminal states
            return (self.health == 0)

        if self.health == 0: # for terminal states, only NONE is valid
            return False

        
        # Now the state is not terminal

        if action == ACTION_SHOOT:
            return (self.arrows != 0 and self.position!=1 and self.position!=3)

        if action == ACTION_HIT:
            return (self.position==0 or self.position==2)

        if action == ACTION_UP:
            return (self.position==0 or self.position==3)
        
        if action == ACTION_DOWN:
            return (self.position==0 or self.position==1)

        if action == ACTION_RIGHT:
            return (self.position==0 or self.position==4)

        if action == ACTION_LEFT:
            return (self.position==0 or self.position==2)

        if action == ACTION_STAY:
            return (1==1)

        if action == ACTION_GATHER:
            return (self.position==3)

        if action == ACTION_CRAFT:
            return (self.position==1 and self.materials!=0 )

    def actions(self):
        actions = []
        for i in range(ACTION_RANGE):
            if self.is_action_valid(i):
                actions.append(i)
        return actions

    def do(self, action):
        # returns list of (probability, state, flag)

        if action not in self.actions():
            raise ValueError

        # the action is valid

        if action == ACTION_NONE:
            return []

        if action == ACTION_SHOOT:
            if(self.arrows>=1):
                if(self.position==0):#Center

                    if(self.monsterState==1):  #Dormant state
                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,0,1) #MM stays dormant : Success of Action
                        # state2= State( max(state.health-1,0) , state.arrows-1, state.materials,0,0) #MM becomes ready : Success of Action
                        # state3= State( state.health, state.arrows-1, state.materials,0,1) #MM stays dormant : Failure of Action
                        # state4= State( state.health , state.arrows-1, state.materials,0,0) #MM becomes ready : Failure of Action

                        state1= State( 0 ,self.materials, self.arrows-1,    1,  max(self.health-1,0)) 
                        state2= State( 0 ,self.materials, self.arrows-1,    0,  max(self.health-1,0)) 
                        state3= State( 0 ,self.materials, self.arrows-1,    1,  self.health) 
                        state4= State( 0 ,self.materials, self.arrows-1,    0,  self.health) 

                        # ShootCost= (0.5 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.5 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.5 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.5 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                        return [
                            (0.5*0.8 , state1,0),
                            (0.5*0.2 , state2,0),
                            (0.5*0.8 , state3,0),
                            (0.5*0.2 , state4,0)
                        ]
                            

                    elif(self.monsterState==0):  #Ready state
                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,0,0) #MM stays ready : Success of Action
                        # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                        # state3= State( state.health , state.arrows-1, state.materials,0,0) #MM stays ready  : Failure of Action
                        # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                        state1= State( 0 ,self.materials, self.arrows-1,    0, max(self.health-1,0)) 
                        state2= State( 0 ,self.materials, 0,    1, min(self.health+1,4)) 
                        state3= State( 0 ,self.materials, self.arrows - 1,  0, self.health) 
                        state4= State( 0 ,self.materials, 0,    1, min(self.health+1,4))    

                        # ShootCost= (0.5 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.5 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                        # + 0.5 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.5 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                        return [
                            (0.5*0.5 , state1,0),
                            (0.5*0.5 , state2,1),
                            (0.5*0.5 , state3,0),
                            (0.5*0.5 , state4,1)
                        ]

                elif(self.position==2): # East
                    
                    if(self.monsterState==1):  #Dormant state
                
                        state1= State(2, self.materials, self.arrows-1 , 1, max(self.health-1,0) )
                        state2= State(2, self.materials, self.arrows-1 , 0, max(self.health-1,0) )
                        state3= State(2, self.materials, self.arrows-1 , 1, self.health)
                        state4= State(2, self.materials, self.arrows-1 , 0, self.health)


                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,2,1) #MM stays dormant : Success of Action
                        # state2= State( max(state.health-1,0) , state.arrows-1, state.materials,2,0) #MM becomes ready : Success of Action
                        # state3= State( state.health,           state.arrows-1, state.materials,2,1) #MM stays dormant : Failure of Action
                        # state4= State( state.health ,          state.arrows-1, state.materials,2,0) #MM becomes ready : Failure of Action

                        # ShootCost= (0.9 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.9 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.1 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.1 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))
                                
                        return [
                            (0.9*0.8 , state1,0),
                            (0.9*0.2 , state2,0),
                            (0.1*0.8 , state3,0),
                            (0.1*0.2 , state4,0)
                        ]

                    elif(self.monsterState==0):  #Ready state
                        
                        state1= State(2, self.materials, self.arrows-1 , 0, max(self.health-1,0) )
                        state2= State(2, self.materials, 0 , 1, min(self.health+1,4) )
                        state3= State(2, self.materials, self.arrows-1 , 0, self.health)
                        state4= State(2, self.materials, 0 , 1, min(self.health+1,4))

                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,2,0) #MM stays ready : Success of Action
                        # state2= State( min(state.health+1,4) , 0,              state.materials,2,1) #MM attacks and become dormant : UNSUCCESSFUL
                        # state3= State( state.health , state.arrows-1,          state.materials,2,0) #MM stays ready  : Failure of Action
                        # state4= State( min(state.health+1,4) , 0,              state.materials,2,1) #MM attacks and become dormant : UNSUCCESSFUL

                        # ShootCost= (0.9 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.9* 0.5 *(COST + REWARD[state2.show()] - 40 + GAMMA*utilities[state2.show()])
                        # + 0.1 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.1 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                        return [
                            (0.9*0.5 , state1,0),
                            (0.9*0.5 , state2,1),
                            (0.1*0.5 , state3,0),
                            (0.1*0.5 , state4,1)
                        ]
                
                elif(self.position==4): #West

                    if(self.monsterState==1): #Dormant
                        state1=State(4, self.materials, self.arrows-1, 1, max(self.health-1,0))
                        state2=State(4, self.materials, self.arrows-1, 1, self.health)
                        state3=State(4, self.materials, self.arrows-1, 0, max(self.health-1,0))
                        state4=State(4, self.materials, self.arrows-1, 0, self.health)


                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,4,1)# Success of Action
                        # state2= State( state.health, state.arrows-1, state.materials,4,1) # Failure of Action
                        # state3= State( max(state.health-1,0) , state.arrows-1, state.materials,4,0)# Success of Action
                        # state4= State( state.health, state.arrows-1, state.materials,4,0) # Failure of Action
                    
                        # ShootCost= (0.25*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.75*0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.25*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.75*0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                        return [
                            (0.25*0.8 , state1,0),
                            (0.75*0.8 , state2,0),
                            (0.25*0.2 , state3,0),
                            (0.75*0.2 , state4,0)
                          ]

                    elif(self.monsterState==0): #Ready
                        
                        state1=State(4, self.materials, self.arrows-1, 0, max(self.health-1,0))
                        state2=State(4, self.materials, self.arrows-1, 0, self.health)
                        state3=State(4, self.materials, self.arrows-1, 1, max(self.health-1,0))
                        state4=State(4, self.materials, self.arrows-1, 1, self.health)

                        # state1= State( max(state.health-1,0) , state.arrows-1, state.materials,4,0)# Success of Action
                        # state2= State( state.health,           state.arrows-1, state.materials,4,0) # Failure of Action
                        # state3= State( max(state.health-1,0) , state.arrows-1, state.materials,4,1)# Success of Action
                        # state4= State( state.health,           state.arrows-1, state.materials,4,1) # Failure of Action
                    
                        # ShootCost= (0.25*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.75*0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.25*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.75*0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                        return [
                            (0.25*0.5 , state1,0),
                            (0.75*0.5 , state2,0),
                            (0.25*0.5 , state3,0),
                            (0.75*0.5 , state4,0)
                        ]

        if action == ACTION_HIT:

            if(self.position==0): #Center
                if(self.monsterState==1):  #Dormant state
                    # state1= State( max(state.health-2,0) , state.arrows, state.materials,0,1) #MM stays dormant : Success of Action
                    # state2= State( max(state.health-2,0) , state.arrows, state.materials,0,0) #MM becomes ready : Success of Action
                    # state3= State( state.health, state.arrows, state.materials,0,1) #MM stays dormant : Failure of Action 
                    # state4= State( state.health , state.arrows, state.materials,0,0) #MM becomes ready : Failure of Action

                    state1= State(    0,  self.materials,  self.arrows,   1,  max(self.health-2,0)) #MM stays dormant : Success of Action
                    state2= State(    0,  self.materials,  self.arrows,   0,  max(self.health-2,0)) #MM becomes ready : Success of Action
                    state3= State(    0,  self.materials,  self.arrows,   1,  self.health) #MM stays dormant : Failure of Action
                    state4= State(    0,  self.materials,  self.arrows,   0,  self.health) #MM becomes ready : Failure of Action

                    # HitCost= (0.1 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.1 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.9 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.9 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.1*0.8 , state1,0),
                            (0.1*0.2 , state2,0),
                            (0.9*0.8 , state3,0),
                            (0.9*0.2 , state4,0)
                        ]
                            

                elif(self.monsterState==0):  #Ready state
                    # state1= State( max(state.health-2,0) , state.arrows, state.materials,0,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , max(state.arrows-1,0), state.materials,0,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                    state1= State(  0, self.materials,  self.arrows,            0,          max(self.health-2,0)) #MM stays ready : Success of Action
                    state2= State(  0, self.materials,  0 ,                     1 ,                 min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State(  0, self.materials,  max(self.arrows-1,0),   0,        self.health) #MM stays ready  : Failure of Action
                    state4= State(  0, self.materials,  0,                      1,          min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL

                    # HitCost= (0.1 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + ( 0.1 ) * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.9 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + ( 0.9 ) * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.1*0.5 , state1,0),
                            (0.1*0.5 , state2,1),
                            (0.9*0.5 , state3,0),
                            (0.9*0.5 , state4,1)
                        ]

            if(self.position==2): #East
                if(self.monsterState==1):  #Dormant state
                
                    state1= State(2,self.materials, self.arrows, 1, max(self.health-2,0))
                    state2= State(2,self.materials, self.arrows, 0, max(self.health-2,0))
                    state3= State(2,self.materials, self.arrows, 1, self.health)
                    state4= State(2,self.materials, self.arrows, 0, self.health)

                    # state1= State( max(state.health-2,0) , state.arrows, state.materials,2,1) #MM stays dormant : Success of Action
                    # state2= State( max(state.health-2,0) , state.arrows, state.materials,2,0) #MM becomes ready : Success of Action
                    # state3= State( state.health, state.arrows, state.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Failure of Action

                    # HitCost= (0.2 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.2 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.8 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.8 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.2*0.8 , state1,0),
                            (0.2*0.2 , state2,0),
                            (0.8*0.8 , state3,0),
                            (0.8*0.2 , state4,0)
                        ]
                    

                elif(self.monsterState==0):  #Ready state
                    
                    state1= State(2,self.materials, self.arrows, 0, max(self.health-2,0))
                    state2= State(2,self.materials, 0,           1, min(self.health+1,4))
                    state3= State(2,self.materials, self.arrows, 0, self.health)
                    state4= State(2,self.materials, 0,           1, min(self.health+1,4))

                    # state1= State( max(state.health-2,0) , state.arrows, state.materials,2,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,2,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , state.arrows, state.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,2,1) #MM attacks and become dormant : UNSUCCESSFUL

                    # HitCost= (0.2 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + (0.2) * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.8 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + (0.8) * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.2*0.5 , state1,0),
                            (0.2*0.5 , state2,1),
                            (0.8*0.5 , state3,0),
                            (0.8*0.5 , state4,1)
                        ]

        
        if action == ACTION_UP:
            if(self.position==0): #Center
                if(self.monsterState==1):  #Dormant state
                    # state1= State( self.health , self.arrows, self.materials,1,1) #MM stays dormant : Success of Action
                    # state2= State( self.health , self.arrows, self.materials,1,0) #MM becomes ready : Success of Action
                    # state3= State( self.health , self.arrows, self.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( self.health , self.arrows, self.materials,2,0) #MM becomes ready : Failure of Action

                    state1= State( 1 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 1 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # UpCost= (0.85 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.8 *COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))
                            
                    return [
                            (0.85*0.8 , state1,0),
                            (0.85*0.2 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0):  #Ready state
                    # state1= State( self.health , self.arrows, self.materials,1,0) #MM stays ready : Success of Action
                    # state2= State( min(self.health+1,4) , 0, self.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( self.health , self.arrows, self.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(self.health+1,4) , 0, self.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                    state1= State( 1 ,self.materials, self.arrows, 0,self.health) #MM stays ready : Success of Action
                    state2= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State( 2 ,self.materials, self.arrows, 0,self.health) #MM stays ready  : Failure of Action
                    state4= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL

                    # UpCost=( 0.85 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.85*0.5 , state2,1),
                            (0.15*0.5 , state3,0),
                            (0.15*0.5 , state4,1)
                        ]
                

            if(self.position==3): #South
                if(self.monsterState==1): #Dormant

                    state1= State(0, self.materials, self.arrows, 1, self.health)
                    state2= State(2, self.materials, self.arrows, 1, self.health)
                    state3= State(0, self.materials, self.arrows, 0, self.health)
                    state4= State(2, self.materials, self.arrows, 0, self.health)

                    # state1= State( state.health , state.arrows, state.materials,0,1) #Success of Action : Stays in Dormant
                    # state2= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Stays in Dormant
                    # state3= State( state.health , state.arrows, state.materials,0,0) #Success of Action : Becomes ready
                    # state4= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Becomes ready

                    # UpCost=(0.85*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.15*0.8 , state2,0),
                            (0.85*0.2 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0): #Ready
                    
                    state1= State(0, self.materials, self.arrows, 0, self.health)
                    state2= State(2, self.materials, self.arrows, 0, self.health)
                    state3= State(0, self.materials, self.arrows, 1, self.health)
                    state4= State(2, self.materials, self.arrows, 1, self.health)

                    # state1= State( state.health , state.arrows, state.materials,0,0) #Success of Action : Stays Ready
                    # state2= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Stays Ready
                    # state3= State( state.health , state.arrows, state.materials,0,1) #Success of Action : Attack
                    # state4= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Attack

                    # UpCost=(0.85*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.15*0.5 , state2,0),
                            (0.85*0.5 , state3,0),
                            (0.15*0.5 , state4,0)
                        ]

        if action == ACTION_DOWN:
            if(self.position==0):
                if(self.monsterState==1):  #Dormant state
                    state1= State( 3 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 3 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # state1= State( state.health , state.arrows, state.materials,3,1) #MM stays dormant : Success of Action    
                    # state2= State( state.health , state.arrows, state.materials,3,0) #MM becomes ready : Success of Action
                    # state3= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Failure of Action

                    # DownCost=( 0.85 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))
                            
                    return [
                            (0.85*0.8 , state1,0),
                            (0.85*0.2 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0):  #Ready state
                    # state1= State( state.health , state.arrows, state.materials,3,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , state.arrows, state.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                    state1= State( 3 ,self.materials, self.arrows, 0,self.health) #MM stays ready : Success of Action
                    state2= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State( 2 ,self.materials, self.arrows, 0,self.health) #MM stays ready  : Failure of Action
                    state4= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    


                    # DownCost= (0.85 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.85*0.5 , state2,1),
                            (0.15*0.5 , state3,0),
                            (0.15*0.5 , state4,1)
                        ]

            if(self.position==1):
                if(self.monsterState==1): #Dormant
                    # state1= State( state.health , state.arrows, state.materials,0,1) #Success of Action : Stays in Dormant
                    # state2= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Stays in Dormant
                    # state3= State( state.health , state.arrows, state.materials,0,0) #Success of Action : Becomes ready
                    # state4= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Becomes ready

                    state1= State( 0 ,self.materials, self.arrows, 1,self.health) 
                    state2= State( 2 ,self.materials, self.arrows, 1,self.health) 
                    state3= State( 0 ,self.materials, self.arrows, 0,self.health) 
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) 

                    # DownCost=(0.85*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.15*0.8 , state2,0),
                            (0.85*0.2 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0): #Ready
                    # state1= State( state.health , state.arrows, state.materials,0,0) #Success of Action : Stays Ready
                    # state2= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Stays Ready
                    # state3= State( state.health , state.arrows, state.materials,0,1) #Success of Action : Attack
                    # state4= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Attack

                    state1= State( 0 ,self.materials, self.arrows, 0,self.health) 
                    state2= State( 2 ,self.materials, self.arrows, 0,self.health) 
                    state3= State( 0 ,self.materials, self.arrows, 1,self.health) 
                    state4= State( 2 ,self.materials, self.arrows, 1,self.health) 

                    # DownCost=(0.85*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.15*0.5 , state2,0),
                            (0.85*0.5 , state3,0),
                            (0.15*0.5 , state4,0)
                        ]


        if action == ACTION_RIGHT:
            if(self.position==0):
                if(self.monsterState==1):  #Dormant state
                    # state1= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Success of Action
                    # state2= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Success of Action
                    # state3= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Failure of Action

                    state1= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # RightCost= (0.85 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.85*0.2 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]
                            

                elif(self.monsterState==0):  #Ready state
                    # state1= State( state.health , state.arrows, state.materials,2,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , state.arrows, state.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL\


                    state1= State( 2 ,self.materials, self.arrows, 0,self.health) #MM stays ready : Success of Action
                    state2= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State( 2 ,self.materials, self.arrows, 0,self.health) #MM stays ready  : Failure of Action
                    state4= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL

                    # RightCost= (0.85 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.85*0.5 , state2,1),
                            (0.15*0.5 , state3,0),
                            (0.15*0.5 , state4,1)
                        ]

            if(self.position==4):
                #Move Right
                if(self.monsterState==1): #Dormant
                    
                    state1= State(0,self.materials,self.arrows,1,self.health)
                    state2= State(0,self.materials,self.arrows,0,self.health)

                    # state1= State( state.health , state.arrows, state.materials,0,1) # Success of Action
                    # state2= State( state.health , state.arrows, state.materials,0,0) # Success of Action
                    # RightCost= (0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    #              + 0.2*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))

                    return [
                            (0.8 , state1,0),
                            (0.2 , state2,0)
                        ]
                
                elif(self.monsterState==0): #Ready

                    state1=State(0,self.materials,self.arrows,0,self.health)
                    state2=State(0,self.materials,self.arrows,1,self.health)

                    # state1= State( state.health , state.arrows, state.materials,0,0) # Success of Action
                    # state2= State( state.health , state.arrows, state.materials,0,1) # Success of Action
                    
                    # RightCost= (0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    #              + 0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))
                    return [
                            (0.5 , state1,0),
                            (0.5 , state2,0)
                        ]

        if action == ACTION_LEFT:
            
            if(self.position==0):
                if(self.monsterState==1):  #Dormant state
                    # state1= State( state.health , state.arrows, state.materials,4,1) #MM stays dormant : Success of Action
                    # state2= State( state.health , state.arrows, state.materials,4,0) #MM becomes ready : Success of Action
                    # state3= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Failure of Action

                    state1= State( 4 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 4 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # LeftCost= (0.85 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.85*0.2 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]
                            

                elif(self.monsterState==0):  #Ready state
                    # state1= State( state.health , state.arrows, state.materials,4,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , state.arrows, state.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                    state1= State( 4 ,self.materials, self.arrows, 0,self.health) #MM stays ready : Success of Action
                    state2= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State( 2 ,self.materials, self.arrows, 0,self.health) #MM stays ready  : Failure of Action
                    state4= State( 0 ,self.materials, 0, 1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL

                    # LeftCost= (0.85 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.85*0.5 , state2,1),
                            (0.15*0.5 , state3,0),
                            (0.15*0.5 , state4,1)
                        ]

            if(self.position==2):
                #Move Left
                if(self.monsterState==1):  #Dormant state
                    
                    state1=State(0,self.materials, self.arrows, 1, self.health)
                    state2=State(0,self.materials, self.arrows, 0, self.health)


                    # state1= State( state.health , state.arrows, state.materials,0,1) #MM stays dormant : Success of Action
                    # state2= State( state.health , state.arrows, state.materials,0,0) #MM becomes ready : Success of Action

                    # LeftCost= (0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))

                    return [
                            (0.8 , state1,0),
                            (0.2 , state2,0)
                        ]
                    
                            

                elif(self.monsterState==0):  #Ready state
                    
                    state1=State(0,self.materials, self.arrows, 0, self.health)
                    state2=State(2,self.materials, 0, 1, min(self.health+1,4))

                    # state1= State( state.health , state.arrows, state.materials,0,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,2,1) #MM attacks and become dormant : UNSUCCESSFUL

                    # LeftCost= (0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()]))

                    return [
                            (0.5 , state1,0),
                            (0.5 , state2,1)
                        ]


        if action == ACTION_STAY:
            if(self.position==0):
                if(self.monsterState==1):  #Dormant state
                    # state1= State( state.health , state.arrows, state.materials,0,1) #MM stays dormant : Success of Action
                    # state2= State( state.health , state.arrows, state.materials,0,0) #MM becomes ready : Success of Action
                    # state3= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Failure of Action
                    # state4= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Failure of Action

                    state1= State( 0 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 0 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 2 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # StayCost= (0.85 * 0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))
                            
                    return [
                            (0.85*0.8 , state1,0),
                            (0.85*0.2 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.15*0.2  ,state4,0)
                        ]

                elif(self.monsterState==0):  #Ready state
                    # state1= State( state.health , state.arrows, state.materials,0,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL
                    # state3= State( state.health , state.arrows, state.materials,2,0) #MM stays ready  : Failure of Action
                    # state4= State( min(state.health+1,4) , 0, state.materials,0,1) #MM attacks and become dormant : UNSUCCESSFUL

                    state1= State( 0 ,self.materials,   self.arrows,                0,self.health) #MM stays ready : Success of Action
                    state2= State( 0 ,self.materials,   0,              1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL
                    state3= State( 2 ,self.materials,   self.arrows,                0,self.health) #MM stays ready  : Failure of Action
                    state4= State( 0 ,self.materials,   0,              1,min(self.health+1,4)) #MM attacks and become dormant : UNSUCCESSFUL

                    # StayCost= (0.85 * 0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.85 * 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15 * 0.5 *(COST + REWARD[state4.show()] -40 + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.85*0.5 , state2,1),
                            (0.15*0.5 , state3,0),
                            (0.15*0.5 , state4,1)
                        ]

            if(self.position==1):
                if(self.monsterState==1): #Dormant
                    # state1= State( state.health , state.arrows, state.materials,1,1) #Success of Action : Stays in Dormant
                    # state2= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Stays in Dormant
                    # state3= State( state.health , state.arrows, state.materials,1,0) #Success of Action : Becomes ready
                    # state4= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Becomes ready

                    state1= State( 1 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Success of Action
                    state2= State( 2 ,self.materials, self.arrows, 1,self.health) #MM becomes ready : Success of Action
                    state3= State( 1 ,self.materials, self.arrows, 0,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Failure of Action

                    # StayCost= (0.85*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.15*0.8 , state2,0),
                            (0.85*0.2 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0): #Ready
                    # state1= State( state.health , state.arrows, state.materials,1,0) #Success of Action : Stays Ready
                    # state2= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Stays Ready
                    # state3= State( state.health , state.arrows, state.materials,1,1) #Success of Action : Attack
                    # state4= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Attack

                    state1= State( 1 ,self.materials, self.arrows, 0,self.health) #MM stays dormant : Success of Action
                    state2= State( 2 ,self.materials, self.arrows, 0,self.health) #MM becomes ready : Success of Action
                    state3= State( 1 ,self.materials, self.arrows, 1,self.health) #MM stays dormant : Failure of Action
                    state4= State( 2 ,self.materials, self.arrows, 1,self.health) #MM becomes ready : Failure of Action

                    # StayCost= (0.85*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.5 , state1,0),
                            (0.15*0.5 , state2,0),
                            (0.85*0.5 , state3,0),
                            (0.15*0.5 , state4,0)
                        ]

            if(self.position==2):
                if(self.monsterState==1):  #Dormant state
                    
                    state1= State(2, self.materials, self.arrows, 1, self.health)
                    state2= State(2, self.materials, self.arrows, 0, self.health)


                    # state1= State( state.health , state.arrows, state.materials,2,1) #MM stays dormant : Success of Action
                    # state2= State( state.health , state.arrows, state.materials,2,0) #MM becomes ready : Success of Action

                    # StayCost= (0.8 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # +0.2 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))
                            
                    return [
                            (0.8 , state1,0),
                            (0.2 , state2,0)
                        ]

                elif(self.monsterState==0):  #Ready state
                    
                    state1= State(2, self.materials, self.arrows, 0, self.health)
                    state2= State(2, self.materials, 0, 1, min(self.health+1,4))

                    # state1= State( state.health , state.arrows, state.materials,2,0) #MM stays ready : Success of Action
                    # state2= State( min(state.health+1,4) , 0, state.materials,2,1)  #MM attacks and become dormant : UNSUCCESSFUL

                    # StayCost= (0.5 *(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.5 *(COST + REWARD[state2.show()] - 40+ GAMMA*utilities[state2.show()]))

                    return [
                            (0.5 , state1,0),
                            (0.5 , state2,1)
                        ]

            if(self.position==3):
                if(self.monsterState==1): #Dormant
                    
                    state1=State(3, self.materials, self.arrows, 1, self.health)
                    state2=State(2, self.materials, self.arrows, 1, self.health)
                    state3=State(3, self.materials, self.arrows, 0, self.health)
                    state4=State(2, self.materials, self.arrows, 0, self.health)

                    # state1= State( state.health , state.arrows, state.materials,3,1) #Success of Action : Stays in Dormant
                    # state2= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Stays in Dormant
                    # state3= State( state.health , state.arrows, state.materials,3,0) #Success of Action : Becomes ready
                    # state4= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Becomes ready

                    # StayCost= (0.85*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.85*0.8 , state1,0),
                            (0.15*0.8 , state2,0),
                            (0.85*0.2 , state3,0),
                            (0.15*0.2 , state4,0)
                        ]

                elif(self.monsterState==0): #Ready
                    
                    state1=State(3, self.materials, self.arrows, 0, self.health)
                    state2=State(2, self.materials, self.arrows, 0, self.health)
                    state3=State(3, self.materials, self.arrows, 1, self.health)
                    state4=State(2, self.materials, self.arrows, 1, self.health)


                    # state1= State( state.health , state.arrows, state.materials,3,0) #Success of Action : Stays Ready
                    # state2= State( state.health , state.arrows, state.materials,2,0) #Failure of Action : Stays Ready
                    # state3= State( state.health , state.arrows, state.materials,3,1) #Success of Action : Attack
                    # state4= State( state.health , state.arrows, state.materials,2,1) #Failure of Action : Attack

                    # StayCost= (0.85*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.15*0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.85*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.15*0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))
                    
                    return [
                            (0.85*0.5 , state1,0),
                            (0.15*0.5 , state2,0),
                            (0.85*0.5 , state3,0),
                            (0.15*0.5 , state4,0)
                        ]

            if(self.position==4):
                if(self.monsterState==1): #Dormant

                    state1=State(4,self.materials,self.arrows,1,self.health)
                    state2=State(4,self.materials,self.arrows,0,self.health)

                    # state1= State( state.health , state.arrows, state.materials,4,1) # Success of Action
                    # state2= State( state.health , state.arrows, state.materials,4,0) # Success of Action
                    # StayCost= (0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    #              + 0.2*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))

                    return [
                            (0.8, state1,0),
                            (0.2 , state2,0)
                        ]

                elif(self.monsterState==0): #Ready

                    state1=State(4,self.materials,self.arrows,0,self.health)
                    state2=State(4,self.materials,self.arrows,1,self.health)

                    # state1= State( state.health , state.arrows, state.materials,4,0) # Success of Action
                    # state2= State( state.health , state.arrows, state.materials,4,1) # Success of Action
                    
                    # StayCost= (0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    #              + 0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()]))

                    return [
                            (0.5 ,state1, 0),
                            (0.5 ,state2, 0)
                        ]

        if action == ACTION_GATHER:
            if(self.position==3):
                if(self.monsterState==1): #Dormant
                    state1=State(3,min(self.materials+1,2), self.arrows,1,self.health)
                    state2=State(3,self.materials, self.arrows,1,self.health)
                    state3=State(3,min(self.materials+1,2), self.arrows,0,self.health)
                    state4=State(3,self.materials, self.arrows,0,self.health)

                    # state1= State( state.health , state.arrows, min(state.materials+1,2),3,1)
                    # state2= State( state.health , state.arrows, state.materials,3,1) 
                    # state3= State( state.health , state.arrows, min(state.materials+1,2),3,0)
                    # state4= State( state.health , state.arrows, state.materials,3,0) 
                
                    # GatherCost = (0.75*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.25 *0.8*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.75*0.2*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.25 *0.2*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.75*0.8 , state1,0),
                            (0.25*0.8 , state2,0),
                            (0.75*0.2 , state3,0),
                            (0.25*0.2 , state4,0)
                        ]

                elif(self.monsterState==0):# Ready

                    state1=State(3,min(self.materials+1,2), self.arrows,0,self.health)
                    state2=State(3,self.materials, self.arrows,0,self.health)
                    state3=State(3,min(self.materials+1,2), self.arrows,1,self.health)
                    state4=State(3,self.materials, self.arrows,1,self.health)


                    # state1= State( state.health , state.arrows, min(state.materials+1,2),3,0)
                    # state2= State( state.health , state.arrows, state.materials,3,0) 
                    # state3= State( state.health , state.arrows, min(state.materials+1,2),3,1)
                    # state4= State( state.health , state.arrows, state.materials,3,1) 
                
                    # GatherCost = (0.75*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                    # + 0.25 *0.5*(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                    # + 0.75*0.5*(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                    # + 0.25 *0.5*(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()]))

                    return [
                            (0.75*0.5 , state1,0),
                            (0.25*0.5 , state2,0),
                            (0.75*0.5 , state3,0),
                            (0.25*0.5 , state4,0)
                        ]


        if action == ACTION_CRAFT:
            if(self.position==1):
                if(self.materials>=1):
                    if(self.monsterState==1): #Dormant
                        
                        state1=State(1, self.materials-1,min(3,self.arrows+1),1,self.health )
                        state2=State(1, self.materials-1,min(3,self.arrows+2),1,self.health )
                        state3=State(1, self.materials-1,min(3,self.arrows+3),1,self.health )
                        state4=State(1, self.materials-1,min(3,self.arrows+1),0,self.health )
                        state5=State(1, self.materials-1,min(3,self.arrows+2),0,self.health )
                        state6=State(1, self.materials-1,min(3,self.arrows+3),0,self.health )

                        # state1= State( state.health , min(3,state.arrows+1), state.materials-1,1,1)
                        # state2= State( state.health , min(3,state.arrows+2), state.materials-1,1,1) 
                        # state3= State( state.health , min(3,state.arrows+3), state.materials-1,1,1)
                        # state4= State( state.health , min(3,state.arrows+1), state.materials-1,1,0)
                        # state5= State( state.health , min(3,state.arrows+2), state.materials-1,1,0) 
                        # state6= State( state.health , min(3,state.arrows+3), state.materials-1,1,0)
                        # CraftCost= (0.5*0.8*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.35 *0.8 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.15 *0.8 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.5 * 0.2 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()])
                        # + 0.35 *0.2*(COST + REWARD[state5.show()] + GAMMA*utilities[state5.show()])
                        # + 0.15 *0.2*(COST + REWARD[state6.show()] + GAMMA*utilities[state6.show()]))

                        return [
                            (0.5*0.8 , state1,0),
                            (0.35*0.8 , state2,0),
                            (0.15*0.8 , state3,0),
                            (0.5*0.2 , state4,0),
                            (0.35*0.2 , state5,0),
                            (0.15*0.2 , state6,0)
                        ]

                    elif(self.monsterState==0): #Ready

                        state1=State(1, self.materials-1,min(3,self.arrows+1),0,self.health )
                        state2=State(1, self.materials-1,min(3,self.arrows+2),0,self.health )
                        state3=State(1, self.materials-1,min(3,self.arrows+3),0,self.health )
                        state4=State(1, self.materials-1,min(3,self.arrows+1),1,self.health )
                        state5=State(1, self.materials-1,min(3,self.arrows+2),1,self.health )
                        state6=State(1, self.materials-1,min(3,self.arrows+3),1,self.health )

                        # state1= State( state.health , min(3,state.arrows+1), state.materials-1,1,0)
                        # state2= State( state.health , min(3,state.arrows+2), state.materials-1,1,0) 
                        # state3= State( state.health , min(3,state.arrows+3), state.materials-1,1,0)
                        # state4= State( state.health , min(3,state.arrows+1), state.materials-1,1,1)
                        # state5= State( state.health , min(3,state.arrows+2), state.materials-1,1,1) 
                        # state6= State( state.health , min(3,state.arrows+3), state.materials-1,1,1)
                        # CraftCost= (0.5*0.5*(COST + REWARD[state1.show()] + GAMMA*utilities[state1.show()])
                        # + 0.35 *0.5 *(COST + REWARD[state2.show()] + GAMMA*utilities[state2.show()])
                        # + 0.15 *0.5 *(COST + REWARD[state3.show()] + GAMMA*utilities[state3.show()])
                        # + 0.5 * 0.5 *(COST + REWARD[state4.show()] + GAMMA*utilities[state4.show()])
                        # + 0.35 *0.5*(COST + REWARD[state5.show()] + GAMMA*utilities[state5.show()])
                        # + 0.15 *0.5*(COST + REWARD[state6.show()] + GAMMA*utilities[state6.show()]))

                        return [
                            (0.5*0.5 ,  state1,0),
                            (0.35*0.5 , state2,0),
                            (0.15*0.5 , state3,0),
                            (0.5*0.5,   state4,0),
                            (0.35*0.5 , state5,0),
                            (0.15*0.5 , state6,0)
                        ]
    

    @classmethod
    def from_hash(self, num):
        if type(num) != int:
            raise ValueError

        if not (0 <= num < NUM_STATES):
            raise ValueError
        
        # (self.position *(MATERIALS_RANGE * ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE) + 
        #         self.materials *(ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE)+
        #         self.arrows*(MONSTER_STATES_RANGE* HEALTH_RANGE)+
        #         self.monsterState * (HEALTH_RANGE)+
        #         self.health)

        # health = num // (ARROWS_RANGE * STAMINA_RANGE)
        # num = num % (ARROWS_RANGE * STAMINA_RANGE)

        # arrows = num // STAMINA_RANGE
        # num = num % STAMINA_RANGE

        # stamina = num

        v_position = num // (MATERIALS_RANGE * ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE )# 3*4*2*5
        num= num % (MATERIALS_RANGE * ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE)

        v_materials=num// (ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE)# 4*2*5
        num= num%(ARROWS_RANGE * MONSTER_STATES_RANGE* HEALTH_RANGE)

        v_arrows=num // (MONSTER_STATES_RANGE* HEALTH_RANGE)# 2*5
        num= num %(MONSTER_STATES_RANGE* HEALTH_RANGE)

        v_monsterState = num // HEALTH_RANGE#5
        num= num %(HEALTH_RANGE)

        v_health =num

        return State(v_position, v_materials, v_arrows, v_monsterState, v_health)


# class Lero:
class IndianaJones:
    def __init__(self):
        self.dim = self.get_dimensions()
        self.r = self.get_r()
        self.a = self.get_a()
        self.alpha = self.get_alpha()
        self.x = self.quest()
        self.policy = []
        self.solution_dict = {}
        self.objective = 0.0
    
    def get_dimensions(self):
        dim = 0
        for i in range(NUM_STATES):
            dim = dim + len(State.from_hash(i).actions())
            # print(i,dim)
        return dim

    def get_a(self):
        a = np.zeros((NUM_STATES, self.dim), dtype=np.float64)
        next_state=[]
        idx = 0
        for i in range(NUM_STATES):
            s = State.from_hash(i)
            actions = s.actions()

            for action in actions:
                a[i][idx] += 1
                next_states = s.do(action)
                
                for next_state in next_states:
                    a[next_state[1].get_hash()][idx] -= next_state[0]
                    if(next_state[2]==1):
                        self.r[0][idx] += (-40)*next_state[0]
                # increment idx
                idx += 1

        return a

    def get_r(self):
        r = np.full((1, self.dim), COST)

        idx = 0
        for i in range(NUM_STATES):
            actions = State.from_hash(i).actions()

            for action in actions:
                if action == ACTION_NONE:
                    r[0][idx] = 0
                idx += 1
        
        return r

    def get_alpha(self):
        alpha = np.zeros((NUM_STATES, 1))
        s = State(0,2,3,0,4).get_hash()
        alpha[s][0] = 1
        return alpha

    def quest(self):
        x = cp.Variable((self.dim, 1), 'x')
        
        constraints = [
            cp.matmul(self.a, x) == self.alpha,
            x >= 0
        ]

        objective = cp.Maximize(cp.matmul(self.r, x))
        problem = cp.Problem(objective, constraints)

        solution = problem.solve()
        self.objective = solution
        arr = list(x.value)
        l = [ float(val) for val in arr]
        return l

    def get_policy(self):
        idx = 0
        for i in range(NUM_STATES):
            s = State.from_hash(i)
            actions = s.actions()
            act_idx = np.argmax(self.x[idx : idx+len(actions)])
            idx += len(actions)
            best_action = actions[act_idx]
            local = []
            local.append(s.as_list())
            local.append(ACTION_NAMES[best_action])
            self.policy.append(local)

    def generate_dict(self):
        self.solution_dict["a"] = self.a.tolist()
        r = [float(val) for val in np.transpose(self.r)]
        self.solution_dict["r"] = r
        alp = [float(val) for val in self.alpha]
        self.solution_dict["alpha"] = alp
        self.solution_dict["x"] = self.x
        self.solution_dict["policy"] = self.policy
        self.solution_dict["objective"] = float(self.objective)
        
    def write_output(self):
        path = "outputs/part_3_output.json"
        json_object = json.dumps(self.solution_dict, indent=4)
        with open(path, 'w+') as f:
          f.write(json_object)

    def execute(self):
        os.makedirs('outputs', exist_ok=True)
        self.quest()
        self.get_policy()
        self.generate_dict()
        self.write_output()    


indianajones = IndianaJones()
indianajones.execute()
