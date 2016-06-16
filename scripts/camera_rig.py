########################################################################
####################### Camera Rig Script ##############################
########################################################################
#
### Directions ###
##
#
## Camera Rig Setup ##
#
# Properties: 
#   ("Player_Mode" = String) values: "3rd_Person" or "1st_Person"
#   - changes the mode of the script
#   - "1st_Person" will disable zoom and distance properties
#
#   ("Free_Move" = Boolean) values: True or False 
#   - if the mouse movement is connected to the rotation of the player
#   - Highly useful for "1st_Person" mode
#
#   ("Button_Switch" = Boolean) values: True or False 
#   - switches the left and right mouse operation
#
#   ("Mouse_Sensitivity" = Integer) how fast or slow the mouse moves
#   ("Mouse_Angle" = Integer) the height angle cap-lock of the mouse move
#   ("Mouse_Zoom" = Float) how quickly the zoom takes place
#   ("Max_Distance" = Integer) the maximum distance from the player
#   ("Min_Distance" = Integer) the minimum distance from the player
#   ("Zoom_FP" = Boolean) whether to zoom past the Min_Distance property
#
#
## Camera Rig - Append Player ##
#
# 1. Download "Camera Rig Blend" from Blender Ignite
# 2. Click "File" and then "Append" in the drop menu
# 2. Navigate to the file and append the "camera_rig" under group
# 3. Open the text editor and open the "camera_rig.py" script
# -  Add motion to the "Player" object with python or logic bricks
#
## Camera Rig - Create Player ##
#
# 1. Creating the rig is a bit difficult to record on the script
# 2. Email me at "ryan.grzesiak126@gmail.com" and I will help you
#
########################################################################

from bge import logic, render, events
import math

#MOUSE_MODE = ['3rd_Person', '1st_Person']
cont = logic.getCurrentController()
own = cont.owner

# Player Mode #
if own.get('Player_Mode'):
    MODE = own.get('Player_Mode')
else:
    MODE = "3rd_Person"
if own.get('Free_Move'):
    FREE_MOVE = own.get('Free_Move')
else:
    FREE_MOVE = False
if own.get('Button_Switch'):
    BUTTTON_SWITCH = own.get('Button_Switch')
else:
    BUTTTON_SWITCH = False
    
# Mouse Settings #
if own.get('Mouse_Sensitivity'):
    SENSITIVITY = own.get('Mouse_Sensitivity') * .001
else:
    SENSITIVITY = 0.003
if own.get('Mouse_Angle'):
    ANGLE_CAP = (own.get('Mouse_Angle') - 90) * (math.pi/180)
else:
    ANGLE_CAP = (170 - 90) * (math.pi/180)
if own.get('Mouse_Zoom'):
    ZOOM = own.get('Mouse_Zoom')
else:
    ZOOM = 2
if own.get('Max_Distance'):
    MAX_DIST = own.get('Max_Distance')
else:
    MAX_DIST = 30
if own.get('Min_Distance'):
    MIN_DIST = own.get('Min_Distance')
else:
    MIN_DIST = 0.5
if own.get('Min_Distance'):
    FP_MODE = own.get('Zoom_FP')
else:
    FP_MODE = True
    

def run():
    cont = logic.getCurrentController()
    own = cont.owner
    
    ray_front = cont.sensors['rayFront']
    ray_back = cont.sensors['rayBack']    
    mouse = cont.sensors['Mouse']
    mouse_event = logic.mouse.events
    
    initialize_var(own)
    
    '''
    ### Wall Hit ###
    ZSMOOTH = .5
    deative = logic.KX_SENSOR_JUST_DEACTIVATED
    back_hit = own.getDistanceTo(ray_back.hitPosition)
    front_hit = own.getDistanceTo(ray_front.hitPosition)
    num = 0
    if str(ray_front.hitObject) != "Player":        
        own['wsmooth'] = own['wsmooth'] + ZSMOOTH
        
    elif own.localPosition[1] > own['dist'] and back_hit > 1:
        own['wsmooth'] = - ZSMOOTH
    
    
    if abs(own['wsmooth']) > .1:
        own['wsmooth'] = own['wsmooth'] / 2
        own.localPosition[1] += own['wsmooth']
        print(own['wsmooth'])
    else:
        own['wsmooth'] = 0
    '''
    
    # Button Switch #
    left_button = right_button = 0
    if BUTTTON_SWITCH == False:
        left_button = events.LEFTMOUSE
        right_button = events.RIGHTMOUSE
    else:
        left_button = events.RIGHTMOUSE
        right_button = events.LEFTMOUSE
    

    ### Mouse Movement ###
    
    if MODE == "1st_Person":
        if own.localPosition[1] != 0:
            own.localPosition[1] = 0
        if FREE_MOVE == True:
            player_rotate(own, mouse)
        elif FREE_MOVE == False:
            if mouse_event[left_button]:
                camera_rotate(own, mouse)
            elif mouse_event[right_button]:
                player_rotate(own, mouse)
            else:
                static_rotate(own, mouse)
            
    elif MODE == "3rd_Person":
        camera_zoom(own, mouse_event)
        if FREE_MOVE == True:
            player_rotate(own, mouse)
        elif FREE_MOVE == False:
            if mouse_event[left_button]:
                camera_rotate(own, mouse)
            elif mouse_event[right_button]:
                player_rotate(own, mouse)
            else:
                static_rotate(own, mouse)
                
    else:
        print("Please select a Camera Mode!")
        
    
### Functions ###
#
def camera_rotate(own, mouse):
    movement = mouse_move(own, mouse)
    move_cap = angle_cap(own, movement[1])
    own['snap'] -= movement[0]
    player_rot(move_cap, movement[0], 0)
  
def player_rotate(own, mouse):
    movement = mouse_move(own, mouse)        
    move_cap = angle_cap(own, movement[1])
    snap = camera_snap(own)
    player_rot(move_cap, snap, movement[0])

def static_rotate(own, mouse):
    if own['cur_pos'] != 0:
        render.setMousePosition(*own['cur_pos'])
        own['cur_pos'] = 0.0
    render.showMouse(True)
    own['init'] = False
    player_rot(0, 0, 0)

    
def initialize_var(own):
    if not 'init' in own:
        own['init'] = False
        own['cur_pos'] = 0.0
        own['snap_start'] = False
        own['snap'] = 0.0
        own['fp'] = False
        own['dist'] = own.localPosition[1]
        own['cam_degr'] = 0.0
        own['zsmooth'] = 0.0
        own['wsmooth'] = 0.0
        x = render.getWindowWidth() // 2
        y = render.getWindowHeight() // 2
        own['win_size'] = x, y
        render.setMousePosition(*own['win_size'])
        
def mouse_move(own, mouse):
    if own['cur_pos'] == 0:
        own['cur_pos'] = mouse.position
    render.showMouse(False)
    render.setMousePosition(*own['win_size'])
    x_pos = (own['win_size'][0] - mouse.position[0]) * SENSITIVITY
    y_pos = (own['win_size'][1] - mouse.position[1]) * SENSITIVITY
    if own['init'] == False:
        x_pos = y_pos = 0
        own['init'] = True
    return x_pos, y_pos
    
def angle_cap(own, move):
    if abs(own['cam_degr']) <= ANGLE_CAP:
        return move
    elif own['cam_degr'] > ANGLE_CAP and move < 0:
        return move
    elif own['cam_degr'] < ANGLE_CAP and move > 0:
        return move
    else:
        return 0
        
def camera_zoom(own, mouse_event):
    dist = own.localPosition[1]
    zoom_in = events.WHEELUPMOUSE
    zoom_out = events.WHEELDOWNMOUSE
    
    # Zoom In #    
    if mouse_event[zoom_in]:
        if abs(ZOOM + own['zsmooth'] + dist) > MIN_DIST:
            own['zsmooth'] += ZOOM
        elif FP_MODE == True and own['fp'] == False:
            own['zsmooth'] = own.localPosition[1] * -1
            own['fp'] = True
        elif FP_MODE == False:
            own['zsmooth'] = -(MIN_DIST + dist)
            
    # Zoom Out #
    elif mouse_event[zoom_out]:
        if own['fp'] == True:
            own['zsmooth'] -= MIN_DIST / 2
            own['fp'] = False
        elif abs(-ZOOM + own['zsmooth'] + dist) < MAX_DIST:
            own['zsmooth'] -= ZOOM
        elif dist < MAX_DIST:
            own['zsmooth'] = -(MAX_DIST + dist)    
    
    ## Zoom Smoothing ##
    if abs(own['zsmooth']) > .1:
        own['zsmooth'] = own['zsmooth'] / 2
        own.localPosition[1] += own['zsmooth']
        own['dist'] = own.localPosition[1]
    else:
        own['zsmooth'] = 0
    
def camera_snap(own):
    
    if own['snap'] != 0:
        own['snap_start'] = True
    if own['snap_start'] == True and own['snap'] > .1:
        own['snap'] = own['snap'] / 2
        return own['snap']
    # Smooth Snap #
    elif own['snap_start'] == True:
        own['snap_start'] = False
        snap = own['snap']
        own['snap'] = 0
        return snap
    else:
        return 0    
    
def player_rot(height, free, width):
    cont = logic.getCurrentController()
    own = cont.owner
    cam_height = cont.actuators['cubeHeight']
    cam_free = cont.actuators['freeRot']
    cam_width = cont.actuators['staticRot']
    cont.activate(cam_height)
    cont.activate(cam_free)
    cont.activate(cam_width)
    cam_height.dRot = [height,0,0]
    cam_free.dRot = [0,0,free]
    cam_width.dRot = [0,0,width]
    own['cam_degr'] += height
    