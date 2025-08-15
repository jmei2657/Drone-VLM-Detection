from djitellopy import Tello
import math
import time

# [!!!] This is partially coded, I will fill in the appropriate areas tommorrow-- but the logic is basically all there (I think?).


# Step 1: Define area and vars
lengt, width = 25, 25
area = [["Unsearched" for _ in range(lengt)] for _ in range(width)]
position = {'x': 0, 'y': 0, 'heading': 0}
corner = ""
heading = 0
heading_radians = math.radians(position['heading'])
bop = True

# Step 2: Check surroundings
#       - Identify and fly to the closest wall
## if caption contains "THIS IS A WALL!"
## move until it reaches a safe-distance (will need to train?)
#       - Rotate, identify, and fly to the next wall
#       - Mark this is as the origin 0,0 on the area grid
def find_wall():
    while vllmpromptfeed !contains "There is a wall" and check_corner == "":
        tello.move_forward(10)
        check_corner(vllmpromptfeed)

#### Check if in a corner and yeah yeah
def check_corner(vvlm_promptin):

    # Checks for walls in all directions
    if promptimo contains WALL: ahead = True
    else: ahead = False
    print(f"Ahead: {ahead}")
    tello.rotate_clockwise(90)

    if promptimo contains WALL: right = True
    else: right = False
    print(f"Right: {right}")
    tello.rotate_clockwise(90)

    if promptimo contains WALL: behind = True
    else: behind = False
    print(f"Behind: {behind}")
    tello.rotate_clockwise(90)

    if promptimo contains WALL: left = True
    else: left = False
    print(f"Left: {left}")
    tello.rotate_clockwise(90)

    # Identifies corner based on ^above^
    if ahead and left:      # Top Left (preferred)
        position = "0,0"
        origin = "0,0"
        heading = 90
        corner = "TL"
        print("TOP LEFT CORNER")
        tello.rotate_clockwise(90)
    elif ahead and right:   # Top Right
        position = "0,25"
        origin = "0,25"
        heading = 180
        corner = "TR"
        print("TOP RIGHT CORNER")
        tello.rotate_counter_clockwise(90)
    elif behind and left:   # Bottom Left
        position = "25,0"
        origin = "25,0"
        heading = 270
        corner = "BL"
        print("BOTTOM LEFT CORNER")
        # do not rotate
    elif behind and right:  # Bottom Right
        position = "25,25"
        origin = "25,25"
        heading = 360
        corner = "BR"
        print("BOTTOM RIGHT CORNER")
        tello.rotate_counter_clockwise(90)
    else:
        corner = ""

    return corner
    
def move_inner():
    # If current position is already visited, move inwards and continue scanning.
    # The movement pattern should look like a peppermint swirl, but as a square.


    if area[position['x']][position['y']] == "Searched":
        corner = check_corner(vlvlm)
        if corner == "TL":
            # Here's what should happen:
            # 1. The drone should rotate inwards
            # - update heading        
            tello.rotate_clockwise(135)
            position['heading'] += 135
            # 2. The drone should move forwards 10cm
            # - update position
            start = getCurrentTime
            tello.move_forward(10)
            # wait until drone is not moving, idk how to write this check
            end = getCurrentTime

            bad_velocity = 10 / start-end
            position['x'] += bad_velocity * math.cos(heading_radians)
            position['y'] += bad_velocity * math.sin(heading_radians)
            # 3. The drone should rotate back to the way it was originally facing
            # - update heading
            tello.rotate_counter_clockwise(135)
            position['heading'] -= 135 #Good

        elif corner == "TR":
            tello.rotate_counter_clockwise(135) # 1
            position['heading'] -= 135
            start = getCurrentTime      # 2
            tello.move_forward(10)
            # wait until drone is not moving, idk how to write this check
            end = getCurrentTime

            bad_velocity = 10 / start-end
            position['x'] += bad_velocity * math.cos(heading_radians)
            position['y'] += bad_velocity * math.sin(heading_radians)
            
            tello.rotateclockwise(135)  # 3
            position['heading'] += 135  #Good

        elif corner == "BL":
            tello.rotate_clockwise(45) # 1
            position['heading'] += 45
            start = getCurrentTime      # 2
            tello.move_forward(10)
            # wait until drone is not moving, idk how to write this check
            end = getCurrentTime

            bad_velocity = 10 / start-end
            position['x'] += bad_velocity * math.cos(heading_radians)
            position['y'] += bad_velocity * math.sin(heading_radians)
            
            tello.rotate_counter_clockwise(45)    # 3
            position['heading'] -= 45

        elif corner == "BR":
            tello.rotate_counter_clockwise(45) # 1
            position['heading'] -= 45
            start = getCurrentTime      # 2
            tello.move_forward(10)
            # wait until drone is not moving, idk how to write this check
            end = getCurrentTime

            bad_velocity = 10 / start-end
            position['x'] += bad_velocity * math.cos(heading_radians)
            position['y'] += bad_velocity * math.sin(heading_radians)
            
            tello.rotate_clockwise(45)   # 3
            position['heading'] += 45

            # 4. The drone should continue the sweep (the function ends)
    else:
        print("Somehow thought I was in a searched area, it was not.")

def update_heading(heading):
    return math.radians(heading)

# Step 3: Move forward by XYZ cm and scan for threats
# tello.move_forward(10)
def move_and_update_position(vllm_feed):
    start = getCurrentTime
    tello.move_forward(10)
    # wait until drone is not moving, idk how to write this check
    end = getCurrentTime

    bad_velocity = 10 / start-end
    position['x'] += bad_velocity * math.cos(heading_radians)
    position['y'] += bad_velocity * math.sin(heading_radians)

    # # # area[position['x']][position['y']] = vvlmfeed (result of what vllm sees)

    # update area grid with the result of what the vllm has recognized
    #       - If no threat, mark area as searched and continue
    if vvlmfeed contains "none" etc:
        area[position['x']][position['y']] = "Searched"
    #       - If threat, mark area with a flag
    elif vllmfeed contains threat in threats:
        area[position['x']][position['y']] = "Flagged"
    #       - If wall, run corner check and move appropriately
    if check_corner(vllm_feeder) in ["TL","TR","BR","BL"] and :
        move_inner()

# Step 4: Repeat until all spaces in the array are either flagged or searched
def main(): # subsitute all of this into main program or import this file
    tello.takeoff()
    while Bop:
        move_and_update_position(vllm)
        # iterate through the area grid, if it does not contain "Unsearched", then Bop = False and break out
    
