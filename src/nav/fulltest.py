from djitellopy import Tello
import time

MAX_ALTITUDE_CM = 91  # 3 feet = ~91.44 cm = 1 yard
HOVER_INCREASE_CM = 5
MOVE_INCREMENT_CM = 25

tello = Tello()
tello.connect()

tello.takeoff()
time.sleep(2)


# Get current height
current_height = tello.get_height()
print(f"Current height: {current_height} cm")

# Hover up a couple of inches (~5 cm) if under limit
if current_height + HOVER_INCREASE_CM <= MAX_ALTITUDE_CM:
    tello.move_up(HOVER_INCREASE_CM)
    print(f"Moved up {HOVER_INCREASE_CM} cm")
else:
    print("Hover increase would exceed 3 ft limit — skipping")



xPos, yPos = 0,0
# Length and Width of the room = ???
# while abs(xPos) <= length of room and abs(yPos) <= width of room

lengt, width = 8, 10
area = [[{'x': 0, 'y': 0, 'state': "U"} for _ in range(lengt)] for _ in range(width)]

# go thru row
row, col = 0, 0

def spiral_traverse(matrix):
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse top row (from left to right)
        for col in range(left, right + 1):
            result.append(matrix[top][col])
            matrix[top][col].update({'x':top,'y':col,'state':"S"})
            tello.move_forward(MOVE_INCREMENT_CM)
        top += 1
        tello.rotate_clockwise(90)
        

        # Traverse right column (top to bottom)
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
            matrix[row][right].update({'x':row,'y':right,'state':"S"})
            tello.move_forward(MOVE_INCREMENT_CM)
        right -= 1
        tello.rotate_clockwise(90)
        

        if top <= bottom:
            # Traverse bottom row (right to left)
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
                matrix[bottom][col].update({'x':bottom,'y':col,'state':"S"})
                tello.move_forward(MOVE_INCREMENT_CM)
            bottom -= 1
            tello.rotate_clockwise(90)

        if left <= right:
            # Traverse left column (bottom to top)
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
                matrix[row][left].update({'x':row,'y':left,'state':"S"})
                tello.move_forward(MOVE_INCREMENT_CM)
            left += 1
            tello.rotate_clockwise(90)

    return result

# Example usage
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]

print(spiral_traverse(area))





tello.rotate_clockwise(90)
time.sleep(5)
tello.land()
