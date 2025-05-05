import cv2
import numpy as np  # Add this line to import NumPy

# Function to draw rectangle and return coordinates


def get_rectangle_coordinates(video_file):
    # Mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        nonlocal pt1, pt2, drawing, rect_points

        if event == cv2.EVENT_LBUTTONDOWN:
            pt1 = (x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                pt2 = (x, y)
                # Define points: top-left, top-right, bottom-right, bottom-left
                rect_points = [(pt1[0], pt1[1]), (pt2[0], pt1[1]),
                               (pt2[0], pt2[1]), (pt1[0], pt2[1])]
        elif event == cv2.EVENT_LBUTTONUP:
            pt2 = (x, y)
            drawing = False
            # Define points: top-left, top-right, bottom-right, bottom-left
            rect_points = [(pt1[0], pt1[1]), (pt2[0], pt1[1]),
                           (pt2[0], pt2[1]), (pt1[0], pt2[1])]
            # Save coordinates when rectangle is drawn
            coordinates.append(rect_points)

    # Global variables
    pt1 = (0, 0)
    pt2 = (0, 0)
    drawing = False
    rect_points = []

    # List to store coordinates
    coordinates = []

    # Read video
    cap = cv2.VideoCapture(video_file)

    # Set mouse callback
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw all saved rectangles on the frame
        for points in coordinates:
            cv2.polylines(frame, [np.array(points)],
                          isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw current rectangle if drawing
        if pt1 and pt2:
            cv2.polylines(frame, [np.array(rect_points)],
                          isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('d'):
            # Save coordinates to a file and exit
            with open('coordinates.txt', 'w') as file:
                for points in coordinates:
                    file.write(f"Coordinates: {points}\n")
            print("Coordinates saved to coordinates.txt")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return the coordinates as a single list
    return coordinates[0] if coordinates else []


# # Example usage:
# # video_file = 'whatever.mp4'
# # coordinates = get_rectangle_coordinates(video_file)
# # print("Coordinates:", coordinates)
