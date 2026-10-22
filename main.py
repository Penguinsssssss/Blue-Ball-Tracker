import cv2
import numpy as np

# Open the input MP4 video
input_video = r"C:\Users\benja\Downloads\linear2.mp4"
cap = cv2.VideoCapture(input_video)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Output video
output_video = r"C:\Users\benja\Downloads\out2 linear.mp4"
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# tracking
framedx = 1
startframe = 0
started = False

heights = []
dists = []

# masks
lower_blue = np.array([103, 30, 30])
upper_blue = np.array([132, 235, 235])

lower_red = np.array([14, 29, 250])
upper_red = np.array([26, 55, 255])

# kernel
kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not started:
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(
            mask_red,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        hy = frame.shape[0]

        for i in contours:
            if cv2.contourArea(i) > 1:
                print(framedx)
                started = True

    else:
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(
            mask_blue,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        hy = frame.shape[0]
        fx = frame.shape[1]

        for i in contours:
            if cv2.contourArea(i) > 100:
                #print(cv2.contourArea(i))
                x, y, w, h = cv2.boundingRect(i)

                hy = min(hy, y)
                fx = min(fx, x+w)

        heights.append(hy)
        dists.append(fx)

        if hy < frame.shape[0]:

            cv2.line(
                frame,
                (0, hy),
                (width - 1, hy),
                (255, 0, 255), # brg
                2
            )

        if fx < frame.shape[1]:
        
            cv2.line(
                frame,
                (fx, 0),
                (fx, width - 1),
                (0, 255, 255), # brg
                2
            )

    out.write(frame)

    framedx += 1

cap.release()
out.release()

print(f"Heights: {heights}")
print(f"Dists: {dists}")
print("Saved to:", output_video)
