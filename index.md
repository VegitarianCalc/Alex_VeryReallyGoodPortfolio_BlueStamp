# Custom Object Detection System
A project that can detect objects, using TensorFlow Lite on the Raspberry Pi, adding convienence to your building experience. 


| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Alex N. | Fremont | Mechanical Engineering | Incoming Junior

![Headstone Image](Alex N.jpeg)



  
# Final Milestone


<iframe width="560" height="315" src="https://www.youtube.com/embed/EV1cqBGiJh8?si=Ei2IFFhwSxzjn1fA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

In my third and final milestone, I developed an object detection software using TensorFlow Lite, enabling real-time object detection on the Raspberry Pi. Leveraging Google's TensorFlow GitHub library, I utilized their optimized computer vision model to achieve swift and efficient object detection and computer vision capabilities on the Raspberry Pi.
- I was able to install a couple of python packages, and RPI vision. I then cloned a github repository and it had a data file that it trained a model that could detect objects. After installing tensorflow and running the command to start the object detection, it started working.
- One of the biggest challenges with this final milestone was trying to run the command that would start the detection software. The terminal would say that it’s missing a module, I would install that module using a PIP command and it would say it’s missing a different module. For me to get it to work, I had to set up the virtual environment so all of my packages were in one place.
- I learned a lot in my time at bluestamp. For example, how to code python on a raspberry pi, a basic understanding of neural networks.
- Going beyond BSE I want to keep on working on projects that explore my intrests, so I can learn new things and build off the things that I've learned at Bluestamp. 


# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/muLqT1Jm2Yo?si=g8nGha933mwRVHtQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Since my previous milestone, I have worked on setting up and testing the camera module and the Adafruit Braincraft HAT for my TensorFlow Lite Object Detection project. 
- Properly connected and configured the Raspberry Pi Camera Module 3 to ensure it captures images correctly. Integrated the Adafruit Braincraft HAT to display the terminal output of the Raspberry Pi.
- I encountered challenges in implementing the video feed from the camera to the Adafruit Braincraft HAT. As a solution, I utilized the main monitor on my computer, which ultimately was beneficial due to its better resolution.
- Moving ahead, my focus will be on installing TensorFlow 2 and RPI Vision libraries on the Raspberry Pi, followed by testing the object detection capabilities.

  
# First Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/p8tFRpc52To?si=-ktnJg8QgnNMPHI5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

My project is Raspberry Pi object detection through machine learning. My first milestone involved building the hardware. The camera module will take a picture and run it through the tensorflow lite python interpreter and that will return what it thinks it sees:
- Materials: Computer, Raspberry Pi 4 (64 bit), Web Cam, Adafruit Braincraft HAT.
- While building my hardware I accidentally touched the input and output pins which caused the raspberry pi to short circuit and I had to wait for claudia to order a new Raspberry pi, which made building the hardware a little difficult. 
- Going further, I plan to start work on the first camera test, and getting the display to show what the camera is seeing. 

# Schematics 

<img width="495" alt="Screenshot 2024-07-09 at 1 05 38 PM" src="https://github.com/VegitarianCalc/Alex_VeryReallyGoodPortfolio_BlueStamp/assets/143454271/2aed104b-b061-4afd-bf0e-1d9c497a2063">


# Code

```c++
import os
import cv2
from picamera2 import Picamera2
import numpy as np


# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Load class names
classNames = []
classFile = "/home/alexnicholls/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "/home/alexnicholls/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/alexnicholls/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres,
nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),
confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(),
(box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)),
(box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    try:
        while True:
            # Capture frame-by-frame
            frame = picam2.capture_array()

            # Convert the frame from 4 channels to 3 channels
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Perform object detection
            result, objectInfo = getObjects(frame, 0.45, 0.2)

            # Display the resulting frame
            cv2.imshow('Output', result)

            # Press 'q' on the keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture and close windows
        picam2.stop()
        cv2.destroyAllWindows()
```

<!---# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |

# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.-->

# Starter Project 

<iframe width="560" height="315" src="https://www.youtube.com/embed/dIFbhf59PQw?si=9BH6f1EvKj84Bn3K" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


This is a retro game arcade, you can play SNES tetris with the up, down, left, right buttons and rotational knob, it has a screen to display the score and the blocks alongside a START and STOP button.
- Put simply, all of the input buttons and controls that the player can use will be sent to the micro controller chip in the center and will output movement on the LED display, sound in the peizo buzzer, and the score seen on the top right.
- Something that was particularly challenging was nailing down the soldering, putting the right amount soldering wire on the circuit and lining it up just straight was hard for me. 
- I have made technical advances in understanding how soldering plays a role into assembly of circuits and projects.
