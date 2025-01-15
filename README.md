This project is a simple real time attendance project

Project Overview

The real-time attendance project is designed to automate the process of tracking and recording attendance using advanced technologies such as facial recognition, computer vision, and database management. The system aims to provide a seamless and efficient method for attendance management, reducing the need for manual record-keeping and minimizing human error. This project is ideal for educational institutions, workplaces, and events where accurate and quick attendance tracking is essential.

Key Components
Facial Recognition Technology:

Face Detection: Identifies and isolates faces from a live video stream using pre-trained models like Haar Cascades, DLIB, or MTCNN.
Face Recognition: Matches the detected face against a stored database of known individuals using deep learning models, such as those built with TensorFlow or Keras.

Camera Integration:

Utilizes a webcam or IP camera to capture real-time video feeds.
Ensures continuous monitoring and detection of faces in various environmental conditions.
Database Management:

MySQL or other RDBMS: Stores user information and attendance records securely.
CRUD Operations: Handles Create, Read, Update, and Delete operations for managing user data and attendance logs.

User Interface:

Graphical User Interface (GUI): Provides a user-friendly interface for administrators and users to interact with the system, view attendance records, and manage profiles.
Real-Time Feedback: Displays instant feedback on attendance status and recognition results.
Notification System:


Sends alerts or notifications upon successful attendance marking or in case of errors.
Technical Workflow
Initialization:

Load the facial recognition model and user data from the database.
Initialize the camera for capturing video feed.
Face Detection and Recognition:


Continuously capture frames from the camera.
Detect faces in the frame and preprocess the images for recognition.
Use the recognition model to identify individuals and match them against the database.
Attendance Logging:

For each recognized individual, check if attendance has already been marked for the current date.
If not, insert a new record into the database with the person's name, date, and time.
Error Handling:

Handle scenarios such as unrecognized faces, multiple faces, or low confidence in recognition results.
Provide error messages or prompts for manual verification.
Data Retrieval and Reporting:

Allow administrators to retrieve attendance records and generate reports based on various filters like date, user, or department.
Features

Real-Time Processing: Detects and recognizes faces in real-time, ensuring instant attendance marking.

High Accuracy: Utilizes advanced deep learning models to achieve high accuracy in face recognition.

Secure Data Storage: Ensures attendance data is securely stored in a database with controlled access.

Scalability: Designed to handle a large number of users and operate efficiently in high-traffic environments.

Cross-Platform Compatibility: Can be accessed from various devices including PCs, tablets, and smartphones.
Challenges and Considerations

Lighting Conditions: Ensure reliable face detection and recognition under different lighting scenarios.

Privacy and Security: Address privacy concerns related to facial data and secure sensitive information.

Performance Optimization: Optimize the system to handle real-time processing without significant delays.

Scalability: Ensure the system can scale to accommodate an increasing number of users and data entries.
Applications

Educational Institutions: Automate student attendance in classrooms and examinations.

Workplaces: Track employee attendance and working hours efficiently.

Events: Manage entry and attendance at large-scale events or conferences.
This real-time attendance project leverages cutting-edge technology to streamline attendance management, enhancing efficiency, accuracy, and user convenience.

TRAINING YOUR MODEL:

Step1: Search in  web browser google teachable machine
![Screenshot 2025-01-09 234718](https://github.com/user-attachments/assets/bf074ef6-30e9-4563-9b46-3e701bdcef1e)

Step2:Click on Teachable Machine
![Screenshot 2025-01-09 234718](https://github.com/user-attachments/assets/71625c39-873d-4714-8cf2-dadee28d6de4)


Step3:Next click on Get started
![Screenshot 2025-01-09 234913](https://github.com/user-attachments/assets/9bd41d7c-94ff-430a-ae85-2820dc847646)

Step4:Click on image model


![Screenshot 2025-01-09 234950](https://github.com/user-attachments/assets/856f23d6-96f7-46d2-ae30-53bf4631927f)

Step5: Click on Standard image option


![Screenshot 2025-01-09 235016](https://github.com/user-attachments/assets/7b6418d9-5c8d-4c35-9e4f-6358e22f04f8)


Step6: Create your model in Class1 as"your name" and class2 as "unknown"
![Screenshot 2025-01-09 235409](https://github.com/user-attachments/assets/bbd78697-f03a-4a97-ae36-aef0c04f9943)


Step7: Train your model
![Screenshot 2025-01-09 235455](https://github.com/user-attachments/assets/15a21609-9fc8-4a0e-a7b6-2d9eed5a0de6)

Step8:Export your model

Step9: download your "opencv keras model"  in "tensorflow"
![Screenshot 2025-01-09 235553](https://github.com/user-attachments/assets/46c66d3c-6c8a-4c8e-809d-fc02754c48c9)

Step10: place the converted keras file in the project directory as "keras.h5" and "labels.txt" in pycharm


![image](https://github.com/user-attachments/assets/ece970bc-b19b-4c9a-b6d7-df5847ea7668)


Step11:Now create a file in pycharm as "main.py" and enter the code
 ![image](https://github.com/user-attachments/assets/e814bf48-7522-4fcd-bdec-6dd073e33f79)

Step 12: Place the converted keras in project directory


![image](https://github.com/user-attachments/assets/e11dce01-f458-4ef1-aaaf-4104404d2151)

Now Download mysql application in your web browser such as
![image](https://github.com/user-attachments/assets/682336b9-c102-4eb4-8e6c-348bf6d2d403)

Step1: Click on mysql website

Step2: Click on downloads
![image](https://github.com/user-attachments/assets/eed5de00-fdab-41c2-a0e0-3b728fde5235)

Step3: Select community edition and click on it .It occurs on the last page of the webpage
![image](https://github.com/user-attachments/assets/8325e18f-d567-4c11-9e52-abc9acb3ed70)

Step4:Click on mysql installer for windows
![image](https://github.com/user-attachments/assets/106a11cc-348a-4248-88d1-d5983540b777)

Step 5:Click on community edition for download (2nd option)
![image](https://github.com/user-attachments/assets/7b663020-1ec0-4f9f-a4c1-fb5ee37d0ab7)


Step6: start downlaoding randomly without any oracle browser

Step7: after installation setup the mysql server accordingly ("Do the steps correctly")

Step8: open your workbench and create a new query and type the given code:
Code:

create database attendance_system;
USE attendance_system;

CREATE TABLE attendance (
   id INT AUTO_INCREMENT PRIMARY KEY,
   person_name VARCHAR(255),
   attendance INT,
   date Date,
   time time
   
   );
   select * from attendance;

  Step9: and open your pycharm create a new project as "attendance_system" and create a new file as main2.py and run the code

Code:
import mysql.connector
from mysql.connector import Error

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="divya",
        password="divyasri@1606",
        database="attendance_system"
    )
    if conn.is_connected():
        print("Database connected successfully.")
except Error as e:
    print(f"Error: {e}")
    exit()

Step 10: when the database is connected, place the files in directory of the project file
![image](https://github.com/user-attachments/assets/a01f0039-1688-4595-be63-094613ecd57d)

Step11: Then a create a main file in the pycharm as "main.py" and run the code


Code:
import cv2
import numpy as np
from keras.models import load_model
import mysql.connector
from datetime import datetime

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Connect to MySQL database
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="divya",  # Your MySQL username
        password="divyasri@1606",  # Your MySQL password
        database="attendance_system"  # Your database name
    )
    cursor = conn.cursor()
    print("Database connected successfully.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

# CAMERA settings
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Camera not accessible.")
    exit()

# Helper function to mark attendance
def mark_attendance(person_name):
    current_date = datetime.now().strftime('%Y-%m-%d')  # Get today's date
    current_time = datetime.now().strftime('%H:%M:%S')  # Get current time

    try:
        # Check if the person has already been marked for the day
        cursor.execute("SELECT * FROM attendance WHERE person_name = %s AND date = %s", (person_name, current_date))
        record = cursor.fetchone()

        if record:
            print(f"Attendance for {person_name} already marked today.")
        else:
            # Mark attendance if not already marked for today
            cursor.execute(""" 
                INSERT INTO attendance (person_name, attendance, date, time)
                VALUES (%s, %s, %s, %s)
            """, (person_name, 1, current_date, current_time))
            conn.commit()
            print(f"Attendance marked for {person_name} at {current_time}")
    except Exception as e:
        print(f"Error marking attendance: {e}")

# Initialize attendance tracking variables
last_detected_person = ""
last_confidence = 0
already_notified = False  # Ensure it's initialized before use

while True:
    # Grab the webcam image
    ret, image = camera.read()

    if not ret:
        print("Failed to grab image.")
        break

    # Resize image to match model input
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    # Prepare the image for prediction
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1  # Normalize image

    # Get predictions (with verbose=0)
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # If confidence is above 95%, mark attendance
    if confidence_score > 0.90:
        if class_name.lower() != "unknown":  # Ensure "unknown" is not marked as present
            if class_name != last_detected_person:
                print(f"Detected: {class_name}, Confidence: {confidence_score * 100:.2f}%")
                mark_attendance(class_name)
                last_detected_person = class_name
                last_confidence = confidence_score
                already_notified = False  # Reset the flag for the new person
            elif not already_notified:
                print(f"{class_name} is still in front of the camera.")
                already_notified = True  # Prevent duplicate notifications
        else:
            print(f"Unknown detected, not marking attendance.")

    # Listen to the keyboard for presses (press ESC to exit)
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:  # ESC key to quit
        break

# Release resources and close window
camera.release()
cv2.destroyAllWindows()
conn.close()

In a real-time attendance project, several potential errors can arise across different components, from hardware and software integration to algorithmic and database management issues. Below are common expected errors and how to address them:

1. Camera Access Errors
Error: Error: Camera not accessible or cv2.VideoCapture(0) failed to open camera.
Cause: The camera is not connected, in use by another application, or permissions are not granted.
Solution: Ensure the camera is connected and not in use. Check system permissions to allow camera access.

2. Face Detection and Recognition Errors
Error: Low confidence in recognition, or failure to detect faces.
Cause: Poor lighting, obstructions, or unusual angles affecting face detection.
Solution: Improve lighting conditions, ensure the camera is positioned correctly, and consider using more robust face detection algorithms like MTCNN or DLIB.

3. Model Prediction Errors
Error: ValueError: Input to model has incorrect shape.
Cause: Mismatch between the input image dimensions and what the model expects.
Solution: Ensure the image is resized and reshaped to the model’s expected input dimensions before prediction.

4. Database Connection Errors
Error: mysql.connector.errors.InterfaceError: Can't connect to MySQL server.
Cause: MySQL server is not running, incorrect connection parameters, or network issues.
Solution: Verify that the MySQL server is running and that the connection parameters (host, user, password, database) are correct.

5. Duplicate Attendance Entry
Error: Attendance is marked multiple times for the same person on the same day.
Cause: Inadequate logic to check for existing attendance records before inserting new ones.
Solution: Implement a check in the database query to ensure attendance is marked only once per person per day.

6. High CPU/GPU Usage
Error: System lags or crashes due to high CPU/GPU usage during real-time face recognition.
Cause: Resource-intensive operations from model inference or image processing.
Solution: Optimize model size, reduce image resolution, or offload processing to a more powerful machine or use a GPU.

7. Network Issues in Distributed Systems
Error: Timeout or ConnectionError when connecting to a remote database or server.
Cause: Network latency or server unavailability.
Solution: Ensure a stable network connection and consider implementing retry logic or local caching for temporary offline functionality.

8. File I/O Errors
Error: FileNotFoundError: No such file or directory: 'model.h5' or similar.
Cause: Missing model or configuration files.
Solution: Verify file paths and ensure all necessary files are available in the specified locations.

9. Incorrect Label Mapping
Error: Incorrect name or label is displayed during recognition.
Cause: Mismatch between the model’s output and the corresponding label.
Solution: Verify the labels file and ensure it matches the order of classes used during model training.

10. Concurrency Issues
Error: Race conditions or data inconsistency in multi-threaded or distributed environments.
Cause: Improper handling of concurrent database access or shared resources.
Solution: Use proper locking mechanisms or database transactions to handle concurrent access.

11. Memory Leaks
Error: Gradual increase in memory usage leading to system slowdowns or crashes.
Cause: Objects such as image frames or database connections not being released properly.
Solution: Ensure proper cleanup of resources by using context managers or try-finally blocks to release memory and close connections.

12. Version Compatibility Issues
Error: ImportError or AttributeError due to incompatibility between library versions.
Cause: Using incompatible versions of dependencies such as TensorFlow, OpenCV, or MySQL connector.
Solution: Check compatibility between the libraries and ensure that the correct versions are installed.

13. Ethical and Privacy Concerns
Error: Legal or ethical concerns raised regarding the use of facial recognition technology.
Cause: Lack of user consent or non-compliance with data protection laws.
Solution: Implement clear privacy policies, obtain user consent, and ensure compliance with relevant laws such as GDPR.

14. User Interface (UI) Errors
Error: Unresponsive or malfunctioning UI during operations.
Cause: Blocking operations in the main UI thread.
Solution: Use asynchronous operations or separate threads for heavy computations to keep the UI responsive.


By anticipating these errors and implementing strategies to handle them, you can create a more robust and reliable real-time attendance system.




















