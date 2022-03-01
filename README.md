# Gesture-Recognition-System

This Gesture Recognition System was one of the first projects I ever created. Using OpenCV and TensorFlow in Python I was able to automate VLC Media player. Possible applications of a Gesture Recognition System are endless. This is a very simple implementation of the Gesture Recognition System. Please have a look and feel free to make your own changes. 

A brief description of each of the files are given below:

  1. collect_data - This file is used to collect the various gestures and save them into their respective directories. The image processing can definitely be improved. I mainly filtered the hand against a white background.
  2. model - Used to create and save the model using a custom VGG architecture. You can replace this with any custom architecture of your choice.
  3. vlc_automation - Finally, the testing of our model upon live feed. Using pyautogui this file tests the live feed for our gestures and triggers various events based upon our choice. 

Use the requirements.txt to install all the dependencies and then run the files in order of their explanations. This should give you the expected output.

Thank you for taking a look at this project!
