RUN
----------------------------
python3 main.py inputs/test_1.jpg

Introduction
----------------------------
Front number plate detection is used in

It reduces manual number plate reading and logging


Our Solution
____________________________
Extracted number plate from the vehicle as an cropped image
Used number plate character set spritesheet for Sri Lankan Vehicle Number Plates
Saved number plate character data in a file
Trained a model with the saved data using KNearest machine learning algorithm
Extracted possible text characters using Contours detection (findContours function in opencv) and height of the Contours
Then using KNearest algorithm find 1 nearest neighbour (k=1)


Contour Detection
----------------------------
Sri Lankan vehicle number plate do consist of 6-9 similar height characters (eg. GO 5943 | CAT 67420)
We classified contours in to 3 size ranges (Large, Medium, Small)
Grab the range which consist with more than 5 charcters
If there are multiple range sets, then we give the priority for larger in size set (Large > Medium > Small)
    eg. Large   6
        Medium  9
    We take Large Set 
Because largest characters in a number plate cropped exactly are the plate numbers


Demostration
_____________________________


Further Enhancements
-----------------------------
For a best result, number plate area cropping can be further improved
We can have more than 1 model
* Different training number plate character spritesheet for each model
* Get the output from each model
* Use most occuring characters 

ex.
model 1 : AC 2732
model 2 : AB 2457
model 3 : RB 2452

answer  : AB 2452


