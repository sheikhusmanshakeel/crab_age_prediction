Data Challenge
=========================

The file data.pdf contains the observation of a study on crabs found around the Boston Area.

The challenge consist of making some sense out of those data. First, extract the data from the dirty PDF file, then perform any kind of analysis you think maybe of interest. We're notably looking for a method to predict the age of a crab given its features.

# Deliverables

## Prediction model

The first deliverable is a program that takes a PDF and generates a CSV of the data.

The program should implement your prediction model for the age of the crab for each line, and run it for each line, outputing the predicted age. The output should have the following fields:

```
Sex, Length, Diameter, Height, Weight, Shucked Weight, Viscera Weight, Shell Weight, Age, Predicted Age
```

and look like:

```
M,0.3875,0.275,0.1,0.43941725,0.18427175,0.0850485,0.1417475,3,3
I,0.4,0.275,0.0625,0.510291,0.18427175,0.15592225,0.1417475,3,4
```
