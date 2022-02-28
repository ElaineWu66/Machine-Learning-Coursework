# HW 1: Decision Trees for CS 349 @ NU
**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's NetID 
here, we will check. The NetID should be placed into that file by itself as the only line, 
with lower case letters and no carriage returns. 

The line that is marked NETID_GOES_HERE should be **REPLACED** with YOUR NetID.

**IF YOU DON'T PUT YOUR NETID INTO THAT FILE PROPERLY YOU WILL NOT RECIEVE CREDIT FOR YOUR WORK!**

In this assignment, you will:
- Understand and implement evaluation measures for machine learning algorithms
- Implement Information Gain and Entropy measures
- Implement a Decision Tree using the ID3 algorithm
- Implement a Prior Probability classifier
- Compare and contrast machine learning approaches on different datasets
- Write up your results in a clear concise report

## Clone this repository

First, let's clone this repository. We'll use `git` for all submissions in this class. New to `git`? Not to worry, it's quite easy! Here's a [helpful guide](https://guides.github.com/activities/hello-world/). 

To clone this repository install GIT on your computer and copy the link of the repository and enter in the command line:

``git clone YOUR-LINK``

Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Environment setup

Make a conda environment for this assignment, and then run:

``pip install -r requirements.txt``

## Running the test cases

The test cases can be run with:

``python -m pytest -s``

at the root directory of the assignment repository.

## Questions? Problems? Issues?

Simply open an issue on the starter code repository for this assignment [here](https://github.com/NUCS349/hw1-decision-trees/issues). Someone from the teaching staff will get back to you through there!

## Helpful Material
[Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8](https://www.youtube.com/watch?v=LDRbO9a6XPU)

[Decision Tree Lecture Series](https://www.youtube.com/playlist?list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO)
1. [How it works](https://www.youtube.com/watch?v=eKD5gxPPeY0&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=2&t=0s)
2. [ID3 Algorithm](https://www.youtube.com/watch?v=_XhOdSLlE5c&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=2)
3. [Which attribute to split on](https://www.youtube.com/watch?v=AmCV4g7_-QM&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=3)
4. [Information Gain and Entropy](https://www.youtube.com/watch?v=AmCV4g7_-QM&list=PLBv09BD7ez_4temBw7vLA19p3tdQH6FYO&index=3)

[A cute (but good) introduction to Decision Trees](https://www.youtube.com/watch?v=DCZ3tsQIoGU)

[ID3-Algorithm : Explanation](https://www.youtube.com/watch?v=UdTKxGQvYdc)
### Entropy
[What is entropy in Data Science (very nice explanaton)](https://www.youtube.com/watch?v=IPkRVpXtbdY)
[Entropy as concept in Physics/Chemistry (only if you're interested)](https://www.youtube.com/watch?v=YM-uykVfq_E)


### Recursion
[Python: Recursion Explained](https://www.youtube.com/watch?v=wMNrSM5RFMc)
[Recursion example](https://www.youtube.com/watch?v=8lhxIOAfDss)
