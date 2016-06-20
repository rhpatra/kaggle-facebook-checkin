# Predicting Facebook Check In Places #

Ever wonder what it's like to work at Facebook? Facebook and Kaggle are launching a machine learning engineering competition for 2016. Trail blaze your way to the top of the leaderboard to earn an opportunity at interviewing for one of the 10+ open roles as a software engineer, working on world class machine learning problems.

## Identify the correct place for check ins based on Facebook data ##

The goal of this competition is to predict which place a person would like to check in to. For the purposes of this competition, Facebook created an artificial world consisting of more than 100,000 places located in a 10 km by 10 km square. For a given set of coordinates, your task is to return a ranked list of the most likely places. Data was fabricated to resemble location signals coming from mobile devices, giving you a flavor of what it takes to work with real data complicated by inaccurate and noisy values. Inconsistent and erroneous location data can disrupt experience for services like Facebook Check In.

## Instructions ##

In this competition, you are going to predict which business a user is checking into based on their location, accuracy, and timestamp.

The train and test dataset are split based on time, and the public/private leaderboard in the test data are split randomly. There is no concept of a person in this dataset. All the row_id's are events, not people.

Note: Some of the columns, such as time and accuracy, are intentionally left vague in their definitions. Please consider them as part of the challenge.

## File descriptions ##

train.csv, test.csv
* row_id: id of the check-in event
* x y: coordinates
* accuracy: location accuracy
* time: timestamp
* place_id: id of the business, this is the target you are predicting
* sample_submission.csv - a sample submission file in the correct format with random predictions
