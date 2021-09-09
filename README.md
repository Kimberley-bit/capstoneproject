# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project by Kimberley

*Note: I was unable to upload all of the datasets that I have used in these notebooks onto github because of how big the size of the file was. But if you have any questions feel free to reach out to me thank you :)*

## Background ##
Worldwide more than 700 000 people die due to suicide every year. Suicide is also the fourth leading cause of death in 15-19-year-olds. Regionally, in Singapore “Suicide cases in Singapore highest in 8 ears amind covid-19 pandemic” this was from an article published on 8th July 2021 by CNA. Singapore recorded 452 suicides n 2020.

## Problem Statement ##
This project aims to apply machine learning abilities in particular text classification techniques in order to detect suicidal tendencies in social media posts. Early detection of these risk factors can help in preventing or reducing the number of suicides and even provide help to parties that urgently need it.

## Datasets used ##
1. Reddit 
2. Twitter

Reddit and Twitter datasets were combined to form a final dataset. 
This final dataset has 4,000 rows and 2 columns (Text, Class).

## Limitations ##
The first limitation is that It is not possible to generalise all human behaviour into simple lines of code. Hence this model won’t be able to capture every single aspect of human behaviour
So what's the point of this project i guess even if we are working with a project that has a accuracy of 70% or 60% and not 100%. With a model that exists we will at least still be able to save some form of human life and in a sense can help to pave the way for other project and would be a step into the right direction.

The second limitation is that the data was limited to only Reddit and Twitter, other social media platforms data can also be introduced for eg, facebook so that the model can cater to an even wider difference of posts. Which can help to improve and enhance the model overall.

## Some key findings from the EDA that was done ##
1. A sentiment analysis was done for Twitter and Reddit. I felt that this was vital to include as it is to ensure that the dataset that it is eventually put into modelling has a balanced sentiment and does not include any biases or skewedes. From the graphs shown, it can be seen that the sentiment for both data is rather balanced which is good.

2. A word count for the posts of Twitter and Reddit was also generated. It is good to note that Twitter has a limited word capacity but as you can see despite that, from the graph for both platforms, suicide has a higher word count as compared to non-suicidal posts. The reason for this could be because suicidal users tend to share their intentions freely, so the length is larger than non-suicidal users. A similar result was also found in a Article in Baghdad Science Journal · December 2020

3. --For Twitter-- A word cloud for Twitter in particular tweets that have been categorised as suicide had these words as the most common: ('kill', 398), ('depressed', 304), ('suicidal', 280), ('hopelessness', 200). Another word cloud was also done for tweets that were categorised as non-suicide. These were the words that was most common: ('love', 58), ('know', 55), ('one', 54).

4. --For Reddit -- A word cloud for Reddit in particular tweets that have been categorised as suicide had these words as the most common: ('people', 420), ('friend', 364), ('really', 288). Another word cloud was also donen for tweets that were categorised as non-suicidal. These were the words that was most common: ('really', 393), ('people', 390), ('thing', 356).

## Modelling and Evaluation ##
Models used:
1. Bernoulli NB
2. Gassian NB
3. Linear Regression
4. Adaboost
5. KNN
6. SVM

## Flasking ##
For the flasking portion of the project, the basic idea is that after inputting a content of a posts the flask model will be able to detect whether the posts was suicide or non-suicide. In addiion to that because my model is trained on 2 different datasets (Reddit & Twitter) this platform is not only designed for a particular social media platform has the capabilities to cater to a wider variety of posts from anywhere basically, and this was one of the key reasons as to why i wanted to include 2 different datasets instead of only one.

## Limitations ##
The first limitation is that it is not possible to generalise all human behaviour into simple lines of code. Hence this model won’t be able to capture every single aspect of human behaviour
So what's the point of this project i guess even if we are working with a project that has a accuracy of 70% or 60% and not 100%. But with a model that exists we will at least still be able to save some form of human life and in a sense can help to pave the way for other project and and would be a step into the right direction.

The second limitation is that the data use for the model was only limited to only Reddit and Twitter, other social media platforms data can also be introduced for eg, facebook so that the model can cater to an even wider difference of posts. Which can help to improve and enhance the model overall.

## Recommendation (and potential future projects) ## 
A given would be to train with a bigger data set this is done to increase accuracy as well as to introduce the model to a wider variety of vocab. We can also train the mode with different languages because for this currennt project this model mainly focuses on english and i did filter every other language out. 

Future projects that we can look towards can be an image classifier and potentially consider a video calssifier. This is because the consumption of media is moving towards being more visual so like how tiktok consists of video, being able to explore image and video would be exciting and an interesting venture not only limiting ourselves to text

## Jupyter Notebooks should be read in this order ##
Book   | Name
-------|-----------
(Bk 1) | Snscrape (Twitter).ipynb
(Bk 2) | Twitter and Reddit.ipynb
(Bk 3) | Twitter & Reddit_EDA.ipynb
(Bk 4) | CV_Modelling.ipynb
(Bk 4) | TFIDF_Modelling.ipynb
(Bk 5) | Random Forest Classifier_CV.ipynb
(Bk 5) | Random Forest Classifier_TFIDF.ipynb
(Bk 6) | Neural Network_CV.ipynb
(Bk 6) | Neural Network_TFIDF.ipynb
(Bk 7) | RNN_CV.ipynb
(Bk 7) | RNN_TF-IDF.ipynb
