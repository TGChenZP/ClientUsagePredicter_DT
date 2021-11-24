# Client Usage Predicter
This was a project undertaken as part of an Internship in July-October 2021 and predicts whether a client will significantly increase or decrease their use of the company's online service this week, based on the data collected in the previous week. 
The module contains three .py scripts (one in the folder Initial.py), and are designed so that the data each week is not only used to predict an outcome but also be included to retrain the decision tree predictor model.

Details such as implementation and the data science theory behind the model can be found in Documentation.pdf.

As part of company protocols all raw company data were removed from this repository, and thus as the code were tailor-made based on the company's data it may be rather difficult to understand what is going on - the "Data Science Decisions" in Documentation.pdf may better represent my ideas. As the output of the scripts contained client names, these were also removed when pushed to Github.

Emperical test-train split testing returned an accuracy of over 67% for predicting true positives (both significant increase and decreases). This is more than double the accuracy compared to randomly allocating "significantly increase", "significantly decrease" and "normal" to clients, given the weekly data.
