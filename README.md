# Client Usage Predicter
This was a project undertaken as part of an Internship in July-October 2021 and predicts whether a client will significantly increase or decrease their use of the company's online service this week, based on the data collected in the previous week. 
The module contains three .py scripts (one more in the folder Initial.py), and are designed so that the data each week is not only used to predict an outcome but also be included to retrain the decision tree predictor model.

Details such as implementation and the data science theory behind the model can be found in Documentation.pdf.

As part of company protocols all raw company data were removed from this repository, and thus as the code were tailor-made based on the company's data it may be rather difficult to understand what is going on - the "Data Science Decisions" in Documentation.pdf may better represent my ideas. As the output of the scripts contained client names, these were also removed when pushed to Github.


The Flask folder contains scripts which support a Flask Web application interface for the predictor

*As this was an early project, undertaken when the author was in his second year of university, some of the code were written inefficiently with use of python default data structures and functions rather than Pandas internal functions, which reduces readability. It is hoped this does not severly impinge the **demonstration of data science ideas being used innovatively, the ideas of continuous-retraining at its primitive stage and the setup of a system for weekly usage**.*
