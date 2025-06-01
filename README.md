## Aviation Accident Data Integration

Information Integration and Analytic Data Processing Project
Faculty of Science, University of Lisbon

### Group 03:

- Tommaso Tragno - fc64699
- Manuel Cardoso - fc56274
- Chen Cheng - fc64872
- Cristian Tedesco - fc65149

#### NTSB Dataset

The NTSB dataset is too big for being uploaded on github. Please download it from the following [link](https://drive.google.com/file/d/1frWczU94UoCY7Cc6OIa43gVOPOkgix2w/view?usp=drive_link) and add the json file to the `data_sources` folder

Same for the weather dataset. Download it from [here](https://drive.google.com/file/d/1UDxntxkzE82WjZuT49ju2nyNbFmkUs4u/view?usp=drive_link)

##### Running the project

Our project is present in project.ipynb. Make sure to install all the required libraries, indicated in the first cell of the notebook.

The file `ntsb_with_zero_shot.csv` shouldn't be deleted, as it is created by another script which takes a lot of computation power (`Bert_text_classification.py`).