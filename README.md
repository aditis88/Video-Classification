A brief intro:- 

I have used UCF50 dataset but I am giving the pre trained model as the dataset is too large and
training requires a lot of time.

So I have trained the dataset on the few activities the are listed in training_specificaton.py folder.
The clasess_list (list contains all the activites that the model is trained on).

How to run the project:-
--------------------------------------------------------------------------------------------------------

Step 1:-Activate the virtual environment

- Inside the Project folder open the terminal
- create and activate the environment by running the command
    ->python -m venv env
    ->.\env\Scripts\activate


Step 2:- Install all the requirements
- go inside the VideoClassification folder
- open the terminal and run the following commands 
  -> pip install -r requirements.txt

  You can manually install all the packages if some packages are creating issue:
  -> pip install opencv-python
  -> pip install pafy youtube-dl moviepy
  -> pip install youtube_dl
  -> pip install ffmpeg-python
  -> pip install matplotlib
  -> pip install scikit-learn
  -> pip install tensorflow
  -> pip install Pillow

Step 3:- Run the program
  -> if You want to train the model again you will have to download the dataset UCF50 and paste it in
     the dataset folder and select the classes(activities) in training_specification.py 

  -> to train the model change the according to the dataset classes present in UCF50 and run
     createmodel.py
  - go into the src folder run the main.py
    -> python main.py



