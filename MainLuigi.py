import os

import gensim
import luigi
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

nltk.download('wordnet')

stemmer = SnowballStemmer("english")

TASK_DATA_FOLDER = './tasksData/'
RESULTS_DATA_FOLDER = './results/'
TRAIN_DATA_FOLDER = './data/20news-bydate-train/'
TEST_DATA_FOLDER = './data/20news-bydate-test/'
KEYWORD_REGEX = 'Lines:\s\d+'


# Before launching the script make sure to install all the libraries with : pip install -r requirements.txt

# Function used to stem the words
def lemmatizeStemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Preprocess the lines
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatizeStemming(token))
    return result


# Purge all the previous data to have a clean start
class StepPurge(luigi.Task):
    # Create the dir if it does not exist
    if not os.path.exists(TASK_DATA_FOLDER):
        os.makedirs(TASK_DATA_FOLDER)

    # Else clear what is inside
    else:
        for folder in os.listdir(TASK_DATA_FOLDER):

            # If it is a dir we go one level deeper
            if os.path.isdir(TASK_DATA_FOLDER + folder):
                for file in os.listdir(TASK_DATA_FOLDER + folder):
                    os.remove(TASK_DATA_FOLDER + folder + "/" + file)

            # Else remove the file
            else:
                os.remove(TASK_DATA_FOLDER + folder)

    # Requires nothing
    def requires(self):
        return None

    # The purge step
    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + 'StepPurge.log')

    # Open teh output file and begin to write inside
    def run(self):
        with self.output().open('w') as outfile:
            outfile.write('Step one begins\n')


# Load the train data in a dataset with only the text data, that is to say we need to get all the lines and when we
# spot the keyword "Lines: xxx" where xxx is a number we need to skip the next paragraph and then we can keep the remaining data
class StepLoadTrainData(luigi.Task):
    # The path of the newsgroup directory where are the news
    currentNewsgroupDirectory = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The dict where we will push the articles without the header by their file name
        self.articles = {"index": [], "data": []}

        # The training model
        self.trainingModel = None

        # The words dictionary
        self.dictionary = None

    # It require the step purge
    def requires(self):
        return StepPurge()

    # The output file
    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + self.currentNewsgroupDirectory + 'StepLoadTrainingData.log')

    def run(self):

        # Open the output file
        with self.output().open('w') as outfile:

            # Log that we begin
            outfile.write('We begin to load the train data\n')

            # Variable used to count the files with a bad encoding
            badEncodingCount = 0

            # Find all the files needed and load them from the TRAIN_DATA_FOLDER and the corresponding newsgroup folder
            for file in os.listdir(TRAIN_DATA_FOLDER + self.currentNewsgroupDirectory):

                # We use a try as the encoding can be different and when don't mind skipping this data
                try:

                    # Open the file
                    with open(TRAIN_DATA_FOLDER + self.currentNewsgroupDirectory + file, 'r') as inpufile:

                        # Read all the lines and get them as a list
                        lines = [line.rstrip() for line in inpufile]

                        # We use this variable to know when we found the line where the keyword is
                        keywordFound = False
                        beginningIndex = 0

                        # We loop through the lines and find a line which has the following regex
                        for (index, line) in enumerate(lines):

                            if re.match(KEYWORD_REGEX, line):
                                # We say that we found the keyword
                                keywordFound = True

                            # If we have already found the keyword and this line is a blank one this is the
                            # starting point of the text we are looking for
                            if keywordFound and line == "":
                                # Set the beginningIndex to the current id + 1 (because we will use the next line as a
                                # starting point)
                                beginningIndex = index + 1

                                # We break as we dont need to go further in the file
                                break

                        # The article will be the lines after the keyword, we just need to join them by a whitespace
                        self.articles["index"].append(file)
                        self.articles["data"].append(' '.join(lines[beginningIndex:]))

                except:
                    badEncodingCount += 1
                    pass

            # Now that we have all the articles in a dict we can create the dataframe
            df = pd.DataFrame(self.articles)
            outfile.write(
                "Training dataframe has been created with %s articles, but %s have not been used due to bad encoding\n" % (
                    len(self.articles["data"]), badEncodingCount))

            # Preprocess the data by stemming and and parsing it
            processed_docs = df['data'].map(preprocess)
            outfile.write("The training dataframe has been preprocessed\n")

            # Get the dictionary of the docs
            self.dictionary = gensim.corpora.Dictionary(processed_docs)
            self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
            outfile.write("The training dictionary has been created and filtered with a size of %s words\n" % len(
                self.dictionary))

            # Make the corpus and create the model
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
            self.trainingModel = gensim.models.LdaMulticore(bow_corpus, num_topics=50, id2word=self.dictionary,
                                                            passes=2, workers=2)

            # Tell that this task is finished
            outfile.write("Model has been created\n")


# Load teh data that will be tested
class StepLoadTestData(luigi.Task):
    # The path of the newsgroup directory where are the news
    currentNewsgroupDirectory = luigi.Parameter()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # The preprocessed documents used to find their topic
        self.documents = {}

        # The dict where we will push we articles without the header by their file name
        self.articles = {}

        # The step used to train the model
        self.stepLoadDataTrain = None

    def requires(self):

        # Create a StepLoadTrainData with the newsgroup directory as a parameter
        self.stepLoadDataTrain = StepLoadTrainData(self.currentNewsgroupDirectory)

        return self.stepLoadDataTrain

    # The output file of the task
    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + self.currentNewsgroupDirectory + 'StepLoadTestData.log')

    def run(self):

        # Open the output file
        with self.output().open('w') as outfile:

            # Log that we begin
            outfile.write('We begin to load the test data\n')

            # Variable used to count the files with a bad encoding
            badEncodingCount = 0

            # Find all the files needed and load them from the TEST_DATA_FOLDER and the corresponding newsgroup folder
            for file in os.listdir(TEST_DATA_FOLDER + self.currentNewsgroupDirectory):

                # We use a try as the encoding can be different and when don't mind skipping this data
                try:
                    with open(TEST_DATA_FOLDER + self.currentNewsgroupDirectory + file, 'r') as inpufile:

                        # Read all the lines and get them as a list
                        lines = [line.rstrip() for line in inpufile]

                        # We use this variable to know when we found the line where the keyword is
                        keywordFound = False
                        beginningIndex = 0

                        # We loop through the lines and find a line which has the following regex
                        for (index, line) in enumerate(lines):

                            if re.match(KEYWORD_REGEX, line):
                                # We say that we found the keyword
                                keywordFound = True

                            # If we have already found the keyword and this line is a blank one this is the
                            # starting point of the text we are looking for
                            if keywordFound and line == "":
                                # Set the beginningIndex to the current id + 1 (because we will use the next line as a
                                # starting point)
                                beginningIndex = index + 1

                                # We break as we dont need to go further in the file
                                break

                        # The article will be the lines after the keyword, we just need to join them by a whitespace
                        self.articles[file] = self.stepLoadDataTrain.dictionary.doc2bow(
                            preprocess(' '.join(lines[beginningIndex:])))

                except:
                    badEncodingCount += 1
                    pass

            outfile.write(
                "The test data has been processed into a bow with a total of %s articles but %d had encoding isues\n" % (
                len(self.articles), badEncodingCount))


class StepPredictTopics(luigi.Task):
    # Current newsgroup directory
    currentNewsgroupDirectory = luigi.Parameter()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # The step used to load the step data
        self.stepLoadDataTest = None

        # The csv result file
        self.resultFile = None

        # The data frame
        self.dataFrame = None

        # The current newsgroup directory
        self.newsGroupDirectory = None

    def requires(self):

        # Create a StepLoadTestData with the newsgroup directory
        self.stepLoadDataTest = StepLoadTestData(self.currentNewsgroupDirectory)
        return self.stepLoadDataTest

    # The output file of the task
    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + self.currentNewsgroupDirectory + 'StepLoadDataframes.log')

    def run(self):

        # The results
        results = {"file": [], "topic": [], "score": [], "words": []}

        # The CSV result file
        self.resultFile = RESULTS_DATA_FOLDER + self.currentNewsgroupDirectory + "result.csv"

        # Save the newsGroupDirectory name
        self.newsGroupDirectory = str(self.currentNewsgroupDirectory)

        # If this newsgroup result directory does not exist we create it
        if not os.path.exists(RESULTS_DATA_FOLDER + self.currentNewsgroupDirectory):
            os.makedirs(RESULTS_DATA_FOLDER + self.currentNewsgroupDirectory)

        with self.output().open('w') as outfile:
            # Log that we begin
            outfile.write('Train step and test step have been done, know getting the topic of the document\n\n')

            # For every documents we give their first topic and the associated score
            for key in self.stepLoadDataTest.articles:
                outfile.write("Model results for file %s :\n" % key)
                for index, score in sorted(
                        self.stepLoadDataTest.stepLoadDataTrain.trainingModel[self.stepLoadDataTest.articles[key]],
                        key=lambda tup: -1 * tup[1]):
                    outfile.write("Score: {}\t Topic: {}\n".format(score,
                                                                   self.stepLoadDataTest.stepLoadDataTrain.trainingModel.print_topic(
                                                                       index, 5)))

                    # Add the results in the results files
                    results["file"].append(key)
                    results["topic"].append(index)
                    results["score"].append(score)
                    # Replace all the unneeded text by nothing
                    topics = re.sub(r'[0-9]+\.[0-9]+\*|"|\s', '',
                                    self.stepLoadDataTest.stepLoadDataTrain.trainingModel.print_topic(index, 5))

                    # Replace all the + by white spaces
                    results["words"].append(re.sub(r'\+', ' ', topics))

                    # For now we only want the first one
                    break

                outfile.write("\n\n")

            # Tell that all the files has been processed
            outfile.write('All the files have been processed\n')

            # Save the file as csv
            self.dataFrame = pd.DataFrame(results)
            self.dataFrame.to_csv(self.resultFile)

            outfile.write('CSV file has been saved to %s\n' % self.resultFile)


class SetpBuildVisualisation(luigi.Task):

    def requires(self):
        # The StepLoadDataframes classes for each directory
        self.stepLoadDataFrames = []

        # If the result directory does not exist we create it
        if not os.path.exists(RESULTS_DATA_FOLDER):
            os.makedirs(RESULTS_DATA_FOLDER)

        # We want to get all the newgroups directories
        for folder in os.listdir(TRAIN_DATA_FOLDER):

            # Check if it is a directory
            if os.path.isdir(TRAIN_DATA_FOLDER + folder):
                self.stepLoadDataFrames.append(StepPredictTopics(folder + "/"))

        # And we return them are requirements
        return self.stepLoadDataFrames

    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + 'SetpBuildVisualisation.log')

    def run(self):

        with self.output().open('w') as outfile:
            outfile.write('All the newsgroups articles have been processed, the visualisation is being built')

            # An array to know which visualisations we get
            visualisationsNames = []

            # Now that we have all the dataframes we can build ou plotly visualisation

            # Loop through the stepLoadDataFrames
            for stepLDF in self.stepLoadDataFrames:
                # Get the dataframe and count the topics and rename the file column to counts
                currentDataframe = stepLDF.dataFrame

                currentDataframe = currentDataframe.groupby('words').count().reset_index()
                currentDataframe = currentDataframe.rename(columns={"file": "counts"})

                print(currentDataframe)

                currentDataframe = currentDataframe.sort_values(by=['counts'])

                # Change the color for the 0.9th quantile population
                counts = currentDataframe["counts"].to_numpy()
                quantile75Percent = currentDataframe["counts"].quantile(0.9)
                color = np.array(['rgb(255,255,255)'] * len(counts))
                color[counts >= quantile75Percent] = 'rgb(204,204, 205)'
                color[counts < quantile75Percent] = 'rgb(130, 0, 0)'

                # Build the figure
                fig = px.bar(currentDataframe, x='words', y='counts', text="counts", color=color, hover_name="words",
                             hover_data=["counts"])
                fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

                # Save the viz as an html file in the file visualisation
                fig.write_html(RESULTS_DATA_FOLDER + stepLDF.newsGroupDirectory + "/visualisation.html")

                # Append the name of the current visualisation
                visualisationsNames.append(stepLDF.newsGroupDirectory)

            # For every visualisations create a button that will open it
            buttons = []
            for name in visualisationsNames:
                buttons.append("<button onclick='changeViz(this)' value='%s'>%s</button>" % (name, name))

            # Crate an HTML script where will be our visualisation
            htmlStr = """
                <script>
                    function changeViz(elt) {
                        document.getElementById('theIframe').src = `results/${elt.value}visualisation.html`
                    }
                </script>
                <html style='margin: 0; overflow: hidden;'>
                    <head>
                        <meta charset='utf-8'/>
                    </head>
                    <body>
                        <div>
                            %s
                            <div id='viz'>
                                <iframe id='theIframe' height='100%%' width='100%%' frameBorder='0' src='results/%svisualisation.html'/>
                            </div>
                        </div>
                    </body>
                </html>
            """ % ("&nbsp".join(buttons), visualisationsNames[0])

            # Now we must create an HTML file that gathers all the different HTML files with an <iframe> tag
            with open("main.html", "w") as file:
                file.write(htmlStr)


# Main class that will give the name for all the newsgroups directories and
class StepMain(luigi.Task):

    def requires(self):
        return SetpBuildVisualisation()

    def output(self):
        return luigi.LocalTarget(TASK_DATA_FOLDER + 'StepMain.log')

    def run(self):
        with self.output().open('w') as outfile:
            outfile.write('Main task is up')


if __name__ == '__main__':
    luigi.run()
