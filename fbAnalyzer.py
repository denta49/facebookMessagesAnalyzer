# %%
from asyncore import read
from datetime import date
import json
from math import comb
from numpy import *
import pandas as pd
from operator import itemgetter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import glob
import os
import matplotlib.pyplot as plt
import matplotlib

# function for open file
def openFile(file):
    with open(file, encoding='utf-8') as messages:
        chat_history = json.load(messages)
        messages.close()
        return chat_history


# function for returning number of df indexes
def getNumberOfMessages(df):
    return len(df.index)


# function to sort files
def sortingPrimary(x):
    return(x[8:-6])

    # TODO: find better sorting atributes


# function for creating JSONs
def createExtractedMessagesJSONs(msgsFolder):

    # creating paths
    extractedDir = msgsFolder + '/extractedMessages/'
    jsonsFilesIdDir = msgsFolder + '/*.json'

    # create new folder for jsons with extracted mssgs
    if not os.path.exists(extractedDir):
        os.makedirs(extractedDir)

        print('Folder for JSON files with extracted messages created.')
        print()

    print('Starting file cleaning process...')
    print('     Looping through messages files...')

    # loop for extracting mssgs and creating new JSONs
    counter = 0
    for file in sorted(glob.glob(jsonsFilesIdDir), key=sortingPrimary):
        counter += 1
        current = openFile(file)
        messages = pd.DataFrame(current['messages'])

        print(f'            Extracting {file}...')

        messages.to_json(extractedDir + 'message_' + str(counter) +
                         '.json', orient='records')

        print('          ' + extractedDir + 'message_' +
              str(counter) + '.json File created.')
    print('     Files with extracted messages created.')
    print()

# function for creating one JSON with all messages only
def createOneJSONforMessagesOnly(msgsFolder):

    theOne = msgsFolder + '/extractedMessages/theOne/'
    extractedJSONsPath = msgsFolder + '/extractedMessages/*.json'

    if not os.path.exists(theOne):
        os.makedirs(theOne)

        print('     Folder for theOne JSON file with extracted messages ONLY created...')
        print()

    # loop for adding files from extracted to list
    print('     Looping through sorted files with extracted messages...')
    print()

    files = []

    for file in sorted(glob.glob(extractedJSONsPath), key=sortingPrimary):
        files.append(file)

    result = list()

    for file in sorted(glob.glob(extractedJSONsPath), key=sortingPrimary):

        with open(file, 'r') as infile:
            result.extend(json.load(infile))

    # create theOne file
    with open(theOne + 'theOne.json', 'w') as output_file:
        json.dump(result, output_file)

        print('     theOne.json created.')
        print()

    # converting theOne to data frame
    print('     Converting theOne.json to data frame...')

    msgs = openFile(theOne + 'theOne.json')
    messages = pd.DataFrame(msgs)

    print('         Done.')
    print('File cleaning process sucessfully ended.')
    print()
    print()
    print()

    return messages

# function for time processing
def createTime(messages):

    # function for converting timestamp to regular pandas date format
    def convert_time(timestamp):
        return pd.to_datetime(timestamp, unit='ms')

    # creating col with date with timestamp function
    print('Starting creating time process...')
    print('     Converting timestamps in theOne file to normal date...')

    messages['date'] = messages['timestamp_ms'].apply(convert_time)

    print('         Done.')
    print()

    # functions for creating day, year, month
    def get_day(date):
        return date.day

    def get_month(date):
        return date.month

    def get_year(date):
        return date.year

    # creating colomns with day, month, year
    print('     Creating day column...')

    messages['day'] = messages['date'].apply(get_day)

    print('         Done.')
    print()
    print('     Creating month column...')

    messages['month'] = messages['date'].apply(get_month)

    print('         Done.')
    print()
    print('     Creating year column...')

    messages['year'] = messages['date'].apply(get_year)

    print('         Done.')
    print('Creating time process ended succesfully.')
    print()
    print()
    print()

# function for extracting one's participant messages
def takeParticipantsMessages(participant1, participant2, messages):

    print('Starting extracting messages for all participants... ')
    print('     Extracting only messages with dates to variable... ')

    # creating copy of df with important data only
    msg = messages[['sender_name', 'content', 'day', 'month', 'year']]

    print('         Messages extracted. ')
    print()

    # TODO: use df.drop insted of creating new df

    # taking participant1 messages
    print('     Extracting messages of first participant... ')
    par1 = msg[msg['sender_name'].isin([participant1])]

    print('         Done. ')
    print()

    # taking participant2 messages
    print('     Extracting messages of second participant... ')
    par2 = msg[msg['sender_name'].isin([participant2])]

    print('         Done. ')
    print('Extracting messages process ended succesfully.')
    print()
    print()
    print()

    return [par1, par2]


def proc(msg1, msg2):
    full = msg1 + msg2
    procMsg1 = ((msg1/full) * 100)
    return procMsg1


# function for printing basic info
def printing(participant1, participant2, msgsPerson1, msgsPerson2):

    numberOfPar1Msgs = getNumberOfMessages(msgsPerson1)
    numberOfPar2Msgs = getNumberOfMessages(msgsPerson2)

    def proc(msg1, msg2):
        full = msg1 + msg2
        procMsg1 = ((msg1/full) * 100)
        return procMsg1

    print('Printing basic info...')
    print('---------------------------------------------------------------')
    print()

    print(
        f'    In chat of {participant1} and {participant2} have been sent {numberOfPar1Msgs + numberOfPar2Msgs} messages.')

    print()

    print(f'    {participant1} has sent: {numberOfPar1Msgs} messages.')

    print()

    print(f'    {participant2} has sent: {numberOfPar2Msgs} messages.')

    print()

    print(
        f'      Which gives such proportions:\n        {participant1}: {proc(numberOfPar1Msgs, numberOfPar2Msgs)}%, {participant2} {proc(numberOfPar2Msgs, numberOfPar1Msgs)}%.')

    print()
    print('---------------------------------------------------------------')
    print('Basic info printed.')
    print()
    print()
    print()


# function with function for files preparation
def filesPreparation(msgsFolder):

    createExtractedMessagesJSONs(msgsFolder)

    messages = createOneJSONforMessagesOnly(msgsFolder)

    createTime(messages)

    return messages


# function for calculling word
def findWords(pers1msgs, pers2msgs, word):

    print('Starting looking for words process...')
    print(
        f'     Taking all messages with word {word} from first participant...')

    df1 = pers1msgs[pers1msgs.content.str.contains(word, na=False)]

    print('         Done.')
    print()

    df2 = pers2msgs[pers2msgs.content.str.contains(
        word, na=False)]  # to use later

    # create list of months in df
    lsOfMonths = []

    print('     Creating list of months in messages...')

    for months in df1.month:
        lsOfMonths.append(months)

    print('         Done.')
    print()

    # create list of years in df
    lsOfYears = []

    print('     Creating list of years in messages...')

    for years in df1.year:
        lsOfYears.append(years)

    print('         Done.')
    print()

    # create dfCounter
    print('     Creating data frame with months and years...')

    dfCounter = pd.DataFrame(list(zip(lsOfMonths, lsOfYears)), columns=[
                             'month', 'year']).drop_duplicates()

    print('         Done.')
    print()

    result = []

    print(f'     Counting word {word} in messages... ')

    for i in dfCounter['year'].drop_duplicates():

        x = df1[df1['year'].isin([i])]

        for j in x['month'].drop_duplicates():

            y = x[x['month'].isin([j])]

            result.append(getNumberOfMessages(y.content))

    print('         Done.')
    print()

    print('     Adding results to data frame...')
    dfCounter['records'] = result
    print('         Done.')
    print()
    print()
    print()

    # converting time for plotting
    dfCounter['DATE'] = pd.to_datetime(
        dfCounter[['month', 'year']].assign(DAY=1))

    dfCounter = dfCounter.drop(columns=['month', 'year'])

    dfCounter['DATE'] = pd.to_datetime(dfCounter["DATE"])

    dfCounter = dfCounter.set_index('DATE')

    return dfCounter


# function for plotting
def plotWord(df):
    fig, ax = plt.subplots(figsize=(30, 30))

    # Add x-axis and y-axis
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m.%Y'))

    ax.plot(df.index.values,
            df['records'],
            color='purple', )

    ax.set(xlabel="Date",
           ylabel="Records",
           title="Word usage")

    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=90, horizontalalignment='right')

    plt.show()


def main():

    print()
    print('Starting Facebook Analyzer...')
    print()

    # here pass names of chat participants
    participant1 = 'Przemys\u00c5\u0082aw Wojenka'
    participant2 = 'Maciej Hy\u00c5\u00bcy'

    # specify dir patch with messages
    msgsFolder = 'E:\VSCode\Facebook\maciejMessages'

    # specify word that You would like to analyze
    word = 'halo'

    messages = filesPreparation(msgsFolder)
    person1Msgs = takeParticipantsMessages(
        participant1, participant2, messages)[0]
    person2Msgs = takeParticipantsMessages(
        participant1, participant2, messages)[1]
    printing(participant1, participant2, person1Msgs, person2Msgs)

    # specify what word You would like to look for
    wrd = findWords(person1Msgs, person2Msgs, word)

    plotWord(wrd)


main()
