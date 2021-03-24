import re
import os
import contractions
import inflect
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# defines what is to be replaced
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def openFile(filename):
    # Read the XML file
    with open(filename, "r", encoding="latin-1") as file:
        # Read each line in the file, readlines() returns a list of lines
        content = file.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)

    return content


# removes the xml formatting and returns only all the text from any posts
def strip_xml(text):
    soup = BeautifulSoup(text, "lxml")
    text = ''
    for post in soup.find_all('post'):
        text = text + post.text

    return text


def alterations(words):
    p = inflect.engine()
    sno = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    for word in words.split():
        new_word = word
        # replaces numbers in numeric form with their alphabetic form
        if word.isnumeric():
            try:
                new_word = p.number_to_words(word)
                new_word = new_word.replace(',', '')
            except:
                new_word = "one"
        stem = sno.stem(new_word)  # stems words to their root
        lemma = lemmatizer.lemmatize(stem, pos='v')  # lemetizes words
        words = words.replace(" " + word + " ", " " + lemma + " ")  # replaces the old word with the altered version

    return words


def clean_text(text):
    text = text.lower()  # lowercase text
    text = contractions.fix(text)  # Replace contractions in string of text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
    text = alterations(text)  # applies alterations to text

    return text


def openAllXml():
    blogsmale = []
    blogsfemale = []
    labelsmale = []
    labelsfemale = []
    male = 0
    female = 0
    limit = 500

    # opens all the xml files found in the blogs folder
    for filename in os.listdir('blogs'):

        gender = filename.split('.')[1]  # obtains the gender of the author

        # arranges the relevent text and places it in the associated array
        if (male < limit) and (gender == 'male'):
            content = openFile("blogs/" + filename)
            text = strip_xml(content)
            cleanText = clean_text(text)
            blogsmale.append(cleanText)
            labelsmale.append(gender)
            male = male + 1

        # arranges the relevent text and places it in the associated array
        elif (female < limit) and (gender == 'female'):
            content = openFile("blogs/" + filename)
            text = strip_xml(content)
            cleanText = clean_text(text)
            blogsfemale.append(cleanText)
            labelsfemale.append(gender)
            female = female + 1

        # exits the loop once all necessary samples are obtained
        if (male >= limit) and (female >= limit):
            break

    return blogsmale, labelsmale, blogsfemale, labelsfemale


blgsM, lblsM, blgsF, lblsF = openAllXml()  # returns the properly formatted data

# saves the arrays to a file to be read from later
np.savez('preProcessed.npz', blogsM=blgsM, labelsM=lblsM, blogsF=blgsF, labelsF=lblsF)

