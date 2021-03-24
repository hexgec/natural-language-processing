# natural-language-processing
Various AI classification methods were carried out allowing for the gender of an author to be identified based on a corpus of text.

Data:

The data used is that obtained from http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm#_blank and is therefore not included. The preProcess.py file makes use of the 'blogs' folder obtained from this site and requires the folder to be able to be run.

preProcess.py:

The preProcess.py file can simply be run and will automatically generate and save the arrays of the cleaned text and labels to a preProcessed.npz file.

Classifiers.py:

The main executable file. When run will display a menu of 5 options. Each option is associated to a different classifier each of which can be run when their associated option is chosen so as to display the respective results achieved.
