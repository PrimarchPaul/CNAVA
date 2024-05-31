'''
Created on Mar 31, 2024

@author: alexpenny, PrimarchPaul
'''
import nltk
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk.tree import Tree

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample subjective notes dataset
with open('subjective note.txt', 'r') as file:
    # Read the entire file content
    file_content = file.read()
    words = file_content.split()
print(file_content)
# Split the file content based on blank lines
array_of_arrays = [block.split('\n') for block in file_content.split('\n\n')]

# Print the array of arrays
'''for array in array_of_arrays:
    print(array)
    print("XDDDDDD")'''

subjective_test = [ "Chief Complaint: Patient presents with a chief complaint of shortness of breath and chest tightness. History of Present Illness (HPI): The patient reports experiencing episodes of shortness of breath, especially during physical exertion, accompanied by tightness in the chest. Symptoms have been gradually worsening over the past week. Past Medical History (PMH): The patient has a history of asthma since childhood, with occasional exacerbations requiring short courses of oral corticosteroids. Medication History (MH): Currently using albuterol inhaler as needed for asthma symptoms. Allergies: No known drug allergies reported. Family History (FH): Family history is significant for asthma in both parents.",
   ]

    # Additional subjective notes...
def tree_format(ne_chunk_output):
    if isinstance(ne_chunk_output, nltk.Tree):
        return nltk.Tree(ne_chunk_output.label(), [tree_format(i) for i in ne_chunk_output])
    elif isinstance(ne_chunk_output, tuple):
        return ne_chunk_output[0]
    else:
        return ne_chunk_output


# Initialize NER tagger
def extract_entities(text):
    sentences = sent_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    entities = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        tagged_words = nltk.pos_tag(lemmatized_words)
        chunked = ne_chunk(tagged_words)
        ner_tree = tree_format(chunked)
       
        for subtree in ner_tree:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([word for word, tag in subtree.leaves()])
                entities.append(entity)
            else:
                entities.append(subtree)
               
    return entities


all_entities = []
for note in words:
    entities = extract_entities(note)
    all_entities.extend(entities)

# Print the extracted entities
result_string = ''
print("Extracted entities:")

for entity in all_entities:
    result_string += entity + " "
    #print(entity)

#print (result_string)
keyword = 'trigger'
separated_strings = result_string.split(keyword)

# Join the separated strings with the keyword 'Next' and a space, adding a line space after each occurrence
result_string = keyword.join(separated_strings)

# Add a line space after each occurrence of the keyword
result_string = result_string.replace(keyword, '\n \n')

# Print the modified result string
print(result_string)
'''for array in array_of_arrays:
    print_entities(array)
    print("I DIT IT LOL") '''  
# Print the extracted entities
'''print("Extracted entities:")
for entity in all_entities:
    print(entity)
    print()'''

