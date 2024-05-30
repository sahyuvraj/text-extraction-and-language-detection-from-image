import mysql.connector
import spacy
import string


# Initialize the NER model
nlp = spacy.load('en_core_web_sm')

# Fetch data from MySQL database
def fetch_data_from_mysql():
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Yuvraj@123",
        database="nagamese_dataset"
    )

    # Create a cursor object
    cursor = connection.cursor()

    # Execute a query
    cursor.execute("SELECT * FROM language_dataset")

    # Fetch all rows
    rows = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return rows

# Remove name from sentence using NER
def remove_name_from_sentence(sentence):
    name=""
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            name=ent.text
            sentence = sentence.replace(ent.text, '')
            # sentence = sentence[:ent.start] + sentence[ent.end:]
            
    return sentence.strip(),name

# Fetch best suitable Nagamese sentences using MySQL database
def fetch_best_nagamese_sentences(query):
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Yuvraj@123",
        database="nagamese_dataset"
    )

    # Create a cursor object
    cursor = connection.cursor()

    # Execute a query to fetch similar sentences based on the provided query
    # cursor.execute("SELECT Nagamese FROM language_dataset WHERE English = %s", (query,))
    cursor.execute("SELECT Nagamese FROM language_dataset WHERE English LIKE %s", ('%' + query + '%',))
    # similar_sentences = cursor.fetchone()

    # Fetch all rows
    similar_sentences = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return similar_sentences


def fetch_Engish_sentences(query):
    # Connect to the MySQL database
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Yuvraj@123",
        database="nagamese_dataset"
    )

    # Create a cursor object
    cursor = connection.cursor()

    # Execute a query to fetch similar sentences based on the provided query
    # cursor.execute("SELECT Nagamese FROM language_dataset WHERE English = %s", (query,))
    cursor.execute("SELECT English FROM language_dataset WHERE Nagamese LIKE %s", ('%' + query + '%',))
    # similar_sentences = cursor.fetchone()

    # Fetch all rows
    similar_sentences = cursor.fetchone()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return similar_sentences


def remove_pun(text):
    for pun in string.punctuation + string.digits:
        text = text.replace(pun, "")
    text = text.lower()
    return text

def replace_words(sentence, old_word, new_word):
    # words = sentence.split()
    # new_sentence = ' '.join([new_word if word == old_word else word for word in words])
    new_sentence = sentence.replace(old_word, new_word)
    return new_sentence




# Main function
# def naga_translation():
def naga_translation(eng_sen):
    # Fetch data from MySQL database
    
    rows = fetch_data_from_mysql()

    # Get the query from the user
    # query = str(input("Enter English sentence: "))
    query = eng_sen

    # Remove the name from the query using NER
    query,name_ = remove_name_from_sentence(query)
    # print(name_)
    name_ = name_.lower()
    # print(name_)

    # Fetch the best suitable Nagamese sentences using MySQL database
    nagamese_sentences = fetch_best_nagamese_sentences(query)

    english_sentence = fetch_Engish_sentences(nagamese_sentences[0][0])
    # print(english_sentence[0])
    english_sentence = english_sentence[0]
    newquery,newname_ = remove_name_from_sentence(english_sentence)
    newname_ = newname_.lower()

    # Add the name to the Nagamese sentences
    nagamese_sentences_with_name = [sentence[0].replace('NAME', rows[0][1]) for sentence in nagamese_sentences]

    # Print the Nagamese sentences with the name
    for sentence in nagamese_sentences_with_name:
        #naga_sen = remove_name_from_sentence(sentence)
        #print("Translate Into Nagamese: ",naga_sen)
        trans = sentence
        # print("Translate Into Nagamese: ",sentence)
        break
        
    if newname_ == "":
        naga_sentence = name_ + trans
    else:
        naga_sentence = replace_words(trans ,newname_, name_)
        
    
    # print("Translate Into new Nagamese: ",naga_sentence)
    return naga_sentence
    

    

# def main():
#     x = naga_translation()
#     print(x)

# # Call the main function
# if __name__ == "__main__":
#     main()

