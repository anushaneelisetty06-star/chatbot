import numpy as np
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
intents = {
    "intents": [
        {
            "tag": "jokes",
            "patterns": ["tell me a joke", "make me laugh 😊", "say something funny", "bye byee"],
            "responses": ["Why don’t programmers like nature? It has too many bugs!",
    "I told my computer I needed a break... it said 'No problem", "I’ll go to sleep!"
  ]
        },
        {
  "tag": "sad",
  "patterns": ["I am sad", "Feeling low", "I'm depressed"],
  "responses": [
    "I'm here for you. Everything will be okay.","don't fell alone - i am with u",
    "Stay strong. Tough times don’t last."
  ]
},
        {
            "tag": "user_info",
            "patterns": ["Who is ur creator?", "who created u?","say something about ur creator?"],
            "responses": ["Anu..","My developer is Anusha, who built me for an intern project.","From  my side my creator 'Anusha'  was a curious user learner and testing how chartborts works 🤖" ]
        }
    ]
}
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # SIMPLE WORD SPLIT (no nltk tokenize)
        word_list = pattern.lower().split()

        words.extend(word_list)
        documents.append((word_list, intent["tag"]))

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# remove punctuation
words = [w for w in words if w not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

print("Words:", words)
print("Classes:", classes)
print("Documents:", len(documents))
import numpy as np

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

training = np.array(training, dtype=object)

x_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

print("Training data ready!")
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=200,
          batch_size=5,
          verbose=1)
model.fit(x_train, y_train,
          epochs=200,
          batch_size=5,
          verbose=1)
import random

def bag_of_words(sentence):
    sentence_words = sentence.lower().split()

    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return classes[results[0][0]]

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

def chatbot():
    print("Chatbot is ready! Type 'see u later' to stop")

    while True:
        message = input("user_info - ANU: ")

        if message.lower() == "quit":
            print("Bot baby: Goodbye!")
            break

        tag = predict_class(message)
        response = get_response(tag)

        print("Bot baby:", response)
chatbot()
