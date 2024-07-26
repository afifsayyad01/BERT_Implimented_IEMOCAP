#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow==2.12.0')


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from transformers import TFBertModel, BertTokenizer
import os


# In[3]:


max_length = 132  # Maximum length of input sentence to the model.
batch_size = 38
epochs = 2


# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Step 1: Load the CSV file
file_path = "/workspace/new_data/new_data.csv"  # Correct file path

# Check if the file exists
if not os.path.exists(file_path):
    print("Error: The file does not exist.")
else:
    df = pd.read_csv(file_path, encoding='utf-8')  # Specify the encoding

    # Step 2: Inspect the Data (Optional)
    print(df.head())
    print(df.info())

# Further steps as before
# Step 3: Reduce the Dataset Size
df_sampled = df.sample(frac=0.50, random_state=50)  # Adjust the fraction as needed

# Splitting the data into features and labels
X = df[['Sentence1', 'Sentence2']]
y = df['Similarity']

# First split: 80% training, 20% remaining (testing + validation)
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.5, random_state=42)

# Second split: 50% of the remaining data for validation, 50% for testing
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

# Convert the splits back to DataFrames for inspection (Optional)
train_df = pd.DataFrame({'Sentence1': X_train['Sentence1'], 'Sentence2': X_train['Sentence2'], 'Similarity': y_train})
val_df = pd.DataFrame({'Sentence1': X_val['Sentence1'], 'Sentence2': X_val['Sentence2'], 'Similarity': y_val})
test_df = pd.DataFrame({'Sentence1': X_test['Sentence1'], 'Sentence2': X_test['Sentence2'], 'Similarity': y_test})

# Display the split datasets
print("Training Data:")
print(train_df.head())

print("\nValidation Data:")
print(val_df.head())

print("\nTesting Data:")
print(test_df.head())


# In[5]:


# Shape of the data
print(f"Total train samples : {train_df.shape[0]}")
print(f"Total validation samples: {val_df.shape[0]}")
print(f"Total test samples: {test_df.shape[0]}")


# In[6]:


print(train_df.columns)


# In[7]:


print(f"Sentence1: {train_df.loc[1, 'Sentence1']}")
print(f"Sentence2: {train_df.loc[1, 'Sentence2']}")
print(f"Similarity: {train_df.loc[1, 'Similarity']}")


# In[8]:


# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)


# In[9]:


print("Train Target Distribution")
print(train_df.Similarity.value_counts())


# In[10]:


print("Validation Target Distribution")
print(val_df.Similarity.value_counts())


# In[11]:


train_df = (
    train_df[train_df.Similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)
valid_df = (
    val_df[val_df.Similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)


# In[12]:


train_df["label"] = train_df["Similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=3)

val_df["label"] = val_df["Similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_val = tf.keras.utils.to_categorical(val_df.label, num_classes=3)

test_df["label"] = test_df["Similarity"].apply(
    lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2
)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=3)


# In[13]:


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


# In[14]:


import tensorflow as tf
from transformers import TFBertModel

# Ensure TensorFlow detects all available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")


num_gpus = strategy.num_replicas_in_sync
batch_size = 32 * num_gpus

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicate to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model.bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

optimizer = tf.keras.optimizers.Adam(1e-4)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)



print(f"Strategy: {strategy}")
model.summary()


# In[15]:


train_data = BertSemanticDataGenerator(
    train_df[["Sentence1", "Sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    val_df[["Sentence1", "Sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)


# In[16]:


epochs = 1
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)


# In[17]:


# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# In[ ]:


epochs = 1
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)


# In[ ]:


test_data = BertSemanticDataGenerator(
    test_df[["Sentence1", "Sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)


# In[ ]:


def check_similarity(Sentence1, Sentence2):
    sentence_pairs = np.array([[str(Sentence1), str(Sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


# In[ ]:


Sentence1 = "Two women are observing something together."
Sentence2 = "Two women are standing with their eyes closed."
check_similarity(Sentence1, Sentence2)


# In[ ]:


Sentence1 = "A smiling costumed woman is holding an umbrella"
Sentence2 = "A happy woman in a fairy costume holds an umbrella"
check_similarity(Sentence1, Sentence2)


# In[ ]:


Sentence1 = "A soccer game with multiple males playing"
Sentence2 = "Some men are playing a sport"
check_similarity(Sentence1, Sentence2)


# In[ ]:


Sentence1 = "BERT Model checks the similarity between two sentences"
Sentence2 = "similarity is being chacked by the model called BERT"
check_similarity(Sentence1, Sentence2)


# In[ ]:


Sentence1 = "It was a fun experience"
Sentence2 = "The experience was very sad to say"
check_similarity(Sentence1, Sentence2)


# In[ ]:


Sentence1 = "Engineering is about multidimensional growth of thinking"
Sentence2 = "He is a doctor"
check_similarity(Sentence1, Sentence2)

