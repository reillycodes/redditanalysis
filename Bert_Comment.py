import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
import numpy as np
import pandas as pd
from User_reddit_collection import UserCollection



def sentiment(dash_comment):
    """
    Each line of the dataset is composed of the review text and its label
    - Data preprocessing consists of transforming text to BERT input features:
    input_word_ids, input_mask, segment_ids
    - In the process, tokenizing the text is done with the provided BERT model tokenizer
    """

    label_list = [0,1]# Label categories
    max_seq_length = 128# maximum length of (token) input sequences
    train_batch_size = 32


    # Get BERT layer and tokenizer:
    # More details here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4

    bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2', trainable=True)

    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file,do_lower_case)

    def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
      example = classifier_data_lib.InputExample(guid=None,
                                                 text_a = text.numpy(),
                                                 text_b = None,
                                                 label = label.numpy())
      feature = classifier_data_lib.convert_single_example(0,example, label_list,max_seq_length,tokenizer)

      return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

    def to_feature_map(text, label):
      input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label],
                                                                    Tout = [tf.int32,tf.int32,tf.int32,tf.int32])
      input_ids.set_shape([max_seq_length])
      input_mask.set_shape([max_seq_length])
      segment_ids.set_shape([max_seq_length])
      label_id.set_shape([])

      x = {
          'input_word_ids': input_ids,
          'input_mask': input_mask,
          'input_type_ids': segment_ids
      }

      return (x, label_id)

    def create_model():
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_type_ids")
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

        drop = tf.keras.layers.Dropout(0.4)(pooled_output)
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(drop)

        model = tf.keras.Model(
            inputs={
                'input_word_ids': input_word_ids,
                'input_mask': input_mask,
                'input_type_ids': input_type_ids

            },
            outputs=output
        )
        return model

    def sentiment_classifier(list):
        return max(set(list), key=list.count)
    # right_test = pd.read_csv('/Users/steve/Google Drive/Dissertation/newtestconservative.csv', usecols=[1,2,3,4,5])
    # left_test = pd.read_csv('/Users/steve/Google Drive/Dissertation/newtestliberal.csv', usecols=[1,2,3,4,5])
    # test_df = pd.concat([right_test, left_test])
    # test_df.columns = ['Comment', 'Date','Score','Subreddit','Label']
    # test_df['Label'] = test_df['Label'].map({'Right-Wing':1, 'Left-Wing':0})
    # test_df.Comment = test_df.Comment.astype(str)

    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics = [tf.keras.metrics.BinaryAccuracy()])
    model.summary()

    print('Model loading')
    model.load_weights('./new_model_all_data_5epochs').expect_partial()
    print('Model loaded')
    sample_example = []
    sample_example.append(dash_comment)
    # # sample_example = ['Trump goes ahead with plans to build wall at the border of Mexico',
    # #                   'Brexit',
    # #                   'The EU is stealing our jobs',
    # #                   'Rep. Marjorie Taylor Greene has been suspended from tweeting for a week after she claimed on Twitter yesterday that vaccines are “failing.”',
    # #                   'All Rabbits should be placed on birth control. The cost of babies is ruining our democracy.',
    # #                   'Journalism is dead.',
    # #                   'fuck the tories',
    # #                   'I Love Joe Biden',
    # #                   'Boris Johnson is a responsible for brexit'
    #                   ]

    test_data = tf.data.Dataset.from_tensor_slices((sample_example, [0]*len(sample_example)))
    test_data = (test_data.map(to_feature_map).batch(1))
    print('predicting')
    preds=model.predict(test_data, verbose=1)
    threshold = 0.5
    result = [1 if pred >= threshold else 0 for pred in preds]
    results = []
    for i in preds:
      if i <=0.25:
        results.append('Strongly Left-wing')
      elif i >= 0.25 and i < 0.40:
        results.append('Somewhat Left-wing')
      elif i >= 0.40 and i < 0.50:
        results. append('Slightly Left-wing')
      elif i == 0.5:
        results.append('Neutral')
      elif i >= 0.5 and i < 0.60:
        results.append('Slightly Right-wing')
      elif i >= 0.60 and i < 0.75:
        results. append('Somewhat Right-wing')
      elif i >= 0.75:
        results.append('Strongly Right-wing')

    return "{scr}".format(scr=sentiment_classifier(results))
    # y_true = [1,1,1,0,1,1,0,0,0]
    # report = classification_report(y_true, result)
    # print(report)
    # print(results)

if __name__ == '__main__':
    print(sentiment('tories are shit'))