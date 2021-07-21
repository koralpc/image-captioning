import tensorflow as tf


class Preprocess:
    @classmethod
    def calc_max_length(cls, tensor):
        return max(len(t) for t in tensor)

    @classmethod
    def tokenize(cls, captions, top_k=5000, filters=None):
        if filters is None:
            filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~'
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=top_k, oov_token="<unk>", filters=filters
        )
        tokenizer.fit_on_texts(captions)
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        train_seqs = tokenizer.texts_to_sequences(captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding="post"
        )
        max_length = cls.calc_max_length(train_seqs)
        return cap_vector, max_length, tokenizer
