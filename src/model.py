from src.utils import load_image
import numpy as np
from src.loss import loss_function
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from src.preprocess import image_features_extract_model
from src.utils import load_image


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    @tf.function(input_signature = [tf.TensorSpec(shape=[1, 64, 2048],)])
    def __call__(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = Attention(self.units)

    @tf.function(input_signature = [tf.TensorSpec(shape=[1, 1], dtype=tf.int32), tf.TensorSpec(shape=[1, 64, 512], dtype=tf.float32),tf.TensorSpec(shape=[1, 1024], dtype=tf.float32)])
    def __call__(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class EDModel(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, tokenizer):
        super(EDModel, self).__init__()
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_func = loss_function
        self.tokenizer = tokenizer
        self.trainable_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
    def __call__(self, inputs, max_length=None, attn_shape=None, mode="train"):
        img_tensor, target = inputs
        if mode == "train":
            loss = 0

            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = self.decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims(
                [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
            )

            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += self.loss_func(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

            return loss, dec_input
        else:
            attention_plot = np.zeros((max_length, attn_shape))

            hidden = self.decoder.reset_state(batch_size=1)
            temp_input = tf.expand_dims(load_image(img_tensor)[0], 0)
            img_tensor_val = image_features_extract_model(temp_input)
            img_tensor_val = tf.reshape(
                img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
            )
            features = self.encoder(img_tensor_val)

            dec_input = tf.expand_dims([self.tokenizer.word_index["<start>"]], 0)
            result = []

            for i in range(max_length):
                predictions, hidden, attention_weights = self.decoder(
                    dec_input, features, hidden
                )

                attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

                predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
                result.append(self.tokenizer.index_word[predicted_id])

                if self.tokenizer.index_word[predicted_id] == "<end>":
                    return result, attention_plot

                dec_input = tf.expand_dims([predicted_id], 0)

            attention_plot = attention_plot[: len(result), :]
            return result, attention_plot

    def plot_attention(self, image, result, attention_plot):
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for i in range(len_result):
            temp_att = np.resize(attention_plot[i], (8, 8))
            grid_size = max(np.ceil(len_result / 2), 2)
            ax = fig.add_subplot(grid_size, grid_size, i + 1)
            ax.set_title(result[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()
