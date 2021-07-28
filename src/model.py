from src.utils import load_image
import numpy as np
from src.loss import loss_function
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import load_image


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden,training=True):
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

    def call(self, x, training=True):
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

    def call(self, x, features, hidden, training=True):
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

class ImageCaptioner(tf.Module):
    def __init__(self, encoder, decoder , tokenizer):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.int32, shape=[1,64,2048])])
    def caption(self, img_tensor, max_length=15, attn_shape=64):
        hidden = self.decoder.reset_state(batch_size=1)
        features = self.encoder(img_tensor)

        dec_input = tf.expand_dims([self.tokenizer.word_index["<start>"]], 0)

        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(
                dec_input, features, hidden
            )
            predicted_id = tf.random.categorical(predictions, 1)[0][0]
            if i == 0:
                attention_plot = tf.reshape(attention_weights, (-1,))
                result = predicted_id
            else:
                attention_plot = tf.stack([attention_plot, tf.reshape(attention_weights, (-1,))],-1)
                result = tf.stack([result,predicted_id],-1)

            # if self.tokenizer.index_word[predicted_id] == "<end>":
            #     return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[: len(result)]
        return {"predictions":result, "attention_plot":attention_plot}

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
