from src.loss import loss_function
from src.utils import load_image
import time
import tensorflow as tf
import numpy as np


class Trainer:
    def __init__(self, checkpoint_path, train_config, encoder, decoder, optimizer, tokenizer) -> None:
        self.checkpoint_path = checkpoint_path
        self.train_config = train_config
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.tokenizer = tokenizer

    def set_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,decoder=self.decoder,optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_path, max_to_keep=5
        )

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])
        dec_input = tf.expand_dims(
            [self.tokenizer.word_index["<start>"]] * target.shape[0], 1
        )
        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def train(
        self,
        train_dataset,
        num_epochs,
        num_steps,
    ):
        loss_plot = []
        start_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            start_epoch = int(self.ckpt_manager.latest_checkpoint.split("-")[-1])
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    print(
                        f"Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}"
                    )
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / num_steps)

            if epoch % 5 == 0:
                self.ckpt_manager.save()

            print(f"Epoch {epoch+1} Loss {total_loss/num_steps:.6f}")
            print(f"Time taken for 1 epoch {time.time()-start:.2f} sec\n")

    def eval_single(self, model, cap_val, img_name_val, visualise=True):
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        caption = cap_val[rid]
        real_caption = " ".join(
            [model.tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]]
        )

        result, attn_plot = model(
            (image, caption),
            mode="eval",
            max_length=self.train_config["max_length"],
            attn_shape=self.train_config["attn_shape"],
        )
        print("Real Caption:", real_caption)
        print("Prediction Caption:", " ".join(result))
        if visualise:
            model.plot_attention(image, result, attn_plot)
