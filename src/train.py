from src.utils import load_image
import time
import tensorflow as tf


class Trainer:
    def __init__(self, checkpoint_path, train_config) -> None:
        self.checkpoint_path = checkpoint_path
        self.train_config = train_config

    def set_checkpoint(self, model):
        ckpt = tf.train.Checkpoint(model=model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, self.checkpoint_path, max_to_keep=5
        )
        return ckpt, ckpt_manager

    @tf.function
    def train_step(self, model, img_tensor, target):
        with tf.GradientTape() as tape:
            loss, _ = model(img_tensor, target)

        total_loss = loss / int(target.shape[1])

        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        model.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss, total_loss

    def train(
        self,
        model,
        train_dataset,
        ckpt,
        ckpt_manager,
        num_epochs,
        num_steps,
    ):
        loss_plot = []
        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)
        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(train_dataset):
                batch_loss, t_loss = self.train_step(model, img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    print(
                        f"Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}"
                    )
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / num_steps)

            if epoch % 5 == 0:
                ckpt_manager.save()

            print(f"Epoch {epoch+1} Loss {total_loss/num_steps:.6f}")
            print(f"Time taken for 1 epoch {time.time()-start:.2f} sec\n")

    def eval(self, model, test_dataset):
        for (batch, (img_tensor, target)) in enumerate(test_dataset):
            real_caption = ' '.join([model.tokenizer.index_word[i]
                                    for i in cap_val[rid] if i not in [0]])
            result, attn_plot = model(
                target,
                mode="eval",
                max_length=self.train_config["max_length"],
                attn_shape=self.train_config["attn_shape"],
            )

            print('Real Caption:', real_caption)
            print('Prediction Caption:', ' '.join(result))
            plot_attention(image, result, attention_plot)

    def eval_single(self, model,cap_val,img_name_val,visualise=True):
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([model.tokenizer.index_word[i]
                        for i in cap_val[rid] if i not in [0]])
        result, attn_plot = model(
            image,
            mode="eval",
            max_length=self.train_config["max_length"],
            attn_shape=self.train_config["attn_shape"],
        )
        print('Real Caption:', real_caption)
        print('Prediction Caption:', ' '.join(result))
        if visualise:
            model.plot_attention(image, result, attn_plot)