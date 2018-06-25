# A Structured Self-attentive Sentence Embedding

Tensorflow Implementation of "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)" (ICLR 2017).

![image](https://user-images.githubusercontent.com/15166794/41864478-21cbf7c8-78e5-11e8-94d2-5aa035a65c8b.png)


## Usage

### Data
* AG's news topic classification dataset.
* The csv files (in my [data directory](https://github.com/roomylee/self-attention-tf/tree/master/data)) were available from [here](https://github.com/mhjabreel/CharCNN/tree/master/data/ag_news_csv).

### Train
* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.
* Display help message:
	```bash
	$ python train.py --help
	```

	```bash
	train.py:
		--[no]allow_soft_placement: Allow device soft device placement
			(default: 'true')
		--batch_size: Batch Size
			(default: '64')
			(an integer)
		--checkpoint_every: Save model after this many steps
			(default: '100')
			(an integer)
		--d_a_size: Size of W_s1 embedding
			(default: '350')
			(an integer)
		--dev_sample_percentage: Percentage of the training data to use for validation
			(default: '0.1')
			(a number)
		--display_every: Number of iterations to display training info.
			(default: '10')
			(an integer)
		--embedding_dim: Dimensionality of word embedding
			(default: '300')
			(an integer)
		--evaluate_every: Evaluate model on dev set after this many steps
			(default: '100')
			(an integer)
		--fc_size: Size of fully connected layer
			(default: '2000')
			(an integer)
		--hidden_size: Size of LSTM hidden layer
			(default: '256')
			(an integer)
		--learning_rate: Which learning rate to start with.
			(default: '0.001')
			(a number)
		--[no]log_device_placement: Log placement of ops on devices
			(default: 'false')
		--max_sentence_length: Max sentence length in train/test data
			(default: '50')
			(an integer)
		--num_checkpoints: Number of checkpoints to store
			(default: '5')
			(an integer)
		--num_epochs: Number of training epochs
			(default: '10')
			(an integer)
		--p_coef: Coefficient for penalty
			(default: '1.0')
			(a number)
		--r_size: Size of W_s2 embedding
			(default: '30')
			(an integer)
		--train_dir: Path of train data
			(default: 'data/train.csv')
		--word2vec: Word2vec file with pre-trained embeddings
	```

* **Train Example (with word2vec):**
    ```bash
	$ python train.py --word2vec "GoogleNews-vectors-negative300.bin"
	```

### Evalutation
* You must give "**checkpoint_dir**" argument, path of checkpoint(trained neural model) file, like below example.

* **Evaluation Example:**
	```bash
	$ python eval.py --checkpoint_dir "runs/1523902663/checkpoints/"
	```

## Results
* **Accuracy for test data = 0.920789**


## Reference
* A Structured Self-attentive Sentence Embedding (ICLR 2017), Z Lin et al. [[paper]](https://arxiv.org/abs/1703.03130)
* flrngel's [Self-Attentive-tensorflow](https://github.com/flrngel/Self-Attentive-tensorflow) github repository
