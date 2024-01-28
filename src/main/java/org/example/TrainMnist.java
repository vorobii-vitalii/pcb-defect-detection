package org.example;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.function.Function;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.google.gson.reflect.TypeToken;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;
import ai.djl.util.Utils;

public final class TrainMnist {

	static class Mlp extends SequentialBlock {

		/**
		 * Create an MLP NeuralNetwork using RELU.
		 *
		 * @param input the size of the input vector
		 * @param output the size of the output vector
		 * @param hidden the sizes of all of the hidden layers
		 */
		public Mlp(int input, int output, int[] hidden) {
			this(input, output, hidden, Activation::relu);
		}

		/**
		 * Create an MLP NeuralNetwork.
		 *
		 * @param input the size of the input vector
		 * @param output the size of the output vector
		 * @param hidden the sizes of all of the hidden layers
		 * @param activation the activation function to use
		 */
		@SuppressWarnings("this-escape")
		public Mlp(int input, int output, int[] hidden, Function<NDList, NDList> activation) {
			add(Blocks.batchFlattenBlock(input));
			for (int hiddenSize : hidden) {
				add(Linear.builder().setUnits(hiddenSize).build());
				add(activation);
			}

			add(Linear.builder().setUnits(output).build());
		}
	}

	private TrainMnist() {}

	public static void main(String[] args) throws IOException, TranslateException {
		TrainMnist.runExample(args);
	}

	public static void runExample(String[] args) throws IOException, TranslateException {
		Arguments arguments = new Arguments().parseArgs(args);
		if (arguments == null) {
			return;
		}
		// Construct neural network
		Block block =
				new Mlp(
						Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
						Mnist.NUM_CLASSES,
						new int[] {128, 64});

		try (Model model = Model.newInstance("mlp")) {
			model.setBlock(block);

			// get training and validation dataset
			RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
			RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

			// setup training configuration
			DefaultTrainingConfig config = setupTrainingConfig(arguments);

			try (Trainer trainer = model.newTrainer(config)) {
				trainer.setMetrics(new Metrics());

				/*
				 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
				 * 1st axis is batch axis, we can use 1 for initialization.
				 */
				Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH);

				// initialize trainer with proper input shape
				trainer.initialize(inputShape);

				EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

				trainer.getTrainingResult();
			}
		}
	}

	private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
		String outputDir = arguments.getOutputDir();
		SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
		listener.setSaveModelCallback(
				trainer -> {
					TrainingResult result = trainer.getTrainingResult();
					Model model = trainer.getModel();
					float accuracy = result.getValidateEvaluation("Accuracy");
					model.setProperty("Accuracy", String.format("%.5f", accuracy));
					model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
				});
		return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
				.addEvaluator(new Accuracy())
				.optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
				.addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
				.addTrainingListeners(listener);
	}

	private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
			throws IOException {
		Mnist mnist =
				Mnist.builder()
						.optUsage(usage)
						.setSampling(arguments.getBatchSize(), true)
						.optLimit(arguments.getLimit())
						.build();
		mnist.prepare(new ProgressBar());
		return mnist;
	}

	static class Arguments {

		protected int epoch;
		protected int batchSize;
		protected int maxGpus;
		protected boolean isSymbolic;
		protected boolean preTrained;
		protected String outputDir;
		protected long limit;
		protected String modelDir;
		protected Map<String, String> criteria;

		protected void initialize() {
			epoch = 2;
			maxGpus = Engine.getInstance().getGpuCount();
			outputDir = "build/model";
			limit = Long.MAX_VALUE;
			modelDir = null;
		}

		protected void setCmd(CommandLine cmd) {
			if (cmd.hasOption("epoch")) {
				epoch = Integer.parseInt(cmd.getOptionValue("epoch"));
			}
			if (cmd.hasOption("max-gpus")) {
				maxGpus = Math.min(Integer.parseInt(cmd.getOptionValue("max-gpus")), maxGpus);
			}
			if (cmd.hasOption("batch-size")) {
				batchSize = Integer.parseInt(cmd.getOptionValue("batch-size"));
			} else {
				batchSize = maxGpus > 0 ? 32 * maxGpus : 32;
			}
			isSymbolic = cmd.hasOption("symbolic-model");
			preTrained = cmd.hasOption("pre-trained");

			if (cmd.hasOption("output-dir")) {
				outputDir = cmd.getOptionValue("output-dir");
			}
			if (cmd.hasOption("max-batches")) {
				limit = Long.parseLong(cmd.getOptionValue("max-batches")) * batchSize;
			}
			if (cmd.hasOption("model-dir")) {
				modelDir = cmd.getOptionValue("model-dir");
			}
			if (cmd.hasOption("criteria")) {
				Type type = new TypeToken<Map<String, Object>>() {}.getType();
				criteria = JsonUtils.GSON.fromJson(cmd.getOptionValue("criteria"), type);
			}
		}

		public Arguments parseArgs(String[] args) {
			initialize();
			Options options = getOptions();
			try {
				DefaultParser parser = new DefaultParser();
				CommandLine cmd = parser.parse(options, args, null, false);
				if (cmd.hasOption("help")) {
					printHelp("./gradlew run --args='[OPTIONS]'", options);
					return null;
				}
				setCmd(cmd);
				return this;
			} catch (ParseException e) {
				printHelp("./gradlew run --args='[OPTIONS]'", options);
			}
			return null;
		}

		public Options getOptions() {
			Options options = new Options();
			options.addOption(
					Option.builder("h").longOpt("help").hasArg(false).desc("Print this help.").build());
			options.addOption(
					Option.builder("e")
							.longOpt("epoch")
							.hasArg()
							.argName("EPOCH")
							.desc("Numbers of epochs user would like to run")
							.build());
			options.addOption(
					Option.builder("b")
							.longOpt("batch-size")
							.hasArg()
							.argName("BATCH-SIZE")
							.desc("The batch size of the training data.")
							.build());
			options.addOption(
					Option.builder("g")
							.longOpt("max-gpus")
							.hasArg()
							.argName("MAXGPUS")
							.desc("Max number of GPUs to use for training")
							.build());
			options.addOption(
					Option.builder("s")
							.longOpt("symbolic-model")
							.argName("SYMBOLIC")
							.desc("Use symbolic model, use imperative model if false")
							.build());
			options.addOption(
					Option.builder("p")
							.longOpt("pre-trained")
							.argName("PRE-TRAINED")
							.desc("Use pre-trained weights")
							.build());
			options.addOption(
					Option.builder("o")
							.longOpt("output-dir")
							.hasArg()
							.argName("OUTPUT-DIR")
							.desc("Use output to determine directory to save your model parameters")
							.build());
			options.addOption(
					Option.builder("m")
							.longOpt("max-batches")
							.hasArg()
							.argName("max-batches")
							.desc(
									"Limit each epoch to a fixed number of iterations to test the"
											+ " training script")
							.build());
			options.addOption(
					Option.builder("d")
							.longOpt("model-dir")
							.hasArg()
							.argName("MODEL-DIR")
							.desc("pre-trained model file directory")
							.build());
			options.addOption(
					Option.builder("r")
							.longOpt("criteria")
							.hasArg()
							.argName("CRITERIA")
							.desc("The criteria used for the model.")
							.build());
			return options;
		}

		public int getBatchSize() {
			return batchSize;
		}

		public int getEpoch() {
			return epoch;
		}

		public int getMaxGpus() {
			return maxGpus;
		}

		public boolean isSymbolic() {
			return isSymbolic;
		}

		public boolean isPreTrained() {
			return preTrained;
		}

		public String getModelDir() {
			return modelDir;
		}

		public String getOutputDir() {
			return outputDir;
		}

		public long getLimit() {
			return limit;
		}

		public Map<String, String> getCriteria() {
			return criteria;
		}

		private void printHelp(String msg, Options options) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.setLeftPadding(1);
			formatter.setWidth(120);
			formatter.printHelp(msg, options);
		}
	}

	public static final class Mnist extends ArrayDataset {

		private static final String ARTIFACT_ID = "mnist";
		private static final String VERSION = "1.0";

		public static final int IMAGE_WIDTH = 28;
		public static final int IMAGE_HEIGHT = 28;
		public static final int NUM_CLASSES = 10;

		private NDManager manager;
		private Usage usage;

		private MRL mrl;
		private boolean prepared;

		private Mnist(Builder builder) {
			super(builder);
			this.manager = builder.manager;
			this.manager.setName("mnist");
			this.usage = builder.usage;
			mrl = builder.getMrl();
		}

		/**
		 * Creates a builder to build a {@link Mnist}.
		 *
		 * @return a new builder
		 */
		public static Builder builder() {
			return new Builder();
		}

		/** {@inheritDoc} */
		@Override
		public void prepare(Progress progress) throws IOException {
			if (prepared) {
				return;
			}

			Artifact artifact = mrl.getDefaultArtifact();
			mrl.prepare(artifact, progress);

			Map<String, Artifact.Item> map = artifact.getFiles();
			Artifact.Item imageItem;
			Artifact.Item labelItem;
			switch (usage) {
				case TRAIN:
					imageItem = map.get("train_data");
					labelItem = map.get("train_labels");
					break;
				case TEST:
					imageItem = map.get("test_data");
					labelItem = map.get("test_labels");
					break;
				case VALIDATION:
				default:
					throw new UnsupportedOperationException("Validation data not available.");
			}
			labels = new NDArray[] {readLabel(labelItem)};
			data = new NDArray[] {readData(imageItem, labels[0].size())};
			prepared = true;
		}

		private NDArray readData(Artifact.Item item, long length) throws IOException {
			try (InputStream is = mrl.getRepository().openStream(item, null)) {
				if (is.skip(16) != 16) {
					throw new AssertionError("Failed skip data.");
				}

				byte[] buf = Utils.toByteArray(is);
				try (NDArray array =
						manager.create(
								ByteBuffer.wrap(buf), new Shape(length, 28, 28, 1), DataType.UINT8)) {
					return array.toType(DataType.FLOAT32, false);
				}
			}
		}

		private NDArray readLabel(Artifact.Item item) throws IOException {
			try (InputStream is = mrl.getRepository().openStream(item, null)) {
				if (is.skip(8) != 8) {
					throw new AssertionError("Failed skip data.");
				}
				byte[] buf = Utils.toByteArray(is);
				try (NDArray array =
						manager.create(ByteBuffer.wrap(buf), new Shape(buf.length), DataType.UINT8)) {
					return array.toType(DataType.FLOAT32, false);
				}
			}
		}

		/** A builder for a {@link Mnist}. */
		public static final class Builder extends RandomAccessDataset.BaseBuilder<Builder> {

			private NDManager manager;
			private Repository repository;
			private String groupId;
			private String artifactId;
			private Dataset.Usage usage;

			interface BasicDatasets {

				String DJL_REPO_URL = "https://mlrepo.djl.ai/";

				Repository REPOSITORY = Repository.newInstance("BasicDataset", DJL_REPO_URL);

				String GROUP_ID = "ai.djl.basicdataset";
			}

			/** Constructs a new builder. */
			Builder() {
				repository = BasicDatasets.REPOSITORY;
				groupId = BasicDatasets.GROUP_ID;
				artifactId = ARTIFACT_ID;
				usage = Dataset.Usage.TRAIN;
				pipeline = new Pipeline(new ToTensor());
				manager = Engine.getInstance().newBaseManager();
			}

			/** {@inheritDoc} */
			@Override
			protected Builder self() {
				return this;
			}

			/**
			 * Sets the optional manager for the dataset (default follows engine default).
			 *
			 * @param manager the manager
			 * @return this builder
			 */
			public Builder optManager(NDManager manager) {
				this.manager.close();
				this.manager = manager.newSubManager();
				return this;
			}

			/**
			 * Sets the optional repository.
			 *
			 * @param repository the repository
			 * @return this builder
			 */
			public Builder optRepository(Repository repository) {
				this.repository = repository;
				return this;
			}

			/**
			 * Sets optional groupId.
			 *
			 * @param groupId the groupId}
			 * @return this builder
			 */
			public Builder optGroupId(String groupId) {
				this.groupId = groupId;
				return this;
			}

			/**
			 * Sets the optional artifactId.
			 *
			 * @param artifactId the artifactId
			 * @return this builder
			 */
			public Builder optArtifactId(String artifactId) {
				if (artifactId.contains(":")) {
					String[] tokens = artifactId.split(":");
					groupId = tokens[0];
					this.artifactId = tokens[1];
				} else {
					this.artifactId = artifactId;
				}
				return this;
			}

			/**
			 * Sets the optional usage.
			 *
			 * @param usage the usage
			 * @return this builder
			 */
			public Builder optUsage(Usage usage) {
				this.usage = usage;
				return this;
			}

			/**
			 * Builds the {@link Mnist}.
			 *
			 * @return the {@link Mnist}
			 */
			public Mnist build() {
				return new Mnist(this);
			}

			MRL getMrl() {
				return repository.dataset(Application.CV.ANY, groupId, artifactId, VERSION);
			}
		}
	}

}
