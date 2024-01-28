package org.example;

import java.io.IOException;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class PCBDefectDetectionModelTrain {
	private static final Logger LOGGER = LoggerFactory.getLogger(PCBDefectDetectionModelTrain.class);

	public static final int BATCH_SIZE = 50;
	public static final String LOCATION = "/work/pcb-defect-detection-histogram-approach/datasets/PCB_DATASET/images";
	public static final String PCB_DATASET = "folder";
	public static final String MODEL_NAME = "ml_pcb_defect_detection";
	public static final int NUM_EPOCH = 2;
	public static final double SPLIT_PERCENT = 0.9;

	public static void main(String[] args) throws TranslateException, IOException {
		try (var model = Model.newInstance(MODEL_NAME)) {
			Block resNet50 =
					ResNetV1.builder() // construct the network
							.setImageShape(new Shape(3, PCBModel.HEIGHT, PCBModel.WIDTH))
							.setNumLayers(18)
							.setOutSize(PCBModel.CLASSES)
							.build();

			// set the neural network to the model
			model.setBlock(resNet50);

//			model.setBlock(new PCBModel());
			var dataset = getDataset();
			RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

			var trainingConfig = getTrainingConfig();
			try (Trainer trainer = model.newTrainer(trainingConfig)) {
				trainer.setMetrics(new Metrics());

				Shape inputShape = new Shape(1, 3, PCBModel.WIDTH, PCBModel.HEIGHT);

				// initialize trainer with proper input shape
				trainer.initialize(inputShape);

				EasyTrain.fit(trainer, NUM_EPOCH, datasets[0], datasets[1]);

				trainer.getTrainingResult();
			}
		}

	}

	private static DefaultTrainingConfig getTrainingConfig() {
		String outputDir = "build/model-pcb";
		SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
		listener.setSaveModelCallback(
				trainer -> {
					var result = trainer.getTrainingResult();
					var model = trainer.getModel();
					var accuracy = result.getValidateEvaluation("Accuracy");
					model.setProperty("Accuracy", String.format("%.5f", accuracy));
					model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
				});
		return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
				.optOptimizer(Optimizer.adadelta().build())
				.addEvaluator(new Accuracy())
				.optDevices(Engine.getInstance().getDevices(0))
				.addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
				.addTrainingListeners(listener);
	}

	private static ImageFolder getDataset() throws IOException, TranslateException {
		var repository = Repository.newInstance(PCB_DATASET, Paths.get(LOCATION));
		var dataset = ImageFolder.builder()
//				.optFlag(Image.Flag.COLOR)
				.setRepository(repository)
				.addTransform(new Resize(PCBModel.WIDTH, PCBModel.HEIGHT))
				.addTransform(new LoggingTransformer("Before: "))
				.addTransform(new ToHistogramTransformer())
				.addTransform(new LoggingTransformer("After resize: "))
				.addTransform(new ToTensor())
				.addTransform(new LoggingTransformer("After toTensor"))
				.setSampling(BATCH_SIZE, true, true)
				.build();
		// call prepare before using
		dataset.prepare(new ProgressBar());

		LOGGER.info("Classes = {}", dataset.getSynset());
		return dataset;
	}

}
