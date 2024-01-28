package org.example;

import java.util.Arrays;
import java.util.function.Function;

import javax.swing.ToolTipManager;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

public class ToHistogramTransformer implements Transform {
	private static final Logger LOGGER = LoggerFactory.getLogger(ToHistogramTransformer.class);
	private final int newMax;

	public ToHistogramTransformer(int newMax) {
		this.newMax = newMax;
	}

	public ToHistogramTransformer() {
		this(16);
	}

	@Override
	public NDArray transform(NDArray array) {
		var converted = array.toType(DataType.FLOAT64, true);
		var toYuvTransformMatrix = array.getManager()
				.create(new double[][] {
						{0.29900},
						{0.58700},
						{0.114001}
				});
		var afterYuvTransform = converted.dot(toYuvTransformMatrix);
		LOGGER.info("After YUV transform shape = {}", afterYuvTransform);
		var maxValue = afterYuvTransform.max();
		LOGGER.info("Max value = {}", Arrays.toString(maxValue.toArray()));
		var normalized = afterYuvTransform.div(maxValue);
		LOGGER.info("Normalized values = {}", normalized.getShape());
		LOGGER.info("New max from normalized = {}", Arrays.toString(normalized.max().toArray()));
		var scaled = normalized.mul(newMax - 1).flatten().toType(DataType.INT64, false);

		var frequencyArray = array.getManager().zeros(new Shape(newMax), DataType.INT32);
		long numPixels = scaled.size();
		LOGGER.info("Num pixels = {}", numPixels);
		for (long i = 0; i < numPixels; i++) {
			var index = new NDIndex(scaled.getLong(i));
			frequencyArray.set(index, ndArray -> ndArray.addi(1));
		}
		LOGGER.info("finished calculation");
		LOGGER.info("Result histogram = {}", frequencyArray);

		// toType(DataType.UINT32, true)

		return array;
	}
}
