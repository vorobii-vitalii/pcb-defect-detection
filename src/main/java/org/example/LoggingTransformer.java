package org.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;

public class LoggingTransformer implements Transform {
	private static final Logger LOGGER = LoggerFactory.getLogger(LoggingTransformer.class);
	private final String logPrefix;

	public LoggingTransformer(String logPrefix) {
		this.logPrefix = logPrefix;
	}

	@Override
	public NDArray transform(NDArray array) {
		LOGGER.info("[{}] Shape = {}", logPrefix, array.getShape());
		return array;
	}
}
