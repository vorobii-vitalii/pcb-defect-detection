package org.example;

import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

public class PCBModel extends SequentialBlock {
	public static final int WIDTH = 300;
	public static final int HEIGHT = 300;
	public static final int CLASSES = 6;
	public static final int BATCH_SIZE = 50;

	public PCBModel() {
		add(Blocks.batchFlattenBlock((long) WIDTH * HEIGHT ));
		add(Linear.builder().setUnits(WIDTH).build());
		add(Activation::sigmoid);
		add(Linear.builder().setUnits(CLASSES).build());
	}

}
