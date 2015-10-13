package edu.cuhk.hccl;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

public class AmazonWordEmbedding extends WordEmbedding {

	private static final String SPACE = " ";

	public AmazonWordEmbedding(String modelPath) {
		super(modelPath);
	}

	@Override
	public void loadWordVectors() throws IOException {

		LineIterator it = FileUtils.lineIterator(new File(modelPath), "UTF-8");
		it.nextLine(); // Skip the first line

		int vectorSize = 50; // default
		try {
			while (it.hasNext()) {
				String line = it.nextLine().trim();
				int split = line.indexOf(SPACE);
				String word = line.substring(0, split);
				String vecStr = line.substring(split + 1);

				String[] vectors = vecStr.split(SPACE);
				float[] embedding = new float[vectors.length];
				vectorSize = vectors.length;
				
				for (int i = 0; i < vectors.length; i++)
					embedding[i] = Float.parseFloat(vectors[i]);

				wordVecMap.put(word, embedding);
			}
			
			// Uniform random generator [0,1]
			Random random = new Random(1234567890);
			float[] unknown = new float[vectorSize];
			for (int m = 0; m < vectorSize; m++)
				unknown[m] = random.nextFloat();
			
			float[] padding = new float[vectorSize];
			for (int n = 0; n < vectorSize; n++)
				unknown[n] = random.nextFloat();
			
			wordVecMap.put(Utility.UNKNOWN, unknown);
			wordVecMap.put(Utility.PADDING, padding);
			
		} finally {
			LineIterator.closeQuietly(it);
		}

	}
}